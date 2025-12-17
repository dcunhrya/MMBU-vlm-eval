import torch
from .base import BaseVLMAdapter
from llava.conversation import conv_templates, Conversation, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.model import LlavaMistralForCausalLM
from PIL import Image
import sys
import os

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

class LlavaMedAdapter(BaseVLMAdapter):
    # "microsoft/llava-med-v1.5-mistral-7b"
    def load(self):
        self.device = "cuda"
        # tokenizer, model, image_processor, _ = load_pretrained_model(
        #     model_path=self.model_name,
        #     model_base=None,
        #     model_name=self.model_name,
        #     device_map="auto",
        #     cache_dir=self.cache_dir
        # )
        kwargs = {}
        kwargs['torch_dtype'] = torch.float16
        kwargs['device_map'] = {"": "cuda"}
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = LlavaMistralForCausalLM.from_pretrained(
            self.model_name,
            # low_cpu_mem_usage=False,
            use_flash_attention_2=False,
            cache_dir = self.cache_dir,
            **kwargs
        )
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=self.device, dtype=torch.float16)
        model.model.mm_projector.to(device=self.device, dtype=torch.float16)
        model.to(device=self.device, dtype=torch.float16)
        image_processor = vision_tower.image_processor
        
        model.eval()
        
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

        return model, None

    def create_template(self, item):
        return item

    def create_template_internal(self, item):
        conv = conv_vicuna_v1.copy()  # always copy so you don’t mutate the base
        conv.append_message(conv.roles[0], ("<image>\n" + item["question"]))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt

    def prepare_inputs(self, messages, processor, model):
        prompts, images = [], []
        for item in messages:
            prompts.append(self.create_template_internal(item))
            img_path = item["image"] if isinstance(item["image"], str) else item["image_path"]
            images.append(Image.open(img_path).convert("RGB"))
        image_tensor = process_images(images, self.image_processor, self.model.config)
        image_tensor = image_tensor.to(dtype=self.model.dtype, device=self.model.device)

        batch_input_ids = []
        for p in prompts:
            enc = tokenizer_image_token(
                p, self.tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            input_ids = enc["input_ids"] if isinstance(enc, dict) else enc
            batch_input_ids.append(input_ids.squeeze(0))
    
        input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(self.model.device)
    
        input_ids = input_ids.to(model.device)
        image_tensor = image_tensor.to(model.device, dtype=model.dtype)
        
        return [input_ids, image_tensor, attention_mask]

    def infer(self, model, processor, inputs, max_new_tokens):
        # answer = output.outputs[0].text
        input_ids, image_tensor, attention_mask = inputs
        with torch.inference_mode():
            out_ids = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        outputs = [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in out_ids]
        return outputs