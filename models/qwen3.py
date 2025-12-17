# -------------------------------
# Qwen3 Adapter
# -------------------------------
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from .base import BaseVLMAdapter

class Qwen3Adapter(BaseVLMAdapter):
    def load(self):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            cache_dir=self.cache_dir
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor

    def create_template(self, item):
        """
        Return a single-sample conversation for Qwen2.5-VL.
        Each sample must be a list of dicts (conversation turns).
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": item["question"]},
                ],
            }
        ]

    # def prepare_inputs(self, messages, processor, model):
    #     """
    #     Qwen2_5-VL requires:
    #     - text prompts (strings)
    #     - images as a list of PIL Image objects
    #     """
    #     texts = processor.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     image_inputs, _ = process_vision_info(messages)

    #     inputs = processor(
    #         text=texts,
    #         images=image_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     ).to(model.device)

    #     return inputs

    def prepare_inputs(self, messages_batch, processor, model):
        """
        messages_batch: list of message dicts for each sample.
        """
        texts = [
            processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
            for msgs in messages_batch
        ]
    
        image_inputs = [
            process_vision_info(msgs)[0]
            for msgs in messages_batch
        ]
    
        inputs = processor(
            text=texts,
            images=image_inputs,
            padding="longest",
            return_tensors="pt"
        ).to(model.device)
    
        return inputs

    def infer(self, model, processor, inputs, max_new_tokens):
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        outputs = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return outputs