# -------------------------------
# GEMMA 3 Adapter
# -------------------------------
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from .base import BaseVLMAdapter

class Gemma3Adapter(BaseVLMAdapter):
    def load(self):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map=self.device,
            cache_dir=self.cache_dir
        ).eval()
        processor = AutoProcessor.from_pretrained(self.model_name)

        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"
            
        return model, processor

    def prepare_inputs(self, messages, processor, model):
        # messages is list[list[dict]] from build_messages()
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        return inputs

    def infer(self, model, processor, inputs, max_new_tokens):
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        outputs = []
        for g in generation:
            g = g[input_len:]
            outputs.append(processor.decode(g, skip_special_tokens=True))
        return outputs

    def stack_inputs(self, input_list, model):    
        # Stack text
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [inp["input_ids"].squeeze(0) for inp in input_list],
            batch_first=True, padding_value=0
        )
        attn_mask = torch.nn.utils.rnn.pad_sequence(
            [inp["attention_mask"].squeeze(0) for inp in input_list],
            batch_first=True, padding_value=0
        )
    
        # Stack pixel values
        pixel_values = torch.stack([inp["pixel_values"].squeeze(0) for inp in input_list], dim=0)
    
        return {
            "input_ids": input_ids.to(model.device),
            "attention_mask": attn_mask.to(model.device),
            "pixel_values": pixel_values.to(model.device, dtype=torch.bfloat16)
        }