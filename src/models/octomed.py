# # -------------------------------
# # OctoMed Adapter
# # -------------------------------
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from .base import BaseVLMAdapter

class OctoMedAdapter(BaseVLMAdapter):
    def load(self):
        model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            cache_dir=self.cache_dir
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor

    def create_template(self, item):
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant. Please reason step-by-step, and put your final answer within \\boxed{}."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": item["question"]},
                ],
            }
        ]

    def prepare_inputs(self, messages, processor, model):
        """
        messages_batch: list of message dicts for each sample.
        """
        inputs = processor.apply_chat_template(
        	messages,
        	add_generation_prompt=True,
        	tokenize=True,
        	return_dict=True,
        	return_tensors="pt",
            padding=True,
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