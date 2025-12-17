import torch
from vllm import LLM, SamplingParams
from .base import BaseVLMAdapter

class LlavaAdapter(BaseVLMAdapter):
    # "llava-hf/llava-1.5-7b-hf"
    def load(self):
        llm = LLM(
            hf_overrides= {"cache_dir": self.cache_dir},
            download_dir=self.cache_dir,
            model=self.model_name,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=4096,
        )

        return llm, None

    def create_template(self, item):
        conversation = {
            "prompt": f"USER: <image>\n{item['question']}\nASSISTANT:",
            "multi_modal_data": {"image": item['image']},
        }
        return conversation

    def prepare_inputs(self, messages, processor, model):
        return messages

    def infer(self, llm, processor, inputs, max_new_tokens):
        # answer = output.outputs[0].text
        sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        with torch.inference_mode():
            outputs = llm.generate(inputs, sampling_params=sampling_params)

        decoded = [res.outputs[0].text if res.outputs else "" for res in outputs]
        return decoded