import torch
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
from huggingface_hub import snapshot_download
from .base import BaseVLMAdapter

class InternVL35Adapter(BaseVLMAdapter):
    def load(self):
        snapshot_download(
            repo_id=self.model_name,
            local_dir=self.cache_dir,
            local_dir_use_symlinks=False
        )

        engine_cfg = PytorchEngineConfig(
            session_len=32768,
            tp=1,
            dtype="bfloat16"
        )

        pipe = pipeline(
            self.cache_dir, 
            backend_config=engine_cfg,
            device=self.device
        )

        return pipe, None

    def create_template(self, item):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": item["question"]},
                ],
            }
        ]
        return conversation

    def prepare_inputs(self, messages, processor, model):
        prompts = [
            (msg[0]["content"][1]["text"], load_image(msg[0]["content"][0]["image"]))
            for msg in messages
        ]

        return prompts

    def infer(self, pipe, processor, inputs, max_new_tokens):
        """
        LMDeploy pipeline returns objects with a `.text` field.
        """
        with torch.inference_mode():
            outputs = pipe(
                inputs,
                max_new_tokens=max_new_tokens
            )

        decoded = [o.text for o in outputs]
        return decoded