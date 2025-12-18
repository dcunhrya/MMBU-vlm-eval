# from transformers import AutoProcessor
import torch
from .base import BaseVLMAdapter
try:
    from .gemma3 import Gemma3Adapter
    from .qwen2 import Qwen2Adapter
    from .qwen2_5 import Qwen2_5Adapter
    from .medvlm import MedVLM_Adapter
    from .lingshu import Lingshu_Adapter
    from .qwen3 import Qwen3Adapter
    from .internvl3_5 import InternVL35Adapter
    from .llava import LlavaAdapter
except:
    pass

try:
    from .llava_med import LlavaMedAdapter
except:
    pass

try:
    MODEL_REGISTRY = {
        "gemma3": Gemma3Adapter,
        "qwen2vl": Qwen2Adapter,
        "qwen2_5vl": Qwen2_5Adapter,
        "medvlm": MedVLM_Adapter,
        "lingshu": Lingshu_Adapter,
        "qwen3vl": Qwen3Adapter,
        "intern": InternVL35Adapter,
        "llava": LlavaAdapter,
        # "llavamed": LlavaMedAdapter,
    }
except:
    MODEL_REGISTRY = {
    # "gemma3": Gemma3Adapter,
    # "qwen2vl": Qwen2Adapter,
    # "qwen2_5vl": Qwen2_5Adapter,
    # "medvlm": MedVLM_Adapter,
    # "lingshu": Lingshu_Adapter,
    # "qwen3vl": Qwen3Adapter,
    # "intern": InternVL35Adapter,
    # "llava": LlavaAdapter,
    "llavamed": LlavaMedAdapter,
}

def load_model_adapter(model_type, model_name, device, cache_dir):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](model_name, device, cache_dir)