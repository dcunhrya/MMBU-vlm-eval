"""Canonical MMBU filesystem paths.

Override any default with the matching environment variable. New code should
import from here instead of hardcoding pasteur absolute paths.
"""

from __future__ import annotations

import os
from pathlib import Path

# Workspace root: /pasteur/u/rdcunha/code/mmbu (parent of this inference repo)
_DEFAULT_WORKSPACE = Path("/pasteur/u/rdcunha/code/mmbu")

_DEFAULT_DATA_ROOT = Path(
    "/pasteur/u/rdcunha/data_cache/mmbu/final_data/subsampled_mmbu_data"
)
_DEFAULT_RESULTS_DIR = _DEFAULT_WORKSPACE / "results_cot_v3"
_DEFAULT_JUDGE_CACHE = (
    _DEFAULT_WORKSPACE / "src/analysis/finalized_analysis/llm_judge_cache"
)
_DEFAULT_HF_STAGING = Path("/pasteur/u/rdcunha/data_cache/mmbu/mmbu_hf_upload")
_DEFAULT_HF_STAGING_CONTEXT = Path(
    "/pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_hf_upload"
)
_DEFAULT_CANONICAL_IMAGES = Path(
    "/pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_dataset_3_18/images"
)
_DEFAULT_XDG_CACHE = Path("/pasteur/u/rdcunha/.cache")
_DEFAULT_HF_HOME = Path("/pasteur/u/rdcunha/models")
_DEFAULT_UV_CACHE = Path("/pasteur/u/rdcunha/uv_cache")


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def workspace_root() -> Path:
    return _env_path("MMBU_WORKSPACE_ROOT", _DEFAULT_WORKSPACE)


def data_root() -> Path:
    return _env_path("MMBU_DATA_ROOT", _DEFAULT_DATA_ROOT)


def results_dir() -> Path:
    return _env_path("MMBU_RESULTS_DIR", _DEFAULT_RESULTS_DIR)


def judge_cache() -> Path:
    return _env_path("MMBU_JUDGE_CACHE", _DEFAULT_JUDGE_CACHE)


def judge_cache_open() -> Path:
    return judge_cache() / "open_vqa"


def judge_cache_closed() -> Path:
    return judge_cache() / "closed_vqa"


def hf_staging() -> Path:
    return _env_path("MMBU_HF_STAGING", _DEFAULT_HF_STAGING)


def hf_staging_context() -> Path:
    return _env_path("MMBU_HF_STAGING_CONTEXT", _DEFAULT_HF_STAGING_CONTEXT)


def canonical_images() -> Path:
    return _env_path("MMBU_CANONICAL_IMAGES", _DEFAULT_CANONICAL_IMAGES)


def xdg_cache() -> Path:
    return _env_path("XDG_CACHE_HOME", _DEFAULT_XDG_CACHE)


def vllm_cache() -> Path:
    return _env_path("VLLM_CACHE_ROOT", xdg_cache() / "vllm")


def hf_home() -> Path:
    return _env_path("HF_HOME", _DEFAULT_HF_HOME)


def uv_cache() -> Path:
    return _env_path("UV_CACHE_DIR", _DEFAULT_UV_CACHE)


def inference_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_CACHE_DIR_KEYS = (
    "XDG_CACHE_HOME",
    "VLLM_CACHE_ROOT",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "UV_CACHE_DIR",
)


def apply_runtime_cache_env() -> dict[str, str]:
    """Export pasteur cache env vars. Does not overwrite a var already set."""
    mapping = {
        "XDG_CACHE_HOME": str(xdg_cache()),
        "VLLM_CACHE_ROOT": str(vllm_cache()),
        "HF_HOME": str(hf_home()),
        "TRANSFORMERS_CACHE": str(hf_home()),
        "HUGGINGFACE_HUB_CACHE": str(hf_home()),
        "UV_CACHE_DIR": str(uv_cache()),
        "TOKENIZERS_PARALLELISM": "false",
    }
    applied: dict[str, str] = {}
    for key, value in mapping.items():
        os.environ.setdefault(key, value)
        applied[key] = os.environ[key]
        if key in _CACHE_DIR_KEYS:
            Path(applied[key]).mkdir(parents=True, exist_ok=True)
    return applied


def load_tasks_config(path: str | Path | None = None) -> dict:
    """Load configs/tasks.yaml (data_root + task registry)."""
    import yaml

    tasks_path = Path(path) if path else inference_repo_root() / "configs" / "tasks.yaml"
    with tasks_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def task_inference_tsvs(path: str | Path | None = None) -> dict[str, str]:
    payload = load_tasks_config(path)
    return {task["name"]: task["inference_tsv"] for task in payload["tasks"]}
