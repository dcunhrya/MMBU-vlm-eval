"""Configuration for v4 Sonnet-judge score PDF."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from mmbu.paths import (
    eval_split_v4_dir,
    eval_split_v4_row_mapping,
    inference_repo_root,
    judge_cache_open,
    results_dir,
    workspace_root,
)

SONNET_V4_MODELS: tuple[str, ...] = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "claude-sonnet-5",
    "claude-opus-5",
    "Qwen3-VL-32B-Instruct",
    "InternVL3_5-8B",
)

MODEL_DISPLAY: dict[str, str] = {
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-opus-5": "Claude Opus 5",
    "Qwen3-VL-32B-Instruct": "Qwen3-VL 32B",
    "InternVL3_5-8B": "InternVL 3.5 8B",
}

OPEN_MACRO_EVAL_TASKS: tuple[str, ...] = (
    "cls_open",
    "fg_det_open",
    "fg_seg_open",
)

IOU_EVAL_TASK = "obj_det_open"

V4_EVAL_TO_LEGACY = {
    "cls_open": "cls_open",
    "fg_det_open": "det_open",
    "fg_seg_open": "seg_open",
    "obj_det_open": "det_bbox_open",
}

LEGACY_TO_JUDGE_TASK = {
    "cls_open": "classification_open_VQA_cot",
    "det_open": "detection_grounding_open_VQA_cot",
    "seg_open": "segmentation_grounding_open_VQA_cot",
    "det_bbox_open": "detection_guess_bbox_open_VQA_cot",
}

PARTITION_COLORS: dict[str, str] = {
    "public": "#E69F00",
    "private_id": "#0072B2",
    "private_ood": "#B19CD9",
}

PARTITION_LABELS: dict[str, str] = {
    "public": "Public (open)",
    "private_id": "Private ID (OK)",
    "private_ood": "Private OOD (license)",
}

EVAL_TASK_LABELS: dict[str, str] = {
    "cls_open": "Classification",
    "fg_det_open": "FG from detection",
    "fg_seg_open": "FG from segmentation",
    "obj_det_open": "Object detection (IoU)",
}

QWEN_JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
SONNET_JUDGE_MODEL = "claude-sonnet-5"


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


@dataclass
class V4SonnetReportConfig:
    results_dir: Path = field(default_factory=results_dir)
    split_mapping_path: Path = field(default_factory=eval_split_v4_row_mapping)
    split_version_dir: Path = field(default_factory=eval_split_v4_dir)
    sonnet_cache_dir: Path = field(default_factory=judge_cache_open)
    qwen_cache_dir: Path = field(default_factory=judge_cache_open)
    sonnet_judge_model: str = SONNET_JUDGE_MODEL
    qwen_judge_model: str = QWEN_JUDGE_MODEL
    models: tuple[str, ...] = SONNET_V4_MODELS
    output_dir: Path = field(
        default_factory=lambda: workspace_root()
        / "src/analysis/finalized_analysis/figures/split_v4_scores"
    )
    cache_path: Path = field(
        default_factory=lambda: workspace_root()
        / "src/analysis/finalized_analysis/split_v4_scores/outputs"
        / "joined_scores_sonnet_judge.parquet"
    )
    stats_cache_path: Path = field(
        default_factory=lambda: workspace_root()
        / "src/analysis/finalized_analysis/split_v4_scores/outputs"
        / "partition_stats_sonnet_judge.csv"
    )
    n_bootstrap: int = 500
    bootstrap_seed: int = 42

    @property
    def pdf_path(self) -> Path:
        return self.output_dir / "public_private_scores_v4_sonnet_judge.pdf"

    def sonnet_judge_cache_root(self) -> Path:
        return self.sonnet_cache_dir / sanitize_name(self.sonnet_judge_model)

    def qwen_judge_cache_root(self) -> Path:
        return self.qwen_cache_dir / sanitize_name(self.qwen_judge_model)

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "V4SonnetReportConfig":
        cfg = cls()
        if raw := os.environ.get("MMBU_RESULTS_DIR"):
            cfg.results_dir = Path(raw)
        if raw := os.environ.get("MMBU_SPLIT_MAPPING"):
            cfg.split_mapping_path = Path(raw)
        if raw := os.environ.get("MMBU_V4_REPORT_OUTPUT"):
            cfg.output_dir = Path(raw)
        return cfg
