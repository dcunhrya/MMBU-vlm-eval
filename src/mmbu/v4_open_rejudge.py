"""Shared helpers for v4 open-VQA Sonnet rejudge scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmbu.paths import eval_split_v4_row_mapping

V4_OPEN_JUDGEABLE = frozenset({"cls_open", "fg_det_open", "fg_seg_open"})

V4_EVAL_TO_JUDGE_TASK = {
    "cls_open": "classification_open_VQA_cot",
    "fg_det_open": "detection_grounding_open_VQA_cot",
    "fg_seg_open": "segmentation_grounding_open_VQA_cot",
}

JUDGE_TASK_TO_V4_EVAL = {v: k for k, v in V4_EVAL_TO_JUDGE_TASK.items()}

DEFAULT_V4_MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "claude-sonnet-5",
    "claude-opus-5",
    "Qwen3-VL-32B-Instruct",
    "InternVL3_5-8B",
)

V4_EVAL_TO_LEGACY = {
    "cls_open": "cls_open",
    "fg_det_open": "det_open",
    "fg_seg_open": "seg_open",
    "obj_det_open": "det_bbox_open",
}

LEGACY_TASK_JSONL = {
    "cls_open": "classification_open_VQA_cot",
    "det_open": "detection_grounding_open_VQA_cot",
    "seg_open": "segmentation_grounding_open_VQA_cot",
    "det_bbox_open": "detection_guess_bbox_open_VQA_cot",
}


def require_eval_stack() -> None:
    try:
        import mmbu.judge  # noqa: F401
        import mmbu.eval.llm_judge_open_vqa  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "mmbu eval stack not found. On Pasteur, ensure src/mmbu/eval/ and "
            "src/mmbu/judge.py exist and PYTHONPATH=src.\n"
            f"Import error: {exc}"
        ) from exc


def index_key(raw) -> str:
    if pd.isna(raw):
        return ""
    try:
        f = float(raw)
        if f == int(f):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(raw).strip()


def load_v4_index_sets(mapping_path: Path | None = None) -> dict[str, set[str]]:
    path = Path(mapping_path) if mapping_path else eval_split_v4_row_mapping()
    if not path.exists():
        raise FileNotFoundError(f"v4 row mapping not found: {path}")
    mapping = pd.read_parquet(path)
    mapping["index"] = mapping["index"].astype(int)
    out: dict[str, set[str]] = {}
    for eval_task, judge_task in V4_EVAL_TO_JUDGE_TASK.items():
        subset = mapping[mapping["eval_task_id"].astype(str) == eval_task]
        out[judge_task] = {index_key(v) for v in subset["index"]}
    return out


def load_v4_mapping(mapping_path: Path | None = None) -> pd.DataFrame:
    path = Path(mapping_path) if mapping_path else eval_split_v4_row_mapping()
    if not path.exists():
        raise FileNotFoundError(f"v4 row mapping not found: {path}")
    mapping = pd.read_parquet(path)
    mapping["index"] = mapping["index"].astype(int)
    return mapping


def filter_to_v4(rows: pd.DataFrame, v4_indexes: set[str]) -> pd.DataFrame:
    if rows.empty:
        return rows
    keys = rows["index"].map(index_key)
    return rows[keys.isin(v4_indexes)].copy()


def parse_csv_filter(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def assign_partition(df: pd.DataFrame) -> pd.Series:
    split = df["split"].astype(str)
    vqa = df.get("vqa_type", pd.Series("", index=df.index)).astype(str).str.lower()
    cu = df.get("commercial_use", pd.Series("", index=df.index)).astype(str)

    part = pd.Series("other", index=df.index, dtype="object")
    part[(split == "public") & (vqa == "open")] = "public"
    part[(split == "private") & (cu == "ok")] = "private_id"
    part[(split == "private") & (cu == "prohibited")] = "private_ood"
    return part
