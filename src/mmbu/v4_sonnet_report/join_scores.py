"""Join v4 mapping + JSONL + Sonnet judge cache into scored rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mmbu.v4_open_rejudge import assign_partition, index_key, load_v4_mapping
from mmbu.v4_sonnet_report.config import (
    IOU_EVAL_TASK,
    LEGACY_TO_JUDGE_TASK,
    OPEN_MACRO_EVAL_TASKS,
    V4SonnetReportConfig,
    V4_EVAL_TO_LEGACY,
    sanitize_name,
)


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_judge_cache(
    cache_root: Path,
    judge_task: str,
    model: str,
) -> pd.DataFrame:
    path = cache_root / sanitize_name(judge_task) / f"{sanitize_name(model)}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "question_type" in df.columns:
        df = df[df["question_type"].astype(str) == "full"].copy()
    return df


def jsonl_path(results_dir: Path, model: str, legacy_task: str) -> Path:
    judge_task = LEGACY_TO_JUDGE_TASK[legacy_task]
    return results_dir / model / f"{model}_{judge_task}.jsonl"


def row_score_from_judge(judge_row: pd.Series | None) -> float:
    if judge_row is None or (isinstance(judge_row, pd.Series) and judge_row.empty):
        return float("nan")
    status = str(judge_row.get("llm_judge_status", "ok"))
    if status not in ("ok", "OK", "Ok"):
        return float("nan")
    val = judge_row.get("llm_judge_correct")
    if pd.isna(val):
        val = judge_row.get("llm_judge_score")
    if pd.isna(val):
        return float("nan")
    try:
        return float(int(val))
    except (TypeError, ValueError):
        return float("nan")


def row_score_from_jsonl(row: pd.Series) -> float:
    for col in ("iou_score", "score", "is_correct?"):
        if col in row.index and pd.notna(row[col]):
            try:
                return float(row[col])
            except (TypeError, ValueError):
                pass
    return float("nan")


def score_model_on_v4(
    model: str,
    config: V4SonnetReportConfig,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    sonnet_root = config.sonnet_judge_cache_root()
    qwen_root = config.qwen_judge_cache_root()
    parts: list[pd.DataFrame] = []

    eval_tasks = list(OPEN_MACRO_EVAL_TASKS) + [IOU_EVAL_TASK]
    for eval_task in eval_tasks:
        legacy = V4_EVAL_TO_LEGACY[eval_task]
        judge_task = LEGACY_TO_JUDGE_TASK[legacy]
        v4_subset = mapping[mapping["eval_task_id"].astype(str) == eval_task].copy()
        if v4_subset.empty:
            continue

        path = jsonl_path(config.results_dir, model, legacy)
        if not path.exists():
            continue

        jsonl = read_jsonl(path)
        if jsonl.empty or "index" not in jsonl.columns:
            continue
        if "question_type" in jsonl.columns:
            jsonl = jsonl[jsonl["question_type"].astype(str) == "full"].copy()
        jsonl["_idx_key"] = jsonl["index"].map(index_key)

        v4_subset = v4_subset.copy()
        v4_subset["_idx_key"] = v4_subset["index"].map(index_key)

        merged = v4_subset.merge(
            jsonl,
            on="_idx_key",
            how="inner",
            suffixes=("_v4", "_jsonl"),
        )
        if merged.empty:
            continue

        if eval_task == IOU_EVAL_TASK:
            merged["row_score"] = merged.apply(row_score_from_jsonl, axis=1)
            merged["score_source"] = "iou_jsonl"
            merged["qwen_row_score"] = float("nan")
        else:
            sonnet_cache = load_judge_cache(sonnet_root, judge_task, model)
            qwen_cache = load_judge_cache(qwen_root, judge_task, model)
            sonnet_by_idx = (
                {index_key(r["index"]): r for _, r in sonnet_cache.iterrows()}
                if not sonnet_cache.empty
                else {}
            )
            qwen_by_idx = (
                {index_key(r["index"]): r for _, r in qwen_cache.iterrows()}
                if not qwen_cache.empty
                else {}
            )

            def _sonnet_score(row):
                hit = sonnet_by_idx.get(index_key(row.get("index_v4", row.get("index"))))
                return row_score_from_judge(hit)

            def _qwen_score(row):
                hit = qwen_by_idx.get(index_key(row.get("index_v4", row.get("index"))))
                return row_score_from_judge(hit)

            merged["row_score"] = merged.apply(_sonnet_score, axis=1)
            merged["qwen_row_score"] = merged.apply(_qwen_score, axis=1)
            merged["score_source"] = "sonnet_judge"

        merged["model"] = model
        merged["eval_task_id"] = eval_task
        merged["legacy_task"] = legacy
        merged["partition"] = assign_partition(merged)
        parts.append(merged)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_joined_scores(config: V4SonnetReportConfig | None = None) -> pd.DataFrame:
    config = config or V4SonnetReportConfig.from_env()
    config.ensure_dirs()
    mapping = load_v4_mapping(config.split_mapping_path)
    frames = [score_model_on_v4(model, config, mapping) for model in config.models]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    joined = pd.concat(frames, ignore_index=True)
    joined.to_parquet(config.cache_path, index=False)
    return joined


def open_scored_subset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["eval_task_id"].isin(list(OPEN_MACRO_EVAL_TASKS))
    return df[mask & df["row_score"].notna()].copy()


def coverage_table(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in df["model"].unique() if not df.empty else []:
        for eval_task in list(OPEN_MACRO_EVAL_TASKS) + [IOU_EVAL_TASK]:
            v4_n = int((mapping["eval_task_id"].astype(str) == eval_task).sum())
            scored = df[
                (df["model"] == model) & (df["eval_task_id"] == eval_task)
            ]
            for part in ("public", "private_id", "private_ood"):
                part_v4 = mapping[
                    (mapping["eval_task_id"].astype(str) == eval_task)
                ].copy()
                part_v4["partition"] = assign_partition(part_v4)
                expected = int((part_v4["partition"] == part).sum())
                got = int(
                    (
                        (scored["partition"] == part) & scored["row_score"].notna()
                    ).sum()
                )
                rows.append(
                    {
                        "model": model,
                        "eval_task_id": eval_task,
                        "partition": part,
                        "v4_rows": v4_n,
                        "expected_partition_rows": expected,
                        "scored_rows": got,
                        "coverage_pct": got / expected if expected else float("nan"),
                    }
                )
    return pd.DataFrame(rows)
