#!/usr/bin/env python3
"""Rejudge v4 public/private open-VQA rows with Sonnet (open_per_dataset_v3).

Filters the 8-task JSONL merge to rows present in the frozen v4 split before calling
the standard open judge. Does not judge the full benchmark.

Requires the full Pasteur eval tree (``src/mmbu/eval/``, ``src/mmbu/judge.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from mmbu.paths import apply_runtime_cache_env, judge_cache_open, results_dir
from mmbu.v4_open_rejudge import (
    DEFAULT_V4_MODELS,
    JUDGE_TASK_TO_V4_EVAL,
    filter_to_v4,
    load_v4_index_sets,
    parse_csv_filter,
    require_eval_stack,
)


def emit_status(args: argparse.Namespace) -> None:
    from mmbu.judge import build_judge_task_configs, get_judgment_progress, merge_model_results

    v4_sets = load_v4_index_sets(args.mapping_path)
    task_filter = parse_csv_filter(args.task_filter)
    model_filter = parse_csv_filter(args.model_filter) or set(DEFAULT_V4_MODELS)

    configs = build_judge_task_configs(kind="open", task_filter=task_filter)
    configs = [c for c in configs if c.task in v4_sets]

    for config in configs:
        v4_indexes = v4_sets[config.task]
        for model in sorted(model_filter):
            rows, source_file = merge_model_results(args.results_dir, config, model)
            if "question_type" in rows.columns:
                rows = rows[rows["question_type"].astype(str) == "full"].copy()
            rows = filter_to_v4(rows, v4_indexes)
            if rows.empty:
                record = {
                    "task": config.task,
                    "v4_eval_task_id": JUDGE_TASK_TO_V4_EVAL.get(config.task),
                    "model": model,
                    "source_file": source_file,
                    "v4_rows": len(v4_indexes),
                    "merged_rows": 0,
                    "status": "source_missing",
                    "missing": 0,
                    "completed": 0,
                }
                print(json.dumps(record, sort_keys=True))
                continue

            progress = get_judgment_progress(
                rows,
                task=config.task,
                model=model,
                cache_dir=args.cache_dir,
                judge_model=args.judge_model,
                prompt_variant=args.prompt_variant,
            )
            record = {
                "task": config.task,
                "v4_eval_task_id": JUDGE_TASK_TO_V4_EVAL.get(config.task),
                "model": model,
                "source_file": source_file,
                "v4_rows": len(v4_indexes),
                "merged_rows": len(rows),
                "status": "ok",
                **progress,
            }
            print(json.dumps(record, sort_keys=True))


def run_judge(args: argparse.Namespace) -> None:
    from mmbu.eval.llm_judge_open_vqa import (
        create_judge_for_model,
        get_missing_judgment_count,
        judge_dataframe,
    )
    from mmbu.judge import build_judge_task_configs, merge_model_results

    v4_sets = load_v4_index_sets(args.mapping_path)
    task_filter = parse_csv_filter(args.task_filter)
    model_filter = parse_csv_filter(args.model_filter) or set(DEFAULT_V4_MODELS)

    configs = build_judge_task_configs(kind="open", task_filter=task_filter)
    configs = [c for c in configs if c.task in v4_sets]

    os.makedirs(args.cache_dir, exist_ok=True)
    judge = None

    for config in configs:
        v4_indexes = v4_sets[config.task]
        print(
            f"Task {config.task} ({JUDGE_TASK_TO_V4_EVAL.get(config.task)}): "
            f"{len(v4_indexes)} v4 indexes"
        )
        for model in sorted(model_filter):
            rows, source_file = merge_model_results(args.results_dir, config, model)
            if "question_type" in rows.columns:
                rows = rows[rows["question_type"].astype(str) == "full"].copy()
            rows = filter_to_v4(rows, v4_indexes)
            if rows.empty:
                print(f"  {model}: no merged v4 rows, skipping")
                continue

            missing = get_missing_judgment_count(
                rows,
                task=config.task,
                model=model,
                cache_dir=args.cache_dir,
                judge_model=args.judge_model,
                prompt_variant=args.prompt_variant,
            )
            if missing == 0:
                print(f"  {model}: all {len(rows)} v4 row(s) cached, skipping")
                continue

            if judge is None:
                judge, _ = create_judge_for_model(
                    args.judge_model,
                    max_new_tokens=args.max_new_tokens,
                )

            cached, cache_path, newly_judged = judge_dataframe(
                rows,
                task=config.task,
                model=model,
                source_result_file=source_file,
                cache_dir=args.cache_dir,
                judge_model=args.judge_model,
                judge=judge,
                batch_size=args.batch_size,
                max_rows=args.max_rows,
                prompt_variant=args.prompt_variant,
                use_gate=args.use_gate,
            )
            print(
                f"  {model}: judged {newly_judged} new v4 row(s), "
                f"{len(cached)} total at {cache_path}"
            )


def build_parser() -> argparse.ArgumentParser:
    from mmbu.eval.llm_judge_open_vqa import OPEN_PROMPT_VARIANT_PER_DATASET_V3

    parser = argparse.ArgumentParser(description="Sonnet v4 open-VQA rejudge (subset)")
    parser.add_argument("--judge-model", default="claude-sonnet-5")
    parser.add_argument(
        "--prompt-variant",
        default=OPEN_PROMPT_VARIANT_PER_DATASET_V3,
        choices=("legacy", OPEN_PROMPT_VARIANT_PER_DATASET_V3),
    )
    parser.add_argument("--results-dir", default=str(results_dir()))
    parser.add_argument("--cache-dir", default=str(judge_cache_open()))
    parser.add_argument("--mapping-path", type=Path, default=None)
    parser.add_argument("--model-filter", default=",".join(DEFAULT_V4_MODELS))
    parser.add_argument("--task-filter", default=None)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--status-json", action="store_true")
    parser.add_argument(
        "--gate/--no-gate",
        dest="use_gate",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    apply_runtime_cache_env()
    require_eval_stack()
    args = build_parser().parse_args(argv)
    if args.status_json:
        emit_status(args)
    else:
        run_judge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
