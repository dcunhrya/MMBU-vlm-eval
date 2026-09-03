#!/usr/bin/env python3
"""Retry unparsed Sonnet/Opus open-VQA judge rows for the v4 subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from mmbu.paths import apply_runtime_cache_env, judge_cache_open, results_dir
from mmbu.v4_open_rejudge import (
    DEFAULT_V4_MODELS,
    V4_EVAL_TO_JUDGE_TASK,
    filter_to_v4,
    load_v4_index_sets,
    parse_csv_filter,
    require_eval_stack,
)

OPEN_TASKS = tuple(V4_EVAL_TO_JUDGE_TASK.values())


def run(args: argparse.Namespace) -> int:
    from mmbu.eval.judge_parser import JUDGE_STATUS_OK
    from mmbu.eval.llm_judge_open_vqa import (
        create_judge_for_model,
        judge_dataframe,
        load_cached_judgments,
    )
    from mmbu.judge import build_judge_task_configs, merge_model_results

    v4_sets = load_v4_index_sets(args.mapping_path)
    model_filter = parse_csv_filter(args.model_filter) or set(DEFAULT_V4_MODELS)
    configs = [c for c in build_judge_task_configs(kind="open") if c.task in OPEN_TASKS]

    judge = None
    for config in configs:
        v4_indexes = v4_sets[config.task]
        for model in sorted(model_filter):
            rows, source_file = merge_model_results(args.results_dir, config, model)
            if "question_type" in rows.columns:
                rows = rows[rows["question_type"].astype(str) == "full"].copy()
            rows = filter_to_v4(rows, v4_indexes)
            if rows.empty:
                continue

            cached = load_cached_judgments(
                config.task,
                model,
                cache_dir=args.cache_dir,
                judge_model=args.judge_model,
                prompt_variant=args.prompt_variant,
            )
            if cached.empty or "llm_judge_status" not in cached.columns:
                continue

            bad = cached[cached["llm_judge_status"].astype(str) != JUDGE_STATUS_OK]
            if bad.empty:
                print(f"  {model}/{config.task}: no unparsed rows")
                continue

            retry_indexes = set(bad["index"].astype(str))
            retry_rows = rows[rows["index"].astype(str).isin(retry_indexes)].copy()
            if retry_rows.empty:
                continue

            print(
                f"  {model}/{config.task}: retrying {len(retry_rows)} unparsed v4 row(s)"
            )
            if judge is None:
                judge, _ = create_judge_for_model(
                    args.judge_model,
                    max_new_tokens=args.max_new_tokens,
                )

            judge_dataframe(
                retry_rows,
                task=config.task,
                model=model,
                source_result_file=source_file,
                cache_dir=args.cache_dir,
                judge_model=args.judge_model,
                judge=judge,
                prompt_variant=args.prompt_variant,
                use_gate=args.use_gate,
            )
    return 0


def main() -> int:
    from mmbu.eval.llm_judge_open_vqa import OPEN_PROMPT_VARIANT_PER_DATASET_V3

    apply_runtime_cache_env()
    require_eval_stack()

    parser = argparse.ArgumentParser(description="Retry unparsed v4 Sonnet judge rows")
    parser.add_argument("--judge-model", default="claude-sonnet-5")
    parser.add_argument("--prompt-variant", default=OPEN_PROMPT_VARIANT_PER_DATASET_V3)
    parser.add_argument("--results-dir", default=str(results_dir()))
    parser.add_argument("--cache-dir", default=str(judge_cache_open()))
    parser.add_argument("--mapping-path", type=Path, default=None)
    parser.add_argument("--model-filter", default=",".join(DEFAULT_V4_MODELS))
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--gate/--no-gate",
        dest="use_gate",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
