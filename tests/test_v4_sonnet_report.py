"""Tests for v4 Sonnet report join + PDF on synthetic fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mmbu.v4_open_rejudge import assign_partition, load_v4_index_sets
from mmbu.v4_sonnet_report.config import V4SonnetReportConfig
from mmbu.v4_sonnet_report.join_scores import build_joined_scores, coverage_table
from mmbu.v4_sonnet_report.stats import compute_partition_stats
from mmbu.v4_sonnet_report.plot_report import render_pdf


@pytest.fixture
def v4_fixture_tree(tmp_path: Path) -> V4SonnetReportConfig:
    mapping = pd.DataFrame(
        [
            {"index": 1, "eval_task_id": "cls_open", "split": "public", "vqa_type": "open", "commercial_use": "ok", "atom_id": "a1", "dataset": "ds_a"},
            {"index": 2, "eval_task_id": "cls_open", "split": "private", "vqa_type": "open", "commercial_use": "ok", "atom_id": "a2", "dataset": "ds_a"},
            {"index": 3, "eval_task_id": "cls_open", "split": "private", "vqa_type": "open", "commercial_use": "prohibited", "atom_id": "a3", "dataset": "ds_b"},
        ]
    )
    map_path = tmp_path / "row_split_mapping.parquet"
    mapping.to_parquet(map_path, index=False)

    model = "gpt-5.6-sol"
    results = tmp_path / "results"
    jsonl_dir = results / model
    jsonl_dir.mkdir(parents=True)
    task = "classification_open_VQA_cot"
    jsonl_path = jsonl_dir / f"{model}_{task}.jsonl"
    with jsonl_path.open("w") as f:
        for idx in (1, 2, 3):
            f.write(json.dumps({"index": idx, "answer": "melanoma", "question_type": "full"}) + "\n")

    cache_root = tmp_path / "judge" / "open_vqa" / "claude-sonnet-5"
    cache_dir = cache_root / "classification_open_VQA_cot"
    cache_dir.mkdir(parents=True)
    judge_df = pd.DataFrame(
        {
            "index": [1, 2, 3],
            "question_type": ["full", "full", "full"],
            "llm_judge_status": ["ok", "ok", "ok"],
            "llm_judge_correct": [1, 0, 1],
        }
    )
    judge_df.to_csv(cache_dir / f"{model}.csv", index=False)

    cfg = V4SonnetReportConfig(
        results_dir=results,
        split_mapping_path=map_path,
        split_version_dir=tmp_path,
        sonnet_cache_dir=tmp_path / "judge" / "open_vqa",
        qwen_cache_dir=tmp_path / "judge" / "open_vqa",
        models=(model,),
        output_dir=tmp_path / "out",
        cache_path=tmp_path / "joined.parquet",
        stats_cache_path=tmp_path / "stats.csv",
        n_bootstrap=50,
    )
    return cfg


def test_assign_partition():
    df = pd.DataFrame(
        [
            {"split": "public", "vqa_type": "open", "commercial_use": "ok"},
            {"split": "private", "vqa_type": "open", "commercial_use": "ok"},
            {"split": "private", "vqa_type": "open", "commercial_use": "prohibited"},
        ]
    )
    parts = assign_partition(df)
    assert list(parts) == ["public", "private_id", "private_ood"]


def test_v4_index_sets_from_fixture(v4_fixture_tree: V4SonnetReportConfig):
    sets = load_v4_index_sets(v4_fixture_tree.split_mapping_path)
    assert sets["classification_open_VQA_cot"] == {"1", "2", "3"}


def test_join_and_pdf(v4_fixture_tree: V4SonnetReportConfig):
    pytest.importorskip("matplotlib")
    joined = build_joined_scores(v4_fixture_tree)
    assert len(joined) == 3
    assert joined["row_score"].notna().all()

    mapping = pd.read_parquet(v4_fixture_tree.split_mapping_path)
    cov = coverage_table(joined, mapping)
    assert not cov.empty

    stats = compute_partition_stats(joined, v4_fixture_tree)
    assert len(stats) == 3  # one model × three partitions

    pdf = render_pdf(joined, stats, cov, v4_fixture_tree)
    assert pdf.exists()
    assert pdf.stat().st_size > 1000
