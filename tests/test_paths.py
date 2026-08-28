from pathlib import Path

import yaml

from mmbu.paths import (
    data_root,
    hf_home,
    inference_repo_root,
    judge_cache,
    load_tasks_config,
    results_dir,
    task_inference_tsvs,
    xdg_cache,
)


def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("MMBU_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("MMBU_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("MMBU_JUDGE_CACHE", str(tmp_path / "judge"))
    assert data_root() == tmp_path / "data"
    assert results_dir() == tmp_path / "results"
    assert judge_cache() == tmp_path / "judge"


def test_tasks_yaml_has_eight_cot_tasks():
    payload = load_tasks_config()
    names = [t["name"] for t in payload["tasks"]]
    assert len(names) == 8
    assert all(n.endswith("_cot") for n in names)
    tsvs = task_inference_tsvs()
    assert tsvs["classification_closed_VQA_cot"] == "final_cot_v2/cls_closed.tsv"
    judgeable = [t["name"] for t in payload["tasks"] if t["supports_judge"]]
    assert len(judgeable) == 6


def test_tasks_yaml_data_root_matches_default():
    payload = load_tasks_config()
    assert Path(payload["data_root"]) == Path(
        "/pasteur/u/rdcunha/data_cache/mmbu/final_data/subsampled_mmbu_data"
    )


def test_apply_runtime_cache_env_respects_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm"))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv"))
    from mmbu.paths import apply_runtime_cache_env

    applied = apply_runtime_cache_env()
    assert applied["XDG_CACHE_HOME"] == str(tmp_path / "xdg")
    assert (tmp_path / "xdg").is_dir()
    assert hf_home() == tmp_path / "hf"
    assert xdg_cache() == tmp_path / "xdg"


def test_inference_repo_root_is_workspace():
    assert (inference_repo_root() / "configs" / "tasks.yaml").is_file()
    with open(inference_repo_root() / "configs" / "tasks.yaml") as f:
        yaml.safe_load(f)
