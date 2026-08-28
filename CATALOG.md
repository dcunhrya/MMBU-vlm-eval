# MMBU workspace catalog

Index for the Pasteur MMBU workspace. Git remotes stay separate. Open `mmbu.code-workspace` (this repo, or the copy under `/pasteur/u/rdcunha/code/mmbu/` after `scripts/install_workspace_index.sh`).

## Git remotes (do not merge)

| Folder | Remote | Role |
|--------|--------|------|
| `inference/` (this repo) | `dcunhrya/MMBU-vlm-eval` | GPU/API infer, judge, score CLI |
| `src/analysis/` | nested git, no remote | figures, eval-split, DuckDB, judge CSV *directory* |
| `src/prepare_final_dataset/` | under empty `src/.git` shell | HF stage / validate / CXR |
| `MBMU-eval/` | `dcunhrya/MBMU-eval` | **Archive.** Pre-8-task VLMEvalKit. Do not use for CoT benchmark. |

Workspace root: `/pasteur/u/rdcunha/code/mmbu` (not a git repo).

## Logical packages

| Package | Job | Home |
|---------|-----|------|
| `mmbu` CLI | infer / submit / judge / score / status | `inference/src/mmbu/` |
| `mmbu-eval` | parsers, metrics, judge engine | `inference/src/mmbu/eval/` (**canonical**) |
| `mmbu-analysis` | plots, DuckDB, eval-split | `src/analysis/mmbu_analysis/` |
| `prepare_final_dataset` | HF ImageFolder stage + image QA | `src/prepare_final_dataset/` |
| `mmbu-research` | `metadata_cot/`, judge factorial | under inference |

**Path config:** `src/mmbu/paths.py`. Env: `MMBU_DATA_ROOT`, `MMBU_RESULTS_DIR`, `MMBU_JUDGE_CACHE`, `MMBU_HF_STAGING`, `MMBU_WORKSPACE_ROOT`.

## Artifacts (not git)

| What | Path |
|------|------|
| JSONL (canonical) | `/pasteur/u/rdcunha/code/mmbu/results_cot_v3/` |
| Judge CSVs | `…/src/analysis/finalized_analysis/llm_judge_cache/` (symlink: `artifacts/llm_judge_cache`) |
| Eval + CoT TSVs | `/pasteur/u/rdcunha/data_cache/mmbu/final_data/subsampled_mmbu_data/` |
| Canonical images | `…/data_cache/mmbu/mmbu_final_dataset_3_18/images/` |
| HF staging | `…/data_cache/mmbu/mmbu_hf_upload/`, `mmbu_context_hf_upload/` |
| Weights / compile | `/pasteur/u/rdcunha/models`, `/pasteur/u/rdcunha/.cache` |
| uv | `/pasteur/u/rdcunha/uv_cache` |

Expected rows: cls 24721, det grounding 4470, det guess_bbox 4238, seg 5250. Hub: `ryandcunha/mmbu` 77358, `ryandcunha/mmbu-context` 25186, **no answers**.

## Archives (keep on disk, ignore for new work)

- `results/`, `results_cot/`, `results_cot_v2/`, `results_hidden/`
- `llm_judge_cache_v1/`, `llm_judge_cache_v2/`
- `src/ryan-src/` (dataset curation notebooks)
- `src/.git` (zero commits — not a monorepo)
- `inference/eval/*_eval.sh` (~100 wrappers). Use `python -m mmbu submit` or `src/run_vlm_eval.py`.
- `inference/src/dataset_check/test_dataset.py` (stale TSV names)

## Venvs (inference)

| Work | Venv |
|------|------|
| Default GPU | `.venv` |
| Qwen3.6 / OctoMed vLLM | `qwen36-vllm/` |
| Judge | `judge/` |
| LLaVA-Med | `llava/` |

`export PYTHONPATH=src`

## Do not

- Copy `eval/*_eval.sh` for a new model.
- Write caches to sailhome `~` (20 GB quota).
- Add features to `src/analysis/src/` (frozen fork of `mmbu.eval`).
- Hub-write without explicit confirmation.
- Index `data_cache` or `results_cot_v3` in Cursor (too large).
