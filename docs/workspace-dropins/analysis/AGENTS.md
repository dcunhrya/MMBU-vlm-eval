# AGENTS.md (analysis)

Paper figures, eval-split, DuckDB metrics cache.

Workspace index: `/pasteur/u/rdcunha/code/mmbu/CATALOG.md` (copy from inference `CATALOG.md`).

## Job

- Plots and tables from `results_cot_v3` JSONL + judge CSVs.
- Competition public/private split under `eval_split/`.
- CLI: `mmbu-analysis plot|cache|eval-split` when installed.

## Paths

Do not hardcode new pasteur strings. Prefer env `MMBU_RESULTS_DIR`, `MMBU_JUDGE_CACHE`, `MMBU_DATA_ROOT` (see inference `src/mmbu/paths.py`).

Judge CSV *files* live in `finalized_analysis/llm_judge_cache/` — that directory is an **artifact**, not source. Inference writes it (`python -m mmbu judge`).

## Frozen fork

`src/` in this tree (`utils_closed.py`, `llm_judge_*.py`, `constants.py`) is a **diverged copy** of `inference/src/mmbu/eval/`. Do not add features here.

New figure code should import `mmbu.eval` / `mmbu.paths` with:

```bash
export PYTHONPATH=/pasteur/u/rdcunha/code/mmbu/inference/src:$PYTHONPATH
```

Legacy `finalized_analysis/run_*_llm_judge.{py,sh}` is retired. Use `python -m mmbu judge` from inference (`judge/` venv, profile `judge_l40s`).
