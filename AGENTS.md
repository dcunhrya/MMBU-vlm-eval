# AGENTS.md

GPU/API inference + scoring + LLM judge for the 8-task MMBU biomedical VLM benchmark.

Workspace index (sibling trees, artifacts, archives): [CATALOG.md](CATALOG.md).

This is a research-engineering repo. Prefer resume over rerun, join existing artifacts over re-inference, and do not invent parallel pipelines.

## Do this

- Paths: `from mmbu.paths import data_root, results_dir, judge_cache, apply_runtime_cache_env`. Do not add new pasteur absolute path literals.
- New models: `configs/runs/*.yaml` + `python -m mmbu submit --stages infer --isolate-tasks --launch` when that CLI exists; otherwise `src/run_vlm_eval.py` with a YAML under `configs/`. Smoke before full 8-task runs.
- Status: `squeue -u $USER` then `python src/results/check_results.py`.
- Secrets in `.env` only. Pasteur cache env on every GPU/`uv` job (`apply_runtime_cache_env()`).
- Join production results on `index`. Check `question_type` before reusing a model dir.

## Do not

- Copy or edit `eval/*_eval*.sh`. Gemma fallback: `run_vlm_eval_gemma.py`.
- Write caches to `~` / sailhome (20 GB quota).
- Drop TSV rows to “fix” NaN stems. Patch eval + CoT + HF metadata together.
- Hub-write (`hf upload`, `upload-large-folder`) without explicit confirmation in that chat.
- Add judge/parser features to `../src/analysis/src/` — that tree is a frozen fork. Canonical eval is `src/mmbu/eval/`.

## Skills (Pasteur working tree)

| Ask | Skill |
|-----|--------|
| Did it finish / queue judge | `.cursor/skills/mmbu-coverage/` |
| Add HF or API model | `.cursor/skills/mmbu-onboard-model/` |
| NaN stems, TSV/image QA | `.cursor/skills/mmbu-dataset-qa/` |
| Push Hub datasets | `.cursor/skills/mmbu-hf-reupload/` |

## Trees

See [CATALOG.md](CATALOG.md). Canonical JSONL: `/pasteur/u/rdcunha/code/mmbu/results_cot_v3`. Judge cache lives under analysis; symlink via `artifacts/`.
