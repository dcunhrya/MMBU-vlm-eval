---
name: mmbu-coverage
description: >-
  Report MMBU inference and judge completion gaps across results_cot_v3 JSONL,
  llm_judge_cache CSVs, Slurm queue, and API batch manifests. Queue missing
  judge jobs only. Use when the user asks if a run finished, what's missing,
  judge coverage, queue judge, squeue status, or batch collect.
---

# MMBU coverage

Answer “did it finish / what’s missing / queue judge” without rediscovering paths.

## Quick path

1. Slurm: `squeue -u "$USER"`
2. Per-config status (if run YAML exists): `python -m mmbu status --config configs/runs/<run>.yaml`
3. Cross-model JSONL: `python src/results/check_results.py` → `inference_status_report.txt`
4. Judge gaps: inspect cache CSVs (see [reference.md](reference.md))
5. Queue only missing judges or submit via pipeline — do not re-run complete tasks

## Environment

```bash
cd /pasteur/u/rdcunha/code/mmbu/inference
source .venv/bin/activate   # or judge/ for judge runs
export PYTHONPATH=src
```

Pasteur cache vars apply for GPU jobs (see workspace cache-paths rule).

## Step 1 — Slurm and silent failures

```bash
squeue -u "$USER"
ls -lt slurm_logs/ | head
```

Jobs that exit instantly leave no JSONL. Check `slurm_logs/` and the sbatch script’s cache exports before resubmitting.

## Step 2 — Per-run pipeline status

When a `configs/runs/*.yaml` exists for the model:

```bash
python -m mmbu status --config configs/runs/<run>.yaml
python -m mmbu status --config configs/runs/<run>.yaml --json   # machine-readable
```

Shows infer / score / judge / aggregate per task for that config.

## Step 3 — Cross-model JSONL completeness

```bash
python src/results/check_results.py
cat inference_status_report.txt
```

Default results root: `/pasteur/u/rdcunha/code/mmbu/results_cot_v3`.

Each line: task, valid rows vs expected, unique indexes. ✖ means incomplete or extra rows.

## Step 4 — Judge coverage

Six tasks require LLM judge (`supports_judge: true` in `configs/tasks.yaml`):

- `classification_{closed,open}_VQA_cot`
- `detection_grounding_{closed,open}_VQA_cot`
- `segmentation_grounding_{closed,open}_VQA_cot`

**Not judged:** `detection_guess_bbox_*` (direct scoring).

Cache roots (partitioned by judge model after migration):

- Closed: `.../llm_judge_cache/closed_vqa/<judge_model>/<task>/<model>.csv`
- Open: `.../llm_judge_cache/open_vqa/<judge_model>/<task>/<model>.csv`

Default judge model: `Qwen/Qwen2.5-32B-Instruct-AWQ` → dir `Qwen_Qwen2.5-32B-Instruct-AWQ`.

Check one model:

```bash
source judge/bin/activate
export PYTHONPATH=src

python -m mmbu judge --kind open --list-models
python -m mmbu judge --kind closed --list-models

python -m mmbu judge --kind open \
  --model-filter claude-sonnet-5 \
  --task-filter classification_open_VQA_cot \
  --status-json
```

If CSV missing or incomplete, run judge for that kind only:

```bash
python -m mmbu judge --kind open \
  --results-dir /pasteur/u/rdcunha/code/mmbu/results_cot_v3 \
  --model-filter <model_stem> \
  --task-filter <task_name>
```

For Slurm judge jobs, use `judge/` venv and profile `judge_l40s` from `configs/cluster.yaml`. Prefer `python -m mmbu submit --config ... --stages judge --launch` over hand-written sbatch.

## Step 5 — API batch models

Anthropic:

```bash
python src/run_anthropic_batch.py status --config configs/anthropic_<model>.yaml
python src/run_anthropic_batch.py collect --config configs/anthropic_<model>.yaml
```

OpenAI: `python src/run_openai_batch.py` with the same subcommands if configured.

After collect confirms JSONL complete → Step 4 for judge gaps.

Long-running frontier campaigns may use sync ticks (`scripts/sol_sync_tick.sh`, `scripts/luna_sync_tick.sh`) or `sbatch eval/anthropic_llm_judge_controller.sh`.

## Step 6 — No-image runs

Directory: `{model_stem}_no_image/`. JSONL filenames still use `{model_stem}_{task}.jsonl` (no `_no_image` in filename). Pass `--no-image` to judge and status commands.

## Step 7 — Output

Print a gap table:

| Model | Task | Infer | Judge | Action |
|-------|------|-------|-------|--------|

Update `.cursor/*_run_status.md` only for long multi-job campaigns (no-image, factorial). For one-off checks, the table is enough.

## Do not

- Copy or edit legacy `eval/*_eval*.sh` for status checks
- Re-run inference for tasks already ✔ in `check_results.py`
- Assume judge cache path is flat `{task}/{model}.csv` — check for judge-model partition first

## Reference

Path tables, row counts, model slug rules: [reference.md](reference.md)
