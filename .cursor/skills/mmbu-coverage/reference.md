# mmbu-coverage reference

## Paths

| Artifact | Location |
|----------|----------|
| JSONL predictions | `/pasteur/u/rdcunha/code/mmbu/results_cot_v3/{model_stem}/{model_stem}_{task}.jsonl` |
| No-image dir | `.../results_cot_v3/{model_stem}_no_image/` |
| Status report | `inference_status_report.txt` (repo root, from `check_results.py`) |
| Slurm logs | `slurm_logs/` |
| Judge cache (closed) | `/pasteur/u/rdcunha/code/mmbu/src/analysis/finalized_analysis/llm_judge_cache/closed_vqa/` |
| Judge cache (open) | `.../llm_judge_cache/open_vqa/` |
| Run configs | `configs/runs/*.yaml` |

## Expected row counts (8 tasks)

| Task | Rows |
|------|-----:|
| classification_open_VQA_cot | 24,721 |
| classification_closed_VQA_cot | 24,721 |
| detection_grounding_open_VQA_cot | 4,470 |
| detection_grounding_closed_VQA_cot | 4,470 |
| detection_guess_bbox_open_VQA_cot | 4,238 |
| detection_guess_bbox_closed_VQA_cot | 4,238 |
| segmentation_grounding_open_VQA_cot | 5,250 |
| segmentation_grounding_closed_VQA_cot | 5,250 |

## Judgeable vs direct

| Task | Judge |
|------|-------|
| classification_* | yes |
| detection_grounding_* | yes |
| segmentation_grounding_* | yes |
| detection_guess_bbox_* | no (direct) |

## Model naming

- **JSONL dir / `--model-filter`:** last path segment of HF id (`Qwen/Qwen2.5-VL-7B-Instruct` → `Qwen2.5-VL-7B-Instruct`)
- **Frontier aliases:** `claude-sonnet-5`, `claude-opus-5`, `gpt-5.6-sol` — match directory name under `results_cot_v3/`
- **Judge CSV path:** `get_judge_cache_path(cache_dir, task, model, judge_model)` → `{cache_dir}/{sanitize(judge_model)}/{task}/{sanitize(model)}.csv`
- **Legacy flat layout:** `{cache_dir}/{task}/{model}.csv` — run `scripts/migrate_judge_cache_by_model.py` if needed

## Judge cache layout (post-migration)

```
llm_judge_cache/open_vqa/
  Qwen_Qwen2.5-32B-Instruct-AWQ/
    classification_open_VQA_cot/
      claude-sonnet-5.csv
    detection_grounding_open_VQA_cot/
      ...
```

## Useful commands

```bash
# List models with JSONL for a task
ls results_cot_v3/*/Qwen2.5-VL-7B-Instruct_classification_open_VQA_cot.jsonl 2>/dev/null

# Count valid JSONL rows
python -c "
from mmbu.artifacts import check_completion
print(check_completion('path/to/file.jsonl', 24721))
"

# Judge missing only (dry inspect)
python -m mmbu judge --kind open --model-filter MODEL --task-filter TASK --status-json
```

## Sync / auto-queue scripts

| Script | Purpose |
|--------|---------|
| `scripts/sol_sync_tick.sh` | OpenAI sol: infer done → queue judge |
| `scripts/luna_sync_tick.sh` | Luna variant |
| `scripts/sol_qt_sync_tick.sh` | Sol QT variant |
| `eval/anthropic_llm_judge_controller.sh` | Anthropic batch → judge monitor |
| `src/monitor_anthropic_llm_judge.py watch --submit` | Poll + submit judge jobs |
