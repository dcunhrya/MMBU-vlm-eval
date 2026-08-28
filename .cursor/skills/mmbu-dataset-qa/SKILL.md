---
name: mmbu-dataset-qa
description: >-
  Audit MMBU benchmark TSVs and Hugging Face metadata for broken question
  stems (empty, nan, trailing " nan"), index parity, missing images, and dark
  CXR quality. Report-only by default; patch all trees in sync before Hub push.
  Use for NaN stems, empty questions, dataset QA, dark CXR, broken TSV, or
  question_type full validation.
---

# MMBU dataset QA

Catch broken questions and images **before** inference or Hub push.

## Trees that must stay in sync

| Tree | Root |
|------|------|
| Eval + inference TSVs | `/pasteur/u/rdcunha/data_cache/mmbu/final_data/subsampled_mmbu_data/` |
| HF metadata (full) | `.../mmbu_final_eccv_data/metadata.tsv` |
| HF metadata (sanitized) | `.../mmbu_final_eccv_data_sanitized_metadata/metadata.tsv` |
| HF metadata (context) | `.../mmbu_context_sanitized/metadata.tsv` |
| Canonical images | `.../mmbu_final_dataset_3_18/images/` |

Task paths: `configs/tasks.yaml` (`inference_tsv` + `evaluation_tsv` per task).

## Quick audit

```bash
cd /pasteur/u/rdcunha/code/mmbu/inference
source .venv/bin/activate
export PYTHONPATH=src

python .cursor/skills/mmbu-dataset-qa/scripts/audit_stems.py
python -m pytest tests/regression/test_task_row_counts.py -v
```

## Step 1 — Stem audit (NaN failure mode)

Run `scripts/audit_stems.py` (report only). Flag every row where `question` is:

- empty or whitespace-only
- literal `nan` / `NaN`
- ends with `" nan"` (missing `clinical VQA task` stringified into `full` stems)

**Root cause:** open `question_type=full` rows concatenate context + `clinical VQA task`; pandas NaN becomes `" nan"`.

**Pairing for fixes:**

- Eval TSVs: open ↔ closed on **`index`**
- HF metadata: **`unique_id`** or **`image_path`**

Do **not** fix by dropping rows.

## Step 2 — Index parity

```bash
python -m pytest tests/regression/test_task_row_counts.py -v
```

Asserts infer TSV index set == eval TSV index set per task. Expected counts:

| Task group | Rows |
|------------|-----:|
| classification_* | 24,721 |
| detection_grounding_* | 4,470 |
| detection_guess_bbox_* | 4,238 |
| segmentation_grounding_* | 5,250 |

## Step 3 — Closed option integrity

On closed eval TSVs from the registry:

- `"None of the above"` present in options
- Answer key appears in options list

Logic mirrors `src/dataset_check/test_dataset.py` but use **registry paths** (v2/v3), not its stale hardcoded file list.

## Step 4 — Missing images

From sibling package (`/pasteur/u/rdcunha/code/mmbu/src`):

```bash
cd /pasteur/u/rdcunha/code/mmbu/src
export XDG_CACHE_HOME="/pasteur/u/rdcunha/.cache"
export HF_HOME="/pasteur/u/rdcunha/models"
export UV_CACHE_DIR="/pasteur/u/rdcunha/uv_cache"

uv run --no-sync python -m prepare_final_dataset.validate_mmbu_hf_dataset \
  --dataset-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_sanitized \
  --report-path /tmp/mmbu_validate_context.json
```

Repeat for `mmbu_final_eccv_data_sanitized_metadata` if staging full MMBU.

## Step 5 — CXR image quality

```bash
cd /pasteur/u/rdcunha/code/mmbu/src
uv run --no-sync python -m prepare_final_dataset.diagnose_image_quality \
  --dataset-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_sanitized
```

Expect **0** `unviewable_cxr` and **0** `borderline_cxr`.

If dirty:

```bash
uv run --no-sync python -m prepare_final_dataset.fix_dark_images --dry-run \
  --dataset-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_sanitized

uv run --no-sync python -m prepare_final_dataset.fix_dark_images --apply \
  --dataset-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_sanitized
```

Fixes apply to **canonical** image store (`mmbu_final_dataset_3_18/images/`). Backups: `data_cache/mmbu/image_fix_backups/`.

Re-run diagnose until clean.

## Step 6 — Leak audit (optional)

For metadata_cot sampling:

```bash
python -m metadata_cot schema
```

## Step 7 — Patch rule

If stems change, patch **all** affected copies:

1. Eval TSVs under `subsampled_mmbu_data/final_{cls,det,seg}/`
2. CoT inference TSVs under `subsampled_mmbu_data/final_cot_v2/` if they embed the same `question`
3. All three HF metadata TSVs listed above
4. Regenerate `metadata.jsonl` from each patched TSV

Never patch Hub without local TSV + JSONL updated.

## Step 8 — Next step

| Change scope | Action |
|--------------|--------|
| HF metadata or images | **mmbu-hf-reupload** after QA clean |
| Eval TSVs only | No Hub push; re-run affected inference if needed |

## Do not

- Reimplement `diagnose_image_quality`, `validate_mmbu_hf_dataset`, or `test_task_row_counts` — call them
- Use `src/dataset_check/test_dataset.py` as-is (stale paths, incomplete task list)
- Patch one tree and forget CoT vs eval copies

## Reference

Join keys, Aug 2026 patch history, file checklist: [reference.md](reference.md)
