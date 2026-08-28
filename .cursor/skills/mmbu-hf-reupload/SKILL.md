---
name: mmbu-hf-reupload
description: >-
  Re-upload sanitized MMBU datasets to Hugging Face (ryandcunha/mmbu and
  ryandcunha/mmbu-context) after local QA passes. Supports metadata-only
  pushes and full image re-staging. Requires explicit user confirmation before
  any Hub write. Use for reupload HF, push mmbu-context, metadata-only push,
  or upload-large-folder.
---

# MMBU Hugging Face reupload

Push local sanitized MMBU to Hub. Scripts live in sibling tree `/pasteur/u/rdcunha/code/mmbu/src/prepare_final_dataset/` — do not copy them into inference.

## Prerequisites

Run **mmbu-dataset-qa** first. Must be clean:

- 0 empty / trailing-`" nan"` open-full questions
- 0 unviewable/borderline CXR (if touching images)
- Sanitized metadata excludes `answer`, `class_label`, `mask_path`

## Hub targets

| Repo | Rows | Content |
|------|-----:|---------|
| `ryandcunha/mmbu` | 77,358 | All `question_type`, no answers |
| `ryandcunha/mmbu-context` | 25,186 | `question_type==full` only, no answers |

Format: **ImageFolder** — `metadata.jsonl` + `images/{classification,detection,segmentation}/...`

## Choose mode

| Mode | When | Upload command |
|------|------|----------------|
| **A — Metadata-only** | Stem/metadata TSV patches only; images unchanged | `hf upload` (small folder) |
| **B — Full images** | CXR fix or any image byte change | Stage with `--copy-images`, then `hf upload-large-folder` |

**Ask the user which mode before proceeding.**

## Environment

```bash
export XDG_CACHE_HOME="/pasteur/u/rdcunha/.cache"
export HF_HOME="/pasteur/u/rdcunha/models"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export UV_CACHE_DIR="/pasteur/u/rdcunha/uv_cache"

cd /pasteur/u/rdcunha/code/mmbu/src
uv run --no-sync hf auth whoami
```

## Mode A — Metadata-only

1. Regenerate `metadata.jsonl` from patched TSV (see mmbu-dataset-qa reference)
2. Stage a tiny push folder:

```bash
STAGE="/pasteur/u/rdcunha/.cache/mmbu_hf_meta_push_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$STAGE/mmbu" "$STAGE/mmbu-context"

cp /pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_eccv_data_sanitized_metadata/metadata.{tsv,jsonl} "$STAGE/mmbu/"
cp /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_sanitized/metadata.{tsv,jsonl} "$STAGE/mmbu-context/"
```

3. **Stop and ask user to confirm upload.**

4. After confirmation:

```bash
uv run --no-sync hf upload ryandcunha/mmbu "$STAGE/mmbu" --repo-type dataset \
  --commit-message "Restore missing full-context question stems (metadata fix)"

uv run --no-sync hf upload ryandcunha/mmbu-context "$STAGE/mmbu-context" --repo-type dataset \
  --commit-message "Restore missing full-context question stems (metadata fix)"
```

Do **not** restage or upload 12k images in metadata-only mode.

## Mode B — Full images

Staging dirs (materialized copies — do not upload symlink trees):

- `/pasteur/u/rdcunha/data_cache/mmbu/mmbu_hf_upload/`
- `/pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_hf_upload/`

Canonical images: `mmbu_final_dataset_3_18/images/`  
Sanitized source TSV: `mmbu_final_eccv_data_sanitized_metadata/metadata.tsv`

```bash
cd /pasteur/u/rdcunha/code/mmbu/src

# Preflight CXR
uv run --no-sync python -m prepare_final_dataset.diagnose_image_quality \
  --dataset-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_sanitized

# Stage full MMBU
uv run --no-sync python -m prepare_final_dataset.prepare_mmbu_hf_dataset \
  --source-tsv /pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_eccv_data_sanitized_metadata/metadata.tsv \
  --source-image-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_dataset_3_18/images \
  --output-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_hf_upload \
  --copy-images

# Stage mmbu-context
uv run --no-sync python -m prepare_final_dataset.prepare_mmbu_context_hf_dataset \
  --source-tsv /pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_eccv_data_sanitized_metadata/metadata.tsv \
  --source-image-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_dataset_3_18/images \
  --output-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_hf_upload \
  --copy-images

# Validate both
uv run --no-sync python -m prepare_final_dataset.validate_mmbu_hf_dataset \
  --dataset-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_hf_upload \
  --report-path /pasteur/u/rdcunha/data_cache/mmbu/mmbu_hf_upload/validation_report.json

uv run --no-sync python -m prepare_final_dataset.validate_mmbu_hf_dataset \
  --dataset-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_hf_upload \
  --report-path /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_hf_upload/validation_report.json
```

**Stop and ask user to confirm upload.**

After confirmation:

```bash
uv run --no-sync hf upload-large-folder ryandcunha/mmbu \
  /pasteur/u/rdcunha/data_cache/mmbu/mmbu_hf_upload \
  --repo-type dataset --num-workers 8

uv run --no-sync hf upload-large-folder ryandcunha/mmbu-context \
  /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_hf_upload \
  --repo-type dataset --num-workers 8
```

## Stale hashed paths on Hub

If Hub has `images/files/...` instead of real paths, delete before re-upload:

```bash
uv run --no-sync python - <<'PY'
from huggingface_hub import HfApi
HfApi().delete_folder(
    repo_id="ryandcunha/mmbu",
    repo_type="dataset",
    path_in_repo="images/files",
    commit_message="Remove stale hashed image paths",
)
PY
```

Then re-run Mode B upload.

## Post-upload verification

```bash
uv run --no-sync python - <<'PY'
from datasets import load_dataset
for repo, n in [("ryandcunha/mmbu", 77358), ("ryandcunha/mmbu-context", 25186)]:
    ds = load_dataset(repo, split="train")
    print(repo, len(ds), "expected", n)
    assert "answer" not in ds.column_names, f"{repo} leaked answer column"
PY
```

Mode B: spot-check one fixed CXR path exists on Hub with `max >= 80`.

## Rules

1. **Never run Hub write commands without explicit user confirmation in that chat**
2. Do not change Hub visibility, gating, or license from CLI (set in Hub UI)
3. Do not upload symlink staging dirs — use `--copy-images` materialized folders
4. Full docs: `/pasteur/u/rdcunha/code/mmbu/src/prepare_final_dataset/README.md`

## Reference

Exact commands, smoke tests, commit messages: [reference.md](reference.md)
