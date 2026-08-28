# mmbu-hf-reupload reference

## Local paths

| Path | Purpose |
|------|---------|
| `mmbu_final_dataset_3_18/images/` | Canonical image bytes (single source of truth) |
| `mmbu_final_eccv_data_sanitized_metadata/` | Sanitized full metadata (no answers) |
| `mmbu_context_sanitized/` | Context subset metadata + symlinked images |
| `mmbu_hf_upload/` | Mode B staging for `ryandcunha/mmbu` |
| `mmbu_context_hf_upload/` | Mode B staging for `ryandcunha/mmbu-context` |
| `image_fix_backups/` | CXR fix backups by date |

All under `/pasteur/u/rdcunha/data_cache/mmbu/` unless noted.

## ImageFolder smoke test (pre-upload)

```bash
uv run --no-sync python - <<'PY'
from datasets import load_dataset
root = "/pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_hf_upload"
ds = load_dataset(
    "imagefolder",
    data_dir=root,
    data_files={"train": [f"{root}/metadata.jsonl"]},
)
print(ds)
print(ds["train"][0].keys())
PY
```

Use explicit `data_files` to avoid Hugging Face inferring wrong splits.

## Mode B — full command block

```bash
cd /pasteur/u/rdcunha/code/mmbu/src
export XDG_CACHE_HOME="/pasteur/u/rdcunha/.cache"
export HF_HOME="/pasteur/u/rdcunha/models"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export UV_CACHE_DIR="/pasteur/u/rdcunha/uv_cache"

uv run --no-sync hf auth whoami

uv run --no-sync python -m prepare_final_dataset.diagnose_image_quality \
  --dataset-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_sanitized

uv run --no-sync python -m prepare_final_dataset.prepare_mmbu_hf_dataset \
  --source-tsv /pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_eccv_data_sanitized_metadata/metadata.tsv \
  --source-image-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_dataset_3_18/images \
  --output-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_hf_upload \
  --copy-images

uv run --no-sync python -m prepare_final_dataset.prepare_mmbu_context_hf_dataset \
  --source-tsv /pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_eccv_data_sanitized_metadata/metadata.tsv \
  --source-image-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_dataset_3_18/images \
  --output-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_hf_upload \
  --copy-images

uv run --no-sync python -m prepare_final_dataset.validate_mmbu_hf_dataset \
  --dataset-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_hf_upload

uv run --no-sync python -m prepare_final_dataset.validate_mmbu_hf_dataset \
  --dataset-root /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_hf_upload

# USER CONFIRMATION REQUIRED BEFORE THESE:
uv run --no-sync hf upload-large-folder ryandcunha/mmbu \
  /pasteur/u/rdcunha/data_cache/mmbu/mmbu_hf_upload \
  --repo-type dataset --num-workers 8

uv run --no-sync hf upload-large-folder ryandcunha/mmbu-context \
  /pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_hf_upload \
  --repo-type dataset --num-workers 8
```

## Mode A — metadata-only command block

```bash
# Regenerate jsonl (repeat for each metadata root that changed)
python3 << 'PY'
import json
from pathlib import Path
import pandas as pd
for src in [
    Path("/pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_eccv_data_sanitized_metadata"),
    Path("/pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_sanitized"),
]:
    df = pd.read_csv(src / "metadata.tsv", sep="\t", dtype=str, keep_default_na=False)
    with (src / "metadata.jsonl").open("w") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
PY

STAGE="/pasteur/u/rdcunha/.cache/mmbu_hf_meta_push_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$STAGE/mmbu" "$STAGE/mmbu-context"
cp .../mmbu_final_eccv_data_sanitized_metadata/metadata.{tsv,jsonl} "$STAGE/mmbu/"
cp .../mmbu_context_sanitized/metadata.{tsv,jsonl} "$STAGE/mmbu-context/"

# USER CONFIRMATION REQUIRED:
uv run --no-sync hf upload ryandcunha/mmbu "$STAGE/mmbu" --repo-type dataset \
  --commit-message "Restore missing full-context question stems (NaN artifact)"
uv run --no-sync hf upload ryandcunha/mmbu-context "$STAGE/mmbu-context" --repo-type dataset \
  --commit-message "Restore missing full-context question stems (NaN artifact)"
```

## Commit message templates

| Fix type | Message |
|----------|---------|
| NaN stems | `Restore missing full-context question stems (NaN artifact)` |
| CXR images | `Re-upload with corrected CXR contrast (46 images)` |
| Stale paths | `Remove stale hashed image paths` + full re-upload |

## Hub tree check (hashed paths)

```bash
uv run --no-sync python - <<'PY'
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile
api = HfApi()
files = [
    item.path
    for item in api.list_repo_tree("ryandcunha/mmbu", repo_type="dataset", recursive=True)
    if isinstance(item, RepoFile)
]
print("hashed:", sum(p.startswith("images/files/") for p in files))
print("classification:", sum(p.startswith("images/classification/") for p in files))
PY
```

## Post-upload checklist

- [ ] `ryandcunha/mmbu` row count 77,358
- [ ] `ryandcunha/mmbu-context` row count 25,186
- [ ] No `answer` column in loaded dataset
- [ ] No `images/files/` stale paths (Mode B)
- [ ] Fixed CXR paths viewable on Hub (Mode B only)
- [ ] Upload logs saved if large (`*_upload.log`)

## Auth

- `uv run --no-sync hf auth login` (interactive, one-time)
- `uv run --no-sync hf auth whoami` (preflight)
- No `HF_TOKEN` in scripts; CLI uses cached token
