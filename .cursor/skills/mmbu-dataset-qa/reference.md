# mmbu-dataset-qa reference

## Join keys

| Context | Key | Notes |
|---------|-----|-------|
| Eval open ↔ closed | `index` | Same image, different VQA format |
| HF metadata rows | `unique_id` | Stable across sanitized/full |
| Image pairing | `image_path` | Used when index differs across exports |
| Question variants | `question_type` | `basic`, `expert`, `full` |
| MMBU-Context subset | `question_type == 'full'` | 25,186 rows |

## Stem defect patterns

| Pattern | Meaning |
|---------|---------|
| `endswith(" nan")` | Missing `clinical VQA task` in full-context open stem |
| `nan nan` | Empty object-detection full-context (known leftover on some closed rows) |
| empty / whitespace | Broken export or merge |
| literal `nan` | pandas NaN stringified |

## Files to patch (Aug 2026 NaN fix — check all if re-auditing)

**Eval TSVs:**

- `final_cls/final_subsampled_cls_open_1_13_v3.tsv`
- `final_det/final_subsampled_det_guess_bbox_open_1_12_v2.tsv`
- `final_det/mmbu_detection_guess_bbox_open_VQA_cot_final.tsv`
- Seg grounding open/closed eval TSVs (v2 paths in registry)

**CoT inference TSVs** (if question text duplicated):

- `final_cot_v2/cls_open.tsv`, `cls_closed.tsv`, etc. per `configs/tasks.yaml`

**HF metadata:**

- `/pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_eccv_data/metadata.tsv`
- `.../mmbu_final_eccv_data_sanitized_metadata/metadata.tsv`
- `.../mmbu_context_sanitized/metadata.tsv`

**Side experiment data:**

- `metadata_cot/open_analysis/data/sampled_questions.csv`

## Regenerate metadata.jsonl

After patching any HF metadata TSV:

```python
import json
from pathlib import Path
import pandas as pd

src = Path("/pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_sanitized")
df = pd.read_csv(src / "metadata.tsv", sep="\t", dtype=str, keep_default_na=False)
with (src / "metadata.jsonl").open("w") as f:
    for rec in df.to_dict(orient="records"):
        f.write(json.dumps(rec, ensure_ascii=True) + "\n")
```

Repeat for each metadata root that changed.

## audit_stems.py output

Sections:

1. **HF metadata** — bad-question counts by `question_type` × open/closed (inferred from `VQA_type` or filename)
2. **Eval TSVs** — per task from registry
3. **Index parity** — infer vs eval set diff sizes
4. **Sample rows** — first N examples per defect class (for agent review)

Exit code 1 if any defects found.

## CXR quality thresholds

From `prepare_final_dataset/image_quality_utils.py`:

| Flag | Condition (x-ray) |
|------|---------------------|
| `unviewable_cxr` | max pixel < 30 |
| `borderline_cxr` | mean < 40 and std < 25 |

## Sanitized HF columns

Public metadata must **not** include: `answer`, `class_label`, `mask_path`.

Expected sanitized row counts:

- Full MMBU: 77,358
- MMBU-Context: 25,186

## External scripts (do not copy into inference)

| Script | Path |
|--------|------|
| validate | `/pasteur/u/rdcunha/code/mmbu/src/prepare_final_dataset/validate_mmbu_hf_dataset.py` |
| diagnose CXR | `.../diagnose_image_quality.py` |
| fix CXR | `.../fix_dark_images.py` |
| stage HF | `.../prepare_mmbu_hf_dataset.py` |
| README | `.../prepare_final_dataset/README.md` |

## Known leftover (verify still present)

~186 closed object-detection rows with `nan nan` stems were documented as unfixed on the closed side; open side was patched. Re-audit with `audit_stems.py`.
