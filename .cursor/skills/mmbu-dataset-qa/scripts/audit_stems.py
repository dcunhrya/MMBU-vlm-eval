#!/usr/bin/env python3
"""Report-only audit for MMBU question stems and index parity.

Reads task registry TSVs and HF metadata files. Does not write fixes.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Allow running from inference repo root with PYTHONPATH=src
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mmbu.tasks import load_task_registry  # noqa: E402

DATA_ROOT = Path("/pasteur/u/rdcunha/data_cache/mmbu/final_data/subsampled_mmbu_data")

HF_METADATA_PATHS = {
    "mmbu_full": Path(
        "/pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_eccv_data/metadata.tsv"
    ),
    "mmbu_sanitized": Path(
        "/pasteur/u/rdcunha/data_cache/mmbu/mmbu_final_eccv_data_sanitized_metadata/metadata.tsv"
    ),
    "mmbu_context": Path(
        "/pasteur/u/rdcunha/data_cache/mmbu/mmbu_context_sanitized/metadata.tsv"
    ),
}


def _classify_question(q: str) -> str | None:
    if q is None:
        return "empty"
    s = str(q).strip()
    if not s:
        return "empty"
    lower = s.lower()
    if lower in {"nan", "none", "null"}:
        return "literal_nan"
    if s.endswith(" nan"):
        return "trailing_nan"
    if lower == "nan nan":
        return "nan_nan"
    return None


def _load_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _infer_vqa_kind(path: Path, df: pd.DataFrame) -> str:
    if "VQA_type" in df.columns:
        return str(df["VQA_type"].iloc[0]).lower()
    name = path.name.lower()
    if "open" in name:
        return "open"
    if "closed" in name:
        return "closed"
    return "unknown"


def audit_file(label: str, path: Path, df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if "question" not in df.columns:
        print(f"  [{label}] SKIP {path}: no 'question' column")
        return counts

    qtype_col = "question_type" if "question_type" in df.columns else None
    vqa = _infer_vqa_kind(path, df)

    for _, row in df.iterrows():
        defect = _classify_question(row.get("question", ""))
        if defect is None:
            continue
        qt = row[qtype_col] if qtype_col else "all"
        key = f"{defect}|vqa={vqa}|qtype={qt}"
        counts[key] += 1

    total_bad = sum(counts.values())
    if total_bad:
        print(f"\n## {label}: {path}")
        print(f"   rows={len(df)} bad_questions={total_bad}")
        for key in sorted(counts):
            print(f"   {key}: {counts[key]}")
    else:
        print(f"\n## {label}: {path} — OK ({len(df)} rows)")

    return counts


def audit_index_parity(registry) -> bool:
    ok = True
    print("\n# Index parity (infer vs eval)")
    for spec in registry.canonical_tasks():
        infer_path = DATA_ROOT / spec.inference_tsv
        eval_path = DATA_ROOT / spec.evaluation_tsv
        if not infer_path.exists() or not eval_path.exists():
            print(f"  SKIP {spec.name}: missing TSV")
            ok = False
            continue
        infer_idx = set(_load_tsv(infer_path)["index"])
        eval_idx = set(_load_tsv(eval_path)["index"])
        if infer_idx != eval_idx:
            only_infer = len(infer_idx - eval_idx)
            only_eval = len(eval_idx - infer_idx)
            print(
                f"  FAIL {spec.name}: infer={len(infer_idx)} eval={len(eval_idx)} "
                f"only_infer={only_infer} only_eval={only_eval}"
            )
            ok = False
        else:
            print(f"  OK   {spec.name}: {len(infer_idx)} indexes match")
    return ok


def sample_defects(path: Path, df: pd.DataFrame, limit: int = 3) -> None:
    if "question" not in df.columns:
        return
    shown: dict[str, int] = defaultdict(int)
    id_col = "index" if "index" in df.columns else "unique_id"
    for _, row in df.iterrows():
        defect = _classify_question(row.get("question", ""))
        if defect is None:
            continue
        if shown[defect] >= limit:
            continue
        shown[defect] += 1
        qpreview = str(row.get("question", ""))[:80].replace("\n", " ")
        print(
            f"    sample [{defect}] {id_col}={row.get(id_col)} "
            f"qtype={row.get('question_type', '?')} q={qpreview!r}..."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        default=2,
        help="Max sample rows printed per defect class per file (0=disable)",
    )
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    registry = load_task_registry()
    all_counts: dict[str, int] = defaultdict(int)
    parity_ok = True

    print("# MMBU stem audit (report only)\n")

    if not args.skip_hf:
        print("# HF metadata")
        for name, path in HF_METADATA_PATHS.items():
            if not path.exists():
                print(f"\n## {name}: MISSING {path}")
                continue
            df = _load_tsv(path)
            counts = audit_file(name, path, df)
            for k, v in counts.items():
                all_counts[k] += v
            if args.sample and counts:
                sample_defects(path, df, limit=args.sample)

    if not args.skip_eval:
        print("\n# Eval + inference TSVs (registry)")
        seen_paths: set[Path] = set()
        for spec in registry.canonical_tasks():
            for rel, tag in (
                (spec.evaluation_tsv, f"eval/{spec.name}"),
                (spec.inference_tsv, f"infer/{spec.name}"),
            ):
                path = DATA_ROOT / rel
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                if not path.exists():
                    print(f"\n## {tag}: MISSING {path}")
                    continue
                df = _load_tsv(path)
                counts = audit_file(tag, path, df)
                for k, v in counts.items():
                    all_counts[k] += v
                if args.sample and counts:
                    sample_defects(path, df, limit=args.sample)

    if DATA_ROOT.is_dir():
        parity_ok = audit_index_parity(registry)
    else:
        print(f"\n# Index parity SKIPPED: {DATA_ROOT} not found")
        parity_ok = False

    total_bad = sum(all_counts.values())
    print(f"\n# Summary: bad_questions={total_bad} index_parity={'OK' if parity_ok else 'FAIL'}")

    if total_bad or not parity_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
