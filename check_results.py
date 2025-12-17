#!/usr/bin/env python3
import os
import json
import pandas as pd
from glob import glob
from collections import defaultdict

ROOT_RESULTS = "/pasteur/u/rdcunha/code/mmbu/results"
ROOT_TSV = "/pasteur/u/rdcunha/data_cache/mmbu/final_data/subsampled_mmbu_data"
OUT_PATH = "inference_status_report.txt"

# -------------------------------------------------------
# Mapping result task → TSV path
# -------------------------------------------------------
TASK_TO_TSV = {
    "classification_closed_VQA": "final_cls/final_subsampled_cls_closed_11_30_25.tsv",
    "classification_open_VQA": "final_cls/final_subsampled_cls_open_11_30_25.tsv",
    "detection_grounding_closed_VQA": "final_det/final_subsampled_det_grounding_closed.tsv",
    "detection_grounding_open_VQA": "final_det/final_subsampled_det_grounding_open.tsv",
    "detection_guess_bbox_closed_VQA": "final_det/final_subsampled_det_guess_bbox_closed.tsv",
    "detection_guess_bbox_open_VQA": "final_det/final_subsampled_det_guess_bbox_open.tsv",
    "segmentation_grounding_closed_VQA": "final_seg/final_subsampled_seg_grounding_closed.tsv",
    "segmentation_grounding_open_VQA": "final_seg/final_subsampled_seg_grounding_open.tsv",
    "segmentation_guess_bbox_open_VQA": "final_seg/final_subsampled_seg_guess_mask_open.tsv",
}

# Cache TSV sizes so we only load them once
TSV_SIZES = {}

def load_tsv_size(task_name):
    """Return the number of rows for the TSV corresponding to a task."""
    if task_name not in TASK_TO_TSV:
        return None

    if task_name in TSV_SIZES:
        return TSV_SIZES[task_name]

    tsv_path = os.path.join(ROOT_TSV, TASK_TO_TSV[task_name])
    df = pd.read_csv(tsv_path, sep="\t")
    TSV_SIZES[task_name] = len(df)
    return TSV_SIZES[task_name]


def count_jsonl_indexes(path):
    """Return number of unique indexes in a JSONL file."""
    idxs = set()
    with open(path, "r") as f:
        for line in f:
            try:
                j = json.loads(line)
                idxs.add(j["index"])
            except:
                pass
    return len(idxs)

def count_filled_answers(path):
    """Return number of JSONL rows where 'answer' is non-empty."""
    count = 0

    with open(path, "r") as f:
        for line in f:
            try:
                j = json.loads(line)
                ans = j.get("answer", None)

                # Treat these as empty:
                if ans is None:
                    continue
                if isinstance(ans, str) and ans.strip().lower() in ["", "none", "null"]:
                    continue

                count += 1
            except:
                pass

    return count

def extract_task_name(filename, model_name):
    """
    Example: filename = 'gemma-3-4b-it_classification_closed_VQA.jsonl'
    Returns: 'classification_closed_VQA'
    """
    stripped = filename.replace(model_name + "_", "")
    return stripped.replace(".jsonl", "")


def main():
    report_lines = []

    # Iterate through each model folder
    model_dirs = sorted(
        d for d in glob(os.path.join(ROOT_RESULTS, "*"))
        if os.path.isdir(d)
    )

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)

        report_lines.append(f"\n=== {model_name} ===")

        # Find all jsonl result files
        jsonl_files = sorted(glob(os.path.join(model_dir, "*.jsonl")))

        if not jsonl_files:
            report_lines.append("  No JSONL results found.")
            continue

        for path in jsonl_files:
            filename = os.path.basename(path)
            task_name = extract_task_name(filename, model_name)

            if task_name not in TASK_TO_TSV:
                report_lines.append(f"  {filename}: Unknown task")
                continue

            # JSONL size
            # n_jsonl = count_jsonl_indexes(path)
            n_jsonl = count_filled_answers(path)

            # TSV size
            n_tsv = load_tsv_size(task_name)

            # Check completion
            ok = abs(n_jsonl - n_tsv) <= 200
            status = "✔" if ok else "✖"

            report_lines.append(
                f"  {task_name:<40} {status}  (JSONL: {n_jsonl} / TSV: {n_tsv})"
            )

    # Write report
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
