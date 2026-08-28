#!/usr/bin/env python3
import os
import sys
from glob import glob
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mmbu.paths import data_root, results_dir, task_inference_tsvs

OUT_PATH = os.environ.get("MMBU_STATUS_REPORT", "inference_status_report.txt")
TASK_TO_TSV = task_inference_tsvs()

# Cache TSV sizes so we only load them once
TSV_SIZES = {}


def load_tsv_size(task_name):
    """Return the number of rows for the TSV corresponding to a task."""
    import pandas as pd

    if task_name not in TASK_TO_TSV:
        return None

    if task_name in TSV_SIZES:
        return TSV_SIZES[task_name]

    tsv_path = data_root() / TASK_TO_TSV[task_name]
    df = pd.read_csv(tsv_path, sep="\t")
    TSV_SIZES[task_name] = len(df)
    return TSV_SIZES[task_name]


def count_jsonl_indexes(path):
    """Return number of unique indexes in a JSONL file."""
    import json

    idxs = set()
    with open(path, "r") as f:
        for line in f:
            try:
                j = json.loads(line)
                idxs.add(j["index"])
            except Exception:
                pass
    return len(idxs)


def count_filled_answers(path):
    """Return number of JSONL rows where 'answer' is non-empty."""
    import json

    count = 0

    with open(path, "r") as f:
        for line in f:
            try:
                j = json.loads(line)
                ans = j.get("answer", None)

                if ans is None:
                    continue
                if isinstance(ans, str) and ans.strip().lower() in ["", "none", "null"]:
                    continue

                count += 1
            except Exception:
                pass

    return count


def extract_task_name(filename, model_name):
    stripped = filename.replace(model_name + "_", "")
    return stripped.replace(".jsonl", "")


def main():
    report_lines = []
    root_results = results_dir()

    model_dirs = sorted(
        d for d in glob(os.path.join(str(root_results), "*"))
        if os.path.isdir(d)
    )

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)

        report_lines.append(f"\n=== {model_name} ===")

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

            n_jsonl = count_filled_answers(path)
            n_tsv = load_tsv_size(task_name)

            ok = n_tsv is not None and n_jsonl == n_tsv
            status = "✔" if ok else "✖"

            report_lines.append(
                f"  {task_name:<40} {status}  (JSONL: {n_jsonl} / TSV: {n_tsv})"
            )

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
