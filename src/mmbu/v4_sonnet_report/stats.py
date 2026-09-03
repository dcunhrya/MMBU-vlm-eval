"""Bootstrap stats for v4 Sonnet-judge score report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmbu.v4_sonnet_report.config import OPEN_MACRO_EVAL_TASKS, V4SonnetReportConfig
from mmbu.v4_sonnet_report.join_scores import open_scored_subset


def _cluster_bootstrap_mean(
    values: pd.Series,
    clusters: pd.Series,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    mask = values.notna()
    if not mask.any():
        return float("nan"), float("nan"), float("nan")

    vals = values[mask].astype(float)
    clust = clusters[mask].astype(str)
    by_cluster = vals.groupby(clust).mean()
    if by_cluster.empty:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    means = []
    clusters_arr = by_cluster.index.to_numpy()
    weights = by_cluster.to_numpy()
    for _ in range(n_boot):
        picks = rng.choice(clusters_arr, size=len(clusters_arr), replace=True)
        boot_vals = weights[[list(clusters_arr).index(p) for p in picks]]
        means.append(float(np.mean(boot_vals)))
    point = float(np.mean(weights))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return point, float(lo), float(hi)


def open_macro_by_partition(df: pd.DataFrame, partition: str) -> float:
    subset = open_scored_subset(df)
    subset = subset[subset["partition"] == partition]
    if subset.empty:
        return float("nan")
    task_means = (
        subset.groupby("eval_task_id")["row_score"].mean().reindex(OPEN_MACRO_EVAL_TASKS)
    )
    valid = task_means.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.mean())


def compute_partition_stats(
    df: pd.DataFrame,
    config: V4SonnetReportConfig | None = None,
) -> pd.DataFrame:
    config = config or V4SonnetReportConfig.from_env()
    rows = []
    atom_col = "atom_id" if "atom_id" in df.columns else "index"

    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        for part in ("public", "private_id", "private_ood"):
            part_df = open_scored_subset(mdf[mdf["partition"] == part])
            if part_df.empty:
                rows.append(
                    {
                        "model": model,
                        "partition": part,
                        "open_macro": float("nan"),
                        "ci_lo": float("nan"),
                        "ci_hi": float("nan"),
                        "n_rows": 0,
                    }
                )
                continue

            task_scores = []
            for task in OPEN_MACRO_EVAL_TASKS:
                tdf = part_df[part_df["eval_task_id"] == task]
                if tdf.empty:
                    continue
                mean, lo, hi = _cluster_bootstrap_mean(
                    tdf["row_score"],
                    tdf[atom_col],
                    n_boot=config.n_bootstrap,
                    seed=config.bootstrap_seed + hash((model, part, task)) % 10_000,
                )
                task_scores.append({"task": task, "mean": mean, "lo": lo, "hi": hi})

            if not task_scores:
                macro = float("nan")
                lo = hi = float("nan")
            else:
                macro = float(np.mean([t["mean"] for t in task_scores]))
                lo = float(np.mean([t["lo"] for t in task_scores]))
                hi = float(np.mean([t["hi"] for t in task_scores]))

            rows.append(
                {
                    "model": model,
                    "partition": part,
                    "open_macro": macro,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_rows": int(len(part_df)),
                }
            )

    stats = pd.DataFrame(rows)
    stats.to_csv(config.stats_cache_path, index=False)
    return stats


def model_verdict(stats: pd.DataFrame, model: str) -> dict[str, float]:
    m = stats[stats["model"] == model]
    out: dict[str, float] = {}
    for part in ("public", "private_id", "private_ood"):
        row = m[m["partition"] == part]
        out[part] = float(row["open_macro"].iloc[0]) if not row.empty else float("nan")
    pub = out.get("public", float("nan"))
    out["delta_id"] = pub - out.get("private_id", float("nan"))
    out["delta_ood"] = pub - out.get("private_ood", float("nan"))
    return out
