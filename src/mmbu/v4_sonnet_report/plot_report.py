"""Multi-page PDF: v4 public vs private scores (Sonnet-5 judge)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from mmbu.v4_open_rejudge import load_v4_mapping
from mmbu.v4_sonnet_report.config import (
    EVAL_TASK_LABELS,
    IOU_EVAL_TASK,
    MODEL_DISPLAY,
    OPEN_MACRO_EVAL_TASKS,
    PARTITION_COLORS,
    PARTITION_LABELS,
    V4SonnetReportConfig,
)
from mmbu.v4_sonnet_report.join_scores import (
    build_joined_scores,
    coverage_table,
    open_scored_subset,
)
from mmbu.v4_sonnet_report.stats import compute_partition_stats, model_verdict


def _display(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def _save(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_title(pdf: PdfPages, cfg: V4SonnetReportConfig, joined: pd.DataFrame) -> None:
    mapping = load_v4_mapping(cfg.split_mapping_path)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    n_pub = int((mapping["split"] == "public").sum())
    n_priv = int((mapping["split"] == "private").sum())
    lines = [
        "MMBU v4 Public vs Private — Sonnet-5 Label-Set Judge",
        f"Split: {cfg.split_version_dir.name}",
        "",
        "Judge: claude-sonnet-5, prompt open_per_dataset_v3",
        "Headline: public-open vs private-ID vs private-OOD (open macro)",
        "obj_det_open: IoU from JSONL (not LLM-judged)",
        "",
        f"Emitted rows: {len(mapping):,} (public {n_pub:,} / private {n_priv:,})",
        f"Joined scored rows: {len(joined):,}",
        "",
        "Join: 8-task JSONLs on (index, eval_task_id) + Sonnet judge cache",
        "counting_open excluded (no JSONL).",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=11, family="monospace")
    _save(pdf, fig)


def page_coverage(pdf: PdfPages, cov: pd.DataFrame) -> None:
    if cov.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 8.5))
    pivot = cov.pivot_table(
        index=["model", "eval_task_id"],
        columns="partition",
        values="coverage_pct",
    )
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
    ax.set_title("Coverage: scored / expected v4 partition rows")
    _save(pdf, fig)


def page_dumbbell(pdf: PdfPages, stats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    models = stats["model"].unique()
    y = np.arange(len(models))
    for i, model in enumerate(models):
        for j, part in enumerate(("public", "private_id", "private_ood")):
            row = stats[(stats["model"] == model) & (stats["partition"] == part)]
            if row.empty:
                continue
            x = float(row["open_macro"].iloc[0]) * 100
            ax.scatter(x, i, color=PARTITION_COLORS[part], s=80, zorder=3)
            if not np.isnan(row["ci_lo"].iloc[0]):
                ax.plot(
                    [row["ci_lo"].iloc[0] * 100, row["ci_hi"].iloc[0] * 100],
                    [i, i],
                    color=PARTITION_COLORS[part],
                    alpha=0.5,
                )
    ax.set_yticks(y)
    ax.set_yticklabels([_display(m) for m in models])
    ax.set_xlabel("Open macro score (%)")
    ax.set_title("Headline: open-macro by partition")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PARTITION_COLORS[p], markersize=10, label=PARTITION_LABELS[p])
        for p in ("public", "private_id", "private_ood")
    ]
    ax.legend(handles=handles, loc="lower right")
    _save(pdf, fig)


def page_delta_forest(pdf: PdfPages, stats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    rows = []
    for model in stats["model"].unique():
        v = model_verdict(stats, model)
        rows.append({"model": model, "gap": "public − private_ID", "delta_pp": v["delta_id"] * 100})
        rows.append({"model": model, "gap": "public − private_OOD", "delta_pp": v["delta_ood"] * 100})
    ddf = pd.DataFrame(rows).dropna(subset=["delta_pp"])
    if ddf.empty:
        _save(pdf, fig)
        return
    sns.barplot(data=ddf, y="model", x="delta_pp", hue="gap", ax=ax)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Δ (percentage points)")
    ax.set_title("Generalization gaps (positive = public higher)")
    ax.set_yticklabels([_display(t.get_text()) for t in ax.get_yticklabels()])
    for label in ax.get_yticklabels():
        label.set_text(_display(label.get_text()))
    ax.set_yticklabels([_display(t.get_text()) for t in ax.get_yticklabels()])
    _save(pdf, fig)


def page_task_heatmap(pdf: PdfPages, joined: pd.DataFrame) -> None:
    open_df = open_scored_subset(joined)
    if open_df.empty:
        return
    agg = (
        open_df.groupby(["model", "eval_task_id", "partition"])["row_score"]
        .mean()
        .reset_index()
    )
    agg["score_pct"] = agg["row_score"] * 100
    for part in ("public", "private_id", "private_ood"):
        sub = agg[agg["partition"] == part]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 6))
        pivot = sub.pivot(index="model", columns="eval_task_id", values="score_pct")
        pivot.index = [_display(m) for m in pivot.index]
        pivot.columns = [EVAL_TASK_LABELS.get(c, c) for c in pivot.columns]
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax, vmin=0, vmax=100)
        ax.set_title(f"Per-task accept % — {PARTITION_LABELS[part]}")
        _save(pdf, fig)


def page_qwen_vs_sonnet(pdf: PdfPages, joined: pd.DataFrame) -> None:
    open_df = open_scored_subset(joined)
    if open_df.empty or "qwen_row_score" not in open_df.columns:
        return
    both = open_df[open_df["qwen_row_score"].notna() & open_df["row_score"].notna()]
    if both.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 8.5))
    delta = (both["row_score"] - both["qwen_row_score"]) * 100
    by_model = both.assign(delta_pp=delta).groupby("model")["delta_pp"].mean().sort_values()
    by_model.index = [_display(m) for m in by_model.index]
    by_model.plot(kind="barh", ax=ax, color="#56B4E9")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Mean Δ accept % (Sonnet − Qwen production judge)")
    ax.set_title("Judge shift on overlapping rows (no extra API)")
    _save(pdf, fig)


def page_model_cards(pdf: PdfPages, joined: pd.DataFrame, stats: pd.DataFrame) -> None:
    for model in joined["model"].unique():
        mdf = joined[joined["model"] == model]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(_display(model), fontsize=14)

        v = model_verdict(stats, model)
        ax = axes[0, 0]
        ax.axis("off")
        lines = [
            f"Public open macro: {v.get('public', float('nan'))*100:.1f}%",
            f"Private ID macro:  {v.get('private_id', float('nan'))*100:.1f}%",
            f"Private OOD macro: {v.get('private_ood', float('nan'))*100:.1f}%",
            f"Δ public−ID:  {v.get('delta_id', float('nan'))*100:+.1f} pp",
            f"Δ public−OOD: {v.get('delta_ood', float('nan'))*100:+.1f} pp",
        ]
        ax.text(0.05, 0.9, "\n".join(lines), va="top", fontsize=11)

        ax = axes[0, 1]
        open_df = open_scored_subset(mdf)
        if not open_df.empty:
            task_means = open_df.groupby("eval_task_id")["row_score"].mean() * 100
            task_means.index = [EVAL_TASK_LABELS.get(t, t) for t in task_means.index]
            task_means.plot(kind="bar", ax=ax, color="#009E73")
            ax.set_ylabel("Accept %")
            ax.set_title("Per open task (all partitions)")
            ax.tick_params(axis="x", rotation=45)

        ax = axes[1, 0]
        iou = mdf[(mdf["eval_task_id"] == IOU_EVAL_TASK) & mdf["row_score"].notna()]
        if not iou.empty:
            for part, color in PARTITION_COLORS.items():
                vals = iou[iou["partition"] == part]["row_score"].sort_values()
                if len(vals):
                    ax.plot(np.linspace(0, 1, len(vals)), vals.values, label=PARTITION_LABELS[part], color=color)
            ax.set_xlabel("ECDF")
            ax.set_ylabel("IoU")
            ax.set_title("obj_det_open IoU")
            ax.legend(fontsize=8)

        ax = axes[1, 1]
        if "dataset" in open_df.columns and not open_df.empty:
            pub = open_df[open_df["partition"] == "public"].groupby("dataset")["row_score"].mean()
            pid = open_df[open_df["partition"] == "private_id"].groupby("dataset")["row_score"].mean()
            shared = pub.index.intersection(pid.index)
            if len(shared):
                gap = (pub[shared] - pid[shared]).sort_values().head(8)
                gap.plot(kind="barh", ax=ax, color="#CC79A7")
                ax.set_xlabel("Δ accept (public − private ID)")
                ax.set_title("Top ID dataset gaps")

        _save(pdf, fig)


def render_pdf(
    joined: pd.DataFrame,
    stats: pd.DataFrame,
    cov: pd.DataFrame,
    cfg: V4SonnetReportConfig,
) -> Path:
    cfg.ensure_dirs()
    with PdfPages(cfg.pdf_path) as pdf:
        page_title(pdf, cfg, joined)
        page_coverage(pdf, cov)
        page_dumbbell(pdf, stats)
        page_delta_forest(pdf, stats)
        page_task_heatmap(pdf, joined)
        page_qwen_vs_sonnet(pdf, joined)
        page_model_cards(pdf, joined, stats)
    return cfg.pdf_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v4 Sonnet-judge score PDF")
    parser.add_argument("--rejoin", action="store_true", help="Rebuild joined parquet")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = V4SonnetReportConfig.from_env()
    if args.output:
        cfg.output_dir = args.output

    if args.rejoin or not cfg.cache_path.exists():
        joined = build_joined_scores(cfg)
    else:
        joined = pd.read_parquet(cfg.cache_path)

    if joined.empty:
        print(
            "No joined scores. Ensure v4 mapping, results JSONLs, and Sonnet judge caches "
            "exist on Pasteur (see docs/open_vqa_per_dataset_judge.md)."
        )
        return 1

    mapping = load_v4_mapping(cfg.split_mapping_path)
    cov = coverage_table(joined, mapping)
    cov.to_csv(cfg.cache_path.parent / "coverage_sonnet_judge.csv", index=False)
    stats = compute_partition_stats(joined, cfg)
    pdf = render_pdf(joined, stats, cov, cfg)
    print(f"Wrote {pdf}")
    print(f"Joined cache: {cfg.cache_path}")
    print(f"Stats: {cfg.stats_cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
