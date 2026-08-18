"""Figure 5: what additional model complexity contributed under formal evaluation."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from diliplus.config import load_settings
from diliplus.reporting.formal_assets import (
    ABLATION_COLORS,
    ABLATION_LABELS,
    ABLATION_ORDER,
    DEEP_MODELS,
    MODEL_COLORS,
    MODEL_LABELS,
    auxiliary_sources,
    configure_publication_style,
    formal_sources,
    load_csv,
    panel_title,
    save_figure,
)


INK = "#333333"
GRID = "#E0E0E0"


def _paired_ablation(ax, paired):
    comparators = list(ABLATION_ORDER[1:])
    rows = paired[
        (paired["Probability_Mode"] == "calibrated") & (paired["metric"] == "AUPRC")
    ].set_index("Comparator")
    if set(comparators) - set(rows.index):
        raise ValueError("Paired ablation AUPRC comparisons are incomplete")
    y = np.arange(len(comparators))
    ax.axvline(0, color="#555555", linewidth=1.0)
    for position, model in zip(y, comparators):
        row = rows.loc[model]
        delta, lower, upper = (float(row[key]) for key in ("delta_primary_minus_comparator", "ci_lower", "ci_upper"))
        ax.errorbar(
            delta,
            position,
            xerr=[[delta - lower], [upper - delta]],
            fmt="o",
            color=ABLATION_COLORS[model],
            markeredgecolor=INK,
            markeredgewidth=0.6,
            markersize=7,
            elinewidth=1.8,
            capsize=4,
        )
        significant = float(row["p_value_holm"]) < 0.05
        ax.text(upper + 0.003, position, f"{delta:+.3f}{'*' if significant else ''}", va="center", fontsize=11.5, fontweight="bold")
    ax.set_yticks(y, [ABLATION_LABELS[model] for model in comparators])
    ax.invert_yaxis()
    ax.set_xlabel("AUPRC difference: full TA-MMT minus ablation (95% CI)")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    panel_title(ax, "A", "Incremental value of modalities and time encoding")
    ax.text(
        0.02,
        0.98,
        "Negative values favour the ablation  ·  * Holm-adjusted P < 0.05",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#666666",
    )


def _head_sensitivity(ax, frame):
    models = ("MultimodalTransformerBaseline", "TimeAwareMultimodalTransformer")
    rows = frame[(frame["Probability_Mode"] == "calibrated") & (frame["metric"] == "AUPRC")].set_index(
        "Model_Architecture"
    )
    if set(models) - set(rows.index):
        raise ValueError("Head-sensitivity AUPRC comparisons are incomplete")
    y = np.arange(len(models))
    ax.axvline(0, color="#555555", linewidth=1.0)
    for position, model in zip(y, models):
        row = rows.loc[model]
        delta, lower, upper = (float(row[key]) for key in ("delta_primary_minus_comparator", "ci_lower", "ci_upper"))
        ax.errorbar(
            delta,
            position,
            xerr=[[delta - lower], [upper - delta]],
            fmt="D",
            color=MODEL_COLORS[model],
            markeredgecolor=INK,
            markeredgewidth=0.6,
            markersize=7,
            elinewidth=1.8,
            capsize=4,
        )
        ax.text(upper + 0.002, position, f"{delta:+.3f}", va="center", fontsize=12, fontweight="bold")
    ax.set_yticks(y, [MODEL_LABELS[model] for model in models])
    ax.invert_yaxis()
    ax.set_xlabel("AUPRC difference: 8 heads minus 4 heads (95% CI)")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    panel_title(ax, "B", "Attention-head sensitivity at 128 dimensions")


def _seed_stability(ax, metrics):
    models = ("MultiModalTextCNN", "TimeAwareMultimodalTransformer")
    frame = metrics[
        (metrics["Probability_Mode"] == "calibrated") & metrics["Model_Architecture"].isin(models)
    ].copy()
    if len(frame) != 6 or set(frame["Seed_Index"].astype(int)) != {0, 1, 2}:
        raise ValueError("Seed-stability panel requires two models across three matched seeds")
    for seed, rows in frame.groupby(frame["Seed_Index"].astype(int)):
        values = []
        for model in models:
            match = rows[rows["Model_Architecture"] == model]
            if len(match) != 1:
                raise ValueError(f"Missing matched seed {seed} result for {model}")
            values.append(float(match["AUPRC"].iloc[0]))
        ax.plot([0, 1], values, color="#BDBDBD", linewidth=1.0, zorder=1)
        for x, model, value in zip([0, 1], models, values):
            ax.scatter(x, value, color=MODEL_COLORS[model], edgecolor=INK, linewidth=0.6, s=52, zorder=3)
    for x, model in zip([0, 1], models):
        values = frame.loc[frame["Model_Architecture"] == model, "AUPRC"].astype(float)
        ax.errorbar(
            x,
            values.mean(),
            yerr=values.std(ddof=1),
            fmt="_",
            color="#222222",
            markersize=18,
            markeredgewidth=2.0,
            elinewidth=1.3,
            capsize=4,
            zorder=4,
        )
        ax.text(x, 0.97, f"{values.mean():.3f} ± {values.std(ddof=1):.3f}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11.5, fontweight="bold")
    ax.set_xticks([0, 1], [MODEL_LABELS[model] for model in models])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("AUPRC")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    panel_title(ax, "C", "Matched three-seed stability")


def _resource_footprint(ax, resource):
    frame = resource[resource["Model_Architecture"].isin(DEEP_MODELS)].copy()
    if set(frame["Model_Architecture"]) != set(DEEP_MODELS):
        raise ValueError("Resource panel requires all four formal deep models")
    grouped = frame.groupby("Model_Architecture", sort=False).agg(
        parameter_count=("Parameter_Count", "max"),
        mean_seconds=("Fold_Duration_Seconds", "mean"),
        peak_gpu_mb=("Peak_GPU_Memory_MB", "max"),
    )
    for model in DEEP_MODELS:
        row = grouped.loc[model]
        x = float(row["parameter_count"]) / 1_000_000.0
        y = float(row["mean_seconds"])
        size = 70 + float(row["peak_gpu_mb"]) / 8.0
        ax.scatter(x, y, s=size, color=MODEL_COLORS[model], edgecolor=INK, linewidth=0.7, alpha=0.9)
        ax.annotate(
            f"{MODEL_LABELS[model]}\n{float(row['peak_gpu_mb']):.0f} MB",
            (x, y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xlabel("Trainable parameters (millions)")
    ax.set_ylabel("Mean fold duration (seconds)")
    ax.set_xlim(4.7, 7.55)
    ax.set_ylim(68, 139)
    ax.grid(color=GRID, linewidth=0.7)
    panel_title(ax, "D", "Deep-model computational footprint")


def generate_robustness_figure(settings=None):
    settings = settings or load_settings()
    configure_publication_style()
    auxiliary = auxiliary_sources(settings)
    formal = formal_sources(settings)
    ablation_paired = load_csv(
        auxiliary["ablation_paired"],
        ("Run_ID", "Probability_Mode", "Comparator", "metric", "delta_primary_minus_comparator", "ci_lower", "ci_upper", "p_value_holm"),
    )
    head_paired = load_csv(
        auxiliary["sensitivity_paired"],
        ("Probability_Mode", "Model_Architecture", "metric", "delta_primary_minus_comparator", "ci_lower", "ci_upper", "p_value_holm"),
    )
    stability = load_csv(
        auxiliary["stability_metrics"],
        ("Seed_Index", "Model_Architecture", "Probability_Mode", "AUPRC"),
    )
    resource = load_csv(
        formal["resource_usage"],
        ("Model_Architecture", "Parameter_Count", "Fold_Duration_Seconds", "Peak_GPU_Memory_MB", "Device"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 12.5), constrained_layout=True)
    _paired_ablation(axes[0, 0], ablation_paired)
    _head_sensitivity(axes[0, 1], head_paired)
    _seed_stability(axes[1, 0], stability)
    _resource_footprint(axes[1, 1], resource)
    png, pdf = save_figure(fig, settings, "Fig_5_Model_Complexity_Audit")
    print(f"[PASS] Figure 5: {png} | {pdf}")
    return png, pdf


if __name__ == "__main__":
    generate_robustness_figure()
