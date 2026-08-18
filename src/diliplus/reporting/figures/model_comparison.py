"""Figure 2: formal discrimination and fixed alert-budget performance."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from diliplus.config import load_settings
from diliplus.reporting.formal_assets import (
    FORMAL_MODELS,
    MODEL_COLORS,
    MODEL_LABELS,
    calibrated_main_frames,
    configure_publication_style,
    panel_title,
    save_figure,
)


def _metric_rows(bootstrap, metric):
    rows = bootstrap[bootstrap["metric"] == metric].set_index("Model_Architecture")
    missing = sorted(set(FORMAL_MODELS) - set(rows.index))
    if missing:
        raise ValueError(f"Missing formal bootstrap {metric} rows: {missing}")
    return rows.loc[list(FORMAL_MODELS)]


def _forest(ax, bootstrap, metric, title, xlabel, letter, reference=None):
    rows = _metric_rows(bootstrap, metric)
    y = np.arange(len(FORMAL_MODELS))
    for position, model in zip(y, FORMAL_MODELS):
        row = rows.loc[model]
        estimate = float(row["estimate"])
        lower = float(row["ci_lower"])
        upper = float(row["ci_upper"])
        ax.errorbar(
            estimate,
            position,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt="o",
            color=MODEL_COLORS[model],
            markeredgecolor="#333333",
            markeredgewidth=0.5,
            markersize=6.5,
            elinewidth=1.6,
            capsize=3,
            zorder=3,
        )
        ax.text(
            upper + 0.006,
            position,
            f"{estimate:.3f}",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_yticks(y, [MODEL_LABELS[model] for model in FORMAL_MODELS])
    ax.invert_yaxis()
    if reference is not None:
        ax.axvline(reference, color="#555555", linestyle="--", linewidth=1.0)
    low = min(float(rows["ci_lower"].min()), reference if reference is not None else 1.0)
    high = float(rows["ci_upper"].max())
    pad = max(0.015, (high - low) * 0.18)
    ax.set_xlim(max(0.0, low - pad), min(1.0, high + 2.2 * pad))
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    panel_title(ax, letter, title)


def _budget_panel(ax, pooled, column_suffix, title, ylabel, letter, prevalence=None):
    budgets = np.array([0.005, 0.01, 0.02, 0.05])
    for model in FORMAL_MODELS:
        row = pooled[pooled["Model_Architecture"] == model].iloc[0]
        values = [float(row[f"Top_{str(value).replace('.', 'p')}_{column_suffix}"]) for value in budgets]
        ax.plot(
            budgets * 100,
            np.asarray(values) * 100,
            color=MODEL_COLORS[model],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=MODEL_LABELS[model],
        )
    if prevalence is not None:
        ax.axhline(prevalence * 100, color="#555555", linestyle="--", linewidth=1.0)
        ax.text(
            0.58,
            prevalence * 100 + 0.5,
            "Cohort prevalence",
            color="#555555",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xticks(budgets * 100, ["0.5", "1", "2", "5"])
    ax.set_xlabel("Flagged encounters (% of cohort)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(color="#E0E0E0", linewidth=0.7)
    panel_title(ax, letter, title)


def generate_advanced_figure_2(settings=None):
    settings = settings or load_settings()
    configure_publication_style()
    pooled, bootstrap = calibrated_main_frames(settings)
    prevalence = float(pooled["Prevalence"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(18, 12.5), constrained_layout=True)
    _forest(
        axes[0, 0], bootstrap, "AUPRC", "Primary discrimination", "AUPRC (95% CI)", "A", reference=prevalence
    )
    _forest(
        axes[0, 1], bootstrap, "AUROC", "Secondary discrimination", "AUROC (95% CI)", "B"
    )
    _budget_panel(
        axes[1, 0], pooled, "Precision", "Positive predictive value by alert budget", "PPV (%)", "C", prevalence
    )
    _budget_panel(
        axes[1, 1], pooled, "Recall", "Sensitivity by alert budget", "Sensitivity (%)", "D"
    )

    handles = [
        Line2D([0], [0], color=MODEL_COLORS[m], marker="o", linewidth=1.8, markersize=4.5, label=MODEL_LABELS[m])
        for m in FORMAL_MODELS
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    png, pdf = save_figure(fig, settings, "Fig_2_Formal_Model_Comparison")
    print(f"[PASS] Figure 2: {png} | {pdf}")
    return png, pdf


if __name__ == "__main__":
    generate_advanced_figure_2()
