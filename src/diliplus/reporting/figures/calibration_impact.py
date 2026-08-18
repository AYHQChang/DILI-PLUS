"""Figure 3: compact formal benchmark across discrimination, calibration, and alerts."""

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


INK = "#333333"
GRID = "#E0E0E0"


def _auprc_panel(ax, bootstrap, prevalence):
    rows = bootstrap[bootstrap["metric"] == "AUPRC"].set_index("Model_Architecture")
    missing = sorted(set(FORMAL_MODELS) - set(rows.index))
    if missing:
        raise ValueError(f"Missing AUPRC bootstrap rows: {missing}")
    rows = rows.loc[list(FORMAL_MODELS)]
    y = np.arange(len(FORMAL_MODELS))
    for position, model in zip(y, FORMAL_MODELS):
        row = rows.loc[model]
        estimate, lower, upper = (float(row[key]) for key in ("estimate", "ci_lower", "ci_upper"))
        ax.errorbar(
            estimate,
            position,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt="o",
            color=MODEL_COLORS[model],
            markeredgecolor=INK,
            markeredgewidth=0.6,
            markersize=7,
            elinewidth=1.8,
            capsize=4,
        )
        ax.text(upper + 0.004, position, f"{estimate:.3f}", va="center", fontsize=12, fontweight="bold")
    ax.axvline(prevalence, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_yticks(y, [MODEL_LABELS[model] for model in FORMAL_MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.16, float(rows["ci_upper"].max()) + 0.025))
    ax.set_xlabel("AUPRC (patient-cluster bootstrap 95% CI)")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    panel_title(ax, "A", "Rare-event discrimination")
    ax.text(
        0.98,
        0.02,
        f"Dashed line = prevalence {100 * prevalence:.3f}%",
        transform=ax.transAxes,
        ha="right",
        fontsize=11,
        fontweight="bold",
        color="#666666",
    )


def _calibration_panel(ax, pooled):
    rows = pooled.set_index("Model_Architecture").loc[list(FORMAL_MODELS)]
    y = np.arange(len(FORMAL_MODELS))
    ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1.1)
    for position, model in zip(y, FORMAL_MODELS):
        row = rows.loc[model]
        slope = float(row["Calibration_Slope"])
        oe = float(row["Observed_Expected_Ratio"])
        ax.plot([slope, oe], [position, position], color=MODEL_COLORS[model], linewidth=1.5, alpha=0.8)
        ax.scatter(slope, position, marker="o", s=48, color=MODEL_COLORS[model], edgecolor=INK, linewidth=0.6)
        ax.scatter(oe, position, marker="s", s=48, facecolor="white", edgecolor=MODEL_COLORS[model], linewidth=1.8)
        ax.text(max(slope, oe) + 0.025, position, f"{slope:.2f} / {oe:.2f}", va="center", fontsize=11.5, fontweight="bold")
    ax.set_yticks(y, [MODEL_LABELS[model] for model in FORMAL_MODELS])
    ax.invert_yaxis()
    low = min(float(rows["Calibration_Slope"].min()), float(rows["Observed_Expected_Ratio"].min()))
    high = max(float(rows["Calibration_Slope"].max()), float(rows["Observed_Expected_Ratio"].max()))
    ax.set_xlim(max(0, low - 0.12), high + 0.22)
    ax.set_xlabel("Calibration estimate (ideal = 1.0)")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    panel_title(ax, "B", "Probability agreement after scaling")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=INK, linestyle="none", label="Slope"),
            Line2D([0], [0], marker="s", markerfacecolor="white", color=INK, linestyle="none", label="Observed:expected"),
        ],
        loc="lower right",
        frameon=False,
    )


def _alert_yield_panel(ax, pooled):
    rows = pooled.set_index("Model_Architecture").loc[list(FORMAL_MODELS)]
    counts = rows["Top_0p01_Count"].astype(int)
    if counts.nunique() != 1:
        raise ValueError("Top-1% alert budget must contain the same alert count for all models")
    alert_count = int(counts.iloc[0])
    y = np.arange(len(FORMAL_MODELS))
    for position, model in zip(y, FORMAL_MODELS):
        row = rows.loc[model]
        true_positive = int(round(alert_count * float(row["Top_0p01_Precision"])))
        false_alert = alert_count - true_positive
        ax.barh(position, alert_count, color="#ECECEC", height=0.58)
        ax.barh(position, true_positive, color=MODEL_COLORS[model], height=0.58)
        ax.text(
            alert_count + 10,
            position,
            f"{true_positive} TP  ·  {false_alert} other alerts",
            va="center",
            fontsize=11.5,
            fontweight="bold",
        )
    ax.set_yticks(y, [MODEL_LABELS[model] for model in FORMAL_MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0, alert_count * 1.38)
    ax.set_xlabel("Flagged encounters at a fixed top-1% budget")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    panel_title(ax, "C", "Operational yield under 447 alerts")
    ax.text(
        0.02,
        0.02,
        "Coloured segment = AHI-proxy events captured; grey = other alerts.",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color="#666666",
    )


def generate_figure_3(settings=None):
    settings = settings or load_settings()
    configure_publication_style()
    pooled, bootstrap = calibrated_main_frames(settings)
    required = {
        "Calibration_Slope",
        "Observed_Expected_Ratio",
        "Top_0p01_Count",
        "Top_0p01_Precision",
        "Prevalence",
    }
    missing = sorted(required - set(pooled.columns))
    if missing:
        raise KeyError(f"Formal pooled metrics are missing Figure 3 columns: {missing}")
    prevalence = float(pooled["Prevalence"].iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.8), constrained_layout=True)
    _auprc_panel(axes[0], bootstrap, prevalence)
    _calibration_panel(axes[1], pooled)
    _alert_yield_panel(axes[2], pooled)
    png, pdf = save_figure(fig, settings, "Fig_3_Compact_Predictive_Benchmark")
    print(f"[PASS] Figure 3: {png} | {pdf}")
    return png, pdf


if __name__ == "__main__":
    generate_figure_3()
