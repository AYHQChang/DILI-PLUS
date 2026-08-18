"""Figure 4: earlier-cutoff information erosion and matched model performance."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from diliplus.config import load_settings
from diliplus.reporting.formal_assets import (
    DEEP_MODELS,
    MODEL_COLORS,
    MODEL_LABELS,
    configure_publication_style,
    formal_sources,
    load_csv,
    panel_title,
    read_json,
    save_figure,
)


MARKERS = {
    "MultiModalTextCNN": "o",
    "MultiModalBiLSTM": "s",
    "MultimodalTransformerBaseline": "^",
    "TimeAwareMultimodalTransformer": "D",
}
MODALITIES = {
    "Medication": {"prefix": "medication", "color": "#99BADF", "marker": "o"},
    "Laboratory": {"prefix": "laboratory", "color": "#F8BF92", "marker": "s"},
    "Diagnosis": {"prefix": "diagnosis", "color": "#99CDCE", "marker": "^"},
}
GRID = "#E0E0E0"


def _load_availability(settings):
    path = settings.paths.manifests / "code09_early_warning_contract.json"
    payload = read_json(path)
    if payload.get("status") != "PASS" or payload.get("contract") != "code09_strict_early_warning_v1":
        raise ValueError(f"Early-warning availability manifest is not PASS: {path}")
    rows = sorted(payload.get("horizon_availability", []), key=lambda row: float(row["effective_horizon_hours_before_index"]))
    horizons = tuple(float(row["effective_horizon_hours_before_index"]) for row in rows)
    if horizons != (24.0, 48.0, 72.0):
        raise ValueError("Figure 4 requires complete 24/48/72-hour availability rows")
    if len({int(row["encounters"]) for row in rows}) != 1:
        raise ValueError("Figure 4 availability rows must refer to one fixed cohort")
    return rows


def _availability_trajectory(ax, rows, value_function, *, letter, title, ylabel, ylim=None, label_mode="all"):
    horizons = np.array([24.0, 48.0, 72.0])
    label_offsets = {"Medication": 14, "Laboratory": -18, "Diagnosis": -1}
    for label, spec in MODALITIES.items():
        values = np.asarray([value_function(row, spec["prefix"]) for row in rows], dtype=float)
        ax.plot(
            horizons,
            values,
            color=spec["color"],
            marker=spec["marker"],
            linewidth=2.2,
            markersize=7,
            label=label,
        )
        points = list(zip(horizons, values))
        if label_mode == "last":
            points = points[-1:]
        elif label_mode != "all":
            raise ValueError(f"Unsupported label mode: {label_mode}")
        for x, value in points:
            ax.annotate(
                f"{value:.1f}",
                (x, value),
                xytext=(0, label_offsets[label]),
                textcoords="offset points",
                ha="center",
                fontsize=11.5,
                fontweight="bold",
            )
    ax.set_xticks(horizons, ["24", "48", "72"])
    ax.set_xlabel("Prediction horizon before index (hours)")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(color=GRID, linewidth=0.7)
    panel_title(ax, letter, title)


def _sequence_length_panel(ax, rows):
    horizons = np.array([24.0, 48.0, 72.0])
    for label, spec in MODALITIES.items():
        prefix = spec["prefix"]
        median = np.asarray([float(row[f"{prefix}_length_median"]) for row in rows])
        q1 = np.asarray([float(row[f"{prefix}_length_q1"]) for row in rows])
        q3 = np.asarray([float(row[f"{prefix}_length_q3"]) for row in rows])
        ax.plot(horizons, median, color=spec["color"], marker=spec["marker"], linewidth=2.2, markersize=7, label=label)
        ax.fill_between(horizons, q1, q3, color=spec["color"], alpha=0.15, linewidth=0)
        for x, value in zip(horizons, median):
            ax.text(x, value + 0.8, f"{value:g}", ha="center", fontsize=11.5, fontweight="bold")
    ax.set_xticks(horizons, ["24", "48", "72"])
    ax.set_xlabel("Prediction horizon before index (hours)")
    ax.set_ylabel("Retained tokens per encounter")
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, linewidth=0.7)
    panel_title(ax, "C", "Median retained sequence length (IQR band)")


def _auprc_panel(ax, bootstrap, prevalence):
    horizons = np.array([24.0, 48.0, 72.0])
    for model in DEEP_MODELS:
        rows = bootstrap[(bootstrap["Model_Architecture"] == model) & (bootstrap["metric"] == "AUPRC")].sort_values(
            "Effective_Horizon_Hours_Before_Index"
        )
        if tuple(rows["Effective_Horizon_Hours_Before_Index"].astype(float)) != tuple(horizons):
            raise ValueError(f"Incomplete strict early-warning AUPRC trajectory for {model}")
        estimate = rows["estimate"].astype(float).to_numpy()
        lower = rows["ci_lower"].astype(float).to_numpy()
        upper = rows["ci_upper"].astype(float).to_numpy()
        ax.plot(
            horizons,
            estimate,
            color=MODEL_COLORS[model],
            marker=MARKERS[model],
            linewidth=2.0,
            markersize=6,
            label=MODEL_LABELS[model],
        )
        ax.fill_between(horizons, lower, upper, color=MODEL_COLORS[model], alpha=0.14, linewidth=0)
    ax.axhline(prevalence, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xticks(horizons, ["24", "48", "72"])
    ax.set_xlabel("Prediction horizon before index (hours)")
    ax.set_ylabel("AUPRC (95% CI)")
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, linewidth=0.7)
    panel_title(ax, "D", "Matched earlier-cutoff discrimination")
    ax.text(
        0.98,
        0.96,
        "Same 24-h checkpoints, temperatures, and test membership; no horizon-specific retraining.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#666666",
    )


def generate_early_warning_figure(settings=None):
    settings = settings or load_settings()
    configure_publication_style()
    availability = _load_availability(settings)
    sources = formal_sources(settings)
    bootstrap = load_csv(
        sources["early_warning_bootstrap"],
        (
            "Run_ID",
            "Probability_Mode",
            "Model_Architecture",
            "Effective_Horizon_Hours_Before_Index",
            "metric",
            "estimate",
            "ci_lower",
            "ci_upper",
        ),
    )
    pooled = load_csv(
        sources["early_warning_pooled"],
        ("Run_ID", "Probability_Mode", "Model_Architecture", "Effective_Horizon_Hours_Before_Index", "Prevalence"),
    )
    run_id = sources["run_id"]
    bootstrap = bootstrap[(bootstrap["Run_ID"] == run_id) & (bootstrap["Probability_Mode"] == "calibrated")].copy()
    pooled = pooled[(pooled["Run_ID"] == run_id) & (pooled["Probability_Mode"] == "calibrated")].copy()
    if set(bootstrap["Model_Architecture"]) != set(DEEP_MODELS):
        raise ValueError("Figure 4 requires all four formal deep models")
    prevalence = float(pooled["Prevalence"].iloc[0])

    base = availability[0]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12.5), constrained_layout=True)
    _availability_trajectory(
        axes[0, 0],
        availability,
        lambda row, prefix: 100.0 * float(row[f"{prefix}_events_retained"]) / float(base[f"{prefix}_events_retained"]),
        letter="A",
        title="Clinical events retained relative to 24 hours",
        ylabel="Retained events (%)",
        ylim=(50, 105),
        label_mode="last",
    )
    _availability_trajectory(
        axes[0, 1],
        availability,
        lambda row, prefix: 100.0 * (1.0 - float(row[f"{prefix}_missing_encounters"]) / float(row["encounters"])),
        letter="B",
        title="Encounters retaining each modality",
        ylabel="Non-empty encounters (%)",
        ylim=(40, 105),
    )
    _sequence_length_panel(axes[1, 0], availability)
    _auprc_panel(axes[1, 1], bootstrap, prevalence)

    modality_handles = [
        Line2D([0], [0], color=spec["color"], marker=spec["marker"], linewidth=2.0, label=label)
        for label, spec in MODALITIES.items()
    ]
    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS[model], marker=MARKERS[model], linewidth=2.0, label=MODEL_LABELS[model])
        for model in DEEP_MODELS
    ]
    fig.legend(handles=modality_handles, loc="outside lower left", ncol=3, frameon=False)
    fig.legend(handles=model_handles, loc="outside lower right", ncol=2, frameon=False)
    png, pdf = save_figure(fig, settings, "Fig_4_Earlier_Cutoff_Information_Erosion")
    print(f"[PASS] Figure 4: {png} | {pdf}")
    return png, pdf


if __name__ == "__main__":
    generate_early_warning_figure()
