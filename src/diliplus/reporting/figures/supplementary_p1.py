"""Supplementary P1 calibration, decision-curve, and subgroup audit figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from diliplus.config import load_settings
from diliplus.reporting.formal_assets import (
    FORMAL_MODELS,
    MODEL_COLORS,
    MODEL_LABELS,
    configure_publication_style,
    formal_sources,
    panel_title,
    save_figure,
)
from diliplus.reporting.p1_contract import (
    AUDIT_COLORS,
    AUDIT_LABELS,
    AUDIT_MODELS,
    PROCESS_MODEL,
    RUN_ID,
)


MARKERS = {
    "LogisticRegression": "o",
    "XGBoost": "s",
    "TimeAwareMultimodalTransformer": "^",
    PROCESS_MODEL: "D",
}


def _load_p1(settings):
    metric_dir = settings.paths.reports / "runs" / RUN_ID / "metrics"
    calibration = pd.read_csv(metric_dir / "calibration_bin_intervals.csv")
    subgroup = pd.read_csv(metric_dir / "subgroup_metrics.csv")
    return calibration, subgroup


def generate_calibration_decision_figure(settings=None):
    settings = settings or load_settings()
    configure_publication_style()
    calibration, _ = _load_p1(settings)
    dca = pd.read_csv(formal_sources(settings)["dca"])
    dca = dca[dca["Probability_Mode"] == "calibrated"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.2), constrained_layout=True)
    ax = axes[0]
    for model in FORMAL_MODELS:
        rows = calibration[calibration["Model_Architecture"] == model].sort_values("bin")
        x = rows["mean_predicted"].to_numpy(dtype=float) * 100
        y = rows["observed_fraction"].to_numpy(dtype=float) * 100
        lower = rows["ci_lower"].to_numpy(dtype=float) * 100
        upper = rows["ci_upper"].to_numpy(dtype=float) * 100
        ax.plot(
            x,
            y,
            color=MODEL_COLORS[model],
            marker="o",
            linewidth=1.7,
            markersize=4.5,
            label=MODEL_LABELS[model],
            zorder=3,
        )
        ax.fill_between(x, lower, upper, color=MODEL_COLORS[model], alpha=0.10, linewidth=0)
    limit = 11.0
    ax.plot([0, limit], [0, limit], color="#444444", linestyle="--", linewidth=1.2, label="Ideal")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted probability (%)")
    ax.set_ylabel("Observed event proportion (%)")
    ax.grid(color="#E0E0E0", linewidth=0.7)
    panel_title(ax, "A", "Calibration across risk quantiles")

    ax = axes[1]
    for model in FORMAL_MODELS:
        rows = dca[dca["Model_Architecture"] == model].sort_values("threshold")
        ax.plot(
            rows["threshold"] * 100,
            rows["net_benefit_model"] * 100,
            color=MODEL_COLORS[model],
            linewidth=1.9,
            marker="o",
            markersize=3.7,
            label=MODEL_LABELS[model],
        )
    reference = dca[["threshold", "net_benefit_treat_all", "net_benefit_treat_none"]].drop_duplicates()
    ax.plot(
        reference["threshold"] * 100,
        reference["net_benefit_treat_all"] * 100,
        color="#8A8A8A",
        linestyle="--",
        linewidth=1.4,
        label="Alert all",
    )
    ax.plot(
        reference["threshold"] * 100,
        reference["net_benefit_treat_none"] * 100,
        color="#333333",
        linestyle=":",
        linewidth=1.4,
        label="Alert none",
    )
    ax.set_xlim(0.5, 5.0)
    ax.set_ylim(-0.10, 0.52)
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Net benefit per 100 encounters")
    ax.grid(color="#E0E0E0", linewidth=0.7)
    panel_title(ax, "B", "Retrospective decision-curve analysis")

    handles = [
        Line2D([0], [0], color=MODEL_COLORS[m], marker="o", linewidth=1.8, markersize=4.5, label=MODEL_LABELS[m])
        for m in FORMAL_MODELS
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#8A8A8A", linestyle="--", linewidth=1.4, label="Alert all"),
            Line2D([0], [0], color="#333333", linestyle=":", linewidth=1.4, label="Alert none"),
        ]
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    png, pdf = save_figure(fig, settings, "Fig_S1_Calibration_Decision_Curves")
    print(f"[PASS] Supplementary Figure S1: {png} | {pdf}")
    return png, pdf


def _subgroup_panel(ax, subgroup, dimension, letter, title):
    rows = subgroup[subgroup["audit_dimension"] == dimension].copy()
    categories = (
        rows[["subgroup", "subgroup_order", "N", "Positive"]]
        .drop_duplicates()
        .sort_values("subgroup_order")
    )
    x = np.arange(len(categories), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(AUDIT_MODELS))
    for offset, model in zip(offsets, AUDIT_MODELS):
        model_rows = rows[rows["Model_Architecture"] == model].set_index("subgroup")
        ordered = model_rows.loc[categories["subgroup"]]
        estimate = ordered["AUPRC"].to_numpy(dtype=float) * 100
        lower = ordered["AUPRC_ci_lower"].to_numpy(dtype=float) * 100
        upper = ordered["AUPRC_ci_upper"].to_numpy(dtype=float) * 100
        ax.errorbar(
            x + offset,
            estimate,
            yerr=[estimate - lower, upper - estimate],
            fmt=MARKERS[model],
            color=AUDIT_COLORS[model],
            markeredgecolor="#333333",
            markeredgewidth=0.5,
            markersize=6.0,
            elinewidth=1.5,
            capsize=3,
            label=AUDIT_LABELS[model],
            zorder=3,
        )
    tick_labels = [
        f"{row.subgroup}\nN={int(row.N):,}; events={int(row.Positive)}"
        for row in categories.itertuples(index=False)
    ]
    ax.set_xticks(x, tick_labels)
    ax.set_ylabel("AUPRC (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.7)
    panel_title(ax, letter, title)


def generate_subgroup_process_figure(settings=None):
    settings = settings or load_settings()
    configure_publication_style()
    _, subgroup = _load_p1(settings)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.2), constrained_layout=True)
    _subgroup_panel(axes[0], subgroup, "Sex", "A", "Recorded-sex heterogeneity audit")
    _subgroup_panel(
        axes[1],
        subgroup,
        "Observation window",
        "B",
        "Observation-window process audit",
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=AUDIT_COLORS[m],
            marker=MARKERS[m],
            linewidth=1.6,
            markersize=5,
            label=AUDIT_LABELS[m],
        )
        for m in AUDIT_MODELS
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    png, pdf = save_figure(fig, settings, "Fig_S2_Subgroup_Process_Audit")
    print(f"[PASS] Supplementary Figure S2: {png} | {pdf}")
    return png, pdf


if __name__ == "__main__":
    generate_calibration_decision_figure()
    generate_subgroup_process_figure()
