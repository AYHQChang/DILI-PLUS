"""Figure 6: local medication-token attribution audit.

The plot is descriptive of one model input. Positive and negative integrated-
gradient values indicate directions in model output, not harmful, protective,
therapeutic, or causal effects.
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from diliplus.config import load_settings
from diliplus.reporting.formal_assets import configure_publication_style, save_figure


OUTPUT_INCREASING = "#DF9E9B"
OUTPUT_DECREASING = "#99CDCE"


def generate_waterfall_chart(settings=None):
    """Render an identifier-free local attribution audit from a saved CSV."""

    settings = settings or load_settings()
    source = settings.paths.reports / "06b_Target_Patient_Attribution.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    required = {"Medication_EN", "Attribution_Score", "Contribution_Pct"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{source} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{source} contains zero rows")

    positive = frame[frame["Attribution_Score"] > 0].sort_values(
        "Attribution_Score", ascending=False
    )
    negative = frame[frame["Attribution_Score"] < 0].sort_values("Attribution_Score")
    display = pd.concat([positive, negative], ignore_index=True)
    if display.empty:
        raise ValueError("No non-zero attribution scores are available")

    labels = [textwrap.fill(str(value), width=16) for value in display["Medication_EN"]]
    scores = display["Attribution_Score"].astype(float).to_numpy()
    percentages = display["Contribution_Pct"].astype(float).to_numpy()
    bottoms = np.concatenate(([0.0], np.cumsum(scores[:-1])))
    colors = [OUTPUT_INCREASING if value > 0 else OUTPUT_DECREASING for value in scores]

    configure_publication_style()
    fig, ax = plt.subplots(figsize=(18, 9), constrained_layout=True)
    bars = ax.bar(
        np.arange(len(labels)),
        scores,
        bottom=bottoms,
        color=colors,
        edgecolor="#333333",
        linewidth=1.0,
        width=0.58,
        zorder=3,
    )
    for index in range(1, len(scores)):
        ax.plot(
            [index - 1, index],
            [bottoms[index], bottoms[index]],
            color="#777777",
            linestyle="--",
            linewidth=1.2,
            zorder=2,
        )
    score_span = max(0.1, float(scores.max() - scores.min()))
    for bar, bottom, score, percentage in zip(bars, bottoms, scores, percentages):
        end = bottom + score
        offset = max(0.002, 0.03 * score_span)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            end + (offset if score > 0 else -offset),
            f"{score:+.3f}\n({percentage:+.1f}%)",
            ha="center",
            va="bottom" if score > 0 else "top",
            fontsize=11.5,
            fontweight="bold",
            color="#333333",
        )

    all_points = np.concatenate((bottoms, bottoms + scores))
    span = max(0.1, float(all_points.max() - all_points.min()))
    ax.set_ylim(float(all_points.min() - 0.28 * span), float(all_points.max() + 0.28 * span))
    ax.axhline(0, color="#333333", linewidth=1.4, zorder=1)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=28, ha="right")
    ax.set_title(
        "Medication-token attribution for one illustrative model input",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Medication tokens")
    ax.set_ylabel("Integrated-gradient attribution score")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.legend(
        handles=(
            Patch(facecolor=OUTPUT_INCREASING, label="Model-output-increasing attribution"),
            Patch(facecolor=OUTPUT_DECREASING, label="Model-output-decreasing attribution"),
        ),
        loc="upper right",
        frameon=False,
    )
    png, pdf = save_figure(fig, settings, "Fig_6_Local_Attribution_Audit")
    print(f"[PASS] Figure 6 local attribution audit: {png} | {pdf}")
    return png, pdf


if __name__ == "__main__":
    generate_waterfall_chart()
