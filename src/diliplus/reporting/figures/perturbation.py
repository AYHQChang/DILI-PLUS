"""Figure 7: medication-token perturbation sensitivity audit.

Embedding attenuation and token substitution probe model behaviour for one
illustrative input. They are not dose-response, treatment, counterfactual, or
causal estimates.
"""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd

from diliplus.config import load_settings
from diliplus.reporting.formal_assets import configure_publication_style, save_figure


TOKEN_PALETTE = (
    "#DF9E9B", "#99CDCE", "#F8BF92", "#99BADF", "#999ACD",
    "#FFB3DD", "#A8E6CF", "#FFD3B6", "#D4A5A5", "#9DC8C8",
    "#B5B8D3", "#F4B6C2", "#CDE5DC", "#E8D5C4", "#A2D5AB",
)
TOKEN_MARKERS = ("o", "s", "^", "D", "v", "p", "*", "h", "X", "<")
ORIGINAL_COLOR = "#999ACD"
SUBSTITUTED_COLOR = "#99CDCE"


def _clean_token(value: str) -> str:
    label = str(value)
    for tag in ("[*]", "[Max Variance]", "[Clinical Target]", "[Clinical Swap]"):
        label = label.replace(tag, "")
    return label.strip()


def generate_simulation_figure(settings=None):
    """Render an identifier-free local medication-token sensitivity audit."""

    settings = settings or load_settings()
    source = settings.paths.reports / "06c_Counterfactual_Trajectory.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    required = {
        "Intervention_Type",
        "Targeted_Medications_EN",
        "Parameter",
        "Predicted_DILI_Risk",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{source} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{source} contains zero rows")
    frame = frame.rename(columns={"Predicted_DILI_Risk": "Model_Probability_Pct"})

    attenuation = frame[frame["Intervention_Type"] == "Dose Tapering"].copy()
    substitution = frame[frame["Intervention_Type"] == "Substitution"].copy()
    if attenuation.empty or substitution.empty:
        raise ValueError("Both attenuation and substitution rows are required")
    attenuation["Alpha"] = attenuation["Parameter"].map(
        lambda value: float(str(value).split("=")[1])
    )

    configure_publication_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 8.5),
        gridspec_kw={"width_ratios": (1.65, 1.0)},
        constrained_layout=True,
    )
    trajectory_ax, substitution_ax = axes
    color_cycle = itertools.cycle(TOKEN_PALETTE)
    marker_cycle = itertools.cycle(TOKEN_MARKERS)
    for token in attenuation["Targeted_Medications_EN"].drop_duplicates():
        rows = attenuation[attenuation["Targeted_Medications_EN"] == token].sort_values(
            "Alpha", ascending=False
        )
        trajectory_ax.plot(
            rows["Alpha"],
            rows["Model_Probability_Pct"],
            marker=next(marker_cycle),
            color=next(color_cycle),
            linewidth=2.2,
            markersize=7,
            label=_clean_token(token),
            markeredgecolor="white",
            zorder=3,
        )
    trajectory_ax.set_xlim(1.05, -0.05)
    trajectory_ax.set_xticks(
        (1.0, 0.75, 0.5, 0.25, 0.0),
        ("1.0\n(original)", "0.75", "0.50", "0.25", "0.0\n(zero vector)"),
    )
    trajectory_ax.set_title(
        "A  Medication-embedding attenuation",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    trajectory_ax.set_xlabel(r"Embedding amplitude multiplier ($\alpha$)")
    trajectory_ax.set_ylabel("Model-predicted AHI-proxy probability (%)")
    trajectory_ax.grid(color="#E0E0E0", linewidth=0.7)
    trajectory_ax.legend(
        loc="upper left",
        frameon=False,
        title="Perturbed medication token",
        title_fontproperties={"weight": "bold", "size": 12},
        fontsize=11,
    )

    baseline_rows = attenuation[
        attenuation["Targeted_Medications_EN"].str.contains(
            "Atorvastatin", case=False, na=False
        )
        & (attenuation["Alpha"] == 1.0)
    ]
    if len(baseline_rows) != 1:
        raise ValueError("Expected one original atorvastatin token row")
    baseline_probability = float(baseline_rows["Model_Probability_Pct"].iloc[0])
    substituted_probability = float(substitution["Model_Probability_Pct"].iloc[0])
    delta = substituted_probability - baseline_probability
    bars = substitution_ax.bar(
        ("Original token\n(atorvastatin)", "Substituted token\n(pravastatin)"),
        (baseline_probability, substituted_probability),
        color=(ORIGINAL_COLOR, SUBSTITUTED_COLOR),
        edgecolor="#333333",
        linewidth=1.0,
        width=0.56,
        zorder=3,
    )
    for bar, probability in zip(bars, (baseline_probability, substituted_probability)):
        substitution_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{probability:.1f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )
    substitution_ax.annotate(
        rf"$\Delta p_{{model}}={delta:+.1f}$ pp",
        xy=(0.78, substituted_probability),
        xytext=(0.5, max(baseline_probability, substituted_probability) * 1.16),
        arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.5},
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    substitution_ax.set_ylim(0, max(baseline_probability, substituted_probability) * 1.32)
    substitution_ax.set_title(
        "B  Predefined token-substitution sensitivity",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    substitution_ax.set_xlabel("Medication-token input")
    substitution_ax.set_ylabel("Model-predicted AHI-proxy probability (%)")
    substitution_ax.grid(axis="y", color="#E0E0E0", linewidth=0.7, zorder=0)

    png, pdf = save_figure(fig, settings, "Fig_7_Medication_Token_Sensitivity")
    print(f"[PASS] Figure 7 token sensitivity audit: {png} | {pdf}")
    return png, pdf


if __name__ == "__main__":
    generate_simulation_figure()
