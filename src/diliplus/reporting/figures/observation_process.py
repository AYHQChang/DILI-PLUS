"""Figure 2: cohort observation and measurement process.

The figure uses only the aggregate Code-08 Table 1 output whose SHA256 is
frozen in the tracked Code-08 manifest.  It does not access row-level data.
"""

from __future__ import annotations

import re

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from diliplus.config import load_settings
from diliplus.reporting.formal_assets import (
    configure_publication_style,
    load_csv,
    panel_title,
    read_json,
    save_figure,
    sha256,
)


NEGATIVE = "#777777"
POSITIVE = "#DF9E9B"
MEDICATION = "#99BADF"
LABORATORY = "#F8BF92"
DIAGNOSIS = "#99CDCE"
PURPLE = "#999ACD"
INK = "#333333"
GRID = "#E0E0E0"

TABLE1_MANIFEST = "code08_cohort_table1.json"
TABLE1_RELATIVE = "p0_08_cohort_table1/table1_characteristics.csv"

CONTINUOUS_PATTERN = re.compile(
    r"^\s*(?P<median>-?\d+(?:\.\d+)?)\s*"
    r"\[\s*(?P<q1>-?\d+(?:\.\d+)?)\s*,\s*(?P<q3>-?\d+(?:\.\d+)?)\s*\]\s*$"
)


def _parse_summary(value: str) -> tuple[float, float, float]:
    match = CONTINUOUS_PATTERN.match(str(value))
    if not match:
        raise ValueError(f"Expected 'median [Q1, Q3]' summary, received: {value!r}")
    return tuple(float(match.group(name)) for name in ("median", "q1", "q3"))


def _load_table1(settings):
    manifest_path = settings.paths.manifests / TABLE1_MANIFEST
    manifest = read_json(manifest_path)
    if manifest.get("contract") != "code08_cohort_table1_v1":
        raise ValueError(f"Unexpected Table 1 contract: {manifest_path}")
    table_path = settings.paths.reports / TABLE1_RELATIVE
    expected = manifest.get("output_sha256", {}).get("table1_characteristics.csv")
    if not expected or sha256(table_path) != str(expected).upper():
        raise ValueError(f"Code-08 Table 1 hash mismatch: {table_path}")
    frame = load_csv(
        table_path,
        (
            "variable",
            "characteristic",
            "type",
            "ahi_proxy_negative",
            "ahi_proxy_positive",
            "standardized_mean_difference",
        ),
    )
    if frame["variable"].duplicated().any():
        raise ValueError("Code-08 Table 1 contains duplicated variable rows")
    return frame.set_index("variable")


def _two_group_interval(ax, row, *, letter, title, xlabel, xscale=None):
    groups = (
        ("AHI-proxy negative", "ahi_proxy_negative", NEGATIVE, "o"),
        ("AHI-proxy positive", "ahi_proxy_positive", POSITIVE, "D"),
    )
    for position, (label, column, color, marker) in enumerate(groups):
        median, q1, q3 = _parse_summary(row[column])
        ax.errorbar(
            median,
            position,
            xerr=[[median - q1], [q3 - median]],
            fmt=marker,
            color=color,
            markerfacecolor="white" if position == 0 else color,
            markeredgecolor=INK,
            markeredgewidth=0.7,
            markersize=8,
            elinewidth=2,
            capsize=5,
        )
        ax.text(q3 * (1.05 if xscale == "log" else 1.0) + (0 if xscale == "log" else 4), position,
                f"{median:g} [{q1:g}, {q3:g}]", va="center", fontsize=12, fontweight="bold")
    ax.set_yticks([0, 1], [item[0] for item in groups])
    ax.invert_yaxis()
    if xscale:
        ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    panel_title(ax, letter, title)


def _multi_characteristic_intervals(ax, frame, variables, *, letter, title, xlabel):
    y = np.arange(len(variables), dtype=float)
    offset = 0.16
    for group_index, (column, label, color, marker) in enumerate(
        (
            ("ahi_proxy_negative", "AHI-proxy negative", NEGATIVE, "o"),
            ("ahi_proxy_positive", "AHI-proxy positive", POSITIVE, "D"),
        )
    ):
        positions = y + (-offset if group_index == 0 else offset)
        for position, variable in zip(positions, variables):
            median, q1, q3 = _parse_summary(frame.loc[variable, column])
            ax.errorbar(
                median,
                position,
                xerr=[[median - q1], [q3 - median]],
                fmt=marker,
                color=color,
                markerfacecolor="white" if group_index == 0 else color,
                markeredgecolor=INK,
                markeredgewidth=0.6,
                markersize=6.5,
                elinewidth=1.6,
                capsize=3,
            )
            ax.text(q3 + 1.7, position, f"{median:g}", va="center", fontsize=11.5, fontweight="bold")
    short_labels = {
        "medication_event_count": "Medication events",
        "laboratory_event_count": "Laboratory events",
        "diagnosis_event_count": "Diagnoses recorded",
        "baseline_alt": "ALT",
        "baseline_ast": "AST",
        "baseline_tbil": "Total bilirubin",
    }
    ax.set_yticks(y, [short_labels.get(variable, frame.loc[variable, "characteristic"]) for variable in variables])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    panel_title(ax, letter, title)


def _smd_panel(ax, frame):
    variables = (
        "laboratory_event_count",
        "medication_event_count",
        "baseline_ast",
        "diagnosis_event_count",
        "observation_hours",
        "baseline_alt",
        "gender_male",
        "sepsis_shock_flag",
        "age",
        "baseline_tbil",
    )
    colors = {
        "laboratory_event_count": LABORATORY,
        "medication_event_count": MEDICATION,
        "diagnosis_event_count": DIAGNOSIS,
        "observation_hours": PURPLE,
        "baseline_ast": POSITIVE,
        "baseline_alt": POSITIVE,
        "baseline_tbil": POSITIVE,
    }
    rows = frame.loc[list(variables)].copy()
    rows["smd"] = rows["standardized_mean_difference"].astype(float).abs()
    rows = rows.sort_values("smd", ascending=True)
    y = np.arange(len(rows))
    bar_colors = [colors.get(variable, "#B8B8B8") for variable in rows.index]
    ax.barh(y, rows["smd"], color=bar_colors, height=0.62)
    for position, value in zip(y, rows["smd"]):
        ax.text(value + 0.012, position, f"{value:.2f}", va="center", fontsize=11.5, fontweight="bold")
    short_labels = {
        "laboratory_event_count": "Laboratory event count",
        "medication_event_count": "Medication event count",
        "baseline_ast": "Baseline AST",
        "diagnosis_event_count": "Diagnosis event count",
        "observation_hours": "Observation window",
        "baseline_alt": "Baseline ALT",
        "gender_male": "Male sex",
        "sepsis_shock_flag": "Sepsis/shock diagnosis",
        "age": "Age",
        "baseline_tbil": "Baseline total bilirubin",
    }
    labels = [short_labels[variable] for variable in rows.index]
    ax.set_yticks(y, labels)
    ax.axvline(0.1, color="#777777", linestyle="--", linewidth=1.0)
    ax.axvline(0.2, color="#777777", linestyle=":", linewidth=1.0)
    ax.set_xlim(0, max(0.82, float(rows["smd"].max()) + 0.10))
    ax.set_xlabel("Absolute standardized mean difference")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    panel_title(ax, "D", "Largest between-group differences")


def generate_observation_process_figure(settings=None):
    settings = settings or load_settings()
    configure_publication_style()
    frame = _load_table1(settings)
    required = {
        "observation_hours",
        "medication_event_count",
        "laboratory_event_count",
        "diagnosis_event_count",
        "baseline_alt",
        "baseline_ast",
        "baseline_tbil",
        "gender_male",
        "sepsis_shock_flag",
        "age",
    }
    if required - set(frame.index):
        raise ValueError(f"Table 1 is missing Figure 2 variables: {sorted(required - set(frame.index))}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 11.5))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.95, bottom=0.095, wspace=0.34, hspace=0.34)
    _two_group_interval(
        axes[0, 0],
        frame.loc["observation_hours"],
        letter="A",
        title="Observation opportunity before prediction",
        xlabel="Observation window, hours (median [Q1, Q3])",
    )
    _multi_characteristic_intervals(
        axes[0, 1],
        frame,
        ("medication_event_count", "laboratory_event_count", "diagnosis_event_count"),
        letter="B",
        title="Recorded event burden",
        xlabel="Events per encounter (median [Q1, Q3])",
    )
    _multi_characteristic_intervals(
        axes[1, 0],
        frame,
        ("baseline_alt", "baseline_ast", "baseline_tbil"),
        letter="C",
        title="Baseline hepatic measurements",
        xlabel="Recorded value (median [Q1, Q3])",
    )
    axes[1, 0].text(
        0.98,
        0.96,
        "All displayed medians are below the 120 U/L cohort baseline threshold.",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#666666",
    )
    _smd_panel(axes[1, 1], frame)

    handles = [
        Line2D([0], [0], marker="o", color=NEGATIVE, markerfacecolor="white", markeredgecolor=INK,
               linewidth=1.5, label="AHI-proxy negative"),
        Line2D([0], [0], marker="D", color=POSITIVE, markerfacecolor=POSITIVE, markeredgecolor=INK,
               linewidth=1.5, label="AHI-proxy positive"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=2, frameon=False)
    png, pdf = save_figure(fig, settings, "Fig_2_Observation_Measurement_Process")
    print(f"[PASS] Figure 2: {png} | {pdf}")
    return png, pdf


if __name__ == "__main__":
    generate_observation_process_figure()
