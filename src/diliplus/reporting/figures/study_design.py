"""Figure 1: formal cohort construction and prediction-time contract.

The figure is generated exclusively from aggregate, tracked manifests.  It
contains no patient- or encounter-level identifiers and does not query the
source database.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from diliplus.config import load_settings
from diliplus.reporting.formal_assets import read_json


LABEL_MANIFEST = "code10_label_rebuild_audit.json"
SPLIT_MANIFEST = "code05_split_protocol.json"
AVAILABILITY_MANIFEST = "code09_early_warning_contract.json"

INK = "#252525"
MUTED = "#666666"
PINK = "#DF9E9B"
TEAL = "#99CDCE"
BLUE = "#99BADF"
PURPLE = "#999ACD"
ORANGE = "#F8BF92"
SOFT_GREY = "#F2F2F2"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class StudyDesignData:
    target_lab_encounters: int
    baseline_eligible: int
    formal_encounters: int
    formal_patients: int
    positive: int
    negative: int
    prediction_gap_hours: float
    modality_rows: tuple[dict, ...]
    split_rows: tuple[dict, ...]
    outer_folds: int
    max_patient_overlap: int


def _integer(value) -> int:
    return int(round(float(value)))


def _load_study_design_data(settings) -> StudyDesignData:
    label_path = settings.paths.manifests / LABEL_MANIFEST
    split_path = settings.paths.manifests / SPLIT_MANIFEST
    availability_path = settings.paths.manifests / AVAILABILITY_MANIFEST
    label = read_json(label_path)
    split = read_json(split_path)
    availability = read_json(availability_path)

    for payload, path in (
        (label, label_path),
        (split, split_path),
        (availability, availability_path),
    ):
        if payload.get("status") != "PASS":
            raise ValueError(f"Study-design source is not PASS: {path}")

    raw = label["deterministic_raw"]
    formal = label["formal_gap_eligible"]
    cohort = split["cohort"]
    formal_encounters = _integer(formal["encounters"])
    positive = _integer(formal["positive"])
    negative = _integer(formal["negative"])
    if (
        formal_encounters != int(cohort["encounters"])
        or positive != int(cohort["positive"])
        or negative != int(cohort["negative"])
    ):
        raise ValueError("Label and split manifests describe different formal cohorts")

    horizon_rows = availability.get("horizon_availability", [])
    matching = [
        row
        for row in horizon_rows
        if float(row["effective_horizon_hours_before_index"])
        == float(settings.prediction.gap_hours)
    ]
    if len(matching) != 1:
        raise ValueError("Expected exactly one formal 24-hour availability row")
    horizon = matching[0]
    if int(horizon["encounters"]) != formal_encounters:
        raise ValueError("Availability manifest differs from the formal cohort")

    modality_rows = (
        {
            "label": "Medication events",
            "events": int(horizon["medication_events_retained"]),
            "missing": int(horizon["medication_missing_encounters"]),
            "median": float(horizon["medication_length_median"]),
            "color": BLUE,
        },
        {
            "label": "Laboratory events",
            "events": int(horizon["laboratory_events_retained"]),
            "missing": int(horizon["laboratory_missing_encounters"]),
            "median": float(horizon["laboratory_length_median"]),
            "color": ORANGE,
        },
        {
            "label": "Diagnosis events",
            "events": int(horizon["diagnosis_events_retained"]),
            "missing": int(horizon["diagnosis_missing_encounters"]),
            "median": float(horizon["diagnosis_length_median"]),
            "color": TEAL,
        },
    )

    folds = split["folds"]
    if len(folds) != int(split["protocol"]["outer_folds"]):
        raise ValueError("Split manifest is missing an outer fold")
    role_names = ("training", "selection", "calibration", "test")
    split_rows = []
    for role in role_names:
        counts = [int(fold["roles"][role]["encounters"]) for fold in folds]
        positives = [int(fold["roles"][role]["positive"]) for fold in folds]
        split_rows.append(
            {
                "role": role,
                "minimum": min(counts),
                "maximum": max(counts),
                "positive_minimum": min(positives),
                "positive_maximum": max(positives),
            }
        )
    overlap_values = []
    for fold in folds:
        overlap_values.extend(int(value) for value in fold["group_overlap"].values())
    max_overlap = max(overlap_values, default=0)
    if max_overlap != 0:
        raise ValueError("Formal split manifest contains patient overlap")

    return StudyDesignData(
        target_lab_encounters=int(raw["target_lab_encounters"]),
        baseline_eligible=int(raw["baseline_rule_pass"]),
        formal_encounters=formal_encounters,
        formal_patients=_integer(formal["patients"]),
        positive=positive,
        negative=negative,
        prediction_gap_hours=float(label["rule"]["prediction_gap_hours"]),
        modality_rows=modality_rows,
        split_rows=tuple(split_rows),
        outer_folds=len(folds),
        max_patient_overlap=max_overlap,
    )


def _box(
    ax,
    xy,
    width,
    height,
    *,
    facecolor,
    edgecolor=INK,
    linewidth=1.3,
    radius=0.025,
):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, start, end, *, color=MUTED, linewidth=1.8, mutation_scale=16):
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        shrinkA=2,
        shrinkB=2,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def _panel_title(ax, letter: str, title: str) -> None:
    ax.text(
        0.0,
        1.03,
        f"{letter}  {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )


def _draw_cohort_flow(ax, data: StudyDesignData) -> None:
    _panel_title(ax, "A", "Deterministic cohort construction")
    ax.set_axis_off()
    box_y, box_w, box_h = 0.49, 0.205, 0.30
    xs = (0.00, 0.265, 0.53, 0.795)
    stages = (
        ("Target ALT/AST\nrecords", data.target_lab_encounters, BLUE),
        ("Deterministic\nbaseline", data.baseline_eligible, TEAL),
        ("24-h eligible\ncohort", data.formal_encounters, PURPLE),
        ("Formal AHI\nproxy", data.formal_encounters, PINK),
    )
    for index, (label, count, color) in enumerate(stages):
        _box(ax, (xs[index], box_y), box_w, box_h, facecolor=color + "40", edgecolor=color)
        ax.text(
            xs[index] + box_w / 2,
            box_y + 0.215,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13.5,
            fontweight="bold",
            color=INK,
        )
        count_text = f"{count:,}"
        count_size = 17
        if index == len(stages) - 1:
            count_text = f"{data.positive:,} positive\n{data.negative:,} negative"
            count_size = 12.5
        ax.text(
            xs[index] + box_w / 2,
            box_y + 0.085,
            count_text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=count_size,
            fontweight="bold",
            color=INK,
        )
        if index < len(stages) - 1:
            _arrow(
                ax,
                (xs[index] + box_w + 0.008, box_y + box_h / 2),
                (xs[index + 1] - 0.008, box_y + box_h / 2),
            )

    baseline_excluded = data.target_lab_encounters - data.baseline_eligible
    gap_excluded = data.baseline_eligible - data.formal_encounters
    ax.text(
        0.237,
        0.40,
        f"−{baseline_excluded:,}\nbaseline-\nineligible",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color=MUTED,
    )
    ax.text(
        0.502,
        0.40,
        f"−{gap_excluded:,}\ninsufficient\n24-h history",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color=MUTED,
    )
    prevalence = 100.0 * data.positive / data.formal_encounters
    ax.text(
        0.897,
        0.40,
        f"prevalence {prevalence:.3f}%",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.5,
        0.105,
        f"Formal cohort: {data.formal_patients:,} unique patients",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.0,
        0.025,
        "Baseline: all earliest-timestamp ALT/AST rows below 120 U/L; "
        "outcome: first strictly later ALT/AST value ≥120 U/L.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=MUTED,
    )


def _draw_timeline(ax, data: StudyDesignData) -> None:
    _panel_title(ax, "B", "Prediction-time and target-blinding contract")
    ax.set_axis_off()
    y = 0.54
    first_x, prediction_x, index_x = 0.08, 0.62, 0.90
    ax.add_patch(
        Rectangle(
            (first_x, 0.39),
            prediction_x - first_x,
            0.29,
            transform=ax.transAxes,
            facecolor=TEAL + "45",
            edgecolor="none",
        )
    )
    ax.add_patch(
        Rectangle(
            (prediction_x, 0.39),
            index_x - prediction_x,
            0.29,
            transform=ax.transAxes,
            facecolor=PINK + "38",
            edgecolor="none",
        )
    )
    _arrow(ax, (0.055, y), (0.955, y), color=INK, linewidth=2.0, mutation_scale=18)
    for x, color in ((first_x, BLUE), (prediction_x, PURPLE), (index_x, PINK)):
        ax.plot(
            [x, x],
            [0.30, 0.76],
            transform=ax.transAxes,
            color=color,
            linewidth=2.2,
            linestyle="--",
            solid_capstyle="round",
        )
    ax.text(
        (first_x + prediction_x) / 2,
        0.64,
        "Observed history\nmedications · laboratories · diagnoses",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        (prediction_x + index_x) / 2,
        0.64,
        f"{data.prediction_gap_hours:g}-hour\nblind gap\nexcluded from input",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(first_x, 0.24, "First eligible\nevent", transform=ax.transAxes, ha="center", va="top", fontsize=11.5, fontweight="bold")
    ax.text(prediction_x, 0.24, "Prediction time\nfeatures stop here", transform=ax.transAxes, ha="center", va="top", fontsize=11.5, fontweight="bold")
    ax.text(index_x, 0.24, "Index time\nAHI proxy defined", transform=ax.transAxes, ha="center", va="top", fontsize=11.5, fontweight="bold")
    ax.text(
        0.50,
        0.08,
        "Formal input rule\n"
        r"$t_{event} < t_{prediction}$   ·   $t_{prediction}=t_{index}-24\,h$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )


def _draw_modalities(ax, data: StudyDesignData) -> None:
    _panel_title(ax, "C", "Information available at the 24-hour horizon")
    ax.set_axis_off()
    y_positions = (0.70, 0.44, 0.18)
    bar_x, bar_w, bar_h = 0.31, 0.60, 0.075
    for row, y in zip(data.modality_rows, y_positions):
        available = data.formal_encounters - int(row["missing"])
        fraction = available / data.formal_encounters
        ax.text(
            0.0,
            y + 0.018,
            row["label"],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=INK,
        )
        ax.add_patch(
            Rectangle(
                (bar_x, y - bar_h / 2),
                bar_w,
                bar_h,
                transform=ax.transAxes,
                facecolor=SOFT_GREY,
                edgecolor="none",
            )
        )
        ax.add_patch(
            Rectangle(
                (bar_x, y - bar_h / 2),
                bar_w * fraction,
                bar_h,
                transform=ax.transAxes,
                facecolor=row["color"],
                edgecolor="none",
            )
        )
        ax.text(
            bar_x + 0.015,
            y + 0.085,
            f"{row['events']:,} retained events · median {row['median']:g}/encounter",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
            color=MUTED,
        )
        ax.text(
            bar_x + bar_w + 0.02,
            y,
            f"{100.0 * fraction:.1f}%\nnon-empty",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color=INK,
        )
    ax.text(
        0.0,
        0.03,
        "Availability means at least one retained token after the strict prediction-time cutoff.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=MUTED,
    )


def _range_text(row: dict) -> str:
    encounter = (
        f"{row['minimum']:,}"
        if row["minimum"] == row["maximum"]
        else f"{row['minimum']:,}–{row['maximum']:,}"
    )
    positive = (
        f"{row['positive_minimum']:,}"
        if row["positive_minimum"] == row["positive_maximum"]
        else f"{row['positive_minimum']:,}–{row['positive_maximum']:,}"
    )
    return f"{encounter} encounters\n{positive} positives"


def _draw_split_contract(ax, data: StudyDesignData) -> None:
    _panel_title(ax, "D", "Patient-grouped evaluation roles")
    ax.set_axis_off()
    colors = (BLUE, TEAL, PURPLE, PINK)
    labels = {
        "training": "Training",
        "selection": "Selection",
        "calibration": "Calibration",
        "test": "Outer test",
    }
    subtitles = {
        "training": "parameter fitting only",
        "selection": "early stopping only",
        "calibration": "temperature only",
        "test": "one final inference",
    }
    positions = ((0.00, 0.56), (0.52, 0.56), (0.00, 0.20), (0.52, 0.20))
    box_w, box_h = 0.46, 0.27
    for (x, y), row, color in zip(positions, data.split_rows, colors):
        _box(ax, (x, y), box_w, box_h, facecolor=color + "40", edgecolor=color)
        ax.text(
            x + box_w / 2,
            y + 0.205,
            labels[row["role"]],
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            x + box_w / 2,
            y + 0.115,
            _range_text(row),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            x + box_w / 2,
            y + 0.035,
            subtitles[row["role"]],
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=MUTED,
        )
    ax.text(
        0.5,
        0.105,
        f"{data.outer_folds} outer folds  ·  maximum patient overlap = {data.max_patient_overlap}  ·  "
        "selection and calibration are disjoint",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.5,
        0.035,
        "Outer-test membership is fixed before model fitting.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=MUTED,
    )


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 14,
            "font.weight": "bold",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def generate_study_design_figure(settings=None) -> tuple[Path, Path]:
    """Render Figure 1 from the final label, split, and availability contracts."""

    settings = settings or load_settings()
    data = _load_study_design_data(settings)
    _configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(18, 11.2))
    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.94,
        bottom=0.05,
        wspace=0.10,
        hspace=0.20,
    )
    _draw_cohort_flow(axes[0, 0], data)
    _draw_timeline(axes[0, 1], data)
    _draw_modalities(axes[1, 0], data)
    _draw_split_contract(axes[1, 1], data)

    settings.paths.figures.mkdir(parents=True, exist_ok=True)
    stem = "Fig_1_Study_Design"
    png = settings.paths.figures / f"{stem}.png"
    pdf = settings.paths.figures / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(pdf, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"[PASS] Figure 1 study design: {png}")
    print(f"[PASS] Figure 1 study design: {pdf}")
    return png, pdf


if __name__ == "__main__":
    generate_study_design_figure()
