"""Supplementary Table S1 for subgroup and observation-process audits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from diliplus.config import load_settings
from diliplus.reporting.formal_assets import MODEL_LABELS
from diliplus.reporting.p1_contract import AUDIT_MODELS, PROCESS_MODEL, RUN_ID


DISPLAY = {
    **{model: MODEL_LABELS[model] for model in AUDIT_MODELS if model in MODEL_LABELS},
    PROCESS_MODEL: "Process-intensity baseline",
}


def _latex_escape(text: str) -> str:
    return str(text).replace("&", r"\&").replace("%", r"\%")


def generate_supplementary_table_s1(settings=None) -> tuple[Path, Path]:
    settings = settings or load_settings()
    metric_dir = settings.paths.reports / "runs" / RUN_ID / "metrics"
    source = metric_dir / "subgroup_metrics.csv"
    frame = pd.read_csv(source)
    process = pd.read_csv(metric_dir / "process_pooled_metrics.csv")
    process = process[process["Probability_Mode"] == "calibrated"].iloc[0]
    process_interval = pd.read_csv(metric_dir / "process_bootstrap_intervals.csv")
    process_interval = process_interval[process_interval["metric"] == "AUPRC"].iloc[0]
    overall = pd.DataFrame(
        [
            {
                "audit_dimension": "Overall",
                "subgroup": "All encounters",
                "subgroup_order": 0,
                "Model_Architecture": PROCESS_MODEL,
                "N": int(process["N"]),
                "Positive": int(process["Positive"]),
                "Prevalence": float(process["Prevalence"]),
                "AUPRC": float(process["AUPRC"]),
                "AUPRC_ci_lower": float(process_interval["ci_lower"]),
                "AUPRC_ci_upper": float(process_interval["ci_upper"]),
                "Observed_Expected_Ratio": float(process["Observed_Expected_Ratio"]),
                "Top_0p01_Count": int(process["Top_0p01_Count"]),
                "Top_0p01_Events_Captured": int(
                    round(process["Top_0p01_Recall"] * process["Positive"])
                ),
                "Top_0p01_Recall": float(process["Top_0p01_Recall"]),
            }
        ]
    )
    frame = pd.concat([overall, frame], ignore_index=True)
    dimension_order = {"Overall": 0, "Observation window": 1, "Sex": 2}
    frame["dimension_order"] = frame["audit_dimension"].map(dimension_order)
    frame["model_order"] = frame["Model_Architecture"].map(
        {model: order for order, model in enumerate(AUDIT_MODELS)}
    )
    frame = frame.sort_values(
        ["dimension_order", "subgroup_order", "model_order"], kind="mergesort"
    )
    output_dir = settings.paths.reports / "paper_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "Table_S1_Subgroup_Process_Audit.csv"
    tex_path = output_dir / "Table_S1_Subgroup_Process_Audit.tex"
    display = frame.drop(columns=["dimension_order", "model_order"]).copy()
    display["Model"] = display["Model_Architecture"].map(DISPLAY)
    display["N_events"] = display.apply(
        lambda row: f"{int(row['N']):,} / {int(row['Positive'])}", axis=1
    )
    display["AUPRC_95CI"] = display.apply(
        lambda row: f"{row['AUPRC']:.3f} ({row['AUPRC_ci_lower']:.3f}--{row['AUPRC_ci_upper']:.3f})",
        axis=1,
    )
    display["OE"] = display["Observed_Expected_Ratio"].map(lambda value: f"{value:.3f}")
    display["Lift"] = (display["AUPRC"] / display["Prevalence"]).map(
        lambda value: f"{value:.1f}"
    )
    display["Top1"] = display.apply(
        lambda row: f"{int(row['Top_0p01_Events_Captured'])}/{int(row['Top_0p01_Count'])} ({row['Top_0p01_Recall']:.3f})",
        axis=1,
    )
    display[
        ["audit_dimension", "subgroup", "Model", "N_events", "AUPRC_95CI", "Lift", "OE", "Top1"]
    ].to_csv(csv_path, index=False)

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{\textbf{Post-hoc subgroup and observation-process audit.} AUPRC intervals are 95\% patient-cluster bootstrap intervals from 1,000 replicates. Lift is AUPRC divided by within-stratum prevalence; O:E is the observed-to-expected event ratio. Top-1\% gives events captured/alerts (sensitivity). Recorded sex is a descriptive heterogeneity audit among encounters with available sex; observation-window strata audit the healthcare measurement process and are not fairness attributes.}",
        r"\label{tab:supplementary_subgroup_process}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lllccccc}",
        r"\toprule",
        r"Audit & Subgroup & Model & N / events & AUPRC (95\% CI) & Lift & O:E & Top-1\% \\",
        r"\midrule",
    ]
    previous_dimension = None
    previous_subgroup = None
    for row in display.itertuples(index=False):
        if previous_dimension is not None and row.audit_dimension != previous_dimension:
            lines.append(r"\midrule")
        dimension = _latex_escape(row.audit_dimension) if row.audit_dimension != previous_dimension else ""
        subgroup = _latex_escape(row.subgroup) if (row.audit_dimension, row.subgroup) != (previous_dimension, previous_subgroup) else ""
        lines.append(
            f"{dimension} & {subgroup} & {_latex_escape(row.Model)} & {row.N_events} & "
            f"{row.AUPRC_95CI} & {row.Lift} & {row.OE} & {row.Top1} \\\\"
        )
        previous_dimension = row.audit_dimension
        previous_subgroup = row.subgroup
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[PASS] Supplementary Table S1: {csv_path} | {tex_path}")
    return csv_path, tex_path


if __name__ == "__main__":
    generate_supplementary_table_s1()
