"""Generate formal Table 2 from the frozen Code-10 OOF analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from diliplus.config import load_settings
from diliplus.reporting.formal_assets import (
    FORMAL_MODELS,
    MODEL_LABELS,
    calibrated_main_frames,
    formal_sources,
)


def _ci_lookup(bootstrap: pd.DataFrame, model: str, metric: str) -> tuple[float, float, float]:
    rows = bootstrap[
        (bootstrap["Model_Architecture"] == model) & (bootstrap["metric"] == metric)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one bootstrap row for {model}/{metric}, found {len(rows)}")
    row = rows.iloc[0]
    return float(row["estimate"]), float(row["ci_lower"]), float(row["ci_upper"])


def _format_ci(value: float, lower: float, upper: float) -> str:
    return f"{value:.3f} ({lower:.3f}–{upper:.3f})"


def _latex_escape(value: str) -> str:
    return value.replace("%", r"\%").replace("–", "--")


def generate_table_2(settings=None) -> dict[str, Path]:
    settings = settings or load_settings()
    pooled, bootstrap = calibrated_main_frames(settings)
    sources = formal_sources(settings)
    rows = []
    for model in FORMAL_MODELS:
        match = pooled[pooled["Model_Architecture"] == model]
        if len(match) != 1:
            raise ValueError(f"Expected one calibrated pooled row for {model}")
        point = match.iloc[0]
        auroc = _ci_lookup(bootstrap, model, "AUROC")
        auprc = _ci_lookup(bootstrap, model, "AUPRC")
        brier = _ci_lookup(bootstrap, model, "Brier")
        nll = _ci_lookup(bootstrap, model, "NLL")
        rows.append(
            {
                "Model_Architecture": model,
                "Model": MODEL_LABELS[model],
                "N": int(point["N"]),
                "Events": int(point["Positive"]),
                "AUROC": auroc[0],
                "AUROC_CI_Lower": auroc[1],
                "AUROC_CI_Upper": auroc[2],
                "AUPRC": auprc[0],
                "AUPRC_CI_Lower": auprc[1],
                "AUPRC_CI_Upper": auprc[2],
                "AUPRC_Lift": float(point["AUPRC_Lift"]),
                "Brier": brier[0],
                "Brier_CI_Lower": brier[1],
                "Brier_CI_Upper": brier[2],
                "NLL": nll[0],
                "NLL_CI_Lower": nll[1],
                "NLL_CI_Upper": nll[2],
                "Calibration_Intercept": float(point["Calibration_Intercept"]),
                "Calibration_Slope": float(point["Calibration_Slope"]),
                "Observed_Expected_Ratio": float(point["Observed_Expected_Ratio"]),
                "Top_1pct_PPV": float(point["Top_0p01_Precision"]),
                "Top_1pct_Recall": float(point["Top_0p01_Recall"]),
                "Run_ID": sources["run_id"],
                "Probability_Mode": "calibrated",
            }
        )

    numeric = pd.DataFrame(rows)
    paper = pd.DataFrame(
        {
            "Model": numeric["Model"],
            "AUROC (95% CI)": [
                _format_ci(r.AUROC, r.AUROC_CI_Lower, r.AUROC_CI_Upper)
                for r in numeric.itertuples()
            ],
            "AUPRC (95% CI)": [
                _format_ci(r.AUPRC, r.AUPRC_CI_Lower, r.AUPRC_CI_Upper)
                for r in numeric.itertuples()
            ],
            "AUPRC lift": numeric["AUPRC_Lift"].map(lambda value: f"{value:.1f}×"),
            "Brier score": numeric["Brier"].map(lambda value: f"{value:.4f}"),
            "NLL": numeric["NLL"].map(lambda value: f"{value:.4f}"),
            "Calibration slope": numeric["Calibration_Slope"].map(lambda value: f"{value:.3f}"),
            "O:E ratio": numeric["Observed_Expected_Ratio"].map(lambda value: f"{value:.3f}"),
            "Top 1% PPV": numeric["Top_1pct_PPV"].map(lambda value: f"{100 * value:.1f}%"),
            "Top 1% sensitivity": numeric["Top_1pct_Recall"].map(lambda value: f"{100 * value:.1f}%"),
        }
    )

    output_dir = settings.paths.reports / "paper_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_path = output_dir / "Table_02_Formal_Model_Performance_numeric.csv"
    paper_path = output_dir / "Table_02_Formal_Model_Performance.csv"
    text_path = output_dir / "Table_02_Formal_Model_Performance.txt"
    latex_path = output_dir / "Table_02_Formal_Model_Performance.tex"
    numeric.to_csv(numeric_path, index=False)
    paper.to_csv(paper_path, index=False)
    text_path.write_text(
        "Table 2. Formal five-fold patient-grouped out-of-fold performance\n"
        f"Run: {sources['run_id']}; calibrated probabilities; N={numeric.N.iloc[0]:,}; "
        f"events={numeric.Events.iloc[0]:,}. Intervals are 1,000-replicate patient-cluster bootstrap 95% CIs.\n\n"
        + paper.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    columns = " & ".join(_latex_escape(column) for column in paper.columns) + r" \\"
    body = "\n".join(
        " & ".join(_latex_escape(str(value)) for value in row) + r" \\"
        for row in paper.itertuples(index=False, name=None)
    )
    latex_path.write_text(
        "\\begin{tabular}{lccccccccc}\n\\toprule\n"
        + columns
        + "\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )
    print(f"[PASS] Formal Table 2: {paper_path}")
    return {"numeric": numeric_path, "paper": paper_path, "text": text_path, "latex": latex_path}


if __name__ == "__main__":
    generate_table_2()
