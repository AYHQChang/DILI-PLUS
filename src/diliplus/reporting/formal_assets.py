"""Shared contracts, labels, colours, and loaders for formal paper assets.

Only aggregate Code-10 outputs are consumed here.  The module deliberately
does not search legacy report directories: every source is anchored by a
tracked formal manifest and an explicit run identifier.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


FORMAL_MANIFEST = "code10_formal_run.json"
ABLATION_MANIFEST = "code10_minimum_ablations_seed0_run.json"
SENSITIVITY_MANIFEST = "code10_sensitivity_128d8h_seed0_run.json"
P0_STATISTICAL_MANIFEST = "code13_p0_statistical_refinement.json"

FORMAL_MODELS = (
    "LogisticRegression",
    "XGBoost",
    "MultiModalTextCNN",
    "MultiModalBiLSTM",
    "MultimodalTransformerBaseline",
    "TimeAwareMultimodalTransformer",
)
DEEP_MODELS = (
    "MultiModalTextCNN",
    "MultiModalBiLSTM",
    "MultimodalTransformerBaseline",
    "TimeAwareMultimodalTransformer",
)

MODEL_LABELS = {
    "LogisticRegression": "Logistic regression",
    "XGBoost": "XGBoost",
    "MultiModalTextCNN": "TextCNN",
    "MultiModalBiLSTM": "BiLSTM",
    "MultimodalTransformerBaseline": "Multimodal Transformer",
    "TimeAwareMultimodalTransformer": "TA-MMT",
}

# User-selected palette.  This is the only authoritative model-colour mapping.
MODEL_COLORS = {
    "TimeAwareMultimodalTransformer": "#DF9E9B",
    "MultimodalTransformerBaseline": "#99BADF",
    "MultiModalBiLSTM": "#99CDCE",
    "MultiModalTextCNN": "#F8BF92",
    "XGBoost": "#999ACD",
    "LogisticRegression": "#FFB3DD",
}

ABLATION_ORDER = (
    "TimeAwareMultimodalTransformer",
    "StaticDiagnosisOnly",
    "MedicationOnly",
    "LaboratoryOnly",
    "FullWithoutTimeEncoding",
    "FullWithoutDiagnosis",
)
ABLATION_LABELS = {
    "TimeAwareMultimodalTransformer": "Full TA-MMT",
    "StaticDiagnosisOnly": "Diagnosis only",
    "MedicationOnly": "Medication only",
    "LaboratoryOnly": "Laboratory only",
    "FullWithoutTimeEncoding": "Without time encoding",
    "FullWithoutDiagnosis": "Without diagnosis",
}
ABLATION_COLORS = {
    "TimeAwareMultimodalTransformer": MODEL_COLORS["TimeAwareMultimodalTransformer"],
    "StaticDiagnosisOnly": "#B8B8B8",
    "MedicationOnly": "#D6A85F",
    "LaboratoryOnly": "#78B7A4",
    "FullWithoutTimeEncoding": "#B4A6D7",
    "FullWithoutDiagnosis": "#C69C7B",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{source} is missing required columns: {missing}")


def load_csv(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} contains zero rows")
    require_columns(frame, columns, path)
    return frame


def _formal_contract(settings) -> tuple[dict, Path]:
    manifest_path = settings.paths.manifests / FORMAL_MANIFEST
    manifest = read_json(manifest_path)
    if manifest.get("status") != "COMPLETE" or manifest.get("run_kind") != "formal":
        raise ValueError(f"Formal manifest is not COMPLETE/formal: {manifest_path}")
    if tuple(manifest.get("models", ())) != (
        "MultiModalTextCNN",
        "MultiModalBiLSTM",
        "MultimodalTransformerBaseline",
        "TimeAwareMultimodalTransformer",
        "LogisticRegression",
        "XGBoost",
    ):
        raise ValueError("Formal manifest model registry differs from the frozen contract")
    return manifest, settings.paths.reports / "runs" / manifest["run_id"]


def formal_sources(settings) -> dict[str, Path | str]:
    manifest, run_dir = _formal_contract(settings)
    metric_dir = run_dir / "metrics"
    sources: dict[str, Path | str] = {
        "run_id": manifest["run_id"],
        "manifest": settings.paths.manifests / FORMAL_MANIFEST,
        "pooled": metric_dir / "pooled_metrics.csv",
        "bootstrap": metric_dir / "bootstrap_intervals.csv",
        "paired": metric_dir / "paired_comparisons.csv",
        "calibration_bins": metric_dir / "calibration_bins.csv",
        "dca": metric_dir / "dca_curves.csv",
        "resource_usage": metric_dir / "resource_usage.csv",
        "early_warning_pooled": run_dir / "early_warning" / "metrics" / "pooled_metrics.csv",
        "early_warning_bootstrap": run_dir / "early_warning" / "metrics" / "bootstrap_intervals.csv",
        "early_warning_degradation": run_dir / "early_warning" / "metrics" / "horizon_degradation.csv",
    }
    recorded = manifest.get("metric_files", {})
    for key in ("pooled", "bootstrap", "paired", "calibration_bins", "dca", "resource_usage"):
        path = sources[key]
        assert isinstance(path, Path)
        expected = recorded.get(path.name, {}).get("sha256")
        if expected and sha256(path) != str(expected).upper():
            raise ValueError(f"Formal metric hash mismatch: {path}")
    return sources


def auxiliary_sources(settings) -> dict[str, Path | str]:
    ablation = read_json(settings.paths.manifests / ABLATION_MANIFEST)
    sensitivity = read_json(settings.paths.manifests / SENSITIVITY_MANIFEST)
    if ablation.get("status") != "COMPLETE" or sensitivity.get("status") != "COMPLETE":
        raise ValueError("Ablation and sensitivity manifests must both be COMPLETE")
    ablation_run = str(ablation["run_id"])
    sensitivity_run = str(sensitivity["run_id"])
    return {
        "ablation_run_id": ablation_run,
        "ablation_manifest": settings.paths.manifests / ABLATION_MANIFEST,
        "ablation_pooled": settings.paths.reports / "runs" / ablation_run / "metrics" / "pooled_metrics.csv",
        "ablation_bootstrap": settings.paths.reports / "runs" / ablation_run / "metrics" / "bootstrap_intervals.csv",
        "ablation_paired": settings.paths.reports / "runs" / ablation_run / "metrics" / "paired_comparisons.csv",
        "sensitivity_run_id": sensitivity_run,
        "sensitivity_manifest": settings.paths.manifests / SENSITIVITY_MANIFEST,
        "sensitivity_paired": settings.paths.reports / "runs" / sensitivity_run / "metrics" / "head_sensitivity_paired_comparison.csv",
        "stability_manifest": settings.paths.manifests / "code10_seed_stability.json",
        "stability_metrics": settings.paths.reports / "runs" / "code10_seed_stability_summary" / "calibrated_seed_metrics.csv",
        "stability_summary": settings.paths.reports / "runs" / "code10_seed_stability_summary" / "calibrated_metric_summary.csv",
    }


def calibrated_main_frames(settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = formal_sources(settings)
    pooled = load_csv(
        sources["pooled"],
        ("Run_ID", "Model_Architecture", "Probability_Mode", "AUROC", "AUPRC", "Brier", "NLL"),
    )
    bootstrap = load_csv(
        sources["bootstrap"],
        ("Run_ID", "Model_Architecture", "Probability_Mode", "metric", "estimate", "ci_lower", "ci_upper"),
    )
    run_id = sources["run_id"]
    pooled = pooled[(pooled["Run_ID"] == run_id) & (pooled["Probability_Mode"] == "calibrated")].copy()
    bootstrap = bootstrap[(bootstrap["Run_ID"] == run_id) & (bootstrap["Probability_Mode"] == "calibrated")].copy()
    if set(pooled["Model_Architecture"]) != set(FORMAL_MODELS) or len(pooled) != len(FORMAL_MODELS):
        raise ValueError("Formal calibrated pooled metrics must contain one row per registered model")
    return pooled, bootstrap


def p0_statistical_sources(settings) -> dict[str, Path | str]:
    """Return hash-verified P0 outputs derived from the frozen formal OOF files."""
    manifest_path = settings.paths.manifests / P0_STATISTICAL_MANIFEST
    manifest = read_json(manifest_path)
    if manifest.get("status") != "COMPLETE":
        raise ValueError(f"P0 statistical manifest is not COMPLETE: {manifest_path}")
    if manifest.get("source_run_id") != formal_sources(settings)["run_id"]:
        raise ValueError("P0 statistical source run differs from the frozen formal run")
    run_dir = settings.paths.reports / "runs" / str(manifest["run_id"]) / "metrics"
    paths: dict[str, Path | str] = {
        "run_id": str(manifest["run_id"]),
        "source_run_id": str(manifest["source_run_id"]),
        "manifest": manifest_path,
        "bootstrap": run_dir / "bootstrap_intervals_10000.csv",
        "paired": run_dir / "paired_comparisons_10000.csv",
        "fold_discrimination": run_dir / "fold_discrimination.csv",
    }
    recorded = manifest.get("outputs", {})
    for key in ("bootstrap", "paired", "fold_discrimination"):
        path = paths[key]
        assert isinstance(path, Path)
        expected = recorded.get(path.name, {}).get("sha256")
        if expected is None or sha256(path) != str(expected).upper():
            raise ValueError(f"P0 statistical output hash mismatch: {path}")
    return paths


def raw_discrimination_bootstrap(settings) -> pd.DataFrame:
    sources = p0_statistical_sources(settings)
    frame = load_csv(
        sources["bootstrap"],
        (
            "Model_Architecture",
            "Probability_Mode",
            "metric",
            "estimate",
            "ci_lower",
            "ci_upper",
            "valid_replicates",
        ),
    )
    result = frame[
        (frame["Probability_Mode"] == "raw")
        & frame["metric"].isin(("AUROC", "AUPRC"))
    ].copy()
    expected = {(model, metric) for model in FORMAL_MODELS for metric in ("AUROC", "AUPRC")}
    observed = set(zip(result["Model_Architecture"], result["metric"]))
    if observed != expected or len(result) != len(expected):
        raise ValueError("P0 raw discrimination table must contain one row per model/metric")
    return result


def configure_publication_style() -> None:
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
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#333333",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_title(ax, letter: str, title: str) -> None:
    ax.set_title(
        f"{letter}  {title}",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=10,
    )


def save_figure(fig, settings, stem: str) -> tuple[Path, Path]:
    settings.paths.figures.mkdir(parents=True, exist_ok=True)
    png = settings.paths.figures / f"{stem}.png"
    pdf = settings.paths.figures / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf
