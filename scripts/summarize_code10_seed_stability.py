"""Summarize the prespecified three-seed primary/TextCNN stability analysis."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.artifacts import file_sha256  # noqa: E402
from diliplus.config import load_settings  # noqa: E402


RUNS = (
    {
        "seed_index": 0,
        "process_seed": 20260816,
        "run_id": "code10_formal_128d4h_seed0",
        "aggregate_manifest": "code10_formal_run.json",
    },
    {
        "seed_index": 1,
        "process_seed": 20260817,
        "run_id": "code10_stability_primary_textcnn_seed1",
        "aggregate_manifest": "code10_stability_primary_textcnn_seed1_run.json",
    },
    {
        "seed_index": 2,
        "process_seed": 20260818,
        "run_id": "code10_stability_primary_textcnn_seed2",
        "aggregate_manifest": "code10_stability_primary_textcnn_seed2_run.json",
    },
)
MODELS = ("MultiModalTextCNN", "TimeAwareMultimodalTransformer")
PRIMARY = "TimeAwareMultimodalTransformer"
COMPARATOR = "MultiModalTextCNN"
CORE_METRICS = (
    "AUROC",
    "AUPRC",
    "Brier",
    "NLL",
    "Calibration_Intercept",
    "Calibration_Slope",
    "Observed_Expected_Ratio",
    "Quantile_ECE",
)
MEMBERSHIP_COLUMNS = ("dataset_index", "encounter_id", "patient_id", "y_true")


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_oof(settings, run_id: str, model_name: str) -> pd.DataFrame:
    directory = settings.paths.reports / "runs" / run_id / "predictions" / model_name
    paths = sorted(directory.glob("fold_[0-9][0-9].csv"))
    if len(paths) != settings.evaluation_protocol.outer_folds:
        raise RuntimeError(
            f"{run_id}/{model_name}: expected "
            f"{settings.evaluation_protocol.outer_folds} fold predictions, found {len(paths)}"
        )
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if frame["dataset_index"].duplicated().any():
        raise RuntimeError(f"{run_id}/{model_name}: duplicate dataset_index")
    return frame.sort_values("dataset_index", kind="stable").reset_index(drop=True)


def _python_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> int:
    settings = load_settings(PROJECT_ROOT / "configs" / "default.yaml")
    output_dir = settings.paths.reports / "runs" / "code10_seed_stability_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_membership = None
    calibrated_rows = []
    paired_rows = []
    run_provenance = []

    for run in RUNS:
        run_id = run["run_id"]
        run_dir = settings.paths.reports / "runs" / run_id
        metrics_dir = run_dir / "metrics"
        aggregate_path = settings.paths.manifests / run["aggregate_manifest"]
        recorder_path = settings.paths.reports / "run_logs" / f"{run_id}.json"
        aggregate = _load_json(aggregate_path)
        recorder = _load_json(recorder_path)

        if aggregate.get("status") not in {"PASS", "COMPLETE"}:
            raise RuntimeError(f"{run_id}: aggregate manifest is not complete")
        if recorder.get("exit_code") != 0:
            raise RuntimeError(f"{run_id}: recorded process did not exit successfully")
        if recorder.get("process_seed") != run["process_seed"]:
            raise RuntimeError(f"{run_id}: process seed differs from protocol")
        if recorder.get("git_dirty_at_start"):
            raise RuntimeError(f"{run_id}: run did not start from a clean Git tree")

        for model_name in MODELS:
            oof = _load_oof(settings, run_id, model_name)
            membership = oof[list(MEMBERSHIP_COLUMNS)]
            if reference_membership is None:
                reference_membership = membership
            elif not reference_membership.equals(membership):
                raise RuntimeError(f"{run_id}/{model_name}: OOF membership differs")

        pooled = pd.read_csv(metrics_dir / "pooled_metrics.csv")
        selected = pooled[
            pooled["Model_Architecture"].isin(MODELS)
            & pooled["Probability_Mode"].eq("calibrated")
        ].copy()
        if len(selected) != len(MODELS):
            raise RuntimeError(f"{run_id}: missing calibrated pooled model row")
        selected.insert(0, "Process_Seed", run["process_seed"])
        selected.insert(0, "Seed_Index", run["seed_index"])
        calibrated_rows.append(selected)

        paired = pd.read_csv(metrics_dir / "paired_comparisons.csv")
        paired = paired[
            paired["Probability_Mode"].eq("calibrated")
            & paired["Primary"].eq(PRIMARY)
            & paired["Comparator"].eq(COMPARATOR)
        ].copy()
        if set(paired["metric"]) != {"AUROC", "AUPRC", "Brier", "NLL"}:
            raise RuntimeError(f"{run_id}: incomplete calibrated paired comparison")
        paired.insert(0, "Process_Seed", run["process_seed"])
        paired.insert(0, "Seed_Index", run["seed_index"])
        paired_rows.append(paired)

        run_provenance.append(
            {
                "seed_index": run["seed_index"],
                "process_seed": run["process_seed"],
                "run_id": run_id,
                "training_git_commit": aggregate["training_git_commit"],
                "analysis_git_commit": aggregate["analysis_git_commit"],
                "duration_seconds": recorder["duration_seconds"],
                "recorded_log_sha256": recorder["log_sha256"],
                "aggregate_manifest_sha256": file_sha256(aggregate_path),
            }
        )

    assert reference_membership is not None
    seed_metrics = pd.concat(calibrated_rows, ignore_index=True)
    paired_differences = pd.concat(paired_rows, ignore_index=True)

    invariant_columns = {
        "Seed_Index",
        "Process_Seed",
        "Run_ID",
        "Model_Architecture",
        "Probability_Mode",
        "N",
        "Positive",
        "Prevalence",
        "Reference_Prevalence",
    }
    outcome_columns = [
        column
        for column in seed_metrics.columns
        if column not in invariant_columns
        and pd.api.types.is_numeric_dtype(seed_metrics[column])
    ]
    long_metrics = seed_metrics.melt(
        id_vars=[
            "Seed_Index",
            "Process_Seed",
            "Run_ID",
            "Model_Architecture",
            "Probability_Mode",
        ],
        value_vars=outcome_columns,
        var_name="Metric",
        value_name="Value",
    )
    metric_summary = (
        long_metrics.groupby(["Model_Architecture", "Metric"], sort=False)["Value"]
        .agg(Mean="mean", SD="std", Minimum="min", Maximum="max")
        .reset_index()
    )

    paired_summary = (
        paired_differences.groupby("metric", sort=False)[
            "delta_primary_minus_comparator"
        ]
        .agg(Mean_Delta="mean", SD_Delta="std", Minimum_Delta="min", Maximum_Delta="max")
        .reset_index()
    )
    paired_summary["Seeds_Positive"] = paired_summary["metric"].map(
        paired_differences.groupby("metric")["delta_primary_minus_comparator"]
        .apply(lambda values: int((values > 0).sum()))
        .to_dict()
    )
    paired_summary["Seeds_Negative"] = paired_summary["metric"].map(
        paired_differences.groupby("metric")["delta_primary_minus_comparator"]
        .apply(lambda values: int((values < 0).sum()))
        .to_dict()
    )
    paired_summary["Seeds_Holm_Significant"] = paired_summary["metric"].map(
        paired_differences.groupby("metric")["p_value_holm"]
        .apply(lambda values: int((values < 0.05).sum()))
        .to_dict()
    )
    paired_summary["Direction_Favoring_Primary"] = paired_summary["metric"].map(
        {"AUROC": "positive", "AUPRC": "positive", "Brier": "negative", "NLL": "negative"}
    )
    paired_summary["All_Seeds_Favor_Primary"] = (
        ((paired_summary["Direction_Favoring_Primary"] == "positive") & (paired_summary["Seeds_Positive"] == 3))
        | ((paired_summary["Direction_Favoring_Primary"] == "negative") & (paired_summary["Seeds_Negative"] == 3))
    )

    paths = {
        "seed_metrics": output_dir / "calibrated_seed_metrics.csv",
        "metric_summary": output_dir / "calibrated_metric_summary.csv",
        "paired_differences": output_dir / "calibrated_paired_differences.csv",
        "paired_summary": output_dir / "calibrated_paired_summary.csv",
    }
    seed_metrics.to_csv(paths["seed_metrics"], index=False)
    metric_summary.to_csv(paths["metric_summary"], index=False)
    paired_differences.to_csv(paths["paired_differences"], index=False)
    paired_summary.to_csv(paths["paired_summary"], index=False)

    core_seed = seed_metrics[
        ["Seed_Index", "Process_Seed", "Run_ID", "Model_Architecture", *CORE_METRICS]
    ]
    core_summary = metric_summary[metric_summary["Metric"].isin(CORE_METRICS)]
    payload = {
        "schema_version": 1,
        "contract": "code10_three_seed_stability_v1",
        "status": "PASS",
        "probability_mode": "calibrated",
        "primary_model": PRIMARY,
        "prespecified_comparator_selection": (
            "highest pooled calibrated AUPRC among deep comparators in seed-0 main run"
        ),
        "selected_deep_comparator": COMPARATOR,
        "seeds": [run["process_seed"] for run in RUNS],
        "runs": run_provenance,
        "cohort_contract": {
            "oof_rows": int(len(reference_membership)),
            "positives": int(reference_membership["y_true"].sum()),
            "patient_clusters": int(reference_membership["patient_id"].nunique()),
            "membership_identical_across_all_runs_and_models": True,
        },
        "core_metrics_by_seed": [
            {key: _python_value(value) for key, value in row.items()}
            for row in core_seed.to_dict(orient="records")
        ],
        "core_metric_summary": [
            {key: _python_value(value) for key, value in row.items()}
            for row in core_summary.to_dict(orient="records")
        ],
        "paired_differences_by_seed": [
            {key: _python_value(value) for key, value in row.items()}
            for row in paired_differences.to_dict(orient="records")
        ],
        "paired_stability_summary": [
            {key: _python_value(value) for key, value in row.items()}
            for row in paired_summary.to_dict(orient="records")
        ],
        "artifacts": {
            name: {"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "interpretation_guardrails": [
            "Three seeds characterize optimization stability; they are not independent cohorts.",
            "No confidence interval is computed across only three seed-level point estimates.",
            "Patient-cluster bootstrap confidence intervals remain within-seed uncertainty estimates.",
            "Outer-test results were not used to replace the prespecified 128d/4-head main architecture.",
        ],
        "privacy": "aggregate statistics and hashes only; no patient or encounter identifiers",
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    manifest = settings.paths.manifests / "code10_seed_stability.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
