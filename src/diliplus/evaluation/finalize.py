"""Finalize one complete Code-10 run from saved out-of-fold predictions."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diliplus.artifacts import dataset_fingerprint, file_sha256, run_report_dir
from diliplus.evaluation.metrics import (
    add_holm_adjustment,
    calibration_bins,
    cluster_bootstrap_intervals,
    compute_binary_metrics,
    decision_curve,
    paired_cluster_bootstrap,
)
from diliplus.models.registry import PRIMARY_MODEL_NAME
from diliplus.reproducibility import derive_seed


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, encoding="utf-8", stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def _dca_thresholds(settings):
    protocol = settings.evaluation_protocol
    return np.arange(
        protocol.dca_min_threshold,
        protocol.dca_max_threshold + protocol.dca_step / 2.0,
        protocol.dca_step,
    )


def _load_model_oof(report_root: Path, model_name: str, expected_folds: int) -> pd.DataFrame:
    directory = report_root / "predictions" / model_name
    paths = sorted(directory.glob("fold_*.csv"))
    if len(paths) != expected_folds:
        raise RuntimeError(
            f"{model_name} has {len(paths)} prediction folds; expected {expected_folds}"
        )
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {
        "dataset_index",
        "encounter_id",
        "patient_id",
        "y_true",
        "logit_0",
        "logit_1",
        "y_prob_raw",
        "y_prob_calibrated",
        "fold",
        "reference_prevalence",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{model_name} OOF predictions are missing columns: {missing}")
    if frame["dataset_index"].duplicated().any():
        raise RuntimeError(f"{model_name} outer-test rows are not unique")
    frame = frame.sort_values("dataset_index", kind="stable").reset_index(drop=True)
    frame.to_parquet(directory / "oof_predictions.parquet", index=False)
    return frame


def _common_membership(frames: dict[str, pd.DataFrame]) -> None:
    names = list(frames)
    anchor = frames[names[0]][["dataset_index", "encounter_id", "patient_id", "y_true"]]
    for name in names[1:]:
        candidate = frames[name][["dataset_index", "encounter_id", "patient_id", "y_true"]]
        if not anchor.equals(candidate):
            raise RuntimeError(f"OOF membership or labels differ for model {name}")


def _training_commit(settings, run_id: str) -> str | None:
    """Recover the clean commit captured when the recorded training began.

    The recorded-run JSON is written only after the training child exits, so
    it is absent during the original in-process finalization but available to
    any later aggregate-only reanalysis.
    """
    path = settings.paths.reports / "run_logs" / f"{run_id}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("git_commit")
    return _git("rev-parse", "HEAD")


def _fold_metric_paths(metrics_dir: Path, model_names) -> list[Path]:
    paths = []
    for model_name in model_names:
        filename = (
            "ml_baselines.csv"
            if model_name in {"LogisticRegression", "XGBoost"}
            else f"{model_name}.csv"
        )
        path = metrics_dir / filename
        if path not in paths:
            paths.append(path)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fold metric files: {missing}")
    return paths


def _fold_ranking_invariance(metric_frames: list[pd.DataFrame]) -> dict:
    frame = pd.concat(metric_frames, ignore_index=True)
    required = {
        "Model_Architecture",
        "Fold",
        "Probability_Mode",
        "AUROC",
        "AUPRC",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Fold metrics cannot audit ranking invariance: {missing}")
    pivot = frame.pivot(
        index=["Model_Architecture", "Fold"],
        columns="Probability_Mode",
        values=["AUROC", "AUPRC"],
    )
    modes = set(pivot.columns.get_level_values(1))
    if modes != {"raw", "calibrated"}:
        raise RuntimeError(f"Expected paired raw/calibrated fold metrics, found {modes}")
    difference = (
        pivot.xs("raw", axis=1, level=1)
        - pivot.xs("calibrated", axis=1, level=1)
    ).abs()
    maxima = {metric: float(difference[metric].max()) for metric in difference}
    tolerance = 1e-5
    if any(value > tolerance for value in maxima.values()):
        raise RuntimeError(
            "Within-fold temperature scaling changed discrimination beyond "
            f"floating-point tolerance: {maxima}"
        )
    return {
        "scope": "within each model/fold only",
        "status": "PASS",
        "absolute_tolerance": tolerance,
        "max_absolute_difference": maxima,
        "pooled_note": (
            "Fold-specific temperatures may change cross-fold ordering after OOF "
            "pooling; pooled raw and calibrated discrimination need not be equal."
        ),
    }


def _resource_table(metric_frames: list[pd.DataFrame]) -> pd.DataFrame:
    resource_frames = []
    for frame in metric_frames:
        columns = [
            column
            for column in (
                "Model_Architecture",
                "Fold",
                "Selected_Epoch",
                "Best_Selection_AUPRC",
                "Parameter_Count",
                "Feature_Count",
                "Fold_Duration_Seconds",
                "Peak_GPU_Memory_MB",
                "Device",
            )
            if column in frame.columns
        ]
        resource_frames.append(frame[columns])
    resources = pd.concat(resource_frames, ignore_index=True)
    keys = ["Model_Architecture", "Fold"]
    for column in (value for value in resources.columns if value not in keys):
        conflicts = resources.groupby(keys, dropna=False)[column].nunique(dropna=False)
        if conflicts.gt(1).any():
            raise RuntimeError(
                f"Raw/calibrated rows disagree on resource field {column!r}"
            )
    return (
        resources.sort_values(keys, kind="stable")
        .drop_duplicates(keys, keep="first")
        .reset_index(drop=True)
    )


def finalize_formal_run(settings, run_id: str, model_names) -> dict:
    model_names = tuple(model_names)
    if PRIMARY_MODEL_NAME not in model_names:
        raise ValueError("Formal finalization requires the primary model")
    report_root = run_report_dir(settings, run_id)
    metrics_dir = report_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    protocol = settings.evaluation_protocol
    frames = {
        model: _load_model_oof(report_root, model, protocol.outer_folds)
        for model in model_names
    }
    _common_membership(frames)
    expected_rows = len(frames[model_names[0]])
    if expected_rows == 0:
        raise RuntimeError("Formal OOF predictions are empty")

    pooled_rows = []
    interval_frames = []
    calibration_frames = []
    dca_frames = []
    thresholds = _dca_thresholds(settings)
    for model_name, frame in frames.items():
        for mode in ("raw", "calibrated"):
            probability_column = f"y_prob_{mode}"
            probabilities = frame[probability_column].to_numpy(dtype=float)
            metrics = compute_binary_metrics(
                frame["y_true"],
                probabilities,
                reference_prevalence=frame["reference_prevalence"],
                p_auc_fpr_limits=protocol.p_auc_fpr_limits,
                risk_thresholds=protocol.risk_thresholds,
                alert_budgets=protocol.alert_budgets,
                dca_thresholds=thresholds,
            )
            metrics.update(
                {
                    "Run_ID": run_id,
                    "Model_Architecture": model_name,
                    "Probability_Mode": mode,
                }
            )
            pooled_rows.append(metrics)

            intervals = cluster_bootstrap_intervals(
                frame["y_true"],
                probabilities,
                frame["patient_id"],
                n_replicates=protocol.bootstrap_replicates,
                seed=derive_seed(
                    settings.reproducibility.bootstrap_seed,
                    run_id,
                    model_name,
                    mode,
                    "cluster_ci",
                ),
            )
            intervals.insert(0, "Probability_Mode", mode)
            intervals.insert(0, "Model_Architecture", model_name)
            intervals.insert(0, "Run_ID", run_id)
            interval_frames.append(intervals)

            bins = calibration_bins(frame["y_true"], probabilities)
            bins.insert(0, "Probability_Mode", mode)
            bins.insert(0, "Model_Architecture", model_name)
            bins.insert(0, "Run_ID", run_id)
            calibration_frames.append(bins)

            curve = decision_curve(frame["y_true"], probabilities, thresholds)
            curve.insert(0, "Probability_Mode", mode)
            curve.insert(0, "Model_Architecture", model_name)
            curve.insert(0, "Run_ID", run_id)
            dca_frames.append(curve)

    comparisons = []
    primary = frames[PRIMARY_MODEL_NAME]
    for mode in ("raw", "calibrated"):
        mode_frames = []
        for comparator_name in model_names:
            if comparator_name == PRIMARY_MODEL_NAME:
                continue
            comparison = paired_cluster_bootstrap(
                primary["y_true"],
                primary[f"y_prob_{mode}"],
                frames[comparator_name][f"y_prob_{mode}"],
                primary["patient_id"],
                n_replicates=protocol.bootstrap_replicates,
                seed=derive_seed(
                    settings.reproducibility.bootstrap_seed,
                    run_id,
                    comparator_name,
                    mode,
                    "paired",
                ),
            )
            comparison.insert(0, "Comparator", comparator_name)
            comparison.insert(0, "Primary", PRIMARY_MODEL_NAME)
            comparison.insert(0, "Probability_Mode", mode)
            comparison.insert(0, "Run_ID", run_id)
            mode_frames.append(comparison)
        comparisons.append(add_holm_adjustment(pd.concat(mode_frames, ignore_index=True)))

    outputs = {
        "pooled_metrics.csv": pd.DataFrame(pooled_rows),
        "bootstrap_intervals.csv": pd.concat(interval_frames, ignore_index=True),
        "paired_comparisons.csv": pd.concat(comparisons, ignore_index=True),
        "calibration_bins.csv": pd.concat(calibration_frames, ignore_index=True),
        "dca_curves.csv": pd.concat(dca_frames, ignore_index=True),
    }
    for filename, frame in outputs.items():
        frame.to_csv(metrics_dir / filename, index=False)

    fold_metric_paths = _fold_metric_paths(metrics_dir, model_names)
    fold_metric_frames = [pd.read_csv(path) for path in fold_metric_paths]
    ranking_invariance = _fold_ranking_invariance(fold_metric_frames)
    resources = _resource_table(fold_metric_frames)
    resources.to_csv(metrics_dir / "resource_usage.csv", index=False)

    tracked_payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "run_id": run_id,
        "run_kind": "formal",
        "git_commit": _git("rev-parse", "HEAD"),
        "training_git_commit": _training_commit(settings, run_id),
        "analysis_git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_at_finalization": bool(_git("status", "--porcelain")),
        "models": list(model_names),
        "outer_folds": protocol.outer_folds,
        "oof_rows_per_model": expected_rows,
        "bootstrap_replicates": protocol.bootstrap_replicates,
        "patient_group_source": "analysis.v_patient_encounters.patient_id",
        "selection_metric": settings.training.selection_metric,
        "ranking_invariance": ranking_invariance,
        "dataset_fingerprint": dataset_fingerprint(settings),
        "analysis_implementation_sha256": {
            "src/diliplus/evaluation/finalize.py": file_sha256(Path(__file__)),
            "src/diliplus/evaluation/metrics.py": file_sha256(
                Path(__file__).with_name("metrics.py")
            ),
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_bytes": (
                int(torch.cuda.get_device_properties(0).total_memory)
                if torch.cuda.is_available()
                else None
            ),
            "system_memory_gb_user_reported": 48,
        },
        "metric_files": {
            path.name: {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(metrics_dir.iterdir())
            if path.is_file()
        },
        "pooled_metrics": outputs["pooled_metrics.csv"].to_dict(orient="records"),
        "bootstrap_intervals": outputs["bootstrap_intervals.csv"].to_dict(
            orient="records"
        ),
        "paired_comparisons": outputs["paired_comparisons.csv"].to_dict(
            orient="records"
        ),
    }
    tracked_payload = _json_safe(tracked_payload)
    local_manifest = report_root / "run_manifest.json"
    local_manifest.write_text(
        json.dumps(tracked_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tracked_manifest = settings.paths.manifests / "code10_formal_run.json"
    tracked_manifest.write_text(
        json.dumps(tracked_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return tracked_payload
