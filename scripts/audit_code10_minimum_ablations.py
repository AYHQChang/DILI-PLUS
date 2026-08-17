"""Audit cross-run minimum-ablation OOF results without exposing identifiers."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.artifacts import file_sha256  # noqa: E402
from diliplus.config import load_settings  # noqa: E402
from diliplus.models.registry import (  # noqa: E402
    ABLATION_ONLY_MODEL_NAMES,
    PRIMARY_MODEL_NAME,
)


RUN_ID = "code10_minimum_ablations_seed0"
MAIN_RUN_ID = "code10_formal_128d4h_seed0"
MODELS = (PRIMARY_MODEL_NAME, *ABLATION_ONLY_MODEL_NAMES)


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponent = np.exp(scaled)
    return exponent[:, 1] / exponent.sum(axis=1)


def main() -> int:
    settings = load_settings(PROJECT_ROOT / "configs" / "default.yaml")
    tracked_path = settings.paths.manifests / f"{RUN_ID}_run.json"
    tracked = json.loads(tracked_path.read_text(encoding="utf-8"))
    expected_sources = {
        model_name: MAIN_RUN_ID if model_name == PRIMARY_MODEL_NAME else RUN_ID
        for model_name in MODELS
    }
    if (
        tracked.get("status") != "COMPLETE"
        or tracked.get("run_id") != RUN_ID
        or tracked.get("analysis_kind") != "minimum_ablation"
        or tuple(tracked.get("models", ())) != MODELS
        or tracked.get("model_source_runs") != expected_sources
    ):
        raise RuntimeError("Minimum-ablation tracked manifest violates the protocol")

    metrics_dir = settings.paths.reports / "runs" / RUN_ID / "metrics"
    for filename, metadata in tracked["metric_files"].items():
        path = metrics_dir / filename
        if not path.exists() or file_sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"Metric file hash mismatch: {filename}")

    reference = None
    model_rows = {}
    maximum_probability_error = 0.0
    maximum_ranking_delta = {"AUROC": 0.0, "AUPRC": 0.0}
    artifact_count = 0
    for model_name in MODELS:
        source_run = expected_sources[model_name]
        frames = []
        for fold in range(1, settings.evaluation_protocol.outer_folds + 1):
            prediction_path = (
                settings.paths.reports
                / "runs"
                / source_run
                / "predictions"
                / model_name
                / f"fold_{fold:02d}.csv"
            )
            frame = pd.read_csv(prediction_path)
            if set(frame["fold"].astype(int)) != {fold}:
                raise RuntimeError(f"{model_name}: fold column mismatch")
            if frame["dataset_index"].duplicated().any() or frame["patient_id"].isna().any():
                raise RuntimeError(f"{model_name}: invalid fold membership")
            sidecar = (
                settings.paths.checkpoints
                / "runs"
                / source_run
                / model_name
                / f"fold_{fold:02d}.pt.metadata.json"
            )
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if (
                metadata.get("run_id") != source_run
                or metadata.get("model_name") != model_name
                or int(metadata.get("fold", -1)) != fold
            ):
                raise RuntimeError(f"Artifact metadata mismatch: {sidecar}")
            temperature = float(metadata["temperature"])
            logits = frame[["logit_0", "logit_1"]].to_numpy(dtype=float)
            maximum_probability_error = max(
                maximum_probability_error,
                float(np.max(np.abs(_probabilities(logits, 1.0) - frame["y_prob_raw"]))),
                float(
                    np.max(
                        np.abs(
                            _probabilities(logits, temperature)
                            - frame["y_prob_calibrated"]
                        )
                    )
                ),
            )
            y_true = frame["y_true"].to_numpy(dtype=int)
            for metric_name, metric in (
                ("AUROC", roc_auc_score),
                ("AUPRC", average_precision_score),
            ):
                delta = abs(
                    float(metric(y_true, frame["y_prob_raw"]))
                    - float(metric(y_true, frame["y_prob_calibrated"]))
                )
                maximum_ranking_delta[metric_name] = max(
                    maximum_ranking_delta[metric_name], delta
                )
            frames.append(frame)
            artifact_count += 1

        combined = pd.concat(frames, ignore_index=True).sort_values(
            "dataset_index", kind="stable"
        )
        membership = combined[
            ["dataset_index", "encounter_id", "patient_id", "y_true"]
        ].reset_index(drop=True)
        if combined["dataset_index"].duplicated().any():
            raise RuntimeError(f"{model_name}: OOF rows are not unique")
        if reference is None:
            reference = membership
        elif not reference.equals(membership):
            raise RuntimeError(f"{model_name}: OOF membership differs")
        model_rows[model_name] = len(combined)

    if maximum_probability_error > 1e-6:
        raise RuntimeError("Saved probabilities do not match logits/temperature")
    if any(value > 1e-5 for value in maximum_ranking_delta.values()):
        raise RuntimeError("Within-fold ranking invariance failed")

    pooled = pd.read_csv(metrics_dir / "pooled_metrics.csv")
    intervals = pd.read_csv(metrics_dir / "bootstrap_intervals.csv")
    comparisons = pd.read_csv(metrics_dir / "paired_comparisons.csv")
    resources = pd.read_csv(metrics_dir / "resource_usage.csv")
    expected_shapes = {
        "pooled_metrics": len(MODELS) * 2,
        "bootstrap_intervals": len(MODELS) * 2 * 4,
        "paired_comparisons": (len(MODELS) - 1) * 2 * 4,
        "resource_usage": len(MODELS) * settings.evaluation_protocol.outer_folds,
    }
    observed_shapes = {
        "pooled_metrics": len(pooled),
        "bootstrap_intervals": len(intervals),
        "paired_comparisons": len(comparisons),
        "resource_usage": len(resources),
    }
    if observed_shapes != expected_shapes:
        raise RuntimeError(f"Aggregate output shape mismatch: {observed_shapes}")
    if resources[["Model_Architecture", "Fold"]].duplicated().any():
        raise RuntimeError("Resource table has duplicate model/fold rows")

    recorded_runs = {}
    for source_run in (MAIN_RUN_ID, RUN_ID):
        path = settings.paths.reports / "run_logs" / f"{source_run}.json"
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded.get("exit_code") != 0 or recorded.get("git_dirty_at_start") is not False:
            raise RuntimeError(f"{source_run}: recorded run was not clean and successful")
        recorded_runs[source_run] = {
            "git_commit": recorded["git_commit"],
            "duration_seconds": recorded["duration_seconds"],
            "log_sha256": recorded["log_sha256"],
        }

    assert reference is not None
    payload = {
        "schema_version": 1,
        "contract": "code10_minimum_ablation_audit_v1",
        "status": "PASS",
        "analysis_run_id": RUN_ID,
        "model_source_runs": expected_sources,
        "recorded_runs": recorded_runs,
        "models": list(MODELS),
        "outer_folds": settings.evaluation_protocol.outer_folds,
        "oof_rows_per_model": model_rows,
        "oof_positive": int(reference["y_true"].sum()),
        "unique_patient_clusters": int(reference["patient_id"].nunique()),
        "artifacts_and_sidecars_verified": artifact_count,
        "maximum_saved_probability_absolute_error": maximum_probability_error,
        "fold_level_ranking_invariance": {
            "status": "PASS",
            "absolute_tolerance": 1e-5,
            "max_absolute_difference": maximum_ranking_delta,
        },
        "aggregate_table_rows": observed_shapes,
        "analysis_manifest_sha256": file_sha256(tracked_path),
        "privacy": "aggregate counts and hashes only; no identifiers",
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    output = settings.paths.manifests / "code10_minimum_ablations_audit.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
