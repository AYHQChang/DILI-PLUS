"""Audit Code-10 formal outputs without exposing row identifiers.

The audit verifies OOF coverage, artifact/probability pairing, fold-level
temperature ranking invariance, tracked hashes and aggregate result shapes.
Only aggregate counts and hashes are written to Git-tracked output.
"""

from __future__ import annotations

import argparse
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
from diliplus.models.registry import FORMAL_DEEP_MODEL_NAMES  # noqa: E402


ML_MODELS = ("LogisticRegression", "XGBoost")
FORMAL_MODELS = tuple(FORMAL_DEEP_MODEL_NAMES) + ML_MODELS


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


def _artifact_sidecar(settings, run_id: str, model_name: str, fold: int) -> Path:
    suffix = ".joblib.metadata.json" if model_name in ML_MODELS else ".pt.metadata.json"
    return (
        settings.paths.checkpoints
        / "runs"
        / run_id
        / model_name
        / f"fold_{fold:02d}{suffix}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="code10_formal_128d4h_seed0")
    args = parser.parse_args(argv)
    settings = load_settings(PROJECT_ROOT / "configs" / "default.yaml")
    run_root = settings.paths.reports / "runs" / args.run_id
    metrics_dir = run_root / "metrics"
    tracked_path = settings.paths.manifests / "code10_formal_run.json"
    tracked = json.loads(tracked_path.read_text(encoding="utf-8"))
    if tracked.get("status") != "COMPLETE" or tracked.get("run_id") != args.run_id:
        raise RuntimeError("Tracked formal-run manifest is incomplete or mismatched")
    if tuple(tracked.get("models", ())) != FORMAL_MODELS:
        raise RuntimeError("Tracked formal model order does not match the registry contract")

    for filename, metadata in tracked["metric_files"].items():
        path = metrics_dir / filename
        if not path.exists() or file_sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"Metric file hash mismatch: {filename}")

    anchor_membership = None
    anchor_labels = None
    model_rows = {}
    artifact_count = 0
    maximum_probability_error = 0.0
    maximum_fold_ranking_delta = {"AUROC": 0.0, "AUPRC": 0.0}
    for model_name in FORMAL_MODELS:
        frames = []
        for fold in range(1, settings.evaluation_protocol.outer_folds + 1):
            prediction_path = (
                run_root / "predictions" / model_name / f"fold_{fold:02d}.csv"
            )
            frame = pd.read_csv(prediction_path)
            if set(frame["fold"].astype(int)) != {fold}:
                raise RuntimeError(f"{model_name} fold column mismatch at fold {fold}")
            if frame["dataset_index"].duplicated().any():
                raise RuntimeError(f"{model_name} fold {fold} has duplicate test rows")
            if frame["patient_id"].isna().any():
                raise RuntimeError(f"{model_name} fold {fold} has missing patient groups")

            sidecar = _artifact_sidecar(settings, args.run_id, model_name, fold)
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if (
                metadata.get("run_id") != args.run_id
                or metadata.get("model_name") != model_name
                or int(metadata.get("fold", -1)) != fold
            ):
                raise RuntimeError(f"Artifact metadata mismatch: {sidecar}")
            temperature = float(metadata["temperature"])
            if not np.isfinite(temperature) or temperature <= 0:
                raise RuntimeError(f"Invalid temperature in {sidecar}")
            logits = frame[["logit_0", "logit_1"]].to_numpy(dtype=float)
            expected_raw = _probabilities(logits, 1.0)
            expected_calibrated = _probabilities(logits, temperature)
            maximum_probability_error = max(
                maximum_probability_error,
                float(np.max(np.abs(expected_raw - frame["y_prob_raw"]))),
                float(
                    np.max(
                        np.abs(
                            expected_calibrated - frame["y_prob_calibrated"]
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
                maximum_fold_ranking_delta[metric_name] = max(
                    maximum_fold_ranking_delta[metric_name], delta
                )
            frames.append(frame)
            artifact_count += 1

        combined = pd.concat(frames, ignore_index=True).sort_values(
            "dataset_index", kind="stable"
        )
        if combined["dataset_index"].duplicated().any():
            raise RuntimeError(f"{model_name} OOF rows are not unique")
        membership = combined["dataset_index"].to_numpy(dtype=int)
        labels = combined["y_true"].to_numpy(dtype=int)
        if anchor_membership is None:
            anchor_membership = membership
            anchor_labels = labels
        elif not np.array_equal(membership, anchor_membership) or not np.array_equal(
            labels, anchor_labels
        ):
            raise RuntimeError(f"OOF membership/labels differ for {model_name}")
        model_rows[model_name] = int(len(combined))

    if maximum_probability_error > 1e-6:
        raise RuntimeError(
            f"Saved probabilities do not match logits/temperature: {maximum_probability_error}"
        )
    if any(value > 1e-5 for value in maximum_fold_ranking_delta.values()):
        raise RuntimeError(
            f"Fold-level temperature ranking invariance failed: {maximum_fold_ranking_delta}"
        )

    pooled = pd.read_csv(metrics_dir / "pooled_metrics.csv")
    intervals = pd.read_csv(metrics_dir / "bootstrap_intervals.csv")
    comparisons = pd.read_csv(metrics_dir / "paired_comparisons.csv")
    resources = pd.read_csv(metrics_dir / "resource_usage.csv")
    if len(pooled) != len(FORMAL_MODELS) * 2:
        raise RuntimeError("Pooled metrics do not contain two modes for six models")
    if len(intervals) != len(FORMAL_MODELS) * 2 * 4:
        raise RuntimeError("Bootstrap interval table has an unexpected shape")
    if len(comparisons) != (len(FORMAL_MODELS) - 1) * 2 * 4:
        raise RuntimeError("Paired-comparison table has an unexpected shape")
    if len(resources) != len(FORMAL_MODELS) * settings.evaluation_protocol.outer_folds:
        raise RuntimeError("Resource table must contain one row per model/fold")
    if resources[["Model_Architecture", "Fold"]].duplicated().any():
        raise RuntimeError("Resource table contains duplicate model/fold rows")

    recorded_path = settings.paths.reports / "run_logs" / f"{args.run_id}.json"
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    if recorded.get("exit_code") != 0 or recorded.get("git_dirty_at_start") is not False:
        raise RuntimeError("Recorded training did not start cleanly and exit successfully")

    payload = {
        "schema_version": 1,
        "contract": "code10_formal_run_audit_v1",
        "status": "PASS",
        "run_id": args.run_id,
        "training_git_commit": recorded.get("git_commit"),
        "training_duration_seconds": recorded.get("duration_seconds"),
        "formal_models": list(FORMAL_MODELS),
        "outer_folds": settings.evaluation_protocol.outer_folds,
        "oof_rows_per_model": model_rows,
        "oof_positive": int(np.sum(anchor_labels)),
        "unique_patient_clusters": int(tracked["bootstrap_intervals"][0]["clusters"]),
        "artifacts_and_sidecars_verified": artifact_count,
        "maximum_saved_probability_absolute_error": maximum_probability_error,
        "fold_level_ranking_invariance": {
            "status": "PASS",
            "absolute_tolerance": 1e-5,
            "max_absolute_difference": maximum_fold_ranking_delta,
            "scope": "within model/fold; pooled cross-fold ordering may change",
        },
        "aggregate_table_rows": {
            "pooled_metrics": int(len(pooled)),
            "bootstrap_intervals": int(len(intervals)),
            "paired_comparisons": int(len(comparisons)),
            "resource_usage": int(len(resources)),
        },
        "formal_manifest_sha256": file_sha256(tracked_path),
        "recorded_log_sha256": recorded.get("log_sha256"),
        "privacy": "aggregate counts and hashes only; no patient or encounter identifiers",
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    output = settings.paths.manifests / "code10_formal_run_audit.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
