"""Audit strict early-warning predictions, metrics and tracked hashes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.artifacts import file_sha256  # noqa: E402
from diliplus.config import load_settings  # noqa: E402
from diliplus.models.registry import FORMAL_DEEP_MODEL_NAMES  # noqa: E402


MODEL_RUN_ID = "code10_formal_128d4h_seed0"
RECORDED_RUN_ID = "code09_early_warning_formal_calibrated"
HORIZONS = (24, 48, 72)


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def main() -> int:
    settings = load_settings(PROJECT_ROOT / "configs" / "default.yaml")
    manifest_path = settings.paths.manifests / "code09_early_warning_performance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("run_id") != MODEL_RUN_ID
        or tuple(manifest.get("models", ())) != tuple(FORMAL_DEEP_MODEL_NAMES)
        or tuple(map(int, manifest.get("effective_horizons_hours", ()))) != HORIZONS
        or manifest.get("probability_mode") != "calibrated"
    ):
        raise RuntimeError("Early-warning performance manifest violates the protocol")

    root = settings.paths.reports / "runs" / MODEL_RUN_ID / "early_warning"
    metrics_root = root / "metrics"
    for filename, metadata in manifest["metric_files"].items():
        path = metrics_root / filename
        if not path.exists() or file_sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"Metric file hash mismatch: {filename}")

    anchor = None
    prediction_rows = {}
    maximum_24h_probability_error = 0.0
    prediction_count = 0
    for model_name in FORMAL_DEEP_MODEL_NAMES:
        main_frames = [
            pd.read_csv(
                settings.paths.reports
                / "runs"
                / MODEL_RUN_ID
                / "predictions"
                / model_name
                / f"fold_{fold:02d}.csv"
            )
            for fold in range(1, settings.evaluation_protocol.outer_folds + 1)
        ]
        main_oof = pd.concat(main_frames, ignore_index=True).sort_values(
            "dataset_index", kind="stable"
        )
        for horizon in HORIZONS:
            path = (
                root
                / "predictions"
                / model_name
                / f"horizon_{horizon:03d}h.csv"
            )
            tracked_key = path.relative_to(PROJECT_ROOT).as_posix()
            if (
                tracked_key not in manifest["prediction_file_hashes"]
                or file_sha256(path) != manifest["prediction_file_hashes"][tracked_key]
            ):
                raise RuntimeError(f"Prediction hash mismatch: {tracked_key}")
            frame = pd.read_csv(path).sort_values("dataset_index", kind="stable")
            if (
                frame["dataset_index"].duplicated().any()
                or frame["patient_id"].isna().any()
                or not np.isfinite(frame["y_prob"]).all()
                or not frame["y_prob"].between(0.0, 1.0).all()
            ):
                raise RuntimeError(f"Invalid prediction rows: {model_name}/{horizon}h")
            membership = frame[
                ["dataset_index", "encounter_id", "patient_id", "y_true"]
            ].reset_index(drop=True)
            if anchor is None:
                anchor = membership
            elif not anchor.equals(membership):
                raise RuntimeError("Early-warning OOF membership differs across outputs")
            if horizon == 24:
                if not membership.equals(
                    main_oof[
                        ["dataset_index", "encounter_id", "patient_id", "y_true"]
                    ].reset_index(drop=True)
                ):
                    raise RuntimeError(f"{model_name}: 24h membership differs from main OOF")
                maximum_24h_probability_error = max(
                    maximum_24h_probability_error,
                    float(
                        np.max(
                            np.abs(
                                frame["y_prob"].to_numpy(dtype=float)
                                - main_oof["y_prob_calibrated"].to_numpy(dtype=float)
                            )
                        )
                    ),
                )
            prediction_rows[f"{model_name}/{horizon}h"] = len(frame)
            prediction_count += 1

    if maximum_24h_probability_error > 1e-7:
        raise RuntimeError("24h early-warning probabilities do not reproduce main OOF")

    pooled = pd.read_csv(metrics_root / "pooled_metrics.csv")
    intervals = pd.read_csv(metrics_root / "bootstrap_intervals.csv")
    primary_comparisons = pd.read_csv(metrics_root / "primary_vs_comparators.csv")
    degradation = pd.read_csv(metrics_root / "horizon_degradation.csv")
    bins = pd.read_csv(metrics_root / "calibration_bins.csv")
    curves = pd.read_csv(metrics_root / "dca_curves.csv")
    expected_shapes = {
        "pooled_metrics": len(FORMAL_DEEP_MODEL_NAMES) * len(HORIZONS),
        "bootstrap_intervals": len(FORMAL_DEEP_MODEL_NAMES) * len(HORIZONS) * 4,
        "primary_vs_comparators": len(HORIZONS) * (len(FORMAL_DEEP_MODEL_NAMES) - 1) * 4,
        "horizon_degradation": len(FORMAL_DEEP_MODEL_NAMES) * (len(HORIZONS) - 1) * 4,
        "calibration_bins": len(FORMAL_DEEP_MODEL_NAMES) * len(HORIZONS) * 10,
        "dca_curves": len(FORMAL_DEEP_MODEL_NAMES)
        * len(HORIZONS)
        * len(
            np.arange(
                settings.evaluation_protocol.dca_min_threshold,
                settings.evaluation_protocol.dca_max_threshold
                + settings.evaluation_protocol.dca_step / 2.0,
                settings.evaluation_protocol.dca_step,
            )
        ),
    }
    observed_shapes = {
        "pooled_metrics": len(pooled),
        "bootstrap_intervals": len(intervals),
        "primary_vs_comparators": len(primary_comparisons),
        "horizon_degradation": len(degradation),
        "calibration_bins": len(bins),
        "dca_curves": len(curves),
    }
    if observed_shapes != expected_shapes:
        raise RuntimeError(f"Early-warning aggregate shape mismatch: {observed_shapes}")
    if set(intervals["clusters"].astype(int)) != {44611}:
        raise RuntimeError("Early-warning intervals do not use source patient clusters")

    recorded_path = settings.paths.reports / "run_logs" / f"{RECORDED_RUN_ID}.json"
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    if recorded.get("exit_code") != 0 or recorded.get("git_dirty_at_start") is not False:
        raise RuntimeError("Recorded early-warning process was not clean and successful")

    assert anchor is not None
    payload = {
        "schema_version": 1,
        "contract": "code09_early_warning_performance_audit_v1",
        "status": "PASS",
        "model_run_id": MODEL_RUN_ID,
        "recorded_run_id": RECORDED_RUN_ID,
        "evaluation_git_commit": recorded["git_commit"],
        "duration_seconds": recorded["duration_seconds"],
        "recorded_log_sha256": recorded["log_sha256"],
        "models": list(FORMAL_DEEP_MODEL_NAMES),
        "horizons": list(HORIZONS),
        "prediction_files_verified": prediction_count,
        "prediction_rows": prediction_rows,
        "oof_positive": int(anchor["y_true"].sum()),
        "unique_patient_clusters": int(anchor["patient_id"].nunique()),
        "maximum_24h_probability_absolute_error_vs_main_oof": maximum_24h_probability_error,
        "aggregate_table_rows": observed_shapes,
        "performance_manifest_sha256": file_sha256(manifest_path),
        "privacy": "aggregate counts and hashes only; no identifiers",
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    output = settings.paths.manifests / "code09_early_warning_performance_audit.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
