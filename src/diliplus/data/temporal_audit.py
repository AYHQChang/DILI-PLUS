"""Audit the P0-02 prediction-time and dynamic-event leakage contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diliplus.config import load_settings


TARGET_LAB_MARKERS = ("谷丙转氨酶", "谷草转氨酶")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, np.ndarray)):
        return list(value)
    return []


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _is_target_lab(name: Any) -> bool:
    text = str(name)
    upper = text.upper()
    return any(marker in text for marker in TARGET_LAB_MARKERS) or "ALT" in upper or "AST" in upper


def build_temporal_audit(
    label_path: str | Path,
    sequence_path: str | Path,
    expected_gap_hours: float,
) -> dict[str, Any]:
    """Return a JSON-serialisable audit and fail only in the wrapper after saving it."""
    label_path = Path(label_path)
    sequence_path = Path(sequence_path)
    labels = pd.read_parquet(label_path)
    sequences = pd.read_parquet(sequence_path)

    for frame in (labels, sequences):
        for column in ("first_med_time", "index_time", "prediction_time", "t_onset"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_localize(None)

    label_ids = set(labels["encounter_id"].astype(str))
    sequence_ids = set(sequences["encounter_id"].astype(str))
    realised_gap = (
        (labels["index_time"] - labels["prediction_time"]).dt.total_seconds() / 3600.0
    )
    positive = labels["label_ahi_proxy"].eq(1)

    violations: dict[str, int] = {
        "duplicate_label_encounters": int(labels["encounter_id"].duplicated().sum()),
        "duplicate_sequence_encounters": int(sequences["encounter_id"].duplicated().sum()),
        "labels_missing_sequences": len(label_ids - sequence_ids),
        "sequences_without_labels": len(sequence_ids - label_ids),
        "prediction_not_after_first_med": int(
            (labels["prediction_time"] <= labels["first_med_time"]).sum()
        ),
        "realised_gap_mismatch": int(
            (~np.isclose(realised_gap, float(expected_gap_hours), atol=1.0 / 3600.0)).sum()
        ),
        "positive_index_not_onset": int(
            (
                positive
                & (
                    (labels["index_time"] - labels["t_onset"])
                    .dt.total_seconds()
                    .abs()
                    .gt(1.0)
                )
            ).sum()
        ),
        "med_event_at_or_after_prediction": 0,
        "lab_event_at_or_after_prediction": 0,
        "target_threshold_lab_in_input": 0,
        "med_sequence_length_mismatch": 0,
        "lab_sequence_length_mismatch": 0,
    }

    encounters_without_pre_prediction_labs = 0
    total_med_events = 0
    total_lab_events = 0
    latest_med_margin_hours: float | None = None
    latest_lab_margin_hours: float | None = None

    for row in sequences.itertuples(index=False):
        prediction_time = _timestamp(row.prediction_time)
        med_tokens = _as_list(row.med_tokens)
        med_times = [_timestamp(value) for value in _as_list(row.med_event_times)]
        lab_tokens = _as_list(row.lab_tokens)
        lab_values = _as_list(row.lab_values)
        lab_times = [_timestamp(value) for value in _as_list(row.lab_event_times)]

        if len(med_tokens) != len(med_times):
            violations["med_sequence_length_mismatch"] += 1
        if not (len(lab_tokens) == len(lab_values) == len(lab_times)):
            violations["lab_sequence_length_mismatch"] += 1
        if not lab_times:
            encounters_without_pre_prediction_labs += 1

        total_med_events += len(med_times)
        total_lab_events += len(lab_times)

        for event_time in med_times:
            if event_time >= prediction_time:
                violations["med_event_at_or_after_prediction"] += 1
            margin = (prediction_time - event_time).total_seconds() / 3600.0
            latest_med_margin_hours = (
                margin if latest_med_margin_hours is None else min(latest_med_margin_hours, margin)
            )

        for token, value, event_time in zip(lab_tokens, lab_values, lab_times):
            if event_time >= prediction_time:
                violations["lab_event_at_or_after_prediction"] += 1
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = float("nan")
            if _is_target_lab(token) and np.isfinite(numeric_value) and numeric_value >= 120.0:
                violations["target_threshold_lab_in_input"] += 1
            margin = (prediction_time - event_time).total_seconds() / 3600.0
            latest_lab_margin_hours = (
                margin if latest_lab_margin_hours is None else min(latest_lab_margin_hours, margin)
            )

    failed_contracts = [name for name, count in violations.items() if count != 0]
    return {
        "audit_status": "PASS" if not failed_contracts else "FAIL",
        "expected_prediction_gap_hours": float(expected_gap_hours),
        "label_encounters": int(len(labels)),
        "sequence_encounters": int(len(sequences)),
        "ahi_proxy_positive": int(positive.sum()),
        "ahi_proxy_negative": int((~positive).sum()),
        "ahi_proxy_prevalence_pct": round(100.0 * float(positive.mean()), 6),
        "total_med_events": int(total_med_events),
        "total_lab_events": int(total_lab_events),
        "encounters_without_pre_prediction_labs": int(encounters_without_pre_prediction_labs),
        "minimum_med_time_margin_hours": latest_med_margin_hours,
        "minimum_lab_time_margin_hours": latest_lab_margin_hours,
        "violations": violations,
        "failed_contracts": failed_contracts,
        "label_artifact": str(label_path.resolve()),
        "sequence_artifact": str(sequence_path.resolve()),
        "label_sha256": _sha256(label_path),
        "sequence_sha256": _sha256(sequence_path),
    }


def audit_prediction_time_contract(settings=None):
    settings = settings or load_settings()
    label_path = settings.model_data_dir / "02_dili_labels_censored.parquet"
    sequence_path = settings.model_data_dir / "03_dili_dual_stream_tensors.parquet"
    audit_dir = settings.prediction_audit_dir
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_json = audit_dir / "temporal_leakage_audit.json"
    output_csv = audit_dir / "corrected_dataset_summary.csv"

    audit = build_temporal_audit(
        label_path, sequence_path, settings.prediction.gap_hours
    )
    output_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                key: value
                for key, value in audit.items()
                if key not in {"violations", "failed_contracts"}
            }
        ]
    ).to_csv(output_csv, index=False)

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"[DILI-PLUS] Temporal audit saved: {output_json}")
    if audit["audit_status"] != "PASS":
        raise RuntimeError(
            "Prediction-time leakage contract failed: "
            + ", ".join(audit["failed_contracts"])
        )
    return audit


if __name__ == "__main__":
    audit_prediction_time_contract()
