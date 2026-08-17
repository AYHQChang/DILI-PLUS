"""Formal-versus-pilot model-data lineage guard."""

from __future__ import annotations

import json


def load_model_data_lineage(settings) -> dict:
    path = settings.model_data_dir / "data_lineage.json"
    if not path.exists():
        raise FileNotFoundError(f"Model-data lineage manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported model-data lineage schema")
    return payload


def validate_model_data_lineage(settings, run_kind: str) -> dict:
    if run_kind not in {"formal", "pilot_legacy"}:
        raise ValueError("run_kind must be formal or pilot_legacy")
    try:
        payload = load_model_data_lineage(settings)
    except FileNotFoundError:
        if run_kind == "pilot_legacy":
            return {
                "schema_version": 0,
                "label_source": "legacy_frozen_pre_lineage",
                "formal_eligible": False,
            }
        raise
    if run_kind == "formal":
        if payload.get("label_source") != "deterministic_rebuild":
            raise ValueError("Formal training requires deterministic_rebuild labels")
        if payload.get("formal_eligible") is not True:
            raise ValueError("Model-data lineage is not marked formal_eligible")
        if payload.get("patient_group_source") != "analysis.v_patient_encounters.patient_id":
            raise ValueError("Formal training requires the source patient_id group key")
    else:
        if payload.get("formal_eligible") is True:
            raise ValueError("pilot_legacy may not run against formal deterministic data")
    return payload
