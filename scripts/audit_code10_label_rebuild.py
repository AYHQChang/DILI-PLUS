"""Aggregate-only audit of the deterministic Code-10 label reconstruction."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.artifacts import file_sha256
from diliplus.config import load_settings
from diliplus.data.labels import (
    _build_encounter_patient_map,
    _build_prediction_label_table,
    _build_raw_label_table,
)
from diliplus.database import connect_source_database


def main() -> int:
    settings = load_settings()
    aligned = settings.paths.data_cache / "01_aligned_dili_labs.parquet"
    legacy = settings.paths.data_cache / "02_dili_labels_censored.parquet"
    if not aligned.exists() or not legacy.exists():
        raise FileNotFoundError("Both aligned laboratory and frozen legacy artifacts are required")

    connection = connect_source_database(settings)
    try:
        _build_raw_label_table(
            connection, aligned, settings.prediction.baseline_threshold_u_l
        )
        _build_encounter_patient_map(connection)
        _build_prediction_label_table(
            connection,
            settings.prediction.gap_hours,
            settings.prediction.pseudo_index_seed,
            "deterministic_rebuild",
        )
        deterministic = connection.execute(
            """
            SELECT
                COUNT(*)::BIGINT AS target_lab_encounters,
                SUM(baseline_rule_pass)::BIGINT AS baseline_rule_pass,
                SUM(NOT baseline_rule_pass)::BIGINT AS baseline_rule_fail,
                SUM(baseline_target_rows > 1)::BIGINT AS tied_baseline_encounters,
                SUM(baseline_distinct_target_items > 1)::BIGINT
                    AS both_alt_ast_at_baseline,
                SUM(baseline_max_value >= 120)::BIGINT
                    AS baseline_numeric_at_or_above_threshold,
                SUM(label_ahi_proxy = 1)::BIGINT AS raw_positive,
                SUM(label_ahi_proxy = 0)::BIGINT AS raw_negative,
                SUM(label_ahi_proxy IS NULL)::BIGINT AS raw_excluded,
                SUM(label_ahi_proxy = 1 AND t_onset > last_med_time)::BIGINT
                    AS positive_onset_after_last_medication
            FROM temp_raw_ahi_labels
            """
        ).df().iloc[0].to_dict()
        formal = connection.execute(
            """
            SELECT
                COUNT(*)::BIGINT AS encounters,
                COUNT(DISTINCT patient_id)::BIGINT AS patients,
                SUM(label_ahi_proxy = 1)::BIGINT AS positive,
                SUM(label_ahi_proxy = 0)::BIGINT AS negative,
                SUM(patient_id IS NULL)::BIGINT AS missing_patient_id,
                COUNT(*) - COUNT(DISTINCT patient_id)::BIGINT AS repeat_encounter_excess,
                MIN(EPOCH(index_time) - EPOCH(prediction_time)) / 3600.0
                    AS minimum_realised_gap_hours,
                MAX(EPOCH(index_time) - EPOCH(prediction_time)) / 3600.0
                    AS maximum_realised_gap_hours
            FROM temp_ahi_prediction_labels
            """
        ).df().iloc[0].to_dict()
        legacy_sql = legacy.as_posix().replace("'", "''")
        comparison = connection.execute(
            f"""
            WITH legacy AS (
                SELECT
                    TRIM(CAST(encounter_id AS VARCHAR)) AS encounter_id,
                    CAST(label_dili AS INTEGER) AS legacy_label
                FROM read_parquet('{legacy_sql}')
                WHERE label_dili IS NOT NULL
            ), deterministic AS (
                SELECT encounter_id, label_ahi_proxy
                FROM temp_raw_ahi_labels
                WHERE label_ahi_proxy IS NOT NULL
            )
            SELECT
                (SELECT COUNT(*) FROM legacy)::BIGINT AS legacy_encounters,
                (SELECT SUM(legacy_label = 1) FROM legacy)::BIGINT AS legacy_positive,
                COUNT(*)::BIGINT AS overlapping_eligible_encounters,
                SUM(legacy_label <> label_ahi_proxy)::BIGINT AS label_disagreements,
                SUM(legacy_label = 1 AND label_ahi_proxy = 0)::BIGINT
                    AS legacy_positive_deterministic_negative,
                SUM(legacy_label = 0 AND label_ahi_proxy = 1)::BIGINT
                    AS legacy_negative_deterministic_positive
            FROM legacy
            INNER JOIN deterministic USING (encounter_id)
            """
        ).df().iloc[0].to_dict()
    finally:
        connection.close()

    def normalize(mapping):
        return {
            key: (float(value) if isinstance(value, float) else int(value))
            for key, value in mapping.items()
        }

    deterministic = normalize(deterministic)
    formal = normalize(formal)
    comparison = normalize(comparison)
    status = (
        "PASS"
        if formal["positive"] > 0
        and formal["negative"] > 0
        and formal["missing_patient_id"] == 0
        and formal["minimum_realised_gap_hours"] == settings.prediction.gap_hours
        and formal["maximum_realised_gap_hours"] == settings.prediction.gap_hours
        else "FAIL"
    )
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": "aggregate-only deterministic label reconstruction audit",
        "privacy": "no encounter_id or patient_id values are stored",
        "rule": {
            "baseline": (
                "all ALT/AST rows at earliest target timestamp have normal/low "
                "status and all available numeric values are below threshold"
            ),
            "onset": "first strictly later ALT/AST numeric value at or above threshold",
            "threshold_u_l": settings.prediction.baseline_threshold_u_l,
            "prediction_gap_hours": settings.prediction.gap_hours,
            "group_key": "analysis.v_patient_encounters.patient_id",
        },
        "deterministic_raw": deterministic,
        "formal_gap_eligible": formal,
        "legacy_comparison": comparison,
        "inputs": {
            "aligned_labs_sha256": file_sha256(aligned),
            "legacy_labels_sha256": file_sha256(legacy),
            "config_sha256": file_sha256(settings.config_path),
            "implementation_sha256": file_sha256(
                PROJECT_ROOT / "src" / "diliplus" / "data" / "labels.py"
            ),
        },
    }
    tracked = settings.paths.manifests / "code10_label_rebuild_audit.json"
    local_dir = settings.paths.reports / "p0_10_experiment_protocol"
    local_dir.mkdir(parents=True, exist_ok=True)
    tracked.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (local_dir / "label_rebuild_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
