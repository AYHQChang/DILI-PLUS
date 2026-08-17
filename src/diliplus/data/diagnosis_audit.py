"""Audit diagnosis encounter linkage and prediction-time availability contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diliplus.config import load_settings
from diliplus.database import connect_source_database


TARGET_DIAGNOSIS_NAME_MARKERS = (
    "药物性肝",
    "毒性肝",
    "中毒性肝",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _duckdb_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def diagnosis_time_sql(alias: str = "d") -> str:
    """Conservatively parse the diagnosis row creation timestamp."""
    value = f"{alias}.create_time"
    numeric_text = f"CAST(TRY_CAST({value} AS BIGINT) AS VARCHAR)"
    return (
        "COALESCE("
        f"TRY_STRPTIME({numeric_text}, '%Y%m%d%H%M%S'), "
        f"TRY_STRPTIME({numeric_text}, '%Y%m%d')"
        ")"
    )


def target_diagnosis_sql(code: str, name: str) -> str:
    return (
        f"UPPER(TRIM(COALESCE({code}, ''))) LIKE 'K71%' "
        f"OR COALESCE({name}, '') LIKE '%药物性肝%' "
        f"OR COALESCE({name}, '') LIKE '%毒性肝%' "
        f"OR COALESCE({name}, '') LIKE '%中毒性肝%'"
    )


def _write_flat_csv(audit: dict[str, Any], path: Path) -> None:
    flat: dict[str, Any] = {}
    for key, value in audit.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat[f"{key}.{nested_key}"] = nested_value
        elif not isinstance(value, list):
            flat[key] = value
    pd.DataFrame([flat]).to_csv(path, index=False)


def audit_diagnosis_source(settings=None) -> dict[str, Any]:
    """Quantify the legacy patient join and the strict encounter/time boundary."""
    settings = settings or load_settings()
    cohort_path = settings.model_data_dir / "03_dili_dual_stream_tensors.parquet"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Corrected cohort artifact not found: {cohort_path}")

    audit_dir = settings.diagnosis_audit_dir
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_json = audit_dir / "diagnosis_source_audit.json"
    output_csv = audit_dir / "diagnosis_source_audit.csv"

    cohort_sql_path = _duckdb_path(cohort_path)
    parsed_time = diagnosis_time_sql("d")
    source_target = target_diagnosis_sql("d.diagnosis_c", "d.diagnosis_n")
    exact_target = target_diagnosis_sql("icd_code", "diag_name")
    conn = connect_source_database(settings)
    try:
        conn.execute(
            f"""
            CREATE TEMP VIEW p003_cohort AS
            SELECT
                TRIM(CAST(encounter_id AS VARCHAR)) AS encounter_id,
                CAST(prediction_time AS TIMESTAMP) AS prediction_time,
                CAST(index_time AS TIMESTAMP) AS index_time,
                CAST(label_ahi_proxy AS INTEGER) AS label_ahi_proxy
            FROM read_parquet('{cohort_sql_path}')
            """
        )
        conn.execute(
            """
            CREATE TEMP VIEW p003_exact_bridge AS
            SELECT DISTINCT
                c.encounter_id,
                c.prediction_time,
                c.index_time,
                c.label_ahi_proxy,
                TRIM(CAST(b.inpatient_f AS VARCHAR)) AS inpatient_f
            FROM p003_cohort c
            INNER JOIN analysis.v_patient_encounters e
                ON TRIM(CAST(e.encounter_id AS VARCHAR)) = c.encounter_id
            INNER JOIN refined.v_discharge_summary b
                ON TRIM(CAST(e.visit_number AS VARCHAR))
                 = TRIM(CAST(b.business_uu AS VARCHAR))
            WHERE b.inpatient_f IS NOT NULL
              AND TRIM(CAST(b.inpatient_f AS VARCHAR)) <> ''
            """
        )

        cohort_profile = conn.execute(
            """
            SELECT
                COUNT(*) AS cohort_rows,
                COUNT(DISTINCT encounter_id) AS cohort_encounters,
                (SELECT COUNT(DISTINCT encounter_id) FROM p003_exact_bridge)
                    AS exact_bridge_encounters,
                (SELECT COUNT(*) FROM p003_exact_bridge) AS exact_bridge_rows
            FROM p003_cohort
            """
        ).df().iloc[0].to_dict()

        exact_profile = conn.execute(
            f"""
            WITH exact_rows AS (
                SELECT DISTINCT
                    b.encounter_id,
                    b.prediction_time,
                    b.index_time,
                    b.label_ahi_proxy,
                    TRIM(CAST(d.diagnosis_c AS VARCHAR)) AS icd_code,
                    TRIM(CAST(d.diagnosis_n AS VARCHAR)) AS diag_name,
                    {parsed_time} AS diagnosis_time
                FROM p003_exact_bridge b
                INNER JOIN refined.v_in_medical_record_diag d
                    ON TRIM(CAST(d.inpatient_f AS VARCHAR)) = b.inpatient_f
                WHERE d.diagnosis_c IS NOT NULL
                  AND TRIM(CAST(d.diagnosis_c AS VARCHAR)) <> ''
            )
            SELECT
                COUNT(*) AS exact_diagnosis_rows,
                COUNT(DISTINCT encounter_id) AS encounters_with_exact_diagnosis,
                COUNT(*) FILTER (WHERE diagnosis_time IS NULL) AS missing_time_rows,
                COUNT(DISTINCT encounter_id) FILTER (WHERE diagnosis_time IS NULL)
                    AS encounters_with_missing_time,
                COUNT(*) FILTER (WHERE diagnosis_time < prediction_time)
                    AS pre_prediction_rows,
                COUNT(DISTINCT encounter_id) FILTER (
                    WHERE diagnosis_time < prediction_time
                ) AS encounters_with_pre_prediction_diagnosis,
                COUNT(*) FILTER (WHERE diagnosis_time = prediction_time)
                    AS at_prediction_rows,
                COUNT(*) FILTER (WHERE diagnosis_time > prediction_time)
                    AS post_prediction_rows,
                COUNT(DISTINCT encounter_id) FILTER (
                    WHERE diagnosis_time >= prediction_time
                ) AS encounters_with_at_or_post_prediction_diagnosis,
                COUNT(*) FILTER (
                    WHERE diagnosis_time >= prediction_time
                      AND NOT ({exact_target})
                ) AS non_target_at_or_post_prediction_rows,
                COUNT(DISTINCT encounter_id) FILTER (
                    WHERE diagnosis_time >= prediction_time
                      AND NOT ({exact_target})
                ) AS encounters_with_non_target_at_or_post_prediction_diagnosis,
                COUNT(*) FILTER (WHERE diagnosis_time >= index_time)
                    AS at_or_post_index_rows,
                COUNT(DISTINCT encounter_id) FILTER (
                    WHERE diagnosis_time >= index_time
                ) AS encounters_with_at_or_post_index_diagnosis,
                COUNT(*) FILTER (
                    WHERE label_ahi_proxy = 1 AND diagnosis_time >= prediction_time
                ) AS positive_at_or_post_prediction_rows,
                COUNT(DISTINCT encounter_id) FILTER (
                    WHERE label_ahi_proxy = 1 AND diagnosis_time >= prediction_time
                ) AS positive_encounters_with_at_or_post_prediction_diagnosis,
                COUNT(*) FILTER (
                    WHERE label_ahi_proxy = 1
                      AND diagnosis_time >= prediction_time
                      AND NOT ({exact_target})
                ) AS positive_non_target_at_or_post_prediction_rows,
                COUNT(DISTINCT encounter_id) FILTER (
                    WHERE label_ahi_proxy = 1
                      AND diagnosis_time >= prediction_time
                      AND NOT ({exact_target})
                ) AS positive_encounters_with_non_target_at_or_post_prediction_diagnosis,
                COUNT(*) FILTER (
                    WHERE label_ahi_proxy = 1 AND diagnosis_time >= index_time
                ) AS positive_at_or_post_onset_rows,
                COUNT(DISTINCT encounter_id) FILTER (
                    WHERE label_ahi_proxy = 1 AND diagnosis_time >= index_time
                ) AS positive_encounters_with_at_or_post_onset_diagnosis,
                COUNT(*) FILTER (
                    WHERE label_ahi_proxy = 1
                      AND diagnosis_time >= index_time
                      AND NOT ({exact_target})
                ) AS positive_non_target_at_or_post_onset_rows,
                COUNT(DISTINCT encounter_id) FILTER (
                    WHERE label_ahi_proxy = 1
                      AND diagnosis_time >= index_time
                      AND NOT ({exact_target})
                ) AS positive_encounters_with_non_target_at_or_post_onset_diagnosis,
                COUNT(*) FILTER (WHERE {exact_target}) AS explicit_target_rows,
                COUNT(*) FILTER (
                    WHERE diagnosis_time < prediction_time AND ({exact_target})
                ) AS pre_prediction_explicit_target_rows,
                COUNT(*) FILTER (
                    WHERE diagnosis_time < prediction_time AND NOT ({exact_target})
                ) AS eligible_rows,
                COUNT(DISTINCT encounter_id) FILTER (
                    WHERE diagnosis_time < prediction_time AND NOT ({exact_target})
                ) AS encounters_with_eligible_diagnosis
            FROM exact_rows
            """
        ).df().iloc[0].to_dict()

        # Reproduce the old patient-only join and determine how many assignments do
        # not belong to the encounter's inpatient record. This is aggregate-only.
        conn.execute(
            """
            CREATE TEMP VIEW p003_legacy_bridge AS
            SELECT DISTINCT
                c.encounter_id,
                TRIM(CAST(m.patient_id AS VARCHAR)) AS health_reco
            FROM p003_cohort c
            INNER JOIN analysis.feature_medications m
                ON TRIM(CAST(m.encounter_id AS VARCHAR)) = c.encounter_id
            WHERE m.patient_id IS NOT NULL
              AND TRIM(CAST(m.patient_id AS VARCHAR)) <> ''
            """
        )
        legacy_profile = conn.execute(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE NOT ({source_target})) AS legacy_retained_rows,
                COUNT(DISTINCT l.encounter_id) FILTER (WHERE NOT ({source_target}))
                    AS legacy_retained_encounters,
                COUNT(*) FILTER (
                    WHERE NOT ({source_target}) AND eb.inpatient_f IS NULL
                ) AS legacy_cross_encounter_rows,
                COUNT(DISTINCT l.encounter_id) FILTER (
                    WHERE NOT ({source_target}) AND eb.inpatient_f IS NULL
                ) AS encounters_with_cross_encounter_assignment,
                COUNT(*) FILTER (WHERE {source_target}) AS legacy_explicit_target_rows
            FROM p003_legacy_bridge l
            INNER JOIN in_medical_record_diag d
                ON TRIM(CAST(d.health_reco AS VARCHAR)) = l.health_reco
            LEFT JOIN p003_exact_bridge eb
                ON eb.encounter_id = l.encounter_id
               AND eb.inpatient_f = TRIM(CAST(d.inpatient_f AS VARCHAR))
            WHERE d.diagnosis_c IS NOT NULL
              AND TRIM(CAST(d.diagnosis_c AS VARCHAR)) <> ''
            """
        ).df().iloc[0].to_dict()
    finally:
        conn.close()

    def integers(values: dict[str, Any]) -> dict[str, int]:
        return {key: int(value or 0) for key, value in values.items()}

    cohort_profile = integers(cohort_profile)
    exact_profile = integers(exact_profile)
    legacy_profile = integers(legacy_profile)
    total_exact = exact_profile["exact_diagnosis_rows"]
    legacy_retained = legacy_profile["legacy_retained_rows"]
    audit = {
        "audit_scope": "aggregate-only pre-fix diagnosis source audit",
        "diagnosis_time_definition": "refined.v_in_medical_record_diag.create_time",
        "time_rule": "diagnosis_time < prediction_time (strict)",
        "encounter_rule": (
            "encounter_id -> v_patient_encounters.visit_number -> "
            "v_discharge_summary.business_uu -> inpatient_f"
        ),
        "cohort": cohort_profile,
        "exact_encounter_source": exact_profile,
        "legacy_patient_join": legacy_profile,
        "derived_percentages": {
            "exact_time_parseable_pct": round(
                100.0
                * (total_exact - exact_profile["missing_time_rows"])
                / total_exact,
                6,
            )
            if total_exact
            else 0.0,
            "exact_rows_at_or_after_prediction_pct": round(
                100.0
                * (
                    exact_profile["at_prediction_rows"]
                    + exact_profile["post_prediction_rows"]
                )
                / total_exact,
                6,
            )
            if total_exact
            else 0.0,
            "legacy_cross_encounter_rows_pct": round(
                100.0
                * legacy_profile["legacy_cross_encounter_rows"]
                / legacy_retained,
                6,
            )
            if legacy_retained
            else 0.0,
        },
        "derived_counts": {
            "exact_non_target_rows_all_times": (
                exact_profile["exact_diagnosis_rows"]
                - exact_profile["explicit_target_rows"]
            ),
            "exact_non_target_rows_excluded_by_time_rule": (
                exact_profile["exact_diagnosis_rows"]
                - exact_profile["explicit_target_rows"]
                - exact_profile["eligible_rows"]
            ),
            "legacy_repeated_or_duplicate_assignments_beyond_distinct_exact_rows": max(
                0,
                legacy_profile["legacy_retained_rows"]
                - (
                    exact_profile["exact_diagnosis_rows"]
                    - exact_profile["explicit_target_rows"]
                ),
            ),
        },
        "cohort_artifact": str(cohort_path.resolve()),
        "cohort_sha256": _sha256(cohort_path),
    }
    output_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_flat_csv(audit, output_csv)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"[DILI-PLUS] Diagnosis source audit saved: {output_json}")
    return audit


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, np.ndarray)):
        return list(value)
    return []


def _is_target_diagnosis(code: Any, name: Any) -> bool:
    code_text = str(code).strip().upper()
    name_text = str(name)
    return code_text.startswith("K71") or any(
        marker in name_text for marker in TARGET_DIAGNOSIS_NAME_MARKERS
    )


def build_diagnosis_tensor_audit(
    cohort_path: str | Path,
    diagnosis_path: str | Path,
) -> dict[str, Any]:
    """Validate a built diagnosis tensor without consulting the source database."""
    cohort_path = Path(cohort_path)
    diagnosis_path = Path(diagnosis_path)
    cohort = pd.read_parquet(
        cohort_path,
        columns=["encounter_id", "prediction_time", "label_ahi_proxy"],
    )
    diagnosis = pd.read_parquet(diagnosis_path)

    required = {
        "encounter_id",
        "icd_codes",
        "diag_names",
        "diag_event_times",
    }
    missing_columns = sorted(required - set(diagnosis.columns))
    if missing_columns:
        raise ValueError(f"Diagnosis artifact is missing columns: {missing_columns}")

    cohort["encounter_id"] = cohort["encounter_id"].astype(str)
    diagnosis["encounter_id"] = diagnosis["encounter_id"].astype(str)
    prediction_by_encounter = {
        row.encounter_id: pd.Timestamp(row.prediction_time)
        for row in cohort.itertuples(index=False)
    }
    label_by_encounter = {
        row.encounter_id: int(row.label_ahi_proxy)
        for row in cohort.itertuples(index=False)
    }
    cohort_ids = set(prediction_by_encounter)
    diagnosis_ids = set(diagnosis["encounter_id"])

    violations = {
        "duplicate_cohort_encounters": int(cohort["encounter_id"].duplicated().sum()),
        "duplicate_diagnosis_encounters": int(
            diagnosis["encounter_id"].duplicated().sum()
        ),
        "cohort_encounters_missing_diagnosis_row": len(cohort_ids - diagnosis_ids),
        "diagnosis_rows_outside_cohort": len(diagnosis_ids - cohort_ids),
        "diagnosis_sequence_length_mismatch": 0,
        "unparseable_diagnosis_event_time": 0,
        "diagnosis_at_or_after_prediction": 0,
        "explicit_target_diagnosis_in_input": 0,
    }
    total_events = 0
    encounters_with_diagnosis = 0
    positive_events = 0
    positive_encounters_with_diagnosis = 0
    minimum_margin_hours: float | None = None

    for row in diagnosis.itertuples(index=False):
        codes = _as_list(row.icd_codes)
        names = _as_list(row.diag_names)
        times = _as_list(row.diag_event_times)
        if codes:
            encounters_with_diagnosis += 1
            if label_by_encounter.get(row.encounter_id) == 1:
                positive_encounters_with_diagnosis += 1
        if not (len(codes) == len(names) == len(times)):
            violations["diagnosis_sequence_length_mismatch"] += 1
        total_events += len(times)
        if label_by_encounter.get(row.encounter_id) == 1:
            positive_events += len(times)
        prediction_time = prediction_by_encounter.get(row.encounter_id)
        if prediction_time is None:
            continue
        for code, name, raw_time in zip(codes, names, times):
            event_time = pd.to_datetime(raw_time, errors="coerce")
            if pd.isna(event_time):
                violations["unparseable_diagnosis_event_time"] += 1
                continue
            if event_time >= prediction_time:
                violations["diagnosis_at_or_after_prediction"] += 1
            margin = (prediction_time - event_time).total_seconds() / 3600.0
            minimum_margin_hours = (
                margin if minimum_margin_hours is None else min(minimum_margin_hours, margin)
            )
            if _is_target_diagnosis(code, name):
                violations["explicit_target_diagnosis_in_input"] += 1

    failed_contracts = [name for name, count in violations.items() if count]
    return {
        "audit_status": "PASS" if not failed_contracts else "FAIL",
        "contract": (
            "same encounter; parsed create_time strictly before prediction_time; "
            "K71/drug-induced-toxic liver labels excluded"
        ),
        "cohort_encounters": int(len(cohort)),
        "diagnosis_rows": int(len(diagnosis)),
        "encounters_with_diagnosis": int(encounters_with_diagnosis),
        "encounters_without_diagnosis": int(len(diagnosis) - encounters_with_diagnosis),
        "total_diagnosis_events": int(total_events),
        "ahi_proxy_positive_encounters": int(sum(label_by_encounter.values())),
        "positive_encounters_with_diagnosis": int(
            positive_encounters_with_diagnosis
        ),
        "positive_diagnosis_events": int(positive_events),
        "minimum_diagnosis_time_margin_hours": minimum_margin_hours,
        "violations": violations,
        "failed_contracts": failed_contracts,
        "cohort_artifact": str(cohort_path.resolve()),
        "diagnosis_artifact": str(diagnosis_path.resolve()),
        "cohort_sha256": _sha256(cohort_path),
        "diagnosis_sha256": _sha256(diagnosis_path),
    }


def audit_diagnosis_time_contract(settings=None) -> dict[str, Any]:
    settings = settings or load_settings()
    cohort_path = settings.model_data_dir / "03_dili_dual_stream_tensors.parquet"
    diagnosis_path = settings.model_data_dir / "03b_diag_tensors.parquet"
    audit_dir = settings.diagnosis_audit_dir
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_json = audit_dir / "diagnosis_tensor_audit.json"
    output_csv = audit_dir / "diagnosis_tensor_summary.csv"

    audit = build_diagnosis_tensor_audit(cohort_path, diagnosis_path)
    output_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_flat_csv(audit, output_csv)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"[DILI-PLUS] Diagnosis tensor audit saved: {output_json}")
    if audit["audit_status"] != "PASS":
        raise RuntimeError(
            "Diagnosis leakage contract failed: "
            + ", ".join(audit["failed_contracts"])
        )
    return audit


if __name__ == "__main__":
    audit_diagnosis_time_contract()
