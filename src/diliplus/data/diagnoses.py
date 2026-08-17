"""Build diagnosis tensors under a strict encounter and prediction-time contract.

Only diagnoses linked to the same inpatient encounter through the curated
visit-number bridge are eligible. A diagnosis must have a parseable source-row
``create_time`` strictly earlier than ``prediction_time``. Explicit K71 and
drug-induced/toxic liver diagnosis labels are excluded as target information.
"""

from __future__ import annotations

import time
from pathlib import Path

from diliplus.config import load_settings
from diliplus.database import connect_source_database
from diliplus.data.diagnosis_audit import (
    diagnosis_time_sql,
    target_diagnosis_sql,
)


def _duckdb_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def build_diag_tensors(settings=None) -> Path:
    settings = settings or load_settings()
    data_dir = settings.model_data_dir
    cohort_path = data_dir / "03_dili_dual_stream_tensors.parquet"
    output_path = data_dir / "03b_diag_tensors.parquet"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Corrected cohort artifact not found: {cohort_path}")
    data_dir.mkdir(parents=True, exist_ok=True)

    cohort_sql_path = _duckdb_path(cohort_path)
    diagnosis_time = diagnosis_time_sql("d")
    target_diagnosis = target_diagnosis_sql("d.diagnosis_c", "d.diagnosis_n")
    query = f"""
    WITH cohort AS (
        SELECT
            TRIM(CAST(encounter_id AS VARCHAR)) AS encounter_id,
            CAST(prediction_time AS TIMESTAMP) AS prediction_time
        FROM read_parquet('{cohort_sql_path}')
    ),
    encounter_bridge AS (
        SELECT DISTINCT
            c.encounter_id,
            c.prediction_time,
            TRIM(CAST(b.inpatient_f AS VARCHAR)) AS inpatient_f
        FROM cohort c
        INNER JOIN analysis.v_patient_encounters e
            ON TRIM(CAST(e.encounter_id AS VARCHAR)) = c.encounter_id
        INNER JOIN refined.v_discharge_summary b
            ON TRIM(CAST(e.visit_number AS VARCHAR))
             = TRIM(CAST(b.business_uu AS VARCHAR))
        WHERE b.inpatient_f IS NOT NULL
          AND TRIM(CAST(b.inpatient_f AS VARCHAR)) <> ''
    ),
    eligible_diagnoses AS (
        SELECT DISTINCT
            b.encounter_id,
            TRIM(CAST(d.diagnosis_c AS VARCHAR)) AS icd_code,
            TRIM(CAST(d.diagnosis_n AS VARCHAR)) AS diag_name,
            {diagnosis_time} AS diagnosis_time
        FROM encounter_bridge b
        INNER JOIN refined.v_in_medical_record_diag d
            ON TRIM(CAST(d.inpatient_f AS VARCHAR)) = b.inpatient_f
        WHERE d.diagnosis_c IS NOT NULL
          AND TRIM(CAST(d.diagnosis_c AS VARCHAR)) <> ''
          AND {diagnosis_time} IS NOT NULL
          AND {diagnosis_time} < b.prediction_time
          AND NOT ({target_diagnosis})
    ),
    grouped AS (
        SELECT
            encounter_id,
            LIST(icd_code ORDER BY diagnosis_time, icd_code, diag_name) AS icd_codes,
            LIST(diag_name ORDER BY diagnosis_time, icd_code, diag_name) AS diag_names,
            LIST(diagnosis_time ORDER BY diagnosis_time, icd_code, diag_name)
                AS diag_event_times
        FROM eligible_diagnoses
        GROUP BY encounter_id
    )
    SELECT
        c.encounter_id,
        COALESCE(g.icd_codes, []::VARCHAR[]) AS icd_codes,
        COALESCE(g.diag_names, []::VARCHAR[]) AS diag_names,
        COALESCE(g.diag_event_times, []::TIMESTAMP[]) AS diag_event_times,
        LENGTH(COALESCE(g.icd_codes, []::VARCHAR[])) AS diagnosis_count,
        'refined.v_in_medical_record_diag.create_time' AS diagnosis_time_source,
        'same_encounter_and_diagnosis_time_strictly_before_prediction_time'
            AS diagnosis_contract
    FROM cohort c
    LEFT JOIN grouped g USING (encounter_id)
    ORDER BY c.encounter_id
    """

    print("[DILI-PLUS] Building strict encounter/time-bounded diagnosis tensors...")
    started = time.perf_counter()
    conn = connect_source_database(settings)
    try:
        diagnosis = conn.execute(query).df()
    finally:
        conn.close()

    diagnosis.to_parquet(output_path, index=False)
    total_events = int(diagnosis["diagnosis_count"].sum())
    with_diagnosis = int(diagnosis["diagnosis_count"].gt(0).sum())
    print(
        "[DILI-PLUS] Strict diagnosis tensors built: "
        f"{len(diagnosis)} encounters, {with_diagnosis} with eligible diagnoses, "
        f"{total_events} diagnosis events."
    )
    print(f"[DILI-PLUS] Tensor exported to: {output_path}")
    print(f"[DILI-PLUS] Elapsed seconds: {time.perf_counter() - started:.3f}")
    return output_path


if __name__ == "__main__":
    build_diag_tensors()
