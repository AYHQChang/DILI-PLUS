"""
DILI-PLUS | Biochemical AHI proxy labels and prediction-time construction.

The positive index time is the first post-baseline ALT or AST measurement at or
above 120 U/L. The model prediction time is defined explicitly as
``index_time - prediction_gap``. Negative encounters receive a deterministic,
seeded pseudo-index within an eligible medication window. Corrected artifacts
are isolated from legacy onset-time data under ``settings.model_data_dir``.

This stage creates labels and prediction times only. Dynamic event inclusion is
enforced by ``sequences.py`` using the strict rule ``event_time < prediction_time``.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from diliplus.config import load_settings
from diliplus.database import connect_source_database


UINT64_MAX = 18446744073709551615


def _interval_sql(hours: float) -> str:
    return f"INTERVAL '{float(hours)} hours'"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _build_raw_label_table(
    conn, input_parquet: Path, baseline_threshold_u_l: float
) -> None:
    """Rebuild the proxy label without row-order-dependent baseline selection.

    All ALT/AST rows at the earliest target-laboratory timestamp form the
    baseline. The baseline passes only when every row is marked normal/low and
    every available numeric value is below the prespecified threshold. A
    positive onset is the first strictly later ALT/AST value at or above that
    threshold.
    """
    input_sql = input_parquet.as_posix().replace("'", "''")
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE temp_raw_ahi_labels AS
        WITH TargetLabs AS (
            SELECT
                TRIM(CAST(encounter_id AS VARCHAR)) AS encounter_id,
                first_med_time,
                last_med_time,
                COALESCE(TRIM(CAST(abnormal_status AS VARCHAR)), '')
                    AS abnormal_status,
                TRIM(CAST(lab_item AS VARCHAR)) AS lab_item,
                TRY_CAST(lab_value AS DOUBLE) AS lab_value,
                TRY_CAST(lab_time AS TIMESTAMP) AS lab_time
            FROM read_parquet('{input_sql}')
            WHERE lab_time IS NOT NULL
              AND (
                    lab_item LIKE '%谷丙转氨酶%'
                 OR lab_item LIKE '%谷草转氨酶%'
              )
        ),
        FirstTimes AS (
            SELECT
                encounter_id,
                MIN(lab_time) AS baseline_time
            FROM TargetLabs
            GROUP BY encounter_id
        ),
        BaselineSummary AS (
            SELECT
                t.encounter_id,
                f.baseline_time,
                COUNT(*) AS baseline_target_rows,
                COUNT(DISTINCT t.lab_item) AS baseline_distinct_target_items,
                COUNT(t.lab_value) AS baseline_numeric_rows,
                MAX(t.lab_value) AS baseline_max_value,
                BOOL_AND(
                    t.abnormal_status IN ('N', '正常', 'L', '低', '↓')
                    AND (
                        t.lab_value IS NULL
                        OR t.lab_value < {float(baseline_threshold_u_l)}
                    )
                ) AS baseline_rule_pass
            FROM TargetLabs t
            INNER JOIN FirstTimes f
              ON t.encounter_id = f.encounter_id
             AND t.lab_time = f.baseline_time
            GROUP BY t.encounter_id, f.baseline_time
        ),
        EncounterSummary AS (
            SELECT
                t.encounter_id,
                MAX(first_med_time) AS first_med_time,
                MAX(last_med_time) AS last_med_time,
                MIN(
                    CASE
                        WHEN t.lab_time > b.baseline_time
                         AND t.lab_value >= {float(baseline_threshold_u_l)}
                        THEN t.lab_time
                    END
                ) AS t_onset,
                COUNT(*) AS target_lab_frequency
            FROM TargetLabs t
            INNER JOIN BaselineSummary b USING (encounter_id)
            GROUP BY t.encounter_id
        )
        SELECT
            e.encounter_id,
            e.first_med_time,
            e.last_med_time,
            e.target_lab_frequency,
            b.baseline_time,
            b.baseline_target_rows,
            b.baseline_distinct_target_items,
            b.baseline_numeric_rows,
            b.baseline_max_value,
            b.baseline_rule_pass,
            CASE
                WHEN b.baseline_rule_pass AND e.t_onset IS NOT NULL THEN 1
                WHEN b.baseline_rule_pass AND e.t_onset IS NULL THEN 0
                ELSE NULL
            END AS label_ahi_proxy,
            e.t_onset
        FROM EncounterSummary e
        INNER JOIN BaselineSummary b USING (encounter_id);
        """
    )


def _load_frozen_legacy_label_table(conn, legacy_label_path: Path) -> None:
    """Preserve the already reported cohort while replacing only its time design."""
    legacy_sql = legacy_label_path.as_posix().replace("'", "''")
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE temp_raw_ahi_labels AS
        SELECT
            TRIM(CAST(encounter_id AS VARCHAR)) AS encounter_id,
            first_med_time,
            last_med_time,
            lab_frequency AS target_lab_frequency,
            NULL::TIMESTAMP AS baseline_time,
            NULL::BIGINT AS baseline_target_rows,
            NULL::BIGINT AS baseline_distinct_target_items,
            NULL::BIGINT AS baseline_numeric_rows,
            NULL::DOUBLE AS baseline_max_value,
            NULL::BOOLEAN AS baseline_rule_pass,
            label_dili AS label_ahi_proxy,
            t_onset
        FROM read_parquet('{legacy_sql}')
        WHERE label_dili IS NOT NULL;
        """
    )


def _build_encounter_patient_map(conn) -> None:
    conn.execute(
        """
        CREATE OR REPLACE TEMP TABLE temp_encounter_patient AS
        SELECT
            TRIM(CAST(encounter_id AS VARCHAR)) AS encounter_id,
            MIN(TRIM(CAST(patient_id AS VARCHAR))) AS patient_id,
            COUNT(DISTINCT TRIM(CAST(patient_id AS VARCHAR))) AS patient_id_count
        FROM analysis.v_patient_encounters
        WHERE encounter_id IS NOT NULL
          AND patient_id IS NOT NULL
          AND TRIM(CAST(patient_id AS VARCHAR)) <> ''
        GROUP BY 1;
        """
    )
    conflicts = int(
        conn.execute(
            "SELECT COUNT(*) FROM temp_encounter_patient WHERE patient_id_count <> 1"
        ).fetchone()[0]
    )
    if conflicts:
        raise RuntimeError(
            f"Encounter-patient bridge contains {conflicts} conflicting encounters"
        )


def _prediction_gap_feasibility(conn, horizons: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for hours in horizons:
        interval = _interval_sql(hours)
        counts = conn.execute(
            f"""
            SELECT
                SUM(
                    CASE WHEN label_ahi_proxy = 1
                              AND t_onset > first_med_time + {interval}
                         THEN 1 ELSE 0 END
                ) AS positive_eligible,
                SUM(
                    CASE WHEN label_ahi_proxy = 0
                              AND last_med_time > first_med_time + {interval}
                         THEN 1 ELSE 0 END
                ) AS negative_eligible,
                SUM(CASE WHEN label_ahi_proxy = 1 THEN 1 ELSE 0 END) AS positive_total,
                SUM(CASE WHEN label_ahi_proxy = 0 THEN 1 ELSE 0 END) AS negative_total
            FROM temp_raw_ahi_labels
            WHERE label_ahi_proxy IS NOT NULL;
            """
        ).fetchone()
        positive_eligible, negative_eligible, positive_total, negative_total = map(int, counts)
        rows.append(
            {
                "prediction_gap_hours": float(hours),
                "positive_eligible": positive_eligible,
                "negative_eligible": negative_eligible,
                "positive_total": positive_total,
                "negative_total": negative_total,
                "positive_retention_pct": round(100.0 * positive_eligible / positive_total, 4),
                "negative_retention_pct": round(100.0 * negative_eligible / negative_total, 4),
            }
        )
    return pd.DataFrame(rows)


def _build_prediction_label_table(
    conn,
    gap_hours: float,
    pseudo_index_seed: int,
    label_source: str,
) -> None:
    interval = _interval_sql(gap_hours)
    stable_fraction = (
        "CAST(HASH(CAST(encounter_id AS VARCHAR) || "
        f"':{int(pseudo_index_seed)}') AS DOUBLE) / {UINT64_MAX}.0"
    )
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE temp_ahi_prediction_labels AS
        WITH Eligible AS (
            SELECT *
            FROM temp_raw_ahi_labels
            WHERE label_ahi_proxy IS NOT NULL
              AND (
                    (label_ahi_proxy = 1 AND t_onset > first_med_time + {interval})
                 OR (label_ahi_proxy = 0 AND last_med_time > first_med_time + {interval})
              )
        ),
        Indexed AS (
            SELECT
                *,
                CASE
                    WHEN label_ahi_proxy = 1 THEN t_onset
                    WHEN label_ahi_proxy = 0 THEN
                        first_med_time + {interval}
                        + (last_med_time - (first_med_time + {interval}))
                          * ({stable_fraction})
                END AS index_time
            FROM Eligible
        ),
        Timed AS (
            SELECT
                *,
                index_time - {interval} AS prediction_time
            FROM Indexed
        )
        SELECT
            i.encounter_id,
            p.patient_id,
            label_ahi_proxy,
            label_ahi_proxy AS label_dili,
            t_onset,
            index_time,
            prediction_time,
            prediction_time AS censor_time,
            {float(gap_hours)}::DOUBLE AS prediction_gap_hours,
            first_med_time,
            last_med_time,
            target_lab_frequency AS lab_frequency,
            baseline_time,
            baseline_target_rows,
            baseline_distinct_target_items,
            baseline_numeric_rows,
            baseline_max_value,
            baseline_rule_pass,
            '{label_source}' AS label_source,
            CASE
                WHEN label_ahi_proxy = 1 THEN 'observed_ahi_threshold'
                ELSE 'deterministic_bounded_hash'
            END AS index_time_source,
            {int(pseudo_index_seed)}::BIGINT AS pseudo_index_seed
        FROM Timed i
        LEFT JOIN temp_encounter_patient p USING (encounter_id)
        WHERE prediction_time > first_med_time
        ORDER BY i.encounter_id;
        """
    )


def extract_dili_labels_and_censor(settings=None):
    """Build leakage-aware AHI-proxy labels while retaining the legacy entry point."""
    settings = settings or load_settings()
    source_dir = settings.paths.data_cache
    output_dir = settings.model_data_dir
    audit_dir = settings.prediction_audit_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    input_parquet = source_dir / "01_aligned_dili_labs.parquet"
    frozen_label_parquet = source_dir / "02_dili_labels_censored.parquet"
    output_parquet = output_dir / "02_dili_labels_censored.parquet"
    feasibility_csv = audit_dir / "prediction_gap_feasibility.csv"

    if not input_parquet.exists():
        raise FileNotFoundError(f"Aligned laboratory input not found: {input_parquet}")

    print(f"[DILI-PLUS] Building AHI-proxy prediction labels from: {input_parquet}")
    print(
        f"[DILI-PLUS] Primary prediction gap: {settings.prediction.gap_hours:g} hours; "
        f"output isolation: {output_dir}"
    )
    start_time = time.time()
    conn = connect_source_database(settings)
    try:
        if settings.prediction.label_source == "legacy_frozen":
            if not frozen_label_parquet.exists():
                raise FileNotFoundError(
                    f"Frozen legacy labels requested but unavailable: {frozen_label_parquet}"
                )
            print(
                "[DILI-PLUS] LEGACY PILOT ONLY: loading frozen labels from: "
                f"{frozen_label_parquet}"
            )
            _load_frozen_legacy_label_table(conn, frozen_label_parquet)
        else:
            print(
                "[DILI-PLUS] FORMAL: deterministically rebuilding the proxy cohort "
                "from all earliest-timestamp ALT/AST rows."
            )
            _build_raw_label_table(
                conn,
                input_parquet,
                settings.prediction.baseline_threshold_u_l,
            )
        _build_encounter_patient_map(conn)
        raw_summary = conn.execute(
            """
            SELECT
                COUNT(*)::BIGINT AS target_lab_encounters,
                SUM(label_ahi_proxy = 1)::BIGINT AS raw_positive,
                SUM(label_ahi_proxy = 0)::BIGINT AS raw_negative,
                SUM(label_ahi_proxy IS NULL)::BIGINT AS baseline_excluded,
                SUM(COALESCE(baseline_target_rows, 0) > 1)::BIGINT
                    AS tied_baseline_encounters
            FROM temp_raw_ahi_labels
            """
        ).df()
        feasibility = _prediction_gap_feasibility(
            conn, settings.prediction.audit_horizons_hours
        )
        feasibility.to_csv(feasibility_csv, index=False)

        _build_prediction_label_table(
            conn,
            settings.prediction.gap_hours,
            settings.prediction.pseudo_index_seed,
            settings.prediction.label_source,
        )
        missing_patients = int(
            conn.execute(
                "SELECT COUNT(*) FROM temp_ahi_prediction_labels WHERE patient_id IS NULL"
            ).fetchone()[0]
        )
        if missing_patients:
            raise RuntimeError(
                f"Formal cohort has {missing_patients} encounters without patient_id"
            )
        output_sql = output_parquet.as_posix().replace("'", "''")
        conn.execute(
            f"COPY temp_ahi_prediction_labels TO '{output_sql}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )

        summary = conn.execute(
            """
            SELECT
                COUNT(*) AS valid_encounters,
                SUM(label_ahi_proxy = 1)::BIGINT AS ahi_proxy_positive,
                SUM(label_ahi_proxy = 0)::BIGINT AS ahi_proxy_negative,
                AVG(EPOCH(prediction_time) - EPOCH(first_med_time)) / 3600.0
                    AS avg_observation_hours,
                MIN(EPOCH(index_time) - EPOCH(prediction_time)) / 3600.0
                    AS min_realised_gap_hours,
                MAX(EPOCH(index_time) - EPOCH(prediction_time)) / 3600.0
                    AS max_realised_gap_hours
            FROM temp_ahi_prediction_labels;
            """
        ).df()
    finally:
        conn.close()

    print("[DILI-PLUS] Prediction-gap feasibility:")
    print(feasibility.to_string(index=False))
    print("[DILI-PLUS] Corrected label summary:")
    print(summary.T.to_string(header=False))
    print(f"[DILI-PLUS] Labels saved: {output_parquet}")
    lineage = {
        "schema_version": 1,
        "label_source": settings.prediction.label_source,
        "formal_eligible": settings.prediction.label_source == "deterministic_rebuild",
        "target_name": "label_ahi_proxy",
        "outcome_interpretation": "biochemistry-defined AHI proxy; not adjudicated DILI",
        "baseline_rule": (
            "all ALT/AST rows at earliest target-lab timestamp have normal/low "
            "status and every available numeric value is below threshold"
        ),
        "onset_rule": "first strictly later ALT/AST numeric value at or above threshold",
        "baseline_threshold_u_l": settings.prediction.baseline_threshold_u_l,
        "prediction_gap_hours": settings.prediction.gap_hours,
        "patient_group_source": "analysis.v_patient_encounters.patient_id",
        "input_file": str(input_parquet.relative_to(settings.paths.root)).replace("\\", "/"),
        "input_sha256": _sha256(input_parquet),
        "output_file": str(output_parquet.relative_to(settings.paths.root)).replace("\\", "/"),
        "output_sha256": _sha256(output_parquet),
        "raw_summary": {
            key: int(value)
            for key, value in raw_summary.iloc[0].to_dict().items()
        },
        "formal_summary": {
            key: (float(value) if isinstance(value, float) else int(value))
            for key, value in summary.iloc[0].to_dict().items()
        },
    }
    lineage_path = output_dir / "data_lineage.json"
    lineage_path.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DILI-PLUS] Data lineage saved: {lineage_path}")
    print(f"[DILI-PLUS] Feasibility audit saved: {feasibility_csv}")
    print(f"[DILI-PLUS] Label stage completed in {time.time() - start_time:.2f}s")
    return output_parquet


if __name__ == "__main__":
    extract_dili_labels_and_censor()
