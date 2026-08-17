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

import time
from pathlib import Path

import pandas as pd

from diliplus.config import load_settings
from diliplus.database import connect_source_database


UINT64_MAX = 18446744073709551615


def _interval_sql(hours: float) -> str:
    return f"INTERVAL '{float(hours)} hours'"


def _build_raw_label_table(conn, input_parquet: Path) -> None:
    input_sql = input_parquet.as_posix().replace("'", "''")
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE temp_raw_ahi_labels AS
        WITH RankedLabs AS (
            SELECT
                encounter_id,
                first_med_time,
                last_med_time,
                abnormal_status,
                lab_item,
                TRY_CAST(lab_value AS DOUBLE) AS lab_value,
                lab_time,
                ROW_NUMBER() OVER (
                    PARTITION BY encounter_id
                    ORDER BY lab_time ASC
                ) AS seq_num
            FROM read_parquet('{input_sql}')
            WHERE lab_item LIKE '%谷丙转氨酶%'
               OR lab_item LIKE '%谷草转氨酶%'
        ),
        EncounterSummary AS (
            SELECT
                encounter_id,
                MAX(first_med_time) AS first_med_time,
                MAX(last_med_time) AS last_med_time,
                MAX(CASE WHEN seq_num = 1 THEN abnormal_status END) AS baseline_status,
                MAX(
                    CASE WHEN seq_num > 1 AND lab_value >= 120.0 THEN 1 ELSE 0 END
                ) AS has_subsequent_high,
                MIN(
                    CASE WHEN seq_num > 1 AND lab_value >= 120.0 THEN lab_time END
                ) AS t_onset,
                COUNT(*) AS target_lab_frequency
            FROM RankedLabs
            GROUP BY encounter_id
        )
        SELECT
            encounter_id,
            first_med_time,
            last_med_time,
            target_lab_frequency,
            CASE
                WHEN baseline_status IN ('N', '正常', 'L', '低', '↓')
                     AND has_subsequent_high = 1 THEN 1
                WHEN baseline_status IN ('N', '正常', 'L', '低', '↓')
                     AND has_subsequent_high = 0 THEN 0
                ELSE NULL
            END AS label_ahi_proxy,
            t_onset
        FROM EncounterSummary;
        """
    )


def _load_frozen_legacy_label_table(conn, legacy_label_path: Path) -> None:
    """Preserve the already reported cohort while replacing only its time design."""
    legacy_sql = legacy_label_path.as_posix().replace("'", "''")
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE temp_raw_ahi_labels AS
        SELECT
            encounter_id,
            first_med_time,
            last_med_time,
            lab_frequency AS target_lab_frequency,
            label_dili AS label_ahi_proxy,
            t_onset
        FROM read_parquet('{legacy_sql}')
        WHERE label_dili IS NOT NULL;
        """
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


def _build_prediction_label_table(conn, gap_hours: float, pseudo_index_seed: int) -> None:
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
            encounter_id,
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
            CASE
                WHEN label_ahi_proxy = 1 THEN 'observed_ahi_threshold'
                ELSE 'deterministic_bounded_hash'
            END AS index_time_source,
            {int(pseudo_index_seed)}::BIGINT AS pseudo_index_seed
        FROM Timed
        WHERE prediction_time > first_med_time
        ORDER BY encounter_id;
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
        if frozen_label_parquet.exists():
            print(
                "[DILI-PLUS] Preserving the frozen reported cohort from: "
                f"{frozen_label_parquet}"
            )
            _load_frozen_legacy_label_table(conn, frozen_label_parquet)
        else:
            print(
                "[DILI-PLUS] Frozen label artifact unavailable; rebuilding the proxy cohort "
                "from aligned laboratory data."
            )
            _build_raw_label_table(conn, input_parquet)
        feasibility = _prediction_gap_feasibility(
            conn, settings.prediction.audit_horizons_hours
        )
        feasibility.to_csv(feasibility_csv, index=False)

        _build_prediction_label_table(
            conn,
            settings.prediction.gap_hours,
            settings.prediction.pseudo_index_seed,
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
    print(f"[DILI-PLUS] Feasibility audit saved: {feasibility_csv}")
    print(f"[DILI-PLUS] Label stage completed in {time.time() - start_time:.2f}s")
    return output_parquet


if __name__ == "__main__":
    extract_dili_labels_and_censor()
