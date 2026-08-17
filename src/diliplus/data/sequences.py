"""
DILI-PLUS | Leakage-aware medication and laboratory sequence construction.

Every dynamic event must satisfy ``first_med_time <= event_time < prediction_time``.
The strict upper bound excludes the target-defining ALT/AST measurement and all
other events recorded at or after the prediction time. Event timestamps are kept
in the Parquet artifact so the contract can be audited independently of tokens.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from diliplus.config import load_settings
from diliplus.database import connect_source_database


def build_dili_tensors(settings=None):
    """Build corrected dynamic sequences while retaining the legacy entry point."""
    settings = settings or load_settings()
    source_dir = settings.paths.data_cache
    output_dir = settings.model_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    label_path = output_dir / "02_dili_labels_censored.parquet"
    lab_path = source_dir / "01_aligned_dili_labs.parquet"
    output_parquet = output_dir / "03_dili_dual_stream_tensors.parquet"

    if not label_path.exists():
        raise FileNotFoundError(f"Prediction-time labels not found: {label_path}")
    if not lab_path.exists():
        raise FileNotFoundError(f"Aligned laboratory input not found: {lab_path}")

    print("[DILI-PLUS] Building dynamic sequences with event_time < prediction_time")
    start_time = time.time()
    conn = connect_source_database(settings)
    try:
        cohort_columns = [
            "encounter_id",
            "label_ahi_proxy",
            "label_dili",
            "t_onset",
            "index_time",
            "prediction_time",
            "censor_time",
            "prediction_gap_hours",
            "first_med_time",
            "last_med_time",
            "index_time_source",
            "pseudo_index_seed",
        ]
        df_cohort = pd.read_parquet(label_path)[cohort_columns]
        conn.execute(
            "CREATE OR REPLACE TEMP TABLE mem_prediction_cohort AS SELECT * FROM df_cohort"
        )

        med_query = """
            WITH ValidMeds AS (
                SELECT
                    m.encounter_id,
                    m.order_name AS med_item,
                    m.start_time AS med_time,
                    c.prediction_time,
                    c.first_med_time
                FROM analysis.feature_medications m
                INNER JOIN mem_prediction_cohort c ON m.encounter_id = c.encounter_id
                WHERE m.start_time IS NOT NULL
                  AND m.start_time >= c.first_med_time
                  AND m.start_time < c.prediction_time
                  AND m.order_name IS NOT NULL
            ),
            OrderedMeds AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY encounter_id
                        ORDER BY med_time, med_item
                    ) AS event_seq
                FROM ValidMeds
            ),
            SortedMeds AS (
                SELECT
                    encounter_id,
                    med_item,
                    med_time,
                    event_seq,
                    COALESCE(
                        EPOCH(med_time) - EPOCH(
                            LAG(med_time) OVER (
                                PARTITION BY encounter_id ORDER BY event_seq
                            )
                        ),
                        EPOCH(med_time) - EPOCH(first_med_time)
                    ) / 3600.0 AS med_dt_hours
                FROM OrderedMeds
            )
            SELECT
                encounter_id,
                LIST(med_item ORDER BY event_seq) AS med_tokens,
                LIST(med_dt_hours ORDER BY event_seq) AS med_dt_hours,
                LIST(med_time ORDER BY event_seq) AS med_event_times
            FROM SortedMeds
            GROUP BY encounter_id
        """
        df_med = conn.execute(med_query).df()

        lab_sql = lab_path.as_posix().replace("'", "''")
        lab_query = f"""
            WITH ValidLabs AS (
                SELECT
                    l.encounter_id,
                    l.lab_item,
                    TRY_CAST(l.lab_value AS DOUBLE) AS lab_value,
                    l.lab_time,
                    c.prediction_time,
                    c.first_med_time
                FROM read_parquet('{lab_sql}') l
                INNER JOIN mem_prediction_cohort c ON l.encounter_id = c.encounter_id
                WHERE l.lab_time IS NOT NULL
                  AND l.lab_time >= c.first_med_time
                  AND l.lab_time < c.prediction_time
                  AND TRY_CAST(l.lab_value AS DOUBLE) IS NOT NULL
            ),
            OrderedLabs AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY encounter_id
                        ORDER BY lab_time, lab_item, lab_value
                    ) AS event_seq
                FROM ValidLabs
            ),
            SortedLabs AS (
                SELECT
                    encounter_id,
                    lab_item,
                    lab_value,
                    lab_time,
                    event_seq,
                    COALESCE(
                        EPOCH(lab_time) - EPOCH(
                            LAG(lab_time) OVER (
                                PARTITION BY encounter_id ORDER BY event_seq
                            )
                        ),
                        EPOCH(lab_time) - EPOCH(first_med_time)
                    ) / 3600.0 AS lab_dt_hours
                FROM OrderedLabs
            )
            SELECT
                encounter_id,
                LIST(lab_item ORDER BY event_seq) AS lab_tokens,
                LIST(lab_value ORDER BY event_seq) AS lab_values,
                LIST(lab_dt_hours ORDER BY event_seq) AS lab_dt_hours,
                LIST(lab_time ORDER BY event_seq) AS lab_event_times
            FROM SortedLabs
            GROUP BY encounter_id
        """
        df_lab = conn.execute(lab_query).df()
    finally:
        conn.close()

    # Medication data are required for this polypharmacy task. Laboratory data
    # may be absent before the prediction time and are represented as empty lists.
    df_final = pd.merge(df_cohort, df_med, on="encounter_id", how="inner")
    df_final = pd.merge(df_final, df_lab, on="encounter_id", how="left")
    df_final = df_final.sort_values("encounter_id", kind="mergesort").reset_index(
        drop=True
    )

    list_columns = [
        "med_tokens",
        "med_dt_hours",
        "med_event_times",
        "lab_tokens",
        "lab_values",
        "lab_dt_hours",
        "lab_event_times",
    ]
    for column in list_columns:
        df_final[column] = df_final[column].apply(
            lambda value: value if isinstance(value, (np.ndarray, list)) else []
        )

    df_final["dynamic_sequence_contract"] = "first_med_time<=event_time<prediction_time"
    df_final.to_parquet(output_parquet, index=False)

    print(
        f"[DILI-PLUS] Corrected sequences saved: {output_parquet} "
        f"({len(df_final)} encounters; {len(df_med)} with medications; "
        f"{len(df_lab)} with pre-prediction labs)"
    )
    print(f"[DILI-PLUS] Sequence stage completed in {time.time() - start_time:.2f}s")
    return output_parquet


if __name__ == "__main__":
    build_dili_tensors()
