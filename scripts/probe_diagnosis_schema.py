"""Read-only, aggregate-only schema probe for diagnosis data sources."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.config import load_settings
from diliplus.database import connect_source_database


def main() -> int:
    settings = load_settings()
    conn = connect_source_database(settings)
    try:
        diagnosis_tables = conn.execute(
            """
            SELECT DISTINCT table_schema, table_name
            FROM information_schema.columns
            WHERE LOWER(table_name) LIKE '%diag%'
               OR LOWER(column_name) LIKE '%diagnos%'
            ORDER BY table_schema, table_name
            """
        ).df()
        print("DIAGNOSIS-RELATED TABLES")
        print(diagnosis_tables.to_string(index=False))

        target = "in_medical_record_diag"
        print(f"\nSCHEMA: {target}")
        schema = conn.execute(f"DESCRIBE {target}").df()
        print(schema.to_string(index=False))
        print(f"\nROW_COUNT: {conn.execute(f'SELECT COUNT(*) FROM {target}').fetchone()[0]}")

        columns = set(schema["column_name"].astype(str))
        profile_expressions = ["COUNT(*) AS rows"]
        for column in sorted(columns):
            lowered = column.lower()
            if any(marker in lowered for marker in ("encounter", "visit", "record", "health")):
                profile_expressions.append(
                    f"COUNT({column}) AS nonnull_{column}"
                )
                profile_expressions.append(
                    f"COUNT(DISTINCT CAST({column} AS VARCHAR)) AS distinct_{column}"
                )
            elif any(marker in lowered for marker in ("time", "date", "admit", "discharge")):
                profile_expressions.append(
                    f"COUNT({column}) AS nonnull_{column}"
                )
        profile = conn.execute(
            f"SELECT {', '.join(profile_expressions)} FROM {target}"
        ).df()
        print("\nAGGREGATE KEY/TIME COMPLETENESS (NO ROW-LEVEL VALUES)")
        print(profile.T.to_string(header=False))

        print("\nMEDICATION VIEW SCHEMA")
        print(conn.execute("DESCRIBE analysis.feature_medications").df().to_string(index=False))

        for relation in ("analysis.feature_diagnoses", "analysis.feature_diagnosis_seq"):
            print(f"\nSCHEMA: {relation}")
            print(conn.execute(f"DESCRIBE {relation}").df().to_string(index=False))
            print(
                "ROW_COUNT:",
                conn.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0],
            )

        candidate_keys = (
            "business_uu",
            "inpatient_f",
            "report_form",
            "diagnosis_i",
        )
        print("\nCANDIDATE ENCOUNTER-KEY OVERLAP WITH MEDICATION VIEW")
        for column in candidate_keys:
            overlap = conn.execute(
                f"""
                WITH med_encounters AS (
                    SELECT DISTINCT TRIM(CAST(encounter_id AS VARCHAR)) AS encounter_id
                    FROM analysis.feature_medications
                    WHERE encounter_id IS NOT NULL
                ),
                diag_keys AS (
                    SELECT DISTINCT TRIM(CAST({column} AS VARCHAR)) AS candidate
                    FROM in_medical_record_diag
                    WHERE {column} IS NOT NULL AND TRIM(CAST({column} AS VARCHAR)) <> ''
                )
                SELECT
                    (SELECT COUNT(*) FROM diag_keys) AS distinct_diag_keys,
                    COUNT(*) AS matched_distinct_keys
                FROM diag_keys d
                INNER JOIN med_encounters m ON d.candidate = m.encounter_id
                """
            ).fetchone()
            print(
                f"{column}: distinct_diag_keys={int(overlap[0])}, "
                f"matched_distinct_keys={int(overlap[1])}"
            )

        print("\nTIME PARSE COMPLETENESS")
        for column in ("create_time", "modified_ti", "cancel_time", "sqctime", "last_diagno"):
            counts = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS rows,
                    SUM(
                        CASE WHEN COALESCE(
                            TRY_CAST({column} AS TIMESTAMP),
                            TRY_STRPTIME(CAST(TRY_CAST({column} AS BIGINT) AS VARCHAR), '%Y%m%d%H%M%S'),
                            TRY_STRPTIME(CAST(TRY_CAST({column} AS BIGINT) AS VARCHAR), '%Y%m%d'),
                            TRY_STRPTIME(CAST({column} AS VARCHAR), '%Y-%m-%d %H:%M:%S')
                        ) IS NOT NULL THEN 1 ELSE 0 END
                    ) AS parsed
                FROM in_medical_record_diag
                """
            ).fetchone()
            print(f"{column}: parsed={int(counts[1])}/{int(counts[0])}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
