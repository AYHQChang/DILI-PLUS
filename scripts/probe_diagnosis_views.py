"""Read-only schema probe for curated diagnosis analysis views."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.config import load_settings
from diliplus.database import connect_source_database


def main() -> int:
    conn = connect_source_database(load_settings())
    try:
        for relation in ("analysis.feature_diagnoses", "analysis.feature_diagnosis_seq"):
            print(f"SCHEMA: {relation}")
            print(conn.execute(f"DESCRIBE {relation}").df().to_string(index=False))
            print(
                "ROW_COUNT:",
                conn.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0],
            )
            print()
        print("VIEW DEFINITIONS")
        definitions = conn.execute(
            """
            SELECT schema_name, view_name, sql
            FROM duckdb_views()
            WHERE schema_name = 'analysis'
              AND view_name IN ('feature_diagnoses', 'feature_diagnosis_seq')
            ORDER BY view_name
            """
        ).df()
        print(definitions.to_string(index=False))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
