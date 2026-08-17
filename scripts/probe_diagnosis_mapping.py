"""Read-only schema probe for the encounter-to-diagnosis bridge."""

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
        relations = (
            "analysis.v_patient_encounters",
            "refined.v_discharge_summary",
            "refined.v_in_medical_record_diag",
        )
        for relation in relations:
            print(f"SCHEMA: {relation}")
            print(conn.execute(f"DESCRIBE {relation}").df().to_string(index=False))
            print()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
