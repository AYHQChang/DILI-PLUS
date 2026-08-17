"""Synthetic Code-08 tests without reading patient data or the source database."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.reporting.table1 import (  # noqa: E402
    LABEL,
    _checked_left_join,
    build_table_rows,
)


class CohortTable1ContractTests(unittest.TestCase):
    def test_checked_join_preserves_encounter_grain(self):
        left = pd.DataFrame({"encounter_id": ["e1", "e2"], LABEL: [0, 1]})
        right = pd.DataFrame(
            {"encounter_id": ["e1", "e2"], "diagnosis_event_count": [0, 2]}
        )
        merged, audit = _checked_left_join(
            left,
            right,
            stage="synthetic",
            matched_column="diagnosis_event_count",
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(audit["matched_left_encounters"], 2)
        self.assertEqual(audit["row_inflation"], 1.0)

    def test_checked_join_rejects_duplicate_and_zero_match(self):
        left = pd.DataFrame({"encounter_id": ["e1", "e2"], LABEL: [0, 1]})
        duplicate = pd.DataFrame(
            {"encounter_id": ["e1", "e1"], "patient_id": ["p1", "p1"]}
        )
        with self.assertRaisesRegex(ValueError, "one-row-per-encounter"):
            _checked_left_join(
                left,
                duplicate,
                stage="duplicate",
                matched_column="patient_id",
            )
        no_match = pd.DataFrame({"encounter_id": ["e3"], "patient_id": ["p3"]})
        with self.assertRaisesRegex(RuntimeError, "matched zero encounters"):
            _checked_left_join(
                left,
                no_match,
                stage="wrong_key",
                matched_column="patient_id",
            )

    def test_table_rows_use_ahi_proxy_group_names_and_missing_counts(self):
        frame = pd.DataFrame(
            {
                LABEL: [0, 0, 1, 1],
                "age": [40.0, None, 60.0, 70.0],
                "gender_male": [0.0, 1.0, 1.0, 1.0],
            }
        )
        rows = build_table_rows(frame).set_index("variable")
        self.assertIn("ahi_proxy_negative", rows.columns)
        self.assertIn("ahi_proxy_positive", rows.columns)
        self.assertEqual(int(rows.loc["age", "total_missing"]), 1)
        self.assertEqual(int(rows.loc["age", "negative_nonmissing"]), 1)
        self.assertEqual(rows.loc["gender_male", "total"], "3 (75.0%)")


if __name__ == "__main__":
    unittest.main()
