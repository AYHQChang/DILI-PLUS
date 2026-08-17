from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.config import load_settings
from diliplus.data.labels import _build_raw_label_table
from diliplus.splits import ROLE_NAMES, build_nested_grouped_splits


class DeterministicLabelTests(unittest.TestCase):
    def _source(self):
        rows = [
            # Positive: both target rows at the same earliest timestamp are valid.
            ("A", "谷丙转氨酶", 30.0, "N", "2026-01-01"),
            ("A", "谷草转氨酶", 40.0, "正常", "2026-01-01"),
            ("A", "谷丙转氨酶", 130.0, "H", "2026-01-03"),
            # Excluded: one same-time baseline row is high/abnormal.
            ("B", "谷丙转氨酶", 25.0, "N", "2026-01-01"),
            ("B", "谷草转氨酶", 140.0, "H", "2026-01-01"),
            ("B", "谷丙转氨酶", 160.0, "H", "2026-01-03"),
            # Negative: valid baseline and no later threshold crossing.
            ("C", "谷丙转氨酶", 20.0, "L", "2026-01-01"),
            ("C", "谷丙转氨酶", 60.0, "N", "2026-01-03"),
        ]
        frame = pd.DataFrame(
            rows,
            columns=["encounter_id", "lab_item", "lab_value", "abnormal_status", "lab_time"],
        )
        frame["lab_time"] = pd.to_datetime(frame["lab_time"])
        frame["first_med_time"] = pd.Timestamp("2025-12-31")
        frame["last_med_time"] = pd.Timestamp("2026-01-05")
        return frame

    def _build(self, frame, path):
        frame.to_parquet(path, index=False)
        connection = duckdb.connect()
        try:
            _build_raw_label_table(connection, path, 120.0)
            return connection.execute(
                """
                SELECT encounter_id, label_ahi_proxy, baseline_target_rows,
                       baseline_distinct_target_items, baseline_rule_pass, t_onset
                FROM temp_raw_ahi_labels ORDER BY encounter_id
                """
            ).df()
        finally:
            connection.close()

    def test_same_timestamp_rows_are_aggregated_and_order_invariant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._build(self._source(), root / "first.parquet")
            second = self._build(
                self._source().sample(frac=1.0, random_state=9), root / "second.parquet"
            )
        pd.testing.assert_frame_equal(first, second)
        labels = dict(zip(first["encounter_id"], first["label_ahi_proxy"]))
        self.assertEqual(labels["A"], 1)
        self.assertTrue(pd.isna(labels["B"]))
        self.assertEqual(labels["C"], 0)
        self.assertEqual(int(first.loc[first["encounter_id"] == "A", "baseline_target_rows"].iloc[0]), 2)


class SourcePatientGroupTests(unittest.TestCase):
    def test_explicit_source_patient_groups_never_cross_roles(self):
        settings = load_settings()
        encounter_ids = []
        patient_ids = []
        labels = []
        for patient in range(160):
            for encounter in range(2):
                encounter_ids.append(f"encounter-{patient}-{encounter}")
                patient_ids.append(f"source-patient-{patient}")
                labels.append(int(patient % 4 == 0))
        encounters = np.asarray(encounter_ids, dtype=object)
        patients = np.asarray(patient_ids, dtype=object)
        folds = build_nested_grouped_splits(
            encounters, labels, settings, group_ids=patients
        )
        for fold in folds:
            for left_index, left in enumerate(ROLE_NAMES):
                left_groups = set(patients[fold.indices_for(left)])
                for right in ROLE_NAMES[left_index + 1 :]:
                    right_groups = set(patients[fold.indices_for(right)])
                    self.assertFalse(left_groups & right_groups)


if __name__ == "__main__":
    unittest.main()
