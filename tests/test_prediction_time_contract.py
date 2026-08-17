"""Prediction-time and target-laboratory leakage contract tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.data.temporal_audit import build_temporal_audit


class PredictionTimeContractTests(unittest.TestCase):
    def _write_artifacts(self, directory: Path, *, leaking: bool):
        labels = pd.DataFrame(
            {
                "encounter_id": ["positive", "negative"],
                "label_ahi_proxy": [1, 0],
                "first_med_time": pd.to_datetime(["2026-01-01", "2026-01-01"]),
                "index_time": pd.to_datetime(["2026-01-03", "2026-01-04"]),
                "prediction_time": pd.to_datetime(["2026-01-02", "2026-01-03"]),
                "t_onset": pd.to_datetime(["2026-01-03", None]),
            }
        )
        positive_lab_time = "2026-01-02" if leaking else "2026-01-01 12:00:00"
        positive_lab_value = 120.0 if leaking else 30.0
        sequences = pd.DataFrame(
            {
                "encounter_id": ["positive", "negative"],
                "prediction_time": labels["prediction_time"],
                "med_tokens": [["Drug A"], ["Drug B"]],
                "med_event_times": [
                    [pd.Timestamp("2026-01-01 06:00:00")],
                    [pd.Timestamp("2026-01-02")],
                ],
                "lab_tokens": [["谷丙转氨酶"], []],
                "lab_values": [[positive_lab_value], []],
                "lab_event_times": [[pd.Timestamp(positive_lab_time)], []],
            }
        )
        label_path = directory / "labels.parquet"
        sequence_path = directory / "sequences.parquet"
        labels.to_parquet(label_path, index=False)
        sequences.to_parquet(sequence_path, index=False)
        return label_path, sequence_path

    def test_strictly_pre_prediction_events_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            label_path, sequence_path = self._write_artifacts(
                Path(temporary), leaking=False
            )
            audit = build_temporal_audit(label_path, sequence_path, 24.0)
        self.assertEqual(audit["audit_status"], "PASS")
        self.assertEqual(audit["violations"]["target_threshold_lab_in_input"], 0)

    def test_event_at_prediction_time_and_threshold_lab_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            label_path, sequence_path = self._write_artifacts(
                Path(temporary), leaking=True
            )
            audit = build_temporal_audit(label_path, sequence_path, 24.0)
        self.assertEqual(audit["audit_status"], "FAIL")
        self.assertEqual(audit["violations"]["lab_event_at_or_after_prediction"], 1)
        self.assertEqual(audit["violations"]["target_threshold_lab_in_input"], 1)


if __name__ == "__main__":
    unittest.main()
