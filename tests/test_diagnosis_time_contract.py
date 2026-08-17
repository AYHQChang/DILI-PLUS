"""Strict encounter/prediction-time diagnosis tensor contract tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.data.diagnosis_audit import build_diagnosis_tensor_audit


class DiagnosisTimeContractTests(unittest.TestCase):
    def _write_artifacts(self, directory: Path, *, leaking: bool):
        cohort = pd.DataFrame(
            {
                "encounter_id": ["encounter-a", "encounter-b"],
                "prediction_time": pd.to_datetime(
                    ["2026-01-03 00:00:00", "2026-01-04 00:00:00"]
                ),
                "label_ahi_proxy": [1, 0],
            }
        )
        if leaking:
            codes = [["K71.6"], []]
            names = [["药物性肝损伤"], []]
            event_times = [[pd.Timestamp("2026-01-03 00:00:00")], []]
        else:
            codes = [["I10"], []]
            names = [["高血压"], []]
            event_times = [[pd.Timestamp("2026-01-02 12:00:00")], []]
        diagnosis = pd.DataFrame(
            {
                "encounter_id": cohort["encounter_id"],
                "icd_codes": codes,
                "diag_names": names,
                "diag_event_times": event_times,
            }
        )
        cohort_path = directory / "cohort.parquet"
        diagnosis_path = directory / "diagnosis.parquet"
        cohort.to_parquet(cohort_path, index=False)
        diagnosis.to_parquet(diagnosis_path, index=False)
        return cohort_path, diagnosis_path

    def test_strictly_pre_prediction_non_target_diagnoses_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort_path, diagnosis_path = self._write_artifacts(
                Path(temporary), leaking=False
            )
            audit = build_diagnosis_tensor_audit(cohort_path, diagnosis_path)
        self.assertEqual(audit["audit_status"], "PASS")
        self.assertEqual(audit["total_diagnosis_events"], 1)
        self.assertEqual(audit["encounters_without_diagnosis"], 1)

    def test_at_prediction_and_explicit_target_diagnosis_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort_path, diagnosis_path = self._write_artifacts(
                Path(temporary), leaking=True
            )
            audit = build_diagnosis_tensor_audit(cohort_path, diagnosis_path)
        self.assertEqual(audit["audit_status"], "FAIL")
        self.assertEqual(audit["violations"]["diagnosis_at_or_after_prediction"], 1)
        self.assertEqual(
            audit["violations"]["explicit_target_diagnosis_in_input"], 1
        )


if __name__ == "__main__":
    unittest.main()
