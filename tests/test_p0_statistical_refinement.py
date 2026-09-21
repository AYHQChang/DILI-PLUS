import tempfile
import unittest
from pathlib import Path

import pandas as pd

from diliplus.evaluation.p0_statistical_refinement import (
    fold_discrimination,
    load_model_oof,
    validate_common_membership,
)


class P0StatisticalRefinementTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame(
            {
                "dataset_index": [0, 1, 2, 3],
                "encounter_id": ["e0", "e1", "e2", "e3"],
                "patient_id": ["p0", "p1", "p2", "p3"],
                "y_true": [0, 1, 0, 1],
                "y_prob_raw": [0.1, 0.8, 0.2, 0.7],
                "y_prob_calibrated": [0.05, 0.6, 0.1, 0.5],
                "fold": [1, 1, 2, 2],
            }
        )

    def test_common_membership_rejects_label_mismatch(self):
        first = self._frame()
        second = self._frame()
        second.loc[0, "y_true"] = 1
        with self.assertRaisesRegex(ValueError, "y_true"):
            validate_common_membership({"first": first, "second": second})

    def test_fold_discrimination_reports_each_mode_and_fold(self):
        result = fold_discrimination({"model": self._frame()})
        self.assertEqual(len(result), 4)
        self.assertEqual(set(result["Probability_Mode"]), {"raw", "calibrated"})
        self.assertEqual(set(result["Fold"]), {1, 2})

    def test_loader_requires_five_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            model_dir.mkdir()
            self._frame().to_csv(model_dir / "fold_01.csv", index=False)
            with self.assertRaisesRegex(ValueError, "five OOF"):
                load_model_oof(root, "model")


if __name__ == "__main__":
    unittest.main()
