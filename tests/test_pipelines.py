"""Pipeline order and paper-asset inclusion tests without executing stages."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def load_pipeline(filename):
    path = PROJECT_ROOT / "pipelines" / filename
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class _FakeModule:
    def __init__(self, calls, module_name):
        self.calls = calls
        self.module_name = module_name

    def __getattr__(self, function_name):
        def invoke(*, settings, **kwargs):
            self.calls.append((self.module_name, function_name, settings, kwargs))
        return invoke

class PipelineTests(unittest.TestCase):
    def test_dataset_stage_order(self):
        module = load_pipeline("01_build_dataset.py")
        self.assertEqual(
            [stage[0] for stage in module.STAGES],
            [
                "cohort",
                "labels",
                "sequences",
                "temporal_audit",
                "diagnosis_source_audit",
                "diagnoses",
                "diagnosis_audit",
                "vocabulary",
                "drug_mapping",
            ],
        )
        calls = []
        settings = object()
        with patch.object(
            module, "import_module", side_effect=lambda name: _FakeModule(calls, name)
        ):
            module.run(settings)
        self.assertEqual([call[1] for call in calls], [stage[2] for stage in module.STAGES])
        self.assertTrue(all(call[2] is settings for call in calls))

    def test_dataset_stage_selection_preserves_dependency_order(self):
        module = load_pipeline("01_build_dataset.py")
        with patch.object(module, "load_settings", return_value="settings"), patch.object(
            module, "run"
        ) as run:
            exit_code = module.main(
                ["--stages", "temporal_audit", "labels", "sequences"]
            )
        self.assertEqual(exit_code, 0)
        selected = run.call_args.args[1]
        self.assertEqual(
            [stage[0] for stage in selected],
            ["labels", "sequences", "temporal_audit"],
        )

    def test_explainability_order(self):
        module = load_pipeline("04_explain_models.py")
        self.assertEqual([stage[0] for stage in module.STAGES], ["ig_loo", "token_perturbation"])

    def test_paper_pipeline_excludes_mock_figure_1c(self):
        module = load_pipeline("05_build_paper_assets.py")
        serialized = " ".join(" ".join(stage) for stage in module.STAGES).lower()
        self.assertNotIn("figure_1c", serialized)
        self.assertNotIn("mock", serialized)
        self.assertIn("diliplus.reporting.figures.perturbation", serialized)

    def test_training_default_mode_is_calibrated(self):
        module = load_pipeline("02_train_models.py")
        with patch.object(module, "load_settings", return_value="settings"), patch.object(
            module, "run"
        ) as run:
            exit_code = module.main(["--run-id", "unit-run"])
        self.assertEqual(exit_code, 0)
        run.assert_called_once_with("settings", "unit-run", "calibrated")

    def test_training_rejects_separate_uncalibrated_mode(self):
        module = load_pipeline("02_train_models.py")
        with self.assertRaises(ValueError):
            module.run("settings", "unit-run", "uncalibrated")

    def test_evaluation_requires_explicit_run_and_probability_mode(self):
        module = load_pipeline("03_evaluate_models.py")
        with patch.object(module, "load_settings", return_value="settings"), patch.object(
            module, "run"
        ) as run:
            exit_code = module.main(
                ["--run-id", "unit-run", "--probability-mode", "raw"]
            )
        self.assertEqual(exit_code, 0)
        run.assert_called_once_with("settings", "unit-run", "raw")

    def test_explainability_requires_same_run_fold_and_mode(self):
        module = load_pipeline("04_explain_models.py")
        with patch.object(module, "load_settings", return_value="settings"), patch.object(
            module, "run"
        ) as run:
            exit_code = module.main(
                [
                    "--run-id",
                    "unit-run",
                    "--probability-mode",
                    "calibrated",
                    "--fold",
                    "2",
                ]
            )
        self.assertEqual(exit_code, 0)
        run.assert_called_once_with("settings", "unit-run", "calibrated", 2)

if __name__ == "__main__":
    unittest.main()
