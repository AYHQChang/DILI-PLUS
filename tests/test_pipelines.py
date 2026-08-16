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
        def invoke(*, settings):
            self.calls.append((self.module_name, function_name, settings))
        return invoke

class PipelineTests(unittest.TestCase):
    def test_dataset_stage_order(self):
        module = load_pipeline("01_build_dataset.py")
        self.assertEqual(
            [stage[0] for stage in module.STAGES],
            ["cohort", "labels", "sequences", "diagnoses", "vocabulary", "drug_mapping"],
        )
        calls = []
        settings = object()
        with patch.object(
            module, "import_module", side_effect=lambda name: _FakeModule(calls, name)
        ):
            module.run(settings)
        self.assertEqual([call[1] for call in calls], [stage[2] for stage in module.STAGES])
        self.assertTrue(all(call[2] is settings for call in calls))

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
            exit_code = module.main([])
        self.assertEqual(exit_code, 0)
        run.assert_called_once_with("settings", "calibrated")

if __name__ == "__main__":
    unittest.main()
