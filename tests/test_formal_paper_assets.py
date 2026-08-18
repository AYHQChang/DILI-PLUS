"""Static contracts for formal, run-specific paper tables and figures."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.reporting.formal_assets import (  # noqa: E402
    FORMAL_MODELS,
    MODEL_COLORS,
    MODEL_LABELS,
)


class FormalPaperAssetTests(unittest.TestCase):
    def test_every_formal_model_has_one_stable_colour_and_label(self):
        self.assertEqual(set(FORMAL_MODELS), set(MODEL_COLORS))
        self.assertEqual(set(FORMAL_MODELS), set(MODEL_LABELS))
        self.assertEqual(len(set(MODEL_COLORS.values())), len(FORMAL_MODELS))
        self.assertEqual(MODEL_COLORS["TimeAwareMultimodalTransformer"], "#DF9E9B")
        self.assertEqual(MODEL_COLORS["MultimodalTransformerBaseline"], "#99BADF")
        self.assertEqual(MODEL_COLORS["MultiModalBiLSTM"], "#99CDCE")
        self.assertEqual(MODEL_COLORS["MultiModalTextCNN"], "#F8BF92")
        self.assertEqual(MODEL_COLORS["XGBoost"], "#999ACD")
        self.assertEqual(MODEL_COLORS["LogisticRegression"], "#FFB3DD")

    def test_formal_figures_do_not_read_legacy_result_locations(self):
        modules = (
            "observation_process.py",
            "model_comparison.py",
            "calibration_impact.py",
            "early_warning.py",
            "robustness.py",
        )
        figure_dir = PROJECT_ROOT / "src" / "diliplus" / "reporting" / "figures"
        forbidden = (
            "predictions_calibrated",
            "06a_Early_Warning_Decay_Results",
            "Lead_Time_Hours",
            "retention",
        )
        for name in modules:
            source = (figure_dir / name).read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{name} contains legacy token {token}")

    def test_pipeline_declares_formal_assets_before_explanatory_assets(self):
        path = PROJECT_ROOT / "pipelines" / "05_build_paper_assets.py"
        spec = importlib.util.spec_from_file_location("formal_asset_pipeline", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        names = [stage[0] for stage in module.STAGES]
        self.assertLess(names.index("figure_1"), names.index("figure_2"))
        self.assertNotIn("figure_1b", names)
        self.assertNotIn("figure_1d", names)
        self.assertLess(names.index("table_2"), names.index("figure_2"))
        self.assertLess(names.index("figure_5"), names.index("figure_6"))
        self.assertEqual(names[-1], "asset_manifest")
        default_names = [stage[0] for stage in module.DEFAULT_STAGES]
        self.assertIn("figure_1", default_names)
        self.assertIn("figure_5", default_names)
        self.assertNotIn("figure_6", default_names)
        self.assertNotIn("figure_7", default_names)

    def test_main_figure_sequence_matches_the_reframed_narrative(self):
        path = PROJECT_ROOT / "pipelines" / "05_build_paper_assets.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("figures.observation_process", source)
        self.assertIn("generate_observation_process_figure", source)
        manifest = (PROJECT_ROOT / "src" / "diliplus" / "reporting" / "paper_manifest.py").read_text(
            encoding="utf-8"
        )
        for stem in (
            "Fig_2_Observation_Measurement_Process",
            "Fig_3_Compact_Predictive_Benchmark",
            "Fig_4_Earlier_Cutoff_Information_Erosion",
            "Fig_5_Model_Complexity_Audit",
        ):
            self.assertIn(stem, manifest)

    def test_local_audit_figures_are_identifier_free_and_noncausal(self):
        figure_dir = PROJECT_ROOT / "src" / "diliplus" / "reporting" / "figures"
        attribution = (figure_dir / "attribution.py").read_text(encoding="utf-8")
        perturbation = (figure_dir / "perturbation.py").read_text(encoding="utf-8")
        self.assertIn("Fig_6_Local_Attribution_Audit", attribution)
        self.assertIn("Fig_7_Medication_Token_Sensitivity", perturbation)
        for forbidden in (
            "Attribution for Patient",
            "Hepatotoxic Contribution",
            "Hepatoprotective Contribution",
            "Prediction Risk Score",
            "Alternative Regimen",
        ):
            self.assertNotIn(forbidden, attribution + perturbation)

    def test_study_design_uses_only_tracked_aggregate_contracts(self):
        path = (
            PROJECT_ROOT
            / "src"
            / "diliplus"
            / "reporting"
            / "figures"
            / "study_design.py"
        )
        source = path.read_text(encoding="utf-8")
        self.assertIn("code10_label_rebuild_audit.json", source)
        self.assertIn("code05_split_protocol.json", source)
        self.assertIn("code09_early_warning_contract.json", source)
        self.assertNotIn("duckdb", source.lower())
        self.assertNotIn("label_dili", source)


if __name__ == "__main__":
    unittest.main()
