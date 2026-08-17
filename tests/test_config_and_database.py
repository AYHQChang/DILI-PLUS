"""Configuration, output-path, and DuckDB safety contract tests."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.config import load_settings
from diliplus.database import connect_source_database
from diliplus.artifacts import config_snapshot

class ConfigAndDatabaseTests(unittest.TestCase):
    def test_default_paths_are_anchored_to_project_root(self):
        settings = load_settings()
        self.assertEqual(settings.paths.root, PROJECT_ROOT)
        expected = {
            "data_cache": PROJECT_ROOT / "data_cache",
            "manifests": PROJECT_ROOT / "manifests",
            "vocab": PROJECT_ROOT / "vocab",
            "checkpoints": PROJECT_ROOT / "checkpoints",
            "reports": PROJECT_ROOT / "reports",
            "figures": PROJECT_ROOT / "figures",
        }
        for name, path in expected.items():
            self.assertEqual(getattr(settings.paths, name), path)
        self.assertEqual(settings.prediction.gap_hours, 24.0)
        self.assertEqual(settings.prediction.pseudo_index_seed, 20260816)
        self.assertEqual(settings.training.hidden_size, 128)
        self.assertEqual(settings.training.num_heads, 4)
        self.assertEqual(settings.training.dropout, 0.3)
        self.assertEqual(settings.training.diagnosis_modality_dropout_prob, 0.15)
        self.assertEqual(settings.training.focal_gamma, 2.0)
        self.assertEqual(settings.reproducibility.global_seed, 20260816)
        self.assertEqual(settings.reproducibility.split_seed, 20260816)
        self.assertEqual(settings.reproducibility.bootstrap_seed, 20260816)
        self.assertEqual(settings.reproducibility.figure_seed, 20260816)
        self.assertTrue(settings.reproducibility.deterministic_torch)
        self.assertEqual(settings.reproducibility.dataloader_num_workers, 0)
        self.assertEqual(settings.evaluation_protocol.outer_folds, 5)
        self.assertEqual(settings.evaluation_protocol.selection_fraction, 0.15)
        self.assertEqual(settings.evaluation_protocol.calibration_fraction, 0.15)
        self.assertEqual(settings.evaluation_protocol.split_search_attempts, 128)
        self.assertIn(
            settings.prediction.pseudo_index_seed,
            settings.reproducibility.pseudo_index_sensitivity_seeds,
        )
        self.assertEqual(
            settings.model_data_dir, PROJECT_ROOT / "data_cache" / "prediction_gap_24h"
        )
        self.assertEqual(
            settings.prediction_audit_dir,
            PROJECT_ROOT / "reports" / "p0_02_prediction_gap_24h",
        )
        self.assertEqual(
            settings.diagnosis_audit_dir,
            PROJECT_ROOT / "reports" / "p0_03_diagnosis_time_gap_24h",
        )
        self.assertEqual(
            settings.reproducibility_audit_dir,
            PROJECT_ROOT / "reports" / "p0_04_reproducibility",
        )
        self.assertEqual(
            settings.baseline_manifest_path,
            PROJECT_ROOT / "manifests" / "code00_code04_baseline.json",
        )
        implementation_hashes = config_snapshot(settings, {})[
            "contract_implementation_sha256"
        ]
        self.assertTrue(
            {
                "models/diliplus_engine.py",
                "models/baselines.py",
                "models/registry.py",
                "training/losses.py",
                "training/deep_trainer_calibrated.py",
                "training/ml_baselines_calibrated.py",
                "data/dataset.py",
            }.issubset(implementation_hashes)
        )

    def test_default_paths_do_not_depend_on_current_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                settings = load_settings()
            finally:
                os.chdir(original)
        self.assertEqual(settings.paths.reports, PROJECT_ROOT / "reports")

    def test_writable_database_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "unsafe.yaml"
            config.write_text(
                f"project_root: '{PROJECT_ROOT.as_posix()}'\n"
                "database:\n"
                "  path: D:/MedicalAI_Work/duck/medical.duckdb\n"
                "  read_only: false\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_settings(config)

    def test_negative_prediction_gap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "invalid_prediction.yaml"
            config.write_text(
                f"project_root: '{PROJECT_ROOT.as_posix()}'\n"
                "prediction:\n"
                "  gap_hours: -1\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_settings(config)

    def test_connection_factory_always_passes_read_only_true(self):
        settings = load_settings()
        sentinel = object()
        with patch("diliplus.database.duckdb.connect", return_value=sentinel) as connect:
            result = connect_source_database(settings)
        self.assertIs(result, sentinel)
        connect.assert_called_once_with(str(settings.database_path), read_only=True)

    def test_invalid_model_semantics_configuration_is_rejected(self):
        invalid_training_blocks = (
            "  hidden_size: 10\n  num_heads: 4\n",
            "  hidden_size: 3\n  num_heads: 1\n",
            "  hidden_size: 2\n  num_heads: 1\n",
            "  focal_gamma: -1\n",
            "  diagnosis_modality_dropout_prob: 1\n",
        )
        for index, training_block in enumerate(invalid_training_blocks):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as temporary:
                config = Path(temporary) / "invalid_training.yaml"
                config.write_text(
                    f"project_root: '{PROJECT_ROOT.as_posix()}'\n"
                    "training:\n"
                    + training_block,
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    load_settings(config)

    def test_overlapping_inner_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "invalid_protocol.yaml"
            config.write_text(
                f"project_root: '{PROJECT_ROOT.as_posix()}'\n"
                "evaluation_protocol:\n"
                "  selection_fraction: 0.30\n"
                "  calibration_fraction: 0.25\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_settings(config)

if __name__ == "__main__":
    unittest.main()
