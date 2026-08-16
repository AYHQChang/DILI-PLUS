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

class ConfigAndDatabaseTests(unittest.TestCase):
    def test_default_paths_are_anchored_to_project_root(self):
        settings = load_settings()
        self.assertEqual(settings.paths.root, PROJECT_ROOT)
        expected = {
            "data_cache": PROJECT_ROOT / "data_cache",
            "vocab": PROJECT_ROOT / "vocab",
            "checkpoints": PROJECT_ROOT / "checkpoints",
            "reports": PROJECT_ROOT / "reports",
            "figures": PROJECT_ROOT / "figures",
        }
        for name, path in expected.items():
            self.assertEqual(getattr(settings.paths, name), path)

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

    def test_connection_factory_always_passes_read_only_true(self):
        settings = load_settings()
        sentinel = object()
        with patch("diliplus.database.duckdb.connect", return_value=sentinel) as connect:
            result = connect_source_database(settings)
        self.assertIs(result, sentinel)
        connect.assert_called_once_with(str(settings.database_path), read_only=True)

if __name__ == "__main__":
    unittest.main()
