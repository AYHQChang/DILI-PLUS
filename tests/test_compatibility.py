"""Legacy CLI and import shim compatibility tests."""

from __future__ import annotations

import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

class CompatibilityTests(unittest.TestCase):
    def test_dataset_shim_exports_original_symbols(self):
        import Util_model_dataset_init as shim
        from diliplus.data.dataset import DILIPlusDataset, load_vocab_sizes

        self.assertIs(shim.DILIPlusDataset, DILIPlusDataset)
        self.assertIs(shim.load_vocab_sizes, load_vocab_sizes)

    def test_model_shims_export_original_core_classes(self):
        from models.baseline_models import (
            MultiModalBiLSTM,
            MultimodalTransformerBaseline,
        )
        from models.diliplus_engine import (
            DILIPlusEngine,
            TimeAwareMultimodalTransformer,
        )
        from diliplus.models.baselines import (
            MultiModalBiLSTM as PackagedBiLSTM,
            MultimodalTransformerBaseline as PackagedTransformerBaseline,
        )
        from diliplus.models.diliplus_engine import (
            DILIPlusEngine as PackagedEngine,
            TimeAwareMultimodalTransformer as PackagedTimeAware,
        )

        self.assertIs(MultiModalBiLSTM, PackagedBiLSTM)
        self.assertIs(DILIPlusEngine, PackagedEngine)
        self.assertIs(TimeAwareMultimodalTransformer, PackagedTimeAware)
        self.assertIs(
            MultimodalTransformerBaseline, PackagedTransformerBaseline
        )

    def test_legacy_trainer_forwards_command_line_unchanged(self):
        fake_main = Mock()
        fake_module = types.ModuleType("diliplus.training.deep_trainer")
        fake_module.main = fake_main
        argv = [
            "05_train_universal_trainer.py",
            "--model",
            "MultiModalBiLSTM",
            "--epochs",
            "3",
            "--batch_size",
            "8",
            "--lr",
            "0.001",
        ]
        with patch.dict(sys.modules, {"diliplus.training.deep_trainer": fake_module}), patch.object(
            sys, "argv", argv
        ):
            runpy.run_path(str(PROJECT_ROOT / argv[0]), run_name="__main__")
        fake_main.assert_called_once_with()

if __name__ == "__main__":
    unittest.main()
