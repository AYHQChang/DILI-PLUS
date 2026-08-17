"""Source-layout checks that prevent legacy shims from becoming second implementations."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SourceLayoutTests(unittest.TestCase):
    def test_root_model_files_are_thin_shims(self):
        for relative in ("models/baseline_models.py", "models/diliplus_engine.py"):
            tree = ast.parse((PROJECT_ROOT / relative).read_text(encoding="utf-8-sig"))
            class_names = [
                node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ]
            self.assertEqual(class_names, [], relative)

    def test_current_source_uses_truthful_model_class_names(self):
        engine = ast.parse(
            (PROJECT_ROOT / "src/diliplus/models/diliplus_engine.py").read_text(
                encoding="utf-8-sig"
            )
        )
        baseline = ast.parse(
            (PROJECT_ROOT / "src/diliplus/models/baselines.py").read_text(
                encoding="utf-8-sig"
            )
        )
        engine_classes = {
            node.name for node in ast.walk(engine) if isinstance(node, ast.ClassDef)
        }
        baseline_classes = {
            node.name for node in ast.walk(baseline) if isinstance(node, ast.ClassDef)
        }
        self.assertIn("TimeAwareMultimodalTransformer", engine_classes)
        self.assertIn("MultimodalTransformerBaseline", baseline_classes)
        self.assertNotIn("MultiModalBaselineMedBERT", baseline_classes)


if __name__ == "__main__":
    unittest.main()
