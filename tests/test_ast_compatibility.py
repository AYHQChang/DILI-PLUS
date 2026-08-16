"""Algorithm-preservation checks for the moved Dataset and model classes."""

from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "src/diliplus/data/dataset.py": "872d20bb6003aad1275310365c87b3e13694e0d60e9eb474b71f9b754567a970",
    "src/diliplus/models/baselines.py": "b54d45674bf5da64963a953017d6fa18968d586ec6d9391f7b36980bdd10e623",
    "src/diliplus/models/diliplus_engine.py": "9993df3dfd7646471b017f6f1a85d976bfd8064323f10de276d58b776eb10a31",
}

def digest_without_docstrings(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body.pop(0)
    canonical = ast.dump(tree, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

class AstCompatibilityTests(unittest.TestCase):
    def test_core_dataset_and_model_ast_is_unchanged(self):
        actual = {
            relative: digest_without_docstrings(PROJECT_ROOT / relative)
            for relative in EXPECTED
        }
        self.assertEqual(actual, EXPECTED)

if __name__ == "__main__":
    unittest.main()
