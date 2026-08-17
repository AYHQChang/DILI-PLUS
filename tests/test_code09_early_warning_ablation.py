"""Synthetic Code-09 cutoff and minimum-ablation contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.evaluation.early_warning import (  # noqa: E402
    apply_effective_horizon_cutoff,
    validate_effective_horizons,
)
from diliplus.models.baselines import MultiModalTextCNN  # noqa: E402
from diliplus.models.registry import (  # noqa: E402
    MINIMUM_ABLATION_MODEL_NAMES,
    MINIMUM_ABLATION_SPECS,
    build_deep_experiment_model,
)


def _settings():
    return SimpleNamespace(
        hidden_size=8,
        num_heads=2,
        dropout=0.0,
        diagnosis_modality_dropout_prob=0.0,
    )


def _inputs():
    return {
        "x_med": torch.tensor([[2, 3, 4, 0]]),
        "dt_med": torch.tensor([[0.0, 4.0, 8.0, 0.0]]),
        "mask_med": torch.tensor([[1, 1, 1, 0]]),
        "age_med_hours": torch.tensor([[50.0, 24.0, 10.0, 0.0]]),
        "x_lab": torch.tensor([[2, 3, 0]]),
        "v_lab": torch.tensor([[12.0, 22.0, 0.0]]),
        "dt_lab": torch.tensor([[0.0, 5.0, 0.0]]),
        "mask_lab": torch.tensor([[1, 1, 0]]),
        "age_lab_hours": torch.tensor([[30.0, 5.0, 0.0]]),
        "x_diag": torch.tensor([[2, 3, 0]]),
        "mask_diag": torch.tensor([[1, 1, 0]]),
        "age_diag_hours": torch.tensor([[25.0, 4.0, 0.0]]),
    }


def _model_inputs():
    return {key: value for key, value in _inputs().items() if not key.startswith("age_")}


class EarlyWarningContractTests(unittest.TestCase):
    def test_horizons_below_existing_gap_are_rejected(self):
        self.assertEqual(validate_effective_horizons([24, 48, 72], 24), (24.0, 48.0, 72.0))
        with self.assertRaisesRegex(ValueError, "cannot be reconstructed"):
            validate_effective_horizons([0, 24], 24)

    def test_cutoff_physically_zeros_all_three_modalities(self):
        truncated, audit = apply_effective_horizon_cutoff(_inputs(), 48, 24)
        self.assertEqual(truncated["mask_med"].tolist(), [[1, 0, 0, 0]])
        self.assertEqual(truncated["mask_lab"].tolist(), [[1, 0, 0]])
        self.assertEqual(truncated["mask_diag"].tolist(), [[1, 0, 0]])
        self.assertEqual(truncated["x_med"].tolist(), [[2, 0, 0, 0]])
        self.assertEqual(truncated["v_lab"].tolist(), [[12.0, 0.0, 0.0]])
        self.assertEqual(truncated["x_diag"].tolist(), [[2, 0, 0]])
        self.assertEqual(audit["medication_events_retained"], 1)

    def test_textcnn_is_invariant_to_values_behind_zero_masks(self):
        torch.manual_seed(7)
        model = MultiModalTextCNN(12, 12, 12, hidden_size=12, dropout=0.0).eval()
        inputs = _model_inputs()
        inputs["mask_med"] = torch.tensor([[1, 1, 0, 0]])
        inputs["mask_lab"] = torch.tensor([[1, 1, 0]])
        changed = {key: value.clone() for key, value in inputs.items()}
        changed["x_med"][0, 2:] = torch.tensor([8, 9])
        changed["x_lab"][0, 2] = 8
        changed["v_lab"][0, 2] = 9999.0
        with torch.no_grad():
            reference = model(**inputs)["logits"]
            altered = model(**changed)["logits"]
        torch.testing.assert_close(reference, altered)


class MinimumAblationContractTests(unittest.TestCase):
    def setUp(self):
        self.vocab = {
            "vocab_med_size": 12,
            "vocab_lab_size": 12,
            "vocab_diag_size": 12,
        }

    def test_matrix_contains_the_six_prespecified_variants(self):
        self.assertEqual(len(MINIMUM_ABLATION_MODEL_NAMES), 6)
        self.assertEqual(set(MINIMUM_ABLATION_MODEL_NAMES), set(MINIMUM_ABLATION_SPECS))
        self.assertIn("StaticDiagnosisOnly", MINIMUM_ABLATION_MODEL_NAMES)
        self.assertIn("FullWithoutTimeEncoding", MINIMUM_ABLATION_MODEL_NAMES)
        self.assertIn("FullWithoutDiagnosis", MINIMUM_ABLATION_MODEL_NAMES)
        self.assertIn("TimeAwareMultimodalTransformer", MINIMUM_ABLATION_MODEL_NAMES)

    def test_all_variants_return_finite_logits_and_disabled_streams_are_zero(self):
        inputs = _model_inputs()
        for name in MINIMUM_ABLATION_MODEL_NAMES:
            with self.subTest(name=name):
                model = build_deep_experiment_model(name, self.vocab, _settings()).eval()
                with torch.no_grad():
                    output = model(**inputs)
                self.assertEqual(tuple(output["logits"].shape), (1, 2))
                self.assertTrue(torch.isfinite(output["logits"]).all())
                spec = MINIMUM_ABLATION_SPECS[name]
                for enabled, key in (
                    (spec["use_medication"], "h_med"),
                    (spec["use_laboratory"], "h_lab"),
                    (spec["use_diagnosis"], "h_diag"),
                ):
                    if not enabled:
                        self.assertTrue(torch.equal(output[key], torch.zeros_like(output[key])))

    def test_without_time_variant_is_invariant_to_time_deltas(self):
        model = build_deep_experiment_model(
            "FullWithoutTimeEncoding", self.vocab, _settings()
        ).eval()
        inputs = _model_inputs()
        changed = {key: value.clone() for key, value in inputs.items()}
        changed["dt_med"] += 1000.0
        changed["dt_lab"] += 1000.0
        with torch.no_grad():
            first = model(**inputs)["logits"]
            second = model(**changed)["logits"]
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
