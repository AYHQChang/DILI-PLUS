"""Code-07 contracts for the single-task model, loss and canonical names."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.data.dataset import DILIPlusDataset
from diliplus.models.diliplus_engine import LogUniformTime2Vec
from diliplus.models.registry import (
    FORMAL_DEEP_MODEL_NAMES,
    FORMAL_DEEP_MODEL_REGISTRY,
    PRIMARY_MODEL_NAME,
    build_formal_deep_model,
    extract_ahi_proxy_logits,
    get_formal_deep_model,
)
from diliplus.training.losses import UnweightedFocalLoss


def _training_settings(diagnosis_dropout=0.0):
    return SimpleNamespace(
        hidden_size=8,
        num_heads=2,
        dropout=0.0,
        diagnosis_modality_dropout_prob=diagnosis_dropout,
    )


def _model_inputs(batch_size=2):
    return {
        "x_med": torch.tensor([[2, 3, 4, 0, 0], [0, 0, 0, 0, 0]])[:batch_size],
        "dt_med": torch.tensor(
            [[0.0, 1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        )[:batch_size],
        "mask_med": torch.tensor([[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]])[
            :batch_size
        ],
        "x_lab": torch.tensor([[2, 3, 0, 0], [0, 0, 0, 0]])[:batch_size],
        "v_lab": torch.tensor([[10.0, 20.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])[
            :batch_size
        ],
        "dt_lab": torch.tensor([[0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])[
            :batch_size
        ],
        "mask_lab": torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])[:batch_size],
        "x_diag": torch.tensor([[2, 3, 0], [0, 0, 0]])[:batch_size],
        "mask_diag": torch.tensor([[1, 1, 0], [0, 0, 0]])[:batch_size],
    }


class FocalLossTests(unittest.TestCase):
    def test_formula_matches_manual_unweighted_focal_loss(self):
        logits = torch.tensor([[1.2, -0.2], [-0.5, 0.8]], dtype=torch.float64)
        targets = torch.tensor([0, 1])
        gamma = 2.0
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        expected = (1.0 - torch.exp(-cross_entropy)).pow(gamma) * cross_entropy
        actual = UnweightedFocalLoss(gamma=gamma, reduction="none")(
            logits, targets
        )
        torch.testing.assert_close(actual, expected)

    def test_gamma_zero_is_cross_entropy(self):
        logits = torch.tensor([[0.3, -0.1], [0.2, 1.1]])
        targets = torch.tensor([0, 1])
        actual = UnweightedFocalLoss(gamma=0.0)(logits, targets)
        expected = F.cross_entropy(logits, targets)
        torch.testing.assert_close(actual, expected)

    def test_easy_example_is_more_strongly_downweighted(self):
        logits = torch.tensor([[6.0, -6.0], [0.1, 0.0]])
        targets = torch.tensor([0, 0])
        focal = UnweightedFocalLoss(gamma=2.0, reduction="none")(logits, targets)
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        weights = focal / cross_entropy
        self.assertLess(float(weights[0]), float(weights[1]))

    def test_no_scalar_alpha_and_invalid_settings_fail(self):
        self.assertNotIn("alpha", inspect.signature(UnweightedFocalLoss).parameters)
        with self.assertRaises(ValueError):
            UnweightedFocalLoss(gamma=-1)
        with self.assertRaises(ValueError):
            UnweightedFocalLoss(reduction="median")


class ModelContractTests(unittest.TestCase):
    def setUp(self):
        self.vocab_sizes = {
            "vocab_med_size": 12,
            "vocab_lab_size": 9,
            "vocab_diag_size": 7,
        }

    def test_formal_registry_has_no_pretraining_implying_names(self):
        self.assertEqual(PRIMARY_MODEL_NAME, "TimeAwareMultimodalTransformer")
        self.assertEqual(len(FORMAL_DEEP_MODEL_NAMES), 4)
        self.assertFalse(any("bert" in name.lower() for name in FORMAL_DEEP_MODEL_NAMES))
        with self.assertRaises(TypeError):
            FORMAL_DEEP_MODEL_REGISTRY["NotAFormalModel"] = object
        for old_name in (
            "MultiModalTimeAwareMedBERT",
            "MultiModalBaselineMedBERT",
            "DILIPlus",
        ):
            with self.assertRaises(ValueError):
                get_formal_deep_model(old_name)

    def test_all_formal_models_return_finite_two_class_logits(self):
        inputs = _model_inputs()
        for model_name in FORMAL_DEEP_MODEL_NAMES:
            with self.subTest(model=model_name):
                model = build_formal_deep_model(
                    model_name, self.vocab_sizes, _training_settings()
                ).eval()
                with warnings.catch_warnings(record=True) as caught, torch.no_grad():
                    outputs = model(**inputs)
                logits = extract_ahi_proxy_logits(outputs)
                self.assertEqual(tuple(logits.shape), (2, 2))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertFalse(
                    any(
                        "mismatched src_key_padding_mask" in str(item.message)
                        for item in caught
                    )
                )

    def test_eval_forward_is_deterministic_and_time_encoding_is_finite(self):
        model = build_formal_deep_model(
            PRIMARY_MODEL_NAME, self.vocab_sizes, _training_settings(0.15)
        ).eval()
        inputs = _model_inputs(batch_size=1)
        with torch.no_grad():
            first = model(**inputs)["logits"]
            second = model(**inputs)["logits"]
        self.assertTrue(torch.equal(first, second))
        encoded = LogUniformTime2Vec(8)(torch.tensor([[0.0, 1.0, 3.0]]))
        self.assertEqual(tuple(encoded.shape), (1, 3, 8))
        self.assertTrue(torch.isfinite(encoded).all())

    def test_diagnosis_dropout_does_not_rescale_dynamic_streams(self):
        model = build_formal_deep_model(
            PRIMARY_MODEL_NAME, self.vocab_sizes, _training_settings(0.5)
        )
        inputs = _model_inputs(batch_size=1)
        model.eval()
        with torch.no_grad():
            reference = model(**inputs)
        model.train()
        with patch("torch.rand", return_value=torch.zeros((1, 1))), torch.no_grad():
            dropped = model(**inputs)
        torch.testing.assert_close(dropped["h_med"], reference["h_med"])
        torch.testing.assert_close(dropped["h_lab"], reference["h_lab"])
        self.assertTrue(
            torch.equal(dropped["h_diag"], torch.zeros_like(dropped["h_diag"]))
        )

    def test_single_task_output_contract_rejects_tuple(self):
        with self.assertRaises(TypeError):
            extract_ahi_proxy_logits((torch.zeros(1, 2), torch.zeros(1, 2)))


class DatasetLabelContractTests(unittest.TestCase):
    def test_dataset_prefers_and_returns_ahi_proxy_label(self):
        med_lab = pd.DataFrame(
            {
                "encounter_id": ["p1_1"],
                "label_ahi_proxy": [1],
                "med_tokens": [["med"]],
                "med_dt_hours": [[0.0]],
                "lab_tokens": [["lab"]],
                "lab_values": [[12.0]],
                "lab_dt_hours": [[0.0]],
            }
        )
        diagnoses = pd.DataFrame(
            {"encounter_id": ["p1_1"], "icd_codes": [["I10"]]}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vocab = {"[PAD]": 0, "[UNK]": 1, "med": 2, "lab": 3}
            diag_vocab = {"[PAD]": 0, "[UNK]": 1, "I10": 2}
            (root / "vocab_polypharmacy.json").write_text(
                json.dumps(vocab), encoding="utf-8"
            )
            (root / "vocab_diagnosis.json").write_text(
                json.dumps(diag_vocab), encoding="utf-8"
            )
            with patch(
                "diliplus.data.dataset.pd.read_parquet",
                side_effect=[med_lab, diagnoses],
            ):
                dataset = DILIPlusDataset(root, root, 4, 3, 2)
            sample = dataset[0]
        self.assertEqual(dataset.label_column, "label_ahi_proxy")
        self.assertEqual(int(sample["label_ahi_proxy"]), 1)
        self.assertEqual(int(sample["label"]), 1)

    def test_dataset_rejects_legacy_only_label_artifact(self):
        med_lab = pd.DataFrame(
            {
                "encounter_id": ["p1_1"],
                "label_dili": [1],
                "med_tokens": [[]],
                "med_dt_hours": [[]],
                "lab_tokens": [[]],
                "lab_values": [[]],
                "lab_dt_hours": [[]],
            }
        )
        diagnoses = pd.DataFrame(
            {"encounter_id": ["p1_1"], "icd_codes": [[]]}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "vocab_polypharmacy.json").write_text(
                json.dumps({"[PAD]": 0, "[UNK]": 1}), encoding="utf-8"
            )
            (root / "vocab_diagnosis.json").write_text(
                json.dumps({"[PAD]": 0, "[UNK]": 1}), encoding="utf-8"
            )
            with patch(
                "diliplus.data.dataset.pd.read_parquet",
                side_effect=[med_lab, diagnoses],
            ), self.assertRaisesRegex(KeyError, "label_ahi_proxy"):
                DILIPlusDataset(root, root, 4, 3, 2)

    def test_empty_modalities_have_zero_masks_and_temporal_ages_are_exact(self):
        prediction = pd.Timestamp("2026-01-03 00:00:00")
        med_lab = pd.DataFrame(
            {
                "encounter_id": ["p1_1"],
                "label_ahi_proxy": [0],
                "prediction_time": [prediction],
                "prediction_gap_hours": [24.0],
                "med_tokens": [["med"]],
                "med_dt_hours": [[0.0]],
                "med_event_times": [[prediction - pd.Timedelta(hours=30)]],
                "lab_tokens": [[]],
                "lab_values": [[]],
                "lab_dt_hours": [[]],
                "lab_event_times": [[]],
            }
        )
        diagnoses = pd.DataFrame(
            {
                "encounter_id": ["p1_1"],
                "icd_codes": [[]],
                "diag_event_times": [[]],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "vocab_polypharmacy.json").write_text(
                json.dumps({"[PAD]": 0, "[UNK]": 1, "med": 2}), encoding="utf-8"
            )
            (root / "vocab_diagnosis.json").write_text(
                json.dumps({"[PAD]": 0, "[UNK]": 1}), encoding="utf-8"
            )
            with patch(
                "diliplus.data.dataset.pd.read_parquet",
                side_effect=[med_lab, diagnoses],
            ):
                dataset = DILIPlusDataset(
                    root, root, 4, 3, 2, include_temporal_metadata=True
                )
            sample = dataset[0]
        self.assertEqual(sample["mask_lab"].tolist(), [0, 0, 0])
        self.assertEqual(sample["mask_diag"].tolist(), [0, 0])
        self.assertEqual(sample["age_med_hours"].tolist(), [30.0, 0.0, 0.0, 0.0])
        self.assertEqual(float(sample["base_prediction_gap_hours"]), 24.0)


if __name__ == "__main__":
    unittest.main()
