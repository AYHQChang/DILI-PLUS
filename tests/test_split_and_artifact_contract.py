from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from diliplus.artifacts import (
    ArtifactContractError,
    artifact_probabilities,
    build_artifact_metadata,
    load_deep_artifact,
    load_sklearn_artifact,
    save_deep_artifact,
    save_sklearn_artifact,
)
from diliplus.calibration import fit_temperature, probabilities_from_logits
from diliplus.config import load_settings
from diliplus.splits import ROLE_NAMES, build_nested_grouped_splits


class SplitContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_settings()
        encounter_ids = []
        labels = []
        for patient in range(160):
            label = int(patient % 4 == 0)
            for encounter in range(2):
                encounter_ids.append(f"P{patient:04d}_{encounter}")
                labels.append(label)
        cls.encounter_ids = np.asarray(encounter_ids, dtype=object)
        cls.labels = np.asarray(labels, dtype=np.int64)

    def test_four_roles_are_index_and_group_disjoint(self):
        folds = build_nested_grouped_splits(
            self.encounter_ids, self.labels, self.settings
        )
        self.assertEqual(len(folds), self.settings.evaluation_protocol.outer_folds)
        test_membership = []
        for fold in folds:
            role_sets = {
                role: set(fold.indices_for(role).tolist()) for role in ROLE_NAMES
            }
            self.assertEqual(set().union(*role_sets.values()), set(range(len(self.labels))))
            for left_index, left in enumerate(ROLE_NAMES):
                for right in ROLE_NAMES[left_index + 1 :]:
                    self.assertFalse(role_sets[left] & role_sets[right])
                    left_groups = {
                        self.encounter_ids[index].split("_")[0]
                        for index in role_sets[left]
                    }
                    right_groups = {
                        self.encounter_ids[index].split("_")[0]
                        for index in role_sets[right]
                    }
                    self.assertFalse(left_groups & right_groups)
            self.assertEqual(fold.summary["status"], "PASS")
            self.assertTrue(
                all(
                    fold.summary["roles"][role]["positive"] > 0
                    and fold.summary["roles"][role]["negative"] > 0
                    for role in ROLE_NAMES
                )
            )
            test_membership.extend(fold.test.tolist())
        self.assertEqual(sorted(test_membership), list(range(len(self.labels))))

    def test_membership_hashes_do_not_depend_on_input_row_order(self):
        first = build_nested_grouped_splits(
            self.encounter_ids, self.labels, self.settings
        )
        permutation = np.random.default_rng(17).permutation(len(self.labels))
        second = build_nested_grouped_splits(
            self.encounter_ids[permutation], self.labels[permutation], self.settings
        )
        first_hashes = [
            {
                role: fold.summary["roles"][role]["membership_sha256"]
                for role in ROLE_NAMES
            }
            for fold in first
        ]
        second_hashes = [
            {
                role: fold.summary["roles"][role]["membership_sha256"]
                for role in ROLE_NAMES
            }
            for fold in second
        ]
        self.assertEqual(first_hashes, second_hashes)


class ArtifactContractTests(unittest.TestCase):
    def _metadata(self, artifact_type="torch"):
        split = {
            "fold": 1,
            "indices": {
                "training": [0, 1, 2, 3],
                "selection": [4, 5],
                "calibration": [6, 7],
                "test": [8, 9],
            },
            "summary": {"status": "PASS"},
        }
        return build_artifact_metadata(
            artifact_type=artifact_type,
            run_id="unit-contract",
            model_name="TinyModel",
            fold=1,
            selected_epoch=3,
            temperature=1.75,
            split_payload=split,
            dataset={"payload_sha256": "DATASET-A", "files": {}},
            configuration={"training_arguments": {"epochs": 3}},
        )

    def test_deep_artifact_roundtrip_and_dataset_guard(self):
        model = torch.nn.Linear(3, 2)
        original = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold_01.pt"
            metadata = self._metadata()
            save_deep_artifact(path, model, metadata)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            loaded = load_deep_artifact(
                path,
                model,
                expected_run_id="unit-contract",
                expected_model_name="TinyModel",
                expected_fold=1,
                expected_dataset_fingerprint="DATASET-A",
            )
            self.assertEqual(loaded["temperature"], 1.75)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, original[key]))
            with self.assertRaises(ArtifactContractError):
                load_deep_artifact(
                    path,
                    model,
                    expected_dataset_fingerprint="DIFFERENT-DATASET",
                )
            tampered_path = Path(directory) / "tampered.pt"
            tampered = torch.load(path, weights_only=False)
            tampered["metadata"]["temperature"] = 99.0
            torch.save(tampered, tampered_path)
            with self.assertRaises(ArtifactContractError):
                load_deep_artifact(tampered_path, model)

    def test_raw_and_calibrated_are_views_of_the_same_logits(self):
        logits = np.asarray([[0.2, 1.1], [1.4, -0.3], [0.1, 0.4]], dtype=np.float32)
        metadata = self._metadata()
        raw = artifact_probabilities(logits, metadata, "raw")
        calibrated = artifact_probabilities(logits, metadata, "calibrated")
        self.assertTrue(
            np.allclose(raw, probabilities_from_logits(logits, 1.0, "raw"))
        )
        self.assertTrue(
            np.allclose(
                calibrated,
                probabilities_from_logits(logits, metadata["temperature"], "calibrated"),
            )
        )
        self.assertFalse(np.allclose(raw, calibrated))

    def test_temperature_is_positive_and_fitted_without_test_inputs(self):
        calibration_logits = np.asarray(
            [[-1.0, 2.0], [2.2, -0.5], [-0.2, 0.7], [0.9, -0.1]],
            dtype=np.float32,
        )
        labels = np.asarray([1, 0, 1, 0])
        temperature = fit_temperature(calibration_logits, labels, max_iter=20)
        self.assertTrue(np.isfinite(temperature))
        self.assertGreater(temperature, 0.0)

    def test_sklearn_artifact_roundtrip(self):
        estimator = LogisticRegression().fit(
            np.asarray([[0.0], [1.0], [2.0], [3.0]]), np.asarray([0, 0, 1, 1])
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold_01.joblib"
            save_sklearn_artifact(
                path,
                estimator,
                {"med": "vectorizer-placeholder"},
                self._metadata("sklearn"),
            )
            payload = load_sklearn_artifact(
                path,
                expected_run_id="unit-contract",
                expected_model_name="TinyModel",
                expected_fold=1,
                expected_dataset_fingerprint="DATASET-A",
            )
            self.assertEqual(payload["metadata"]["run_id"], "unit-contract")
            self.assertIn("med", payload["vectorizers"])


if __name__ == "__main__":
    unittest.main()
