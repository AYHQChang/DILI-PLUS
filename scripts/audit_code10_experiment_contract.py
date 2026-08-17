"""Freeze and audit the aggregate-only Code-10 formal experiment contract.

This script instantiates models to count parameters and fingerprints formal
inputs/configuration/source code. It does not train, infer on patient rows or
estimate predictive performance.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.artifacts import dataset_fingerprint, file_sha256  # noqa: E402
from diliplus.config import load_settings  # noqa: E402
from diliplus.data.dataset import load_vocab_sizes  # noqa: E402
from diliplus.models.registry import (  # noqa: E402
    FORMAL_DEEP_MODEL_NAMES,
    PRIMARY_MODEL_NAME,
    build_formal_deep_model,
)


FORMAL_ML_MODELS = ("LogisticRegression", "XGBoost")
SOURCE_PATHS = (
    "configs/default.yaml",
    "configs/pilot_legacy.yaml",
    "configs/sensitivity_128d_8heads.yaml",
    "pipelines/02_train_models.py",
    "src/diliplus/config.py",
    "src/diliplus/data/labels.py",
    "src/diliplus/data/lineage.py",
    "src/diliplus/splits.py",
    "src/diliplus/models/registry.py",
    "src/diliplus/training/losses.py",
    "src/diliplus/training/deep_trainer_calibrated.py",
    "src/diliplus/training/ml_baselines_calibrated.py",
    "src/diliplus/training/run_all_calibrated.py",
    "src/diliplus/evaluation/metrics.py",
    "src/diliplus/evaluation/finalize.py",
)


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, found {actual!r}")


def main() -> int:
    settings = load_settings(PROJECT_ROOT / "configs" / "default.yaml")
    pilot = load_settings(PROJECT_ROOT / "configs" / "pilot_legacy.yaml")
    sensitivity = load_settings(
        PROJECT_ROOT / "configs" / "sensitivity_128d_8heads.yaml"
    )

    _assert_equal(settings.prediction.label_source, "deterministic_rebuild", "formal label")
    _assert_equal(pilot.prediction.label_source, "legacy_frozen", "pilot label")
    _assert_equal(settings.training.hidden_size, 128, "formal hidden size")
    _assert_equal(settings.training.num_heads, 4, "primary attention heads")
    _assert_equal(sensitivity.training.hidden_size, 128, "sensitivity hidden size")
    _assert_equal(sensitivity.training.num_heads, 8, "sensitivity attention heads")
    _assert_equal(settings.training.selection_metric, "AUPRC", "selection metric")
    _assert_equal(settings.training.focal_gamma, 2.0, "focal gamma")
    _assert_equal(settings.evaluation_protocol.outer_folds, 5, "outer folds")
    _assert_equal(
        settings.evaluation_protocol.bootstrap_replicates,
        1000,
        "bootstrap replicates",
    )

    vocab_sizes = load_vocab_sizes(settings.paths.vocab)
    deep_models = []
    for name in FORMAL_DEEP_MODEL_NAMES:
        model = build_formal_deep_model(name, vocab_sizes, settings.training)
        deep_models.append(
            {
                "name": name,
                "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
                "trainable_parameters": int(
                    sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                ),
                "from_scratch": True,
            }
        )

    split_manifest_path = settings.paths.manifests / "code05_split_protocol.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    _assert_equal(split_manifest["status"], "PASS", "split manifest status")
    _assert_equal(split_manifest["cohort"]["encounters"], 44631, "split cohort size")
    _assert_equal(
        split_manifest["protocol"]["group_definition"],
        "analysis.v_patient_encounters.patient_id",
        "split patient group",
    )

    source_hashes = {
        relative: file_sha256(PROJECT_ROOT / relative) for relative in SOURCE_PATHS
    }
    payload = {
        "schema_version": 1,
        "contract": "code10_formal_experiment_v1",
        "status": "PASS",
        "git_commit_before_contract_checkpoint": _git_commit(),
        "formal_label_source": settings.prediction.label_source,
        "legacy_label_scope": "pipeline_pilot_only",
        "target": "single-task biochemical AHI proxy; no causal DILI attribution",
        "formal_models": deep_models + [
            {"name": name, "model_family": "traditional_machine_learning"}
            for name in FORMAL_ML_MODELS
        ],
        "primary_model": PRIMARY_MODEL_NAME,
        "architecture": {
            "primary_hidden_size": settings.training.hidden_size,
            "primary_attention_heads": settings.training.num_heads,
            "sensitivity_hidden_size": sensitivity.training.hidden_size,
            "sensitivity_attention_heads": sensitivity.training.num_heads,
        },
        "optimization": {
            "epochs_max": settings.training.epochs,
            "batch_size": settings.training.batch_size,
            "learning_rate": settings.training.learning_rate,
            "weight_decay": settings.training.weight_decay,
            "loss": "UnweightedFocalLoss",
            "focal_gamma": settings.training.focal_gamma,
            "class_or_sample_weighting": "none",
            "selection_metric": settings.training.selection_metric,
            "early_stopping_patience": settings.training.early_stopping_patience,
        },
        "evaluation": {
            "outer_folds": settings.evaluation_protocol.outer_folds,
            "split_roles": ["training", "selection", "calibration", "test"],
            "patient_group_source": "analysis.v_patient_encounters.patient_id",
            "bootstrap_unit": "patient_id cluster",
            "bootstrap_replicates": settings.evaluation_protocol.bootstrap_replicates,
            "p_auc_fpr_limits": list(settings.evaluation_protocol.p_auc_fpr_limits),
            "risk_thresholds": list(settings.evaluation_protocol.risk_thresholds),
            "alert_budgets": list(settings.evaluation_protocol.alert_budgets),
            "holm_adjusted_paired_comparisons": True,
        },
        "prespecified_sequence": [
            "six-model single-seed five-fold primary run",
            "128d/8-head architecture sensitivity",
            "primary plus strongest deep comparator three-seed stability",
            "minimum ablation and early-warning evaluation",
        ],
        "split_manifest_payload_sha256": split_manifest["manifest_payload_sha256"],
        "dataset_fingerprint": dataset_fingerprint(settings),
        "vocab_sizes": {key: int(value) for key, value in vocab_sizes.items()},
        "source_sha256": source_hashes,
        "privacy": "aggregate counts, parameter counts and hashes only; no row identifiers",
        "training_performed": False,
        "performance_estimated": False,
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    output = settings.paths.manifests / "code10_experiment_contract.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[DILI-PLUS][Code-10] Experiment contract PASS: "
        f"{len(FORMAL_DEEP_MODEL_NAMES) + len(FORMAL_ML_MODELS)} formal models, "
        "128d/4-head primary, 128d/8-head sensitivity, AUPRC selection"
    )
    print(f"[DILI-PLUS][Code-10] Contract manifest: {output}")
    print(f"[DILI-PLUS][Code-10] Payload SHA256: {payload['payload_sha256']}")
    return 0


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(main())
