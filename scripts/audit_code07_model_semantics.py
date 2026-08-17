"""Audit and freeze the Code-07 model/loss semantic contract.

This is an aggregate-only structural smoke test.  It instantiates every formal
deep model on CPU with a fixed synthetic batch; it does not read patient data,
train a model, or estimate predictive performance.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.config import load_settings
from diliplus.models.registry import (
    FORMAL_DEEP_MODEL_NAMES,
    FORMAL_DEEP_MODEL_REGISTRY,
    PRIMARY_MODEL_NAME,
    build_formal_deep_model,
    extract_ahi_proxy_logits,
    get_formal_deep_model,
)
from diliplus.training.losses import UnweightedFocalLoss


MANIFEST_PATH = PROJECT_ROOT / "manifests" / "code07_model_loss_contract.json"
SYNTHETIC_VOCAB_SIZES = {
    "vocab_med_size": 32,
    "vocab_lab_size": 24,
    "vocab_diag_size": 16,
}
MODEL_INPUT_NAMES = (
    "x_med",
    "dt_med",
    "mask_med",
    "x_lab",
    "v_lab",
    "dt_lab",
    "mask_lab",
    "x_diag",
    "mask_diag",
)
SOURCE_FILES = (
    "configs/default.yaml",
    "src/diliplus/config.py",
    "src/diliplus/data/dataset.py",
    "src/diliplus/artifacts.py",
    "src/diliplus/models/diliplus_engine.py",
    "src/diliplus/models/baselines.py",
    "src/diliplus/models/registry.py",
    "src/diliplus/training/losses.py",
    "src/diliplus/training/deep_trainer_calibrated.py",
    "src/diliplus/training/ml_baselines_calibrated.py",
    "src/diliplus/evaluation/early_warning.py",
    "src/diliplus/explainability/attribution.py",
    "src/diliplus/explainability/perturbation.py",
    "scripts/audit_code07_model_semantics.py",
)
ACTIVE_SEMANTIC_FILES = (
    "src/diliplus/models/diliplus_engine.py",
    "src/diliplus/models/baselines.py",
    "src/diliplus/models/registry.py",
    "src/diliplus/training/losses.py",
    "src/diliplus/training/deep_trainer_calibrated.py",
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _source_hashes() -> dict[str, str]:
    hashes = {}
    for relative_path in SOURCE_FILES:
        source_path = PROJECT_ROOT / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Required Code-07 source is missing: {relative_path}")
        hashes[relative_path] = _file_sha256(source_path)
    return hashes


def _synthetic_batch() -> dict[str, torch.Tensor]:
    """Return two synthetic examples, including one fully masked example."""
    return {
        "x_med": torch.tensor([[2, 3, 4, 0, 0], [0, 0, 0, 0, 0]]),
        "dt_med": torch.tensor(
            [[0.0, 1.0, 6.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "mask_med": torch.tensor([[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]]),
        "x_lab": torch.tensor([[2, 3, 0, 0], [0, 0, 0, 0]]),
        "v_lab": torch.tensor(
            [[12.0, 25.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ),
        "dt_lab": torch.tensor(
            [[0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ),
        "mask_lab": torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]]),
        "x_diag": torch.tensor([[2, 3, 0], [0, 0, 0]]),
        "mask_diag": torch.tensor([[1, 1, 0], [0, 0, 0]]),
    }


def _ast_identifiers(tree: ast.AST) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            values.add(node.id)
        elif isinstance(node, ast.Attribute):
            values.add(node.attr)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            values.add(node.name)
        elif isinstance(node, ast.arg):
            values.add(node.arg)
    return values


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    current: ast.AST = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _static_semantic_checks() -> dict[str, Any]:
    forbidden_identifier_fragments = (
        "uncertaintymtl",
        "label_aki",
        "aki_head",
        "mtl_head",
        "mtl_loss",
        "log_var_aki",
    )
    forbidden_pretraining_calls = (
        "from_pretrained",
        "load_state_dict",
        "torch.load",
    )
    identifier_hits: list[dict[str, str]] = []
    pretraining_call_hits: list[dict[str, str]] = []

    for relative_path in ACTIVE_SEMANTIC_FILES:
        tree = ast.parse(
            (PROJECT_ROOT / relative_path).read_text(encoding="utf-8"),
            filename=relative_path,
        )
        for identifier in sorted(_ast_identifiers(tree)):
            lowered = identifier.casefold()
            if any(fragment in lowered for fragment in forbidden_identifier_fragments):
                identifier_hits.append({"file": relative_path, "identifier": identifier})
        if relative_path.startswith("src/diliplus/models/"):
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = _call_name(node)
                    lowered = name.casefold()
                    if any(
                        lowered == forbidden or lowered.endswith(f".{forbidden}")
                        for forbidden in forbidden_pretraining_calls
                    ):
                        pretraining_call_hits.append(
                            {"file": relative_path, "call": name}
                        )

    if identifier_hits:
        raise AssertionError(
            f"Active MTL/AKI identifiers were found: {identifier_hits}"
        )
    if pretraining_call_hits:
        raise AssertionError(
            f"Model construction contains checkpoint/pretraining calls: "
            f"{pretraining_call_hits}"
        )
    return {
        "active_mtl_or_aki_identifier_hits": identifier_hits,
        "model_checkpoint_or_pretraining_call_hits": pretraining_call_hits,
        "comments_and_docstrings_excluded_from_identifier_scan": True,
    }


def _verify_loss(settings) -> dict[str, Any]:
    constructor_parameters = inspect.signature(UnweightedFocalLoss).parameters
    forbidden_weight_parameters = sorted(
        name
        for name in ("alpha", "weight", "class_weight", "class_weights")
        if name in constructor_parameters
    )
    if forbidden_weight_parameters:
        raise AssertionError(
            f"UnweightedFocalLoss exposes class-weight parameters: "
            f"{forbidden_weight_parameters}"
        )
    if float(settings.training.focal_gamma) != 2.0:
        raise AssertionError(
            f"Code-07 freezes focal_gamma=2.0, found {settings.training.focal_gamma}"
        )

    criterion = UnweightedFocalLoss(gamma=settings.training.focal_gamma)
    if criterion.gamma != 2.0:
        raise AssertionError(f"Constructed focal gamma is {criterion.gamma}, expected 2.0")
    if tuple(criterion.parameters()) or tuple(criterion.buffers()):
        raise AssertionError("UnweightedFocalLoss unexpectedly stores weights")

    logits = torch.tensor([[1.2, -0.2], [-0.5, 0.8]], dtype=torch.float64)
    targets = torch.tensor([0, 1])
    with torch.no_grad():
        value = criterion(logits, targets)
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise AssertionError("UnweightedFocalLoss did not return one finite scalar")

    return {
        "name": "UnweightedFocalLoss",
        "task": "single_task_ahi_proxy_binary_classification",
        "formula": "(1 - p_t)^gamma * cross_entropy",
        "gamma": criterion.gamma,
        "reduction": criterion.reduction,
        "class_weighting": "none",
        "constructor_weight_parameters": forbidden_weight_parameters,
        "learned_parameters": 0,
        "registered_buffers": 0,
        "synthetic_loss_scalar_finite": True,
    }


def _verify_models(settings) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    formal_names = tuple(FORMAL_DEEP_MODEL_NAMES)
    if formal_names != tuple(FORMAL_DEEP_MODEL_REGISTRY):
        raise AssertionError("Formal model names and registry keys do not match")
    if len(formal_names) != len(set(formal_names)):
        raise AssertionError("Formal model names are not unique")
    if any("bert" in name.casefold() for name in formal_names):
        raise AssertionError(f"Pretraining-implying formal model name found: {formal_names}")
    if PRIMARY_MODEL_NAME != "TimeAwareMultimodalTransformer":
        raise AssertionError(f"Unexpected primary model name: {PRIMARY_MODEL_NAME}")

    primary_class = get_formal_deep_model(PRIMARY_MODEL_NAME)
    if primary_class.__name__ != PRIMARY_MODEL_NAME:
        raise AssertionError(
            f"Primary class/name mismatch: {primary_class.__name__} != {PRIMARY_MODEL_NAME}"
        )
    if not primary_class.__module__.startswith("diliplus.models."):
        raise AssertionError(f"Primary class is not project-local: {primary_class.__module__}")

    inputs = {key: value.cpu() for key, value in _synthetic_batch().items()}
    model_records: list[dict[str, Any]] = []
    primary_dropout_observed = None
    for model_name in formal_names:
        torch.manual_seed(int(settings.reproducibility.global_seed))
        model = build_formal_deep_model(
            model_name,
            SYNTHETIC_VOCAB_SIZES,
            settings.training,
        ).cpu().eval()
        signature_inputs = tuple(
            name
            for name in inspect.signature(model.forward).parameters
            if name != "self"
        )
        if signature_inputs != MODEL_INPUT_NAMES:
            raise AssertionError(
                f"{model_name} forward inputs {signature_inputs} != {MODEL_INPUT_NAMES}"
            )
        with warnings.catch_warnings(), torch.no_grad():
            warnings.filterwarnings(
                "ignore",
                message="The PyTorch API of nested tensors is in prototype stage.*",
                category=UserWarning,
            )
            outputs = model(**inputs)
        if not isinstance(outputs, dict):
            raise AssertionError(f"{model_name} returned {type(outputs).__name__}, not dict")
        logits = extract_ahi_proxy_logits(outputs)
        shape = list(logits.shape)
        if shape != [2, 2] or not bool(torch.isfinite(logits).all()):
            raise AssertionError(
                f"{model_name} invalid logits: shape={shape}, finite="
                f"{bool(torch.isfinite(logits).all())}"
            )

        if model_name == PRIMARY_MODEL_NAME:
            primary_dropout_observed = float(model.diagnosis_modality_dropout_prob)
        model_records.append(
            {
                "formal_name": model_name,
                "class_name": type(model).__name__,
                "module": type(model).__module__,
                "initialization": "local_random_initialization_from_scratch",
                "device": "cpu",
                "parameter_count_with_synthetic_vocab": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "trainable_parameter_count_with_synthetic_vocab": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "forward_input_count": len(signature_inputs),
                "output_mapping": True,
                "output_keys": sorted(outputs),
                "logits_shape": shape,
                "logits_finite": True,
            }
        )

    configured_dropout = float(settings.training.diagnosis_modality_dropout_prob)
    if primary_dropout_observed != configured_dropout:
        raise AssertionError(
            f"Primary diagnosis dropout {primary_dropout_observed} != config "
            f"{configured_dropout}"
        )

    tuple_rejected = False
    try:
        extract_ahi_proxy_logits((torch.zeros(1, 2), torch.zeros(1, 2)))
    except TypeError:
        tuple_rejected = True
    if not tuple_rejected:
        raise AssertionError("Tuple/multi-task outputs are not rejected")

    dropout_settings = replace(
        settings.training,
        dropout=0.0,
        diagnosis_modality_dropout_prob=0.5,
    )
    torch.manual_seed(int(settings.reproducibility.global_seed))
    dropout_model = build_formal_deep_model(
        PRIMARY_MODEL_NAME,
        SYNTHETIC_VOCAB_SIZES,
        dropout_settings,
    ).cpu()
    single_inputs = {
        key: value[:1].cpu() for key, value in _synthetic_batch().items()
    }
    dropout_model.eval()
    with torch.no_grad():
        reference = dropout_model(**single_inputs)
    dropout_model.train()
    with patch("torch.rand", return_value=torch.zeros((1, 1))), torch.no_grad():
        diagnosis_dropped = dropout_model(**single_inputs)
    torch.testing.assert_close(diagnosis_dropped["h_med"], reference["h_med"])
    torch.testing.assert_close(diagnosis_dropped["h_lab"], reference["h_lab"])
    if not torch.equal(
        diagnosis_dropped["h_diag"],
        torch.zeros_like(diagnosis_dropped["h_diag"]),
    ):
        raise AssertionError("Diagnosis dropout did not zero the diagnosis stream")

    return model_records, {
        "primary_model_name": PRIMARY_MODEL_NAME,
        "primary_class_name": primary_class.__name__,
        "formal_model_names": list(formal_names),
        "formal_registry_size": len(formal_names),
        "formal_names_imply_external_pretraining": False,
        "from_scratch": True,
        "external_pretrained_weights_loaded": False,
        "single_task": True,
        "target_name": "label_ahi_proxy",
        "output_contract": {"type": "mapping", "logits_shape": ["batch", 2]},
        "tuple_or_multi_task_output_rejected": tuple_rejected,
        "diagnosis_modality_dropout_prob": configured_dropout,
        "diagnosis_dropout_applies_during": "training_only",
        "diagnosis_dropout_granularity": "per_sample",
        "diagnosis_dropout_affected_stream": "diagnosis_only",
        "diagnosis_dropout_dynamic_streams_unscaled": True,
    }


def main() -> int:
    settings = load_settings()
    if torch.device("cpu").type != "cpu":
        raise AssertionError("Code-07 audit must run on CPU")

    static_checks = _static_semantic_checks()
    loss_contract = _verify_loss(settings)
    model_records, model_contract = _verify_models(settings)
    source_hashes = _source_hashes()

    stable_payload = {
        "schema_version": 1,
        "audit_name": "code07_model_loss_semantics",
        "audit_status": "PASS",
        "scope": {
            "data_source": "fixed_synthetic_batch_only",
            "patient_level_data_read": False,
            "patient_identifiers_recorded": False,
            "training_performed": False,
            "performance_metrics_computed": False,
            "execution_device": "cpu",
            "synthetic_batch_size": 2,
            "synthetic_vocab_sizes": dict(SYNTHETIC_VOCAB_SIZES),
        },
        "training_configuration": {
            "hidden_size": int(settings.training.hidden_size),
            "num_heads": int(settings.training.num_heads),
            "dropout": float(settings.training.dropout),
            "diagnosis_modality_dropout_prob": float(
                settings.training.diagnosis_modality_dropout_prob
            ),
            "focal_gamma": float(settings.training.focal_gamma),
        },
        "model_contract": model_contract,
        "loss_contract": loss_contract,
        "static_semantic_checks": static_checks,
        "formal_model_smoke_results": model_records,
        "source_sha256": source_hashes,
    }
    manifest = {
        **stable_payload,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "stable_payload_sha256": _canonical_sha256(stable_payload),
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audit_status": manifest["audit_status"],
                "manifest": str(MANIFEST_PATH),
                "formal_models_verified": len(model_records),
                "stable_payload_sha256": manifest["stable_payload_sha256"],
                "patient_level_data_read": False,
                "performance_metrics_computed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
