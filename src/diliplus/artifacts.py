"""Versioned model artifacts shared by training, evaluation and explanation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import torch

from diliplus.calibration import ProbabilityMode, probabilities_from_logits
from diliplus.reproducibility import settings_manifest


ARTIFACT_SCHEMA = "diliplus.model_artifact.v1"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ArtifactContractError(RuntimeError):
    pass


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(str(run_id)):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'"
        )
    return str(run_id)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def dataset_fingerprint(settings) -> dict[str, Any]:
    candidates = (
        settings.model_data_dir / "data_lineage.json",
        settings.model_data_dir / "02_dili_labels_censored.parquet",
        settings.model_data_dir / "03_dili_dual_stream_tensors.parquet",
        settings.model_data_dir / "03b_diag_tensors.parquet",
        settings.paths.vocab / "vocab_polypharmacy.json",
        settings.paths.vocab / "vocab_diagnosis.json",
    )
    missing = [str(path) for path in candidates if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot fingerprint missing model inputs: {missing}")
    files = {
        path.resolve().relative_to(settings.paths.root).as_posix(): {
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in candidates
    }
    return {"files": files, "payload_sha256": _canonical_sha256(files)}


def config_snapshot(settings, training_arguments: dict[str, Any]) -> dict[str, Any]:
    protocol = settings.evaluation_protocol
    package_dir = Path(__file__).resolve().parent
    return {
        "config_path": settings.config_path.relative_to(settings.paths.root).as_posix(),
        "config_sha256": file_sha256(settings.config_path),
        "config_sources": {
            path.relative_to(settings.paths.root).as_posix(): file_sha256(path)
            for path in settings.config_sources
        },
        "prediction_gap_hours": settings.prediction.gap_hours,
        "label_source": settings.prediction.label_source,
        "reproducibility": settings_manifest(settings.reproducibility),
        "evaluation_protocol": {
            "outer_folds": protocol.outer_folds,
            "selection_fraction": protocol.selection_fraction,
            "calibration_fraction": protocol.calibration_fraction,
            "split_search_attempts": protocol.split_search_attempts,
            "bootstrap_replicates": protocol.bootstrap_replicates,
            "p_auc_fpr_limits": list(protocol.p_auc_fpr_limits),
            "risk_thresholds": list(protocol.risk_thresholds),
            "alert_budgets": list(protocol.alert_budgets),
            "dca_threshold_range": [
                protocol.dca_min_threshold,
                protocol.dca_max_threshold,
                protocol.dca_step,
            ],
        },
        "contract_implementation_sha256": {
            "artifacts.py": file_sha256(Path(__file__)),
            "calibration.py": file_sha256(package_dir / "calibration.py"),
            "splits.py": file_sha256(package_dir / "splits.py"),
            "models/diliplus_engine.py": file_sha256(
                package_dir / "models" / "diliplus_engine.py"
            ),
            "models/baselines.py": file_sha256(
                package_dir / "models" / "baselines.py"
            ),
            "models/registry.py": file_sha256(
                package_dir / "models" / "registry.py"
            ),
            "training/losses.py": file_sha256(
                package_dir / "training" / "losses.py"
            ),
            "training/deep_trainer_calibrated.py": file_sha256(
                package_dir / "training" / "deep_trainer_calibrated.py"
            ),
            "training/ml_baselines_calibrated.py": file_sha256(
                package_dir / "training" / "ml_baselines_calibrated.py"
            ),
            "data/dataset.py": file_sha256(
                package_dir / "data" / "dataset.py"
            ),
        },
        "training_arguments": training_arguments,
    }


def run_checkpoint_dir(settings, run_id: str) -> Path:
    return settings.paths.checkpoints / "runs" / validate_run_id(run_id)


def run_report_dir(settings, run_id: str) -> Path:
    return settings.paths.reports / "runs" / validate_run_id(run_id)


def deep_artifact_path(settings, run_id: str, model_name: str, fold: int) -> Path:
    return run_checkpoint_dir(settings, run_id) / model_name / f"fold_{int(fold):02d}.pt"


def sklearn_artifact_path(settings, run_id: str, model_name: str, fold: int) -> Path:
    return run_checkpoint_dir(settings, run_id) / model_name / f"fold_{int(fold):02d}.joblib"


def _validate_common_metadata(metadata: dict[str, Any]) -> None:
    required = {
        "schema",
        "artifact_type",
        "run_id",
        "model_name",
        "fold",
        "selected_epoch",
        "temperature",
        "split",
        "dataset_fingerprint",
        "configuration",
        "probability_contract",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ArtifactContractError(f"Artifact metadata is missing keys: {missing}")
    if metadata["schema"] != ARTIFACT_SCHEMA:
        raise ArtifactContractError(f"Unsupported artifact schema: {metadata['schema']}")
    temperature = float(metadata["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ArtifactContractError("Artifact temperature must be positive and finite")
    split = metadata["split"]
    indices = split.get("indices", {})
    if set(indices) != {"training", "selection", "calibration", "test"}:
        raise ArtifactContractError("Artifact must contain all four split roles")
    role_sets = {role: set(map(int, values)) for role, values in indices.items()}
    for left, left_values in role_sets.items():
        for right, right_values in role_sets.items():
            if left < right and left_values & right_values:
                raise ArtifactContractError(f"Artifact split overlap: {left} vs {right}")
    recorded_hash = metadata.get("metadata_payload_sha256")
    if recorded_hash is not None:
        unhashed = {
            key: value
            for key, value in metadata.items()
            if key != "metadata_payload_sha256"
        }
        if recorded_hash != _canonical_sha256(unhashed):
            raise ArtifactContractError("Artifact metadata checksum mismatch")


def build_artifact_metadata(
    *,
    artifact_type: str,
    run_id: str,
    model_name: str,
    fold: int,
    selected_epoch: int,
    temperature: float,
    split_payload: dict[str, Any],
    dataset: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_type": artifact_type,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": validate_run_id(run_id),
        "model_name": str(model_name),
        "fold": int(fold),
        "selected_epoch": int(selected_epoch),
        "temperature": float(temperature),
        "split": split_payload,
        "dataset_fingerprint": dataset,
        "configuration": configuration,
        "probability_contract": {
            "raw": "softmax(model_logits)",
            "calibrated": "softmax(model_logits / temperature)",
            "test_logits_single_pass": True,
            "raw_and_calibrated_share_test_logits": True,
        },
    }
    _validate_common_metadata(metadata)
    metadata["metadata_payload_sha256"] = _canonical_sha256(metadata)
    return metadata


def _write_metadata_sidecar(path: Path, metadata: dict[str, Any]) -> None:
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    sidecar.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_deep_artifact(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> Path:
    _validate_common_metadata(metadata)
    if metadata["artifact_type"] != "torch":
        raise ArtifactContractError("Deep artifacts must use artifact_type='torch'")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "model_state_dict": model.state_dict()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    _write_metadata_sidecar(path, metadata)
    return path


def load_deep_artifact(
    path: Path,
    model: torch.nn.Module,
    *,
    map_location: str | torch.device = "cpu",
    expected_run_id: str | None = None,
    expected_model_name: str | None = None,
    expected_fold: int | None = None,
    expected_dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "metadata" not in payload or "model_state_dict" not in payload:
        raise ArtifactContractError("File is not a versioned DILI-PLUS deep artifact")
    metadata = payload["metadata"]
    _validate_common_metadata(metadata)
    _validate_expected_metadata(
        metadata,
        expected_run_id=expected_run_id,
        expected_model_name=expected_model_name,
        expected_fold=expected_fold,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return metadata


def _validate_expected_metadata(
    metadata: dict[str, Any],
    *,
    expected_run_id: str | None = None,
    expected_model_name: str | None = None,
    expected_fold: int | None = None,
    expected_dataset_fingerprint: str | None = None,
) -> None:
    expected = {
        "run_id": expected_run_id,
        "model_name": expected_model_name,
        "fold": expected_fold,
    }
    for key, value in expected.items():
        if value is not None and metadata[key] != value:
            raise ArtifactContractError(
                f"Artifact {key} mismatch: expected {value!r}, found {metadata[key]!r}"
            )
    if expected_dataset_fingerprint is not None:
        actual = metadata["dataset_fingerprint"]["payload_sha256"]
        if actual != expected_dataset_fingerprint:
            raise ArtifactContractError(
                "Artifact dataset fingerprint does not match current model inputs"
            )


def save_sklearn_artifact(
    path: Path,
    estimator: Any,
    vectorizers: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    _validate_common_metadata(metadata)
    if metadata["artifact_type"] != "sklearn":
        raise ArtifactContractError("Sklearn artifacts must use artifact_type='sklearn'")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "estimator": estimator,
        "vectorizers": vectorizers,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary)
    temporary.replace(path)
    _write_metadata_sidecar(path, metadata)
    return path


def load_sklearn_artifact(
    path: Path,
    *,
    expected_run_id: str | None = None,
    expected_model_name: str | None = None,
    expected_fold: int | None = None,
    expected_dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or not {"metadata", "estimator", "vectorizers"}.issubset(payload):
        raise ArtifactContractError("File is not a versioned DILI-PLUS sklearn artifact")
    _validate_common_metadata(payload["metadata"])
    _validate_expected_metadata(
        payload["metadata"],
        expected_run_id=expected_run_id,
        expected_model_name=expected_model_name,
        expected_fold=expected_fold,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
    )
    return payload


def artifact_probabilities(
    logits,
    metadata: dict[str, Any],
    mode: ProbabilityMode,
):
    if metadata.get("schema") != ARTIFACT_SCHEMA or "temperature" not in metadata:
        raise ArtifactContractError("Invalid model artifact metadata")
    return probabilities_from_logits(logits, float(metadata["temperature"]), mode)
