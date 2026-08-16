"""Central, immutable project configuration and legacy-compatible paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_PROJECT_ROOT / "configs" / "default.yaml"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_cache: Path
    vocab: Path
    checkpoints: Path
    reports: Path
    figures: Path
    saved_models: Path
    tensors: Path
    drug_mapping: Path
    drug_mapping_template: Path

    @classmethod
    def from_mapping(cls, root: Path, values: Mapping[str, Any]) -> "ProjectPaths":
        root = root.resolve()
        return cls(
            root=root,
            data_cache=_resolve(root, values.get("data_cache", "data_cache")),
            vocab=_resolve(root, values.get("vocab", "vocab")),
            checkpoints=_resolve(root, values.get("checkpoints", "checkpoints")),
            reports=_resolve(root, values.get("reports", "reports")),
            figures=_resolve(root, values.get("figures", "figures")),
            saved_models=_resolve(root, values.get("saved_models", "saved_models")),
            tensors=_resolve(root, values.get("tensors", "tensors")),
            drug_mapping=_resolve(
                root, values.get("drug_mapping", "mapping_completed_drugs_checkpoint.csv")
            ),
            drug_mapping_template=_resolve(
                root, values.get("drug_mapping_template", "mapping_template_drugs.csv")
            ),
        )


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-4


@dataclass(frozen=True)
class Settings:
    paths: ProjectPaths
    database_path: Path
    database_read_only: bool
    training: TrainingSettings
    config_path: Path

    def __post_init__(self) -> None:
        if self.database_read_only is not True:
            raise ValueError("DILI-PLUS source DuckDB must remain read-only")


def load_settings(config_path: str | Path | None = None) -> Settings:
    resolved_config = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH
    with resolved_config.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    configured_root = Path(raw.get("project_root", PACKAGE_PROJECT_ROOT))
    if not configured_root.is_absolute():
        configured_root = (resolved_config.parent / configured_root).resolve()

    database = raw.get("database", {})
    training = raw.get("training", {})
    read_only = database.get("read_only", True)
    if read_only is not True:
        raise ValueError("configs/default.yaml may not enable DuckDB writes")

    return Settings(
        paths=ProjectPaths.from_mapping(configured_root, raw.get("paths", {})),
        database_path=_resolve(
            configured_root,
            database.get("path", "D:/MedicalAI_Work/duck/medical.duckdb"),
        ),
        database_read_only=True,
        training=TrainingSettings(
            epochs=int(training.get("epochs", 50)),
            batch_size=int(training.get("batch_size", 256)),
            learning_rate=float(training.get("learning_rate", 1e-4)),
        ),
        config_path=resolved_config,
    )

