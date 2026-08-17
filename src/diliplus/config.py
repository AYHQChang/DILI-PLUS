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
    manifests: Path
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
            manifests=_resolve(root, values.get("manifests", "manifests")),
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
    weight_decay: float = 1e-5
    hidden_size: int = 128
    num_heads: int = 4
    dropout: float = 0.3
    diagnosis_modality_dropout_prob: float = 0.15
    focal_gamma: float = 2.0
    selection_metric: str = "AUPRC"
    early_stopping_patience: int = 7
    scheduler_patience: int = 3

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("training epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("training learning_rate must be positive and weight_decay non-negative")
        if self.hidden_size < 4 or self.num_heads < 1:
            raise ValueError(
                "training hidden_size must be at least 4 and num_heads positive"
            )
        if self.hidden_size % 2 != 0:
            raise ValueError(
                "training hidden_size must be even for all formal model families"
            )
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("training hidden_size must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("training dropout must be in [0, 1)")
        if not 0.0 <= self.diagnosis_modality_dropout_prob < 1.0:
            raise ValueError(
                "training diagnosis_modality_dropout_prob must be in [0, 1)"
            )
        if self.focal_gamma < 0:
            raise ValueError("training focal_gamma must be non-negative")
        if self.selection_metric != "AUPRC":
            raise ValueError("formal training selection_metric must be AUPRC")
        if self.early_stopping_patience < 1 or self.scheduler_patience < 1:
            raise ValueError("training patience values must be positive")


@dataclass(frozen=True)
class PredictionSettings:
    label_source: str = "deterministic_rebuild"
    baseline_threshold_u_l: float = 120.0
    gap_hours: float = 24.0
    pseudo_index_seed: int = 20260816
    audit_horizons_hours: tuple[float, ...] = (0.0, 12.0, 24.0, 48.0, 72.0)
    early_warning_horizons_hours: tuple[float, ...] = (24.0, 48.0, 72.0)

    def __post_init__(self) -> None:
        if self.label_source not in {"deterministic_rebuild", "legacy_frozen"}:
            raise ValueError(
                "prediction.label_source must be deterministic_rebuild or legacy_frozen"
            )
        if self.baseline_threshold_u_l <= 0:
            raise ValueError("prediction.baseline_threshold_u_l must be positive")
        if self.gap_hours < 0:
            raise ValueError("prediction.gap_hours must be non-negative")
        if not self.audit_horizons_hours:
            raise ValueError("prediction.audit_horizons_hours must not be empty")
        if any(hours < 0 for hours in self.audit_horizons_hours):
            raise ValueError("prediction audit horizons must be non-negative")
        if not self.early_warning_horizons_hours:
            raise ValueError("prediction.early_warning_horizons_hours must not be empty")
        if any(hours < self.gap_hours for hours in self.early_warning_horizons_hours):
            raise ValueError(
                "early-warning horizons cannot be earlier than the model-data "
                "prediction gap because later events are absent from the artifact"
            )
        if tuple(sorted(set(self.early_warning_horizons_hours))) != tuple(
            self.early_warning_horizons_hours
        ):
            raise ValueError("early-warning horizons must be unique and increasing")


@dataclass(frozen=True)
class ReproducibilitySettings:
    global_seed: int = 20260816
    split_seed: int = 20260816
    bootstrap_seed: int = 20260816
    figure_seed: int = 20260816
    deterministic_torch: bool = True
    dataloader_num_workers: int = 0
    pseudo_index_sensitivity_seeds: tuple[int, ...] = (
        20260816,
        20260817,
        20260818,
        20260819,
        20260820,
    )

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.global_seed,
                self.split_seed,
                self.bootstrap_seed,
                self.figure_seed,
            )
        ):
            raise ValueError("reproducibility seeds must be non-negative")
        if self.dataloader_num_workers < 0:
            raise ValueError("reproducibility.dataloader_num_workers must be non-negative")
        if not self.pseudo_index_sensitivity_seeds:
            raise ValueError("pseudo_index_sensitivity_seeds must not be empty")
        if len(set(self.pseudo_index_sensitivity_seeds)) != len(
            self.pseudo_index_sensitivity_seeds
        ):
            raise ValueError("pseudo_index_sensitivity_seeds must be unique")


@dataclass(frozen=True)
class EvaluationProtocolSettings:
    outer_folds: int = 5
    selection_fraction: float = 0.15
    calibration_fraction: float = 0.15
    split_search_attempts: int = 128
    bootstrap_replicates: int = 1000
    p_auc_fpr_limits: tuple[float, ...] = (0.05, 0.10, 0.20)
    risk_thresholds: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
    alert_budgets: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
    dca_min_threshold: float = 0.005
    dca_max_threshold: float = 0.05
    dca_step: float = 0.0025

    def __post_init__(self) -> None:
        if self.outer_folds < 2:
            raise ValueError("evaluation_protocol.outer_folds must be at least 2")
        for name, value in (
            ("selection_fraction", self.selection_fraction),
            ("calibration_fraction", self.calibration_fraction),
        ):
            if not 0.0 < value < 0.5:
                raise ValueError(f"evaluation_protocol.{name} must be between 0 and 0.5")
        if self.selection_fraction + self.calibration_fraction >= 0.5:
            raise ValueError(
                "selection_fraction + calibration_fraction must leave most outer-training "
                "data for parameter fitting"
            )
        if self.split_search_attempts < 1:
            raise ValueError("evaluation_protocol.split_search_attempts must be positive")
        if self.bootstrap_replicates < 100:
            raise ValueError("evaluation_protocol.bootstrap_replicates must be at least 100")
        for name, values in (
            ("p_auc_fpr_limits", self.p_auc_fpr_limits),
            ("risk_thresholds", self.risk_thresholds),
            ("alert_budgets", self.alert_budgets),
        ):
            if not values or any(not 0.0 < value < 1.0 for value in values):
                raise ValueError(f"evaluation_protocol.{name} must be within (0, 1)")
            if tuple(sorted(set(values))) != tuple(values):
                raise ValueError(f"evaluation_protocol.{name} must be unique and increasing")
        if not 0.0 < self.dca_min_threshold < self.dca_max_threshold < 1.0:
            raise ValueError("evaluation_protocol DCA bounds must satisfy 0 < min < max < 1")
        if self.dca_step <= 0:
            raise ValueError("evaluation_protocol.dca_step must be positive")


def _hours_tag(hours: float) -> str:
    numeric = float(hours)
    return str(int(numeric)) if numeric.is_integer() else str(numeric).replace(".", "p")


@dataclass(frozen=True)
class Settings:
    paths: ProjectPaths
    database_path: Path
    database_read_only: bool
    training: TrainingSettings
    prediction: PredictionSettings
    reproducibility: ReproducibilitySettings
    evaluation_protocol: EvaluationProtocolSettings
    config_path: Path
    config_sources: tuple[Path, ...]

    def __post_init__(self) -> None:
        if self.database_read_only is not True:
            raise ValueError("DILI-PLUS source DuckDB must remain read-only")

    @property
    def model_data_dir(self) -> Path:
        """Leakage-corrected model data, isolated from legacy onset-time artifacts."""
        return self.paths.data_cache / f"prediction_gap_{_hours_tag(self.prediction.gap_hours)}h"

    @property
    def prediction_audit_dir(self) -> Path:
        return self.paths.reports / f"p0_02_prediction_gap_{_hours_tag(self.prediction.gap_hours)}h"

    @property
    def diagnosis_audit_dir(self) -> Path:
        return self.paths.reports / f"p0_03_diagnosis_time_gap_{_hours_tag(self.prediction.gap_hours)}h"

    @property
    def reproducibility_audit_dir(self) -> Path:
        return self.paths.reports / "p0_04_reproducibility"

    @property
    def cohort_table1_dir(self) -> Path:
        """Local-only Code-08 cohort and Table 1 evidence."""
        return self.paths.reports / "p0_08_cohort_table1"

    @property
    def early_warning_audit_dir(self) -> Path:
        """Local-only Code-09 horizon-contract evidence."""
        return self.paths.reports / "p0_09_early_warning"

    @property
    def baseline_manifest_path(self) -> Path:
        return self.paths.manifests / "code00_code04_baseline.json"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config_with_extends(
    path: Path, seen: tuple[Path, ...] = ()
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Configuration extends cycle detected at {path}")
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    extends = raw.pop("extends", None)
    if extends is None:
        return raw, (path,)
    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base, sources = _read_config_with_extends(base_path, seen + (path,))
    return _deep_merge(base, raw), sources + (path,)


def load_settings(config_path: str | Path | None = None) -> Settings:
    resolved_config = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH
    raw, config_sources = _read_config_with_extends(resolved_config)

    configured_root = Path(raw.get("project_root", PACKAGE_PROJECT_ROOT))
    if not configured_root.is_absolute():
        configured_root = (resolved_config.parent / configured_root).resolve()

    database = raw.get("database", {})
    training = raw.get("training", {})
    prediction = raw.get("prediction", {})
    reproducibility = raw.get("reproducibility", {})
    evaluation_protocol = raw.get("evaluation_protocol", {})
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
            weight_decay=float(training.get("weight_decay", 1e-5)),
            hidden_size=int(training.get("hidden_size", 128)),
            num_heads=int(training.get("num_heads", 4)),
            dropout=float(training.get("dropout", 0.3)),
            diagnosis_modality_dropout_prob=float(
                training.get("diagnosis_modality_dropout_prob", 0.15)
            ),
            focal_gamma=float(training.get("focal_gamma", 2.0)),
            selection_metric=str(training.get("selection_metric", "AUPRC")),
            early_stopping_patience=int(
                training.get("early_stopping_patience", 7)
            ),
            scheduler_patience=int(training.get("scheduler_patience", 3)),
        ),
        prediction=PredictionSettings(
            label_source=str(
                prediction.get("label_source", "deterministic_rebuild")
            ),
            baseline_threshold_u_l=float(
                prediction.get("baseline_threshold_u_l", 120.0)
            ),
            gap_hours=float(prediction.get("gap_hours", 24.0)),
            pseudo_index_seed=int(prediction.get("pseudo_index_seed", 20260816)),
            audit_horizons_hours=tuple(
                float(value)
                for value in prediction.get(
                    "audit_horizons_hours", [0.0, 12.0, 24.0, 48.0, 72.0]
                )
            ),
            early_warning_horizons_hours=tuple(
                float(value)
                for value in prediction.get(
                    "early_warning_horizons_hours", [24.0, 48.0, 72.0]
                )
            ),
        ),
        reproducibility=ReproducibilitySettings(
            global_seed=int(reproducibility.get("global_seed", 20260816)),
            split_seed=int(reproducibility.get("split_seed", 20260816)),
            bootstrap_seed=int(reproducibility.get("bootstrap_seed", 20260816)),
            figure_seed=int(reproducibility.get("figure_seed", 20260816)),
            deterministic_torch=bool(
                reproducibility.get("deterministic_torch", True)
            ),
            dataloader_num_workers=int(
                reproducibility.get("dataloader_num_workers", 0)
            ),
            pseudo_index_sensitivity_seeds=tuple(
                int(value)
                for value in reproducibility.get(
                    "pseudo_index_sensitivity_seeds",
                    [20260816, 20260817, 20260818, 20260819, 20260820],
                )
            ),
        ),
        evaluation_protocol=EvaluationProtocolSettings(
            outer_folds=int(evaluation_protocol.get("outer_folds", 5)),
            selection_fraction=float(
                evaluation_protocol.get("selection_fraction", 0.15)
            ),
            calibration_fraction=float(
                evaluation_protocol.get("calibration_fraction", 0.15)
            ),
            split_search_attempts=int(
                evaluation_protocol.get("split_search_attempts", 128)
            ),
            bootstrap_replicates=int(
                evaluation_protocol.get("bootstrap_replicates", 1000)
            ),
            p_auc_fpr_limits=tuple(
                float(value)
                for value in evaluation_protocol.get(
                    "p_auc_fpr_limits", [0.05, 0.10, 0.20]
                )
            ),
            risk_thresholds=tuple(
                float(value)
                for value in evaluation_protocol.get(
                    "risk_thresholds", [0.005, 0.01, 0.02, 0.05]
                )
            ),
            alert_budgets=tuple(
                float(value)
                for value in evaluation_protocol.get(
                    "alert_budgets", [0.005, 0.01, 0.02, 0.05]
                )
            ),
            dca_min_threshold=float(
                evaluation_protocol.get("dca_min_threshold", 0.005)
            ),
            dca_max_threshold=float(
                evaluation_protocol.get("dca_max_threshold", 0.05)
            ),
            dca_step=float(evaluation_protocol.get("dca_step", 0.0025)),
        ),
        config_path=resolved_config,
        config_sources=config_sources,
    )
