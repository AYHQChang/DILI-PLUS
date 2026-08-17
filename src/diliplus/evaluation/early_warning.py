"""Code-09 strict early-warning evaluation.

The model data are already censored ``gap_hours`` before each encounter's index
time. An effective horizon therefore must be at least that base gap. Additional
cutoff is performed from retained event timestamps, never reconstructed from
inter-event deltas. Medication, laboratory and diagnosis tensors are all
physically zeroed together with their masks.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from diliplus.artifacts import (
    artifact_probabilities,
    dataset_fingerprint,
    deep_artifact_path,
    file_sha256,
    load_deep_artifact,
)
from diliplus.config import load_settings
from diliplus.data.dataset import DILIPlusDataset, load_vocab_sizes
from diliplus.evaluation.metrics import (
    add_holm_adjustment,
    calibration_bins,
    cluster_bootstrap_intervals,
    compute_binary_metrics,
    decision_curve,
    paired_cluster_bootstrap,
)
from diliplus.models.registry import (
    FORMAL_DEEP_MODEL_NAMES,
    build_formal_deep_model,
    extract_ahi_proxy_logits,
)
from diliplus.reproducibility import derive_seed, seed_everything


MODEL_INPUT_KEYS = (
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
MODALITY_CONTRACT = {
    "medication": {
        "mask": "mask_med",
        "age": "age_med_hours",
        "zero": ("x_med", "dt_med", "age_med_hours"),
    },
    "laboratory": {
        "mask": "mask_lab",
        "age": "age_lab_hours",
        "zero": ("x_lab", "v_lab", "dt_lab", "age_lab_hours"),
    },
    "diagnosis": {
        "mask": "mask_diag",
        "age": "age_diag_hours",
        "zero": ("x_diag", "age_diag_hours"),
    },
}


def validate_effective_horizons(horizons, base_gap_hours: float) -> tuple[float, ...]:
    values = tuple(float(value) for value in horizons)
    if not values:
        raise ValueError("At least one early-warning horizon is required")
    if tuple(sorted(set(values))) != values:
        raise ValueError("Early-warning horizons must be unique and increasing")
    invalid = [value for value in values if value < float(base_gap_hours)]
    if invalid:
        raise ValueError(
            f"Effective horizons {invalid} are below the {base_gap_hours:g} h base "
            "prediction gap; their later events are absent and cannot be reconstructed"
        )
    return values


def apply_effective_horizon_cutoff(
    batch: dict[str, torch.Tensor],
    effective_horizon_hours: float,
    base_gap_hours: float,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Return a cloned, physically truncated batch and aggregate retained counts."""
    if float(effective_horizon_hours) < float(base_gap_hours):
        raise ValueError("effective horizon cannot be below the model-data base gap")
    additional_hours = float(effective_horizon_hours) - float(base_gap_hours)
    output = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    audit = {}
    for modality, contract in MODALITY_CONTRACT.items():
        mask_key = contract["mask"]
        age_key = contract["age"]
        if mask_key not in output or age_key not in output:
            raise KeyError(
                f"Early-warning batch lacks {mask_key!r} or {age_key!r} for {modality}"
            )
        original_mask = output[mask_key].bool()
        ages = output[age_key]
        if torch.any(original_mask & (~torch.isfinite(ages) | (ages <= 0))):
            raise ValueError(
                f"Active {modality} events must have finite positive age-to-prediction"
            )
        retained = original_mask & (ages > additional_hours)

        # Events are sorted chronologically, so retained events must be a prefix.
        invalid_reentry = retained & ((~retained).cumsum(dim=1) > 0)
        if torch.any(invalid_reentry):
            raise ValueError(f"{modality} event times are not chronologically ordered")

        output[mask_key] = retained.to(output[mask_key].dtype)
        for key in contract["zero"]:
            output[key] = output[key] * retained.to(output[key].dtype)
        lengths = retained.sum(dim=1)
        audit[f"{modality}_events_retained"] = int(lengths.sum().item())
        audit[f"{modality}_missing_encounters"] = int(lengths.eq(0).sum().item())
    audit["encounters"] = int(next(iter(batch.values())).shape[0])
    return output, audit


def _dca_thresholds(settings):
    protocol = settings.evaluation_protocol
    return np.arange(
        protocol.dca_min_threshold,
        protocol.dca_max_threshold + protocol.dca_step / 2.0,
        protocol.dca_step,
    )


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def build_horizon_availability(
    dataset: DILIPlusDataset,
    horizons,
    *,
    base_gap_hours: float,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Describe real cohort/event availability without loading any model artifact."""
    horizons = validate_effective_horizons(horizons, base_gap_hours)
    accumulators = {
        horizon: {
            "labels": [],
            **{f"{modality}_lengths": [] for modality in MODALITY_CONTRACT},
        }
        for horizon in horizons
    }
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        observed_gaps = batch["base_prediction_gap_hours"].numpy()
        if not np.allclose(observed_gaps, base_gap_hours):
            raise ValueError("Model artifact mixes prediction-gap values")
        labels = batch["label_ahi_proxy"].numpy()
        for horizon in horizons:
            truncated, _ = apply_effective_horizon_cutoff(
                batch, horizon, base_gap_hours
            )
            accumulators[horizon]["labels"].extend(labels.tolist())
            for modality, contract in MODALITY_CONTRACT.items():
                lengths = truncated[contract["mask"]].sum(dim=1).numpy()
                accumulators[horizon][f"{modality}_lengths"].extend(lengths.tolist())

    rows = []
    for horizon in horizons:
        labels = np.asarray(accumulators[horizon]["labels"], dtype=np.int64)
        row = {
            "base_prediction_gap_hours": float(base_gap_hours),
            "effective_horizon_hours_before_index": float(horizon),
            "additional_cutoff_hours_before_prediction": float(horizon - base_gap_hours),
            "encounters": int(len(labels)),
            "positive_encounters": int(labels.sum()),
            "negative_encounters": int(len(labels) - labels.sum()),
            "prevalence_pct": float(100.0 * labels.mean()),
        }
        for modality in MODALITY_CONTRACT:
            lengths = np.asarray(
                accumulators[horizon][f"{modality}_lengths"], dtype=np.int64
            )
            row.update(
                {
                    f"{modality}_events_retained": int(lengths.sum()),
                    f"{modality}_missing_encounters": int((lengths == 0).sum()),
                    f"{modality}_length_median": float(np.median(lengths)),
                    f"{modality}_length_q1": float(np.percentile(lengths, 25)),
                    f"{modality}_length_q3": float(np.percentile(lengths, 75)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_cutoff_audit(base_gap_hours: float) -> dict:
    batch = {
        "x_med": torch.tensor([[2, 3, 4]]),
        "dt_med": torch.tensor([[0.0, 5.0, 5.0]]),
        "mask_med": torch.tensor([[1, 1, 1]]),
        "age_med_hours": torch.tensor([[30.0, 20.0, 5.0]]),
        "x_lab": torch.tensor([[5, 6, 0]]),
        "v_lab": torch.tensor([[10.0, 20.0, 0.0]]),
        "dt_lab": torch.tensor([[0.0, 8.0, 0.0]]),
        "mask_lab": torch.tensor([[1, 1, 0]]),
        "age_lab_hours": torch.tensor([[26.0, 8.0, 0.0]]),
        "x_diag": torch.tensor([[7, 8, 0]]),
        "mask_diag": torch.tensor([[1, 1, 0]]),
        "age_diag_hours": torch.tensor([[40.0, 10.0, 0.0]]),
    }
    horizon = float(base_gap_hours) + 24.0
    truncated, audit = apply_effective_horizon_cutoff(batch, horizon, base_gap_hours)
    expected_masks = {
        "mask_med": [[1, 0, 0]],
        "mask_lab": [[1, 0, 0]],
        "mask_diag": [[1, 0, 0]],
    }
    for key, expected in expected_masks.items():
        if truncated[key].tolist() != expected:
            raise AssertionError(f"Synthetic strict-cutoff audit failed for {key}")
    for key in ("x_med", "dt_med", "x_lab", "v_lab", "dt_lab", "x_diag"):
        mask_key = "mask_med" if "med" in key else ("mask_lab" if "lab" in key else "mask_diag")
        if torch.any(truncated[key][truncated[mask_key] == 0] != 0):
            raise AssertionError(f"Synthetic physical-zero audit failed for {key}")
    return {
        "status": "PASS",
        "effective_horizon_hours": horizon,
        "boundary_rule": "event_age_to_prediction > effective_horizon - base_gap",
        **audit,
    }


def audit_early_warning_contract(settings=None) -> Path:
    """Run Code-09 data/cutoff audits without training or estimating performance."""
    settings = settings or load_settings()
    seed_everything(settings.reproducibility)
    output_dir = settings.early_warning_audit_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = DILIPlusDataset(
        settings.model_data_dir,
        settings.paths.vocab,
        include_temporal_metadata=True,
    )
    horizons = validate_effective_horizons(
        settings.prediction.early_warning_horizons_hours,
        settings.prediction.gap_hours,
    )
    availability = build_horizon_availability(
        dataset,
        horizons,
        base_gap_hours=settings.prediction.gap_hours,
        batch_size=settings.training.batch_size,
    )
    csv_path = output_dir / "horizon_availability.csv"
    availability.to_csv(csv_path, index=False)
    synthetic = _synthetic_cutoff_audit(settings.prediction.gap_hours)

    manifest_path = settings.paths.manifests / "code09_early_warning_contract.json"
    manifest = {
        "contract": "code09_strict_early_warning_v1",
        "status": "PASS",
        "base_prediction_gap_hours": settings.prediction.gap_hours,
        "effective_horizons_hours": list(horizons),
        "unsupported_horizons_below_base_gap": (
            "cannot be reconstructed from already-censored model artifacts"
        ),
        "cutoff_rule": "event_time < index_time - effective_horizon",
        "implemented_as": (
            "event_age_to_prediction > effective_horizon - base_prediction_gap"
        ),
        "modalities_physically_zeroed": {
            modality: list(contract["zero"])
            for modality, contract in MODALITY_CONTRACT.items()
        },
        "horizon_availability": availability.to_dict(orient="records"),
        "synthetic_cutoff_audit": synthetic,
        "input_sha256": {
            path.name: file_sha256(path)
            for path in (
                settings.model_data_dir / "03_dili_dual_stream_tensors.parquet",
                settings.model_data_dir / "03b_diag_tensors.parquet",
            )
        },
        "output_sha256": {csv_path.name: file_sha256(csv_path)},
        "performance_estimated": False,
        "training_performed": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DILI-PLUS][Code-09] Strict temporal cutoff audit PASS")
    print(availability.to_string(index=False))
    print(f"[DILI-PLUS][Code-09] Aggregate manifest: {manifest_path}")
    return manifest_path


@torch.no_grad()
def _predict_horizon_single_fold(
    model,
    dataloader,
    device,
    *,
    effective_horizon_hours,
    base_gap_hours,
    artifact_metadata,
    probability_mode,
):
    model.eval()
    predictions, labels = [], []
    for batch in dataloader:
        truncated, _ = apply_effective_horizon_cutoff(
            batch, effective_horizon_hours, base_gap_hours
        )
        inputs = {key: truncated[key].to(device) for key in MODEL_INPUT_KEYS}
        logits = extract_ahi_proxy_logits(model(**inputs))
        probabilities = artifact_probabilities(
            logits.detach().cpu(), artifact_metadata, probability_mode
        )
        predictions.extend(np.asarray(probabilities).tolist())
        labels.extend(batch["label_ahi_proxy"].numpy().tolist())
    return labels, predictions


def _require_complete_artifacts(settings, run_id: str) -> None:
    missing = [
        str(deep_artifact_path(settings, run_id, model_name, fold))
        for model_name in FORMAL_DEEP_MODEL_NAMES
        for fold in range(1, settings.evaluation_protocol.outer_folds + 1)
        if not deep_artifact_path(settings, run_id, model_name, fold).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Early-warning evaluation requires every formal model/fold artifact; "
            f"missing {len(missing)} files, first={missing[0]}"
        )


def main(settings=None, run_id=None, probability_mode="calibrated"):
    settings = settings or load_settings()
    if not run_id:
        raise ValueError("run_id is required to resolve versioned model artifacts")
    if probability_mode not in ("raw", "calibrated"):
        raise ValueError("probability_mode must be 'raw' or 'calibrated'")
    seed_everything(settings.reproducibility)
    _require_complete_artifacts(settings, run_id)
    audit_early_warning_contract(settings)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DILIPlusDataset(
        settings.model_data_dir,
        settings.paths.vocab,
        include_temporal_metadata=True,
    )
    vocab_sizes = load_vocab_sizes(settings.paths.vocab)
    fingerprint = dataset_fingerprint(settings)["payload_sha256"]
    horizons = validate_effective_horizons(
        settings.prediction.early_warning_horizons_hours,
        settings.prediction.gap_hours,
    )
    protocol = settings.evaluation_protocol
    thresholds = _dca_thresholds(settings)
    result_root = settings.paths.reports / "runs" / run_id / "early_warning"
    prediction_root = result_root / "predictions"
    metrics_root = result_root / "metrics"
    prediction_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    encounter_ids = dataset.data["encounter_id"].astype(str).to_numpy()
    if "patient_id" not in dataset.data or dataset.data["patient_id"].isna().any():
        raise KeyError("Early-warning evaluation requires non-null source patient_id")
    patient_ids = dataset.data["patient_id"].astype(str).to_numpy()
    labels_array = np.asarray(dataset.labels, dtype=np.int64)
    results = []
    interval_frames = []
    calibration_frames = []
    dca_frames = []
    prediction_frames = {}
    prediction_paths = []
    reference_test_indices = None

    for model_name in FORMAL_DEEP_MODEL_NAMES:
        fold_payloads = []
        for fold in range(1, settings.evaluation_protocol.outer_folds + 1):
            model = build_formal_deep_model(
                model_name, vocab_sizes, settings.training
            ).to(device)
            metadata = load_deep_artifact(
                deep_artifact_path(settings, run_id, model_name, fold),
                model,
                map_location=device,
                expected_run_id=run_id,
                expected_model_name=model_name,
                expected_fold=fold,
                expected_dataset_fingerprint=fingerprint,
            )
            test_indices = list(map(int, metadata["split"]["indices"]["test"]))
            training_indices = list(
                map(int, metadata["split"]["indices"]["training"])
            )
            reference_prevalence = float(labels_array[training_indices].mean())
            fold_payloads.append(
                (model, metadata, test_indices, reference_prevalence, fold)
            )
        flattened = [index for _, _, indices, _, _ in fold_payloads for index in indices]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(range(len(dataset))):
            raise RuntimeError(
                f"{model_name} outer-test folds must cover each encounter exactly once"
            )
        if reference_test_indices is None:
            reference_test_indices = [tuple(item[2]) for item in fold_payloads]
        elif reference_test_indices != [tuple(item[2]) for item in fold_payloads]:
            raise RuntimeError("Formal models do not share identical outer-test splits")

        for horizon in horizons:
            fold_frames = []
            for model, metadata, test_indices, reference_prevalence, fold in fold_payloads:
                loader = DataLoader(
                    Subset(dataset, test_indices),
                    batch_size=settings.training.batch_size,
                    shuffle=False,
                    num_workers=settings.reproducibility.dataloader_num_workers,
                )
                labels, predictions = _predict_horizon_single_fold(
                    model,
                    loader,
                    device,
                    effective_horizon_hours=horizon,
                    base_gap_hours=settings.prediction.gap_hours,
                    artifact_metadata=metadata,
                    probability_mode=probability_mode,
                )
                indices = np.asarray(test_indices, dtype=np.int64)
                fold_frames.append(
                    pd.DataFrame(
                        {
                            "dataset_index": indices,
                            "encounter_id": encounter_ids[indices],
                            "patient_id": patient_ids[indices],
                            "y_true": np.asarray(labels, dtype=np.int64),
                            "y_prob": np.asarray(predictions, dtype=np.float64),
                            "fold": int(fold),
                            "reference_prevalence": reference_prevalence,
                        }
                    )
                )
            frame = (
                pd.concat(fold_frames, ignore_index=True)
                .sort_values("dataset_index", kind="stable")
                .reset_index(drop=True)
            )
            if frame["dataset_index"].duplicated().any() or len(frame) != len(dataset):
                raise RuntimeError(
                    f"{model_name}/{horizon:g} h predictions do not cover the cohort once"
                )
            prediction_frames[(model_name, float(horizon))] = frame
            prediction_dir = prediction_root / model_name
            prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = prediction_dir / f"horizon_{int(horizon):03d}h.csv"
            frame.to_csv(prediction_path, index=False)
            prediction_paths.append(prediction_path)

            metrics = compute_binary_metrics(
                frame["y_true"],
                frame["y_prob"],
                reference_prevalence=frame["reference_prevalence"],
                p_auc_fpr_limits=protocol.p_auc_fpr_limits,
                risk_thresholds=protocol.risk_thresholds,
                alert_budgets=protocol.alert_budgets,
                dca_thresholds=thresholds,
            )
            metrics.update(
                {
                    "Run_ID": run_id,
                    "Probability_Mode": probability_mode,
                    "Model_Architecture": model_name,
                    "Base_Prediction_Gap_Hours": settings.prediction.gap_hours,
                    "Effective_Horizon_Hours_Before_Index": horizon,
                }
            )
            results.append(metrics)

            intervals = cluster_bootstrap_intervals(
                frame["y_true"],
                frame["y_prob"],
                frame["patient_id"],
                n_replicates=protocol.bootstrap_replicates,
                seed=derive_seed(
                    settings.reproducibility.bootstrap_seed,
                    run_id,
                    model_name,
                    horizon,
                    probability_mode,
                    "early_warning_ci",
                ),
            )
            intervals.insert(0, "Effective_Horizon_Hours_Before_Index", horizon)
            intervals.insert(0, "Model_Architecture", model_name)
            intervals.insert(0, "Probability_Mode", probability_mode)
            intervals.insert(0, "Run_ID", run_id)
            interval_frames.append(intervals)

            bins = calibration_bins(frame["y_true"], frame["y_prob"])
            bins.insert(0, "Effective_Horizon_Hours_Before_Index", horizon)
            bins.insert(0, "Model_Architecture", model_name)
            bins.insert(0, "Probability_Mode", probability_mode)
            bins.insert(0, "Run_ID", run_id)
            calibration_frames.append(bins)

            curve = decision_curve(frame["y_true"], frame["y_prob"], thresholds)
            curve.insert(0, "Effective_Horizon_Hours_Before_Index", horizon)
            curve.insert(0, "Model_Architecture", model_name)
            curve.insert(0, "Probability_Mode", probability_mode)
            curve.insert(0, "Run_ID", run_id)
            dca_frames.append(curve)
            print(
                f"[DILI-PLUS][Code-09] {model_name} {horizon:g} h: "
                f"AUROC={metrics['AUROC']:.4f}, AUPRC={metrics['AUPRC']:.4f}"
            )

    primary_name = "TimeAwareMultimodalTransformer"
    primary_comparison_frames = []
    for horizon in horizons:
        primary = prediction_frames[(primary_name, float(horizon))]
        horizon_comparisons = []
        for comparator_name in FORMAL_DEEP_MODEL_NAMES:
            if comparator_name == primary_name:
                continue
            comparator = prediction_frames[(comparator_name, float(horizon))]
            if not primary[
                ["dataset_index", "encounter_id", "patient_id", "y_true"]
            ].equals(
                comparator[
                    ["dataset_index", "encounter_id", "patient_id", "y_true"]
                ]
            ):
                raise RuntimeError("Early-warning model memberships differ")
            comparison = paired_cluster_bootstrap(
                primary["y_true"],
                primary["y_prob"],
                comparator["y_prob"],
                primary["patient_id"],
                n_replicates=protocol.bootstrap_replicates,
                seed=derive_seed(
                    settings.reproducibility.bootstrap_seed,
                    run_id,
                    horizon,
                    comparator_name,
                    probability_mode,
                    "early_warning_model_pair",
                ),
            )
            comparison.insert(0, "Comparator", comparator_name)
            comparison.insert(0, "Primary", primary_name)
            comparison.insert(0, "Effective_Horizon_Hours_Before_Index", horizon)
            comparison.insert(0, "Probability_Mode", probability_mode)
            comparison.insert(0, "Run_ID", run_id)
            horizon_comparisons.append(comparison)
        primary_comparison_frames.append(
            add_holm_adjustment(pd.concat(horizon_comparisons, ignore_index=True))
        )

    horizon_degradation_frames = []
    base_horizon = float(horizons[0])
    for model_name in FORMAL_DEEP_MODEL_NAMES:
        reference = prediction_frames[(model_name, base_horizon)]
        model_comparisons = []
        for horizon in horizons[1:]:
            earlier = prediction_frames[(model_name, float(horizon))]
            comparison = paired_cluster_bootstrap(
                earlier["y_true"],
                earlier["y_prob"],
                reference["y_prob"],
                earlier["patient_id"],
                n_replicates=protocol.bootstrap_replicates,
                seed=derive_seed(
                    settings.reproducibility.bootstrap_seed,
                    run_id,
                    model_name,
                    horizon,
                    probability_mode,
                    "early_warning_horizon_pair",
                ),
            ).rename(
                columns={
                    "primary_estimate": "earlier_horizon_estimate",
                    "comparator_estimate": "reference_24h_estimate",
                    "delta_primary_minus_comparator": "delta_earlier_minus_24h",
                }
            )
            comparison.insert(0, "Reference_Horizon_Hours", base_horizon)
            comparison.insert(0, "Earlier_Horizon_Hours", horizon)
            comparison.insert(0, "Model_Architecture", model_name)
            comparison.insert(0, "Probability_Mode", probability_mode)
            comparison.insert(0, "Run_ID", run_id)
            model_comparisons.append(comparison)
        horizon_degradation_frames.append(
            add_holm_adjustment(pd.concat(model_comparisons, ignore_index=True))
        )

    report_path = (
        settings.paths.reports
        / "runs"
        / run_id
        / f"06a_Early_Warning_Strict_{probability_mode}.csv"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pooled_metrics = pd.DataFrame(results)
    bootstrap_intervals = pd.concat(interval_frames, ignore_index=True)
    primary_comparisons = pd.concat(primary_comparison_frames, ignore_index=True)
    horizon_degradation = pd.concat(horizon_degradation_frames, ignore_index=True)
    metric_outputs = {
        "pooled_metrics.csv": pooled_metrics,
        "bootstrap_intervals.csv": bootstrap_intervals,
        "primary_vs_comparators.csv": primary_comparisons,
        "horizon_degradation.csv": horizon_degradation,
        "calibration_bins.csv": pd.concat(calibration_frames, ignore_index=True),
        "dca_curves.csv": pd.concat(dca_frames, ignore_index=True),
    }
    for filename, frame in metric_outputs.items():
        frame.to_csv(metrics_root / filename, index=False)
    pooled_metrics.to_csv(report_path, index=False)

    project_root = Path(settings.paths.root)
    tracked_manifest_path = settings.paths.manifests / "code09_early_warning_performance.json"
    main_manifest_path = settings.paths.manifests / "code10_formal_run.json"
    availability_manifest_path = settings.paths.manifests / "code09_early_warning_contract.json"
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "contract": "code09_strict_early_warning_performance_v1",
        "run_id": run_id,
        "probability_mode": probability_mode,
        "models": list(FORMAL_DEEP_MODEL_NAMES),
        "base_prediction_gap_hours": settings.prediction.gap_hours,
        "effective_horizons_hours": list(horizons),
        "outer_folds": protocol.outer_folds,
        "oof_rows_per_model_horizon": len(dataset),
        "positives": int(labels_array.sum()),
        "patient_clusters": int(pd.Series(patient_ids).nunique()),
        "bootstrap_replicates": protocol.bootstrap_replicates,
        "patient_group_source": "analysis.v_patient_encounters.patient_id",
        "evaluation_git_commit": _git_commit(project_root),
        "training_manifest_sha256": file_sha256(main_manifest_path),
        "availability_manifest_sha256": file_sha256(availability_manifest_path),
        "dataset_fingerprint": dataset_fingerprint(settings),
        "metric_files": {
            filename: {
                "sha256": file_sha256(metrics_root / filename),
                "bytes": (metrics_root / filename).stat().st_size,
            }
            for filename in metric_outputs
        },
        "prediction_file_hashes": {
            path.relative_to(project_root).as_posix(): file_sha256(path)
            for path in prediction_paths
        },
        "pooled_metrics": pooled_metrics.to_dict(orient="records"),
        "bootstrap_intervals": bootstrap_intervals.to_dict(orient="records"),
        "primary_vs_comparators": primary_comparisons.to_dict(orient="records"),
        "horizon_degradation": horizon_degradation.to_dict(orient="records"),
        "interpretation_guardrails": [
            "All horizons reuse the same fold checkpoint and fitted temperature; no horizon-specific retraining or recalibration.",
            "The 24-hour model artifact cannot reconstruct 0-hour or 12-hour inputs.",
            "Earlier-horizon performance is internal retrospective prediction evidence, not proof of clinical benefit or causality.",
        ],
        "privacy": "tracked manifest contains aggregate statistics and file hashes only",
    }
    payload = _json_safe(payload)
    local_manifest_path = result_root / "run_manifest.json"
    encoded_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    local_manifest_path.write_text(encoded_payload, encoding="utf-8")
    tracked_manifest_path.write_text(encoded_payload, encoding="utf-8")
    print(f"[DILI-PLUS][Code-09] Strict early-warning results: {report_path}")
    print(f"[DILI-PLUS][Code-09] Aggregate manifest: {tracked_manifest_path}")
    return report_path


if __name__ == "__main__":
    main()
