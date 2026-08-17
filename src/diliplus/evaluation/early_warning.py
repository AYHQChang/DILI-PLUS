"""Code-09 strict early-warning evaluation.

The model data are already censored ``gap_hours`` before each encounter's index
time. An effective horizon therefore must be at least that base gap. Additional
cutoff is performed from retained event timestamps, never reconstructed from
inter-event deltas. Medication, laboratory and diagnosis tensors are all
physically zeroed together with their masks.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
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
from diliplus.models.registry import (
    FORMAL_DEEP_MODEL_NAMES,
    build_formal_deep_model,
    extract_ahi_proxy_logits,
)
from diliplus.reproducibility import DEFAULT_SEED, derive_seed, seed_everything


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


def _metric_summary(y_true, y_prob, *, bootstrap_seed=DEFAULT_SEED, n_bootstraps=1000):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(y_true) == 0 or set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Early-warning metrics require nonempty binary outcomes")
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    rng = np.random.default_rng(bootstrap_seed)
    boot_auroc, boot_auprc = [], []
    for _ in range(int(n_bootstraps)):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        boot_auroc.append(roc_auc_score(y_true[indices], y_prob[indices]))
        boot_auprc.append(average_precision_score(y_true[indices], y_prob[indices]))
    return {
        "AUROC": float(auroc),
        "AUROC_95CI_Lower": float(np.percentile(boot_auroc, 2.5)),
        "AUROC_95CI_Upper": float(np.percentile(boot_auroc, 97.5)),
        "AUPRC": float(auprc),
        "AUPRC_95CI_Lower": float(np.percentile(boot_auprc, 2.5)),
        "AUPRC_95CI_Upper": float(np.percentile(boot_auprc, 97.5)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
    }


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
    results = []
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
            fold_payloads.append((model, metadata, test_indices))
        flattened = [index for _, _, indices in fold_payloads for index in indices]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(range(len(dataset))):
            raise RuntimeError(
                f"{model_name} outer-test folds must cover each encounter exactly once"
            )
        if reference_test_indices is None:
            reference_test_indices = [tuple(item[2]) for item in fold_payloads]
        elif reference_test_indices != [tuple(item[2]) for item in fold_payloads]:
            raise RuntimeError("Formal models do not share identical outer-test splits")

        for horizon in horizons:
            pooled_labels, pooled_predictions = [], []
            for model, metadata, test_indices in fold_payloads:
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
                pooled_labels.extend(labels)
                pooled_predictions.extend(predictions)
            metrics = _metric_summary(
                pooled_labels,
                pooled_predictions,
                bootstrap_seed=derive_seed(
                    settings.reproducibility.bootstrap_seed,
                    model_name,
                    horizon,
                    probability_mode,
                ),
            )
            metrics.update(
                {
                    "Run_ID": run_id,
                    "Probability_Mode": probability_mode,
                    "Model_Architecture": model_name,
                    "Base_Prediction_Gap_Hours": settings.prediction.gap_hours,
                    "Effective_Horizon_Hours_Before_Index": horizon,
                    "N": len(pooled_labels),
                    "Positive_N": int(np.sum(pooled_labels)),
                    "Prevalence": float(np.mean(pooled_labels)),
                }
            )
            results.append(metrics)
            print(
                f"[DILI-PLUS][Code-09] {model_name} {horizon:g} h: "
                f"AUROC={metrics['AUROC']:.4f}, AUPRC={metrics['AUPRC']:.4f}"
            )

    report_path = (
        settings.paths.reports
        / "runs"
        / run_id
        / f"06a_Early_Warning_Strict_{probability_mode}.csv"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(report_path, index=False)
    print(f"[DILI-PLUS][Code-09] Strict early-warning results: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
