"""P0 statistical refinement from frozen Code-10 OOF predictions.

This module never fits or updates a predictive model.  It reloads the
preserved outer-fold predictions, verifies common membership, and repeats the
prespecified patient-cluster uncertainty analysis at higher Monte Carlo
resolution.  Raw discrimination and calibrated probability assessment remain
separate analysis modes because fold-specific temperatures can alter ranking
when folds are pooled even though they are monotone within each fold.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from diliplus.config import load_settings
from diliplus.evaluation.metrics import (
    add_holm_adjustment,
    cluster_bootstrap_intervals,
    paired_cluster_bootstrap,
)
from diliplus.reproducibility import derive_seed
from diliplus.reporting.formal_assets import FORMAL_MODELS, formal_sources, sha256


RUN_ID = "code13_p0_statistical_refinement"
SOURCE_RUN_ID = "code10_formal_128d4h_seed0"
PRIMARY_MODEL = "TimeAwareMultimodalTransformer"
MODES = ("raw", "calibrated")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816
REQUIRED_PREDICTION_COLUMNS = (
    "dataset_index",
    "encounter_id",
    "patient_id",
    "y_true",
    "y_prob_raw",
    "y_prob_calibrated",
    "fold",
)


@dataclass(frozen=True)
class BootstrapTask:
    kind: str
    model: str
    mode: str
    prediction_root: str
    comparator: str | None = None


def _prediction_files(prediction_root: Path, model: str) -> list[Path]:
    paths = sorted((prediction_root / model).glob("fold_*.csv"))
    if len(paths) != 5:
        raise ValueError(f"Expected five OOF prediction files for {model}, found {len(paths)}")
    return paths


def load_model_oof(prediction_root: Path, model: str) -> pd.DataFrame:
    paths = _prediction_files(prediction_root, model)
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = sorted(set(REQUIRED_PREDICTION_COLUMNS) - set(frame.columns))
        if missing:
            raise KeyError(f"{path} is missing required columns: {missing}")
        frames.append(frame.loc[:, REQUIRED_PREDICTION_COLUMNS])
    result = pd.concat(frames, ignore_index=True).sort_values("dataset_index").reset_index(drop=True)
    if result["dataset_index"].duplicated().any():
        raise ValueError(f"Duplicate dataset_index values in {model} OOF predictions")
    return result


def validate_common_membership(frames: dict[str, pd.DataFrame]) -> None:
    reference_name = next(iter(frames))
    reference = frames[reference_name]
    columns = ("dataset_index", "encounter_id", "patient_id", "y_true", "fold")
    for model, frame in frames.items():
        if len(frame) != len(reference):
            raise ValueError(f"OOF row count differs for {reference_name} and {model}")
        for column in columns:
            if not np.array_equal(reference[column].to_numpy(), frame[column].to_numpy()):
                raise ValueError(f"OOF membership mismatch in {column}: {reference_name} vs {model}")


def fold_discrimination(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, frame in frames.items():
        for fold, fold_frame in frame.groupby("fold", sort=True):
            for mode in MODES:
                probabilities = fold_frame[f"y_prob_{mode}"].to_numpy(dtype=float)
                labels = fold_frame["y_true"].to_numpy(dtype=int)
                rows.append(
                    {
                        "Run_ID": RUN_ID,
                        "Source_Run_ID": SOURCE_RUN_ID,
                        "Model_Architecture": model,
                        "Probability_Mode": mode,
                        "Fold": int(fold),
                        "N": int(len(fold_frame)),
                        "Positive": int(labels.sum()),
                        "AUROC": float(roc_auc_score(labels, probabilities)),
                        "AUPRC": float(average_precision_score(labels, probabilities)),
                    }
                )
    return pd.DataFrame(rows)


def _seed(task: BootstrapTask) -> int:
    parts = [RUN_ID, task.model, task.mode, task.kind]
    if task.comparator is not None:
        parts.append(task.comparator)
    return derive_seed(BOOTSTRAP_SEED, *parts)


def _run_bootstrap_task(task: BootstrapTask) -> pd.DataFrame:
    root = Path(task.prediction_root)
    primary = load_model_oof(root, task.model)
    seed = _seed(task)
    if task.kind == "interval":
        result = cluster_bootstrap_intervals(
            primary["y_true"],
            primary[f"y_prob_{task.mode}"],
            primary["patient_id"],
            n_replicates=BOOTSTRAP_REPLICATES,
            seed=seed,
        )
        result.insert(0, "Probability_Mode", task.mode)
        result.insert(0, "Model_Architecture", task.model)
    elif task.kind == "paired":
        if task.comparator is None:
            raise ValueError("Paired bootstrap task requires a comparator")
        comparator = load_model_oof(root, task.comparator)
        validate_common_membership({task.model: primary, task.comparator: comparator})
        result = paired_cluster_bootstrap(
            primary["y_true"],
            primary[f"y_prob_{task.mode}"],
            comparator[f"y_prob_{task.mode}"],
            primary["patient_id"],
            n_replicates=BOOTSTRAP_REPLICATES,
            seed=seed,
        )
        result.insert(0, "Comparator", task.comparator)
        result.insert(0, "Primary", task.model)
        result.insert(0, "Probability_Mode", task.mode)
    else:
        raise ValueError(f"Unknown bootstrap task kind: {task.kind}")
    result.insert(0, "Bootstrap_Seed", seed)
    result.insert(0, "Source_Run_ID", SOURCE_RUN_ID)
    result.insert(0, "Run_ID", RUN_ID)
    return result


def _file_contract(path: Path) -> dict[str, object]:
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8"
    ).strip()


def _metric_outputs(settings) -> dict[str, Path]:
    output_dir = settings.paths.reports / "runs" / RUN_ID / "metrics"
    return {
        "bootstrap_intervals_10000": output_dir / "bootstrap_intervals_10000.csv",
        "paired_comparisons_10000": output_dir / "paired_comparisons_10000.csv",
        "fold_discrimination": output_dir / "fold_discrimination.csv",
    }


def _write_manifest(settings, frames: dict[str, pd.DataFrame], outputs: dict[str, Path]) -> Path:
    for path in outputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    prediction_root = settings.paths.reports / "runs" / SOURCE_RUN_ID / "predictions"
    prediction_paths = [
        path
        for model in FORMAL_MODELS
        for path in _prediction_files(prediction_root, model)
    ]
    source_files = {
        path.relative_to(settings.paths.root).as_posix(): _file_contract(path)
        for path in prediction_paths
    }
    implementation_paths = (
        settings.paths.root / "src" / "diliplus" / "evaluation" / "metrics.py",
        settings.paths.root / "src" / "diliplus" / "evaluation" / "p0_statistical_refinement.py",
        settings.paths.root / "pipelines" / "07_run_p0_statistical_refinement.py",
    )
    reference = frames[FORMAL_MODELS[0]]
    manifest_path = settings.paths.manifests / "code13_p0_statistical_refinement.json"
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "run_id": RUN_ID,
        "analysis_kind": "aggregate-only statistical refinement from frozen OOF predictions",
        "source_run_id": SOURCE_RUN_ID,
        "analysis_git_commit": _git("rev-parse", "HEAD", cwd=settings.paths.root),
        "git_dirty_at_manifest_refresh": bool(
            _git("status", "--porcelain", cwd=settings.paths.root)
        ),
        "models": list(FORMAL_MODELS),
        "primary_model": PRIMARY_MODEL,
        "probability_modes": list(MODES),
        "oof_rows_per_model": int(len(reference)),
        "events": int(reference["y_true"].sum()),
        "patient_clusters": int(reference["patient_id"].astype(str).nunique()),
        "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
        "bootstrap_seed_base": BOOTSTRAP_SEED,
        "interval_seed_contract": "derived independently by model and probability mode",
        "paired_seed_contract": "derived independently by comparator and probability mode",
        "confidence_interval": "2.5th and 97.5th percentiles of valid patient-cluster replicates",
        "paired_p_value": "two-sided empirical tail probability with +1 numerator/denominator correction",
        "multiplicity": {
            "method": "Holm",
            "families": [
                "raw: 5 comparators x 4 core metrics (20 hypotheses)",
                "calibrated: 5 comparators x 4 core metrics (20 hypotheses)",
            ],
        },
        "interpretation_contract": {
            "raw": "primary pooled and fold-wise discrimination",
            "calibrated": "probability agreement, threshold summaries, and operational alert analyses",
            "conditional_uncertainty": "bootstrap conditions on the fitted fold models, selected hyperparameters, and temperature estimates",
        },
        "training_performed": False,
        "model_selection_performed": False,
        "source_prediction_files": source_files,
        "analysis_implementation_sha256": {
            path.relative_to(settings.paths.root).as_posix(): sha256(path)
            for path in implementation_paths
        },
        "source_formal_manifest": _file_contract(
            settings.paths.manifests / "code10_formal_run.json"
        ),
        "outputs": {path.name: _file_contract(path) for path in outputs.values()},
    }
    stable_payload = dict(payload)
    stable_payload.pop("generated_utc")
    payload["stable_payload_sha256"] = hashlib.sha256(
        json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] P0 manifest: {manifest_path}")
    return manifest_path


def refresh_p0_manifest(settings=None) -> Path:
    """Refresh hashes/lineage without repeating the 10,000 bootstrap samples."""
    settings = settings or load_settings()
    sources = formal_sources(settings)
    if sources["run_id"] != SOURCE_RUN_ID:
        raise ValueError(f"Expected frozen source run {SOURCE_RUN_ID}, found {sources['run_id']}")
    prediction_root = settings.paths.reports / "runs" / SOURCE_RUN_ID / "predictions"
    frames = {model: load_model_oof(prediction_root, model) for model in FORMAL_MODELS}
    validate_common_membership(frames)
    return _write_manifest(settings, frames, _metric_outputs(settings))


def run_p0_statistical_refinement(settings=None, *, max_workers: int = 8) -> dict[str, Path]:
    settings = settings or load_settings()
    sources = formal_sources(settings)
    if sources["run_id"] != SOURCE_RUN_ID:
        raise ValueError(f"Expected frozen source run {SOURCE_RUN_ID}, found {sources['run_id']}")
    source_run_dir = settings.paths.reports / "runs" / SOURCE_RUN_ID
    prediction_root = source_run_dir / "predictions"
    frames = {model: load_model_oof(prediction_root, model) for model in FORMAL_MODELS}
    validate_common_membership(frames)

    tasks = [
        BootstrapTask("interval", model, mode, str(prediction_root))
        for model in FORMAL_MODELS
        for mode in MODES
    ]
    tasks.extend(
        BootstrapTask("paired", PRIMARY_MODEL, mode, str(prediction_root), comparator)
        for mode in MODES
        for comparator in FORMAL_MODELS
        if comparator != PRIMARY_MODEL
    )
    completed: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {executor.submit(_run_bootstrap_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            completed.append(result)
            print(
                f"[PASS] {task.kind}: {task.model} vs {task.comparator or '-'} "
                f"({task.mode}; {BOOTSTRAP_REPLICATES:,} replicates)"
            )

    interval_frames = [frame for frame in completed if "Model_Architecture" in frame.columns]
    paired_frames = [frame for frame in completed if "Primary" in frame.columns]
    intervals = pd.concat(interval_frames, ignore_index=True).sort_values(
        ["Probability_Mode", "Model_Architecture", "metric"]
    )
    paired_by_mode = []
    paired_all = pd.concat(paired_frames, ignore_index=True)
    for mode in MODES:
        family = paired_all[paired_all["Probability_Mode"] == mode].copy()
        family = add_holm_adjustment(family)
        family["Holm_Family"] = f"{mode}: 5 comparators x 4 core metrics (20 hypotheses)"
        family["Monte_Carlo_P_Correction"] = "+1 numerator and denominator correction"
        paired_by_mode.append(family)
    paired = pd.concat(paired_by_mode, ignore_index=True).sort_values(
        ["Probability_Mode", "metric", "Comparator"]
    )
    folds = fold_discrimination(frames).sort_values(
        ["Probability_Mode", "Model_Architecture", "Fold"]
    )

    output_dir = settings.paths.reports / "runs" / RUN_ID / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _metric_outputs(settings)
    intervals.to_csv(outputs["bootstrap_intervals_10000"], index=False)
    paired.to_csv(outputs["paired_comparisons_10000"], index=False)
    folds.to_csv(outputs["fold_discrimination"], index=False)

    manifest_path = _write_manifest(settings, frames, outputs)
    outputs["manifest"] = manifest_path
    return outputs
