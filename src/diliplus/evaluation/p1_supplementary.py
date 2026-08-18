"""Post-hoc P1 reporting audits for the formal Code-10 experiment.

This module never changes the six-model formal benchmark.  It consumes the
frozen out-of-fold predictions, builds one lightweight observation-process
baseline on the identical grouped folds, and produces aggregate calibration
and subgroup audit files.  Patient identifiers are used only for clustered
resampling and are not written to tracked aggregate outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diliplus.calibration import (
    fit_temperature,
    probabilities_from_logits,
    probabilities_to_logits,
)
from diliplus.config import load_settings
from diliplus.evaluation.metrics import (
    calibration_bins,
    cluster_bootstrap_intervals,
    compute_binary_metrics,
)
from diliplus.reporting.formal_assets import FORMAL_MODELS, formal_sources
from diliplus.reporting.p1_contract import AUDIT_MODELS, PROCESS_MODEL, RUN_ID
from diliplus.reporting.table1 import (
    LABEL,
    MODEL_FILE,
    DIAGNOSIS_FILE,
    _add_demographics,
    _checked_left_join,
    _load_cohort,
    _load_diagnoses,
    _query_encounter_dimension,
    _query_patient_profile,
    _verify_source_schema,
)
from diliplus.database import connect_source_database
from diliplus.reproducibility import derive_seed
from diliplus.splits import build_nested_grouped_splits


PROCESS_FEATURES = (
    "log_observation_hours",
    "log_medication_event_count",
    "log_laboratory_event_count",
    "log_diagnosis_event_count",
    "log_medication_events_per_day",
    "log_laboratory_events_per_day",
    "log_diagnosis_events_per_day",
    "laboratory_available",
    "diagnosis_available",
)


def _require_columns(frame: pd.DataFrame, required, source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise KeyError(f"{source} is missing required columns: {missing}")


def _formal_feature_frame(settings) -> pd.DataFrame:
    """Rebuild the Code-08 encounter-grain features without exporting rows."""
    cohort = _load_cohort(settings.model_data_dir / MODEL_FILE)
    cohort.insert(0, "dataset_index", np.arange(len(cohort), dtype=np.int64))
    diagnoses = _load_diagnoses(settings.model_data_dir / DIAGNOSIS_FILE)
    cohort, _ = _checked_left_join(
        cohort,
        diagnoses,
        stage="p1_formal_cohort_to_time_bounded_diagnoses",
        matched_column="diagnosis_event_count",
    )

    conn = connect_source_database(settings)
    try:
        _verify_source_schema(conn)
        encounters = _query_encounter_dimension(conn, cohort)
        cohort, _ = _checked_left_join(
            cohort,
            encounters,
            stage="p1_formal_cohort_to_encounter_dimension",
            matched_column="patient_id",
        )
        profiles = _query_patient_profile(conn, encounters)
        cohort = _add_demographics(cohort, profiles)
    finally:
        conn.close()

    if len(cohort) != 44631 or cohort["encounter_id"].nunique() != len(cohort):
        raise RuntimeError("P1 feature joins changed the formal 44,631-encounter grain")
    if not np.array_equal(cohort["dataset_index"].to_numpy(), np.arange(len(cohort))):
        raise RuntimeError("P1 feature joins changed the model-artifact row order")
    if cohort["patient_id"].isna().any():
        raise RuntimeError("P1 formal cohort contains missing source patient groups")

    duration_days = cohort["observation_hours"] / 24.0
    counts = {
        "medication": cohort["medication_event_count"].astype(float),
        "laboratory": cohort["laboratory_event_count"].astype(float),
        "diagnosis": cohort["diagnosis_event_count"].fillna(0).astype(float),
    }
    cohort["log_observation_hours"] = np.log1p(cohort["observation_hours"])
    for modality, count in counts.items():
        cohort[f"log_{modality}_event_count"] = np.log1p(count)
        cohort[f"log_{modality}_events_per_day"] = np.log1p(count / duration_days)
    cohort["laboratory_available"] = counts["laboratory"].gt(0).astype(float)
    cohort["diagnosis_available"] = counts["diagnosis"].gt(0).astype(float)
    if not np.isfinite(cohort[list(PROCESS_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("Observation-process features must be finite")
    return cohort


def _formal_prediction_frame(settings, cohort: pd.DataFrame, model: str) -> pd.DataFrame:
    sources = formal_sources(settings)
    path = (
        settings.paths.reports
        / "runs"
        / str(sources["run_id"])
        / "predictions"
        / model
        / "oof_predictions.parquet"
    )
    frame = pd.read_parquet(path)
    _require_columns(
        frame,
        ("dataset_index", "encounter_id", "patient_id", "y_true", "y_prob_calibrated"),
        str(path),
    )
    frame = frame.sort_values("dataset_index", kind="mergesort").reset_index(drop=True)
    expected_index = np.arange(len(cohort), dtype=np.int64)
    if not np.array_equal(frame["dataset_index"].to_numpy(dtype=np.int64), expected_index):
        raise RuntimeError(f"{model} OOF predictions do not cover each dataset row once")
    if not np.array_equal(frame["encounter_id"].astype(str), cohort["encounter_id"].astype(str)):
        raise RuntimeError(f"{model} OOF encounter alignment failed")
    if not np.array_equal(frame["patient_id"].astype(str), cohort["patient_id"].astype(str)):
        raise RuntimeError(f"{model} OOF patient-group alignment failed")
    if not np.array_equal(frame["y_true"].to_numpy(dtype=int), cohort[LABEL].to_numpy(dtype=int)):
        raise RuntimeError(f"{model} OOF label alignment failed")
    return frame


def _metric_kwargs(settings, reference_prevalence):
    protocol = settings.evaluation_protocol
    thresholds = np.arange(
        protocol.dca_min_threshold,
        protocol.dca_max_threshold + protocol.dca_step / 2.0,
        protocol.dca_step,
    )
    return {
        "reference_prevalence": reference_prevalence,
        "p_auc_fpr_limits": protocol.p_auc_fpr_limits,
        "risk_thresholds": protocol.risk_thresholds,
        "alert_budgets": protocol.alert_budgets,
        "dca_thresholds": thresholds,
    }


def fit_observation_process_baseline(settings, cohort: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Fit a no-content process-intensity logistic baseline on the formal folds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions" / PROCESS_MODEL
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metric_dir = output_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    x = cohort[list(PROCESS_FEATURES)].to_numpy(dtype=float)
    y = cohort[LABEL].to_numpy(dtype=np.int64)
    encounter_ids = cohort["encounter_id"].astype(str).to_numpy()
    patient_ids = cohort["patient_id"].astype(str).to_numpy()
    folds = build_nested_grouped_splits(
        encounter_ids,
        y,
        settings,
        group_ids=patient_ids,
    )
    raw = np.full(len(cohort), np.nan, dtype=float)
    calibrated = np.full(len(cohort), np.nan, dtype=float)
    fold_number = np.zeros(len(cohort), dtype=np.int64)
    reference = np.full(len(cohort), np.nan, dtype=float)
    temperatures = []
    coefficients = []

    for split in folds:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight=None,
                        random_state=settings.reproducibility.global_seed,
                    ),
                ),
            ]
        )
        model.fit(x[split.training], y[split.training])
        calibration_prob = model.predict_proba(x[split.calibration])
        calibration_logits = probabilities_to_logits(calibration_prob)
        temperature = fit_temperature(calibration_logits, y[split.calibration])
        test_logits = probabilities_to_logits(model.predict_proba(x[split.test]))
        raw[split.test] = probabilities_from_logits(test_logits, temperature, "raw")
        calibrated[split.test] = probabilities_from_logits(
            test_logits, temperature, "calibrated"
        )
        fold_number[split.test] = split.fold
        reference[split.test] = float(y[split.training].mean())
        temperatures.append({"fold": split.fold, "temperature": temperature})
        coef = model.named_steps["classifier"].coef_[0]
        coefficients.extend(
            {
                "fold": split.fold,
                "feature": feature,
                "standardized_log_odds_coefficient": float(value),
            }
            for feature, value in zip(PROCESS_FEATURES, coef)
        )

    if not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
        raise RuntimeError("Process baseline failed to produce complete finite OOF predictions")
    predictions = pd.DataFrame(
        {
            "dataset_index": cohort["dataset_index"].astype(int),
            "encounter_id": encounter_ids,
            "patient_id": patient_ids,
            "y_true": y,
            "y_prob_raw": raw,
            "y_prob_calibrated": calibrated,
            "fold": fold_number,
            "reference_prevalence": reference,
            "run_id": RUN_ID,
            "model": PROCESS_MODEL,
        }
    )
    predictions.to_parquet(prediction_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame(temperatures).to_csv(metric_dir / "process_temperatures.csv", index=False)
    pd.DataFrame(coefficients).to_csv(metric_dir / "process_coefficients.csv", index=False)

    rows = []
    for mode, probabilities in (("raw", raw), ("calibrated", calibrated)):
        row = compute_binary_metrics(y, probabilities, **_metric_kwargs(settings, reference))
        row.update(
            {
                "Run_ID": RUN_ID,
                "Model_Architecture": PROCESS_MODEL,
                "Probability_Mode": mode,
            }
        )
        rows.append(row)
    pooled = pd.DataFrame(rows)
    pooled.to_csv(metric_dir / "process_pooled_metrics.csv", index=False)
    intervals = cluster_bootstrap_intervals(
        y,
        calibrated,
        patient_ids,
        n_replicates=settings.evaluation_protocol.bootstrap_replicates,
        seed=derive_seed(settings.reproducibility.bootstrap_seed, RUN_ID, "process"),
    )
    intervals.insert(0, "Model_Architecture", PROCESS_MODEL)
    intervals.insert(0, "Run_ID", RUN_ID)
    intervals.to_csv(metric_dir / "process_bootstrap_intervals.csv", index=False)
    return predictions


def calibration_bin_intervals(
    y_true,
    y_prob,
    groups,
    *,
    n_replicates: int,
    seed: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Fixed-quantile-bin observed fractions with patient-cluster intervals."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=float)
    group_values = np.asarray([str(value) for value in groups], dtype=object)
    bin_codes = pd.qcut(p, q=n_bins, labels=False, duplicates="drop").astype(int)
    group_codes, unique_groups = pd.factorize(group_values, sort=True)
    point = calibration_bins(y, p, n_bins=n_bins)
    samples = np.full((n_replicates, len(point)), np.nan, dtype=float)
    rng = np.random.default_rng(seed)
    for replicate in range(n_replicates):
        drawn = rng.integers(0, len(unique_groups), len(unique_groups))
        group_weights = np.bincount(drawn, minlength=len(unique_groups))
        row_weights = group_weights[group_codes]
        denominator = np.bincount(bin_codes, weights=row_weights, minlength=len(point))
        numerator = np.bincount(
            bin_codes,
            weights=row_weights * y,
            minlength=len(point),
        )
        samples[replicate] = np.divide(
            numerator,
            denominator,
            out=np.full(len(point), np.nan),
            where=denominator > 0,
        )
    point["ci_lower"] = np.nanquantile(samples, 0.025, axis=0)
    point["ci_upper"] = np.nanquantile(samples, 0.975, axis=0)
    point["valid_replicates"] = np.sum(np.isfinite(samples), axis=0)
    return point


def _subgroup_assignments(cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    groups = []
    known_sex = cohort["gender_male"].notna()
    for value, name, order in ((0.0, "Female", 1), (1.0, "Male", 2)):
        mask = known_sex & cohort["gender_male"].eq(value)
        groups.append(
            pd.DataFrame(
                {
                    "dataset_index": cohort.loc[mask, "dataset_index"],
                    "audit_dimension": "Sex",
                    "subgroup": name,
                    "subgroup_order": order,
                }
            )
        )
    cut_1, cut_2 = cohort["observation_hours"].quantile([1 / 3, 2 / 3]).tolist()
    observation_group = pd.cut(
        cohort["observation_hours"],
        bins=[-np.inf, cut_1, cut_2, np.inf],
        labels=["Short", "Intermediate", "Long"],
        include_lowest=True,
    )
    for order, name in enumerate(("Short", "Intermediate", "Long"), start=1):
        mask = observation_group.eq(name)
        groups.append(
            pd.DataFrame(
                {
                    "dataset_index": cohort.loc[mask, "dataset_index"],
                    "audit_dimension": "Observation window",
                    "subgroup": name,
                    "subgroup_order": order,
                }
            )
        )
    definitions = {
        "sex": {
            "known_encounters": int(known_sex.sum()),
            "missing_encounters": int((~known_sex).sum()),
            "interpretation": "descriptive heterogeneity/fairness audit among recorded binary sex values",
        },
        "observation_window_hours": {
            "tertile_cutpoints": [float(cut_1), float(cut_2)],
            "interpretation": "healthcare-process and information-availability audit, not a fairness attribute",
        },
    }
    return pd.concat(groups, ignore_index=True), definitions


def _subgroup_metrics(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    settings,
) -> pd.DataFrame:
    rows = []
    y_all = cohort[LABEL].to_numpy(dtype=int)
    patients_all = cohort["patient_id"].astype(str).to_numpy()
    for keys, membership in assignments.groupby(
        ["audit_dimension", "subgroup", "subgroup_order"], sort=True
    ):
        dimension, subgroup, subgroup_order = keys
        indices = membership["dataset_index"].to_numpy(dtype=int)
        y = y_all[indices]
        patients = patients_all[indices]
        if y.sum() < 10 or (len(y) - y.sum()) < 10:
            raise ValueError(f"Subgroup {dimension}/{subgroup} has too few events for audit")
        group_codes, unique_groups = pd.factorize(patients, sort=True)
        rng = np.random.default_rng(
            derive_seed(
                settings.reproducibility.bootstrap_seed,
                RUN_ID,
                str(dimension),
                str(subgroup),
            )
        )
        samples = {model: [] for model in AUDIT_MODELS}
        for _ in range(settings.evaluation_protocol.bootstrap_replicates):
            drawn = rng.integers(0, len(unique_groups), len(unique_groups))
            weights = np.bincount(drawn, minlength=len(unique_groups))[group_codes]
            if np.unique(y[weights > 0]).size < 2:
                continue
            for model in AUDIT_MODELS:
                samples[model].append(
                    average_precision_score(y, probabilities[model][indices], sample_weight=weights)
                )
        for model in AUDIT_MODELS:
            p = probabilities[model][indices]
            values = np.asarray(samples[model], dtype=float)
            alert_count = max(1, int(np.ceil(len(y) * 0.01)))
            top = np.argsort(-p, kind="stable")[:alert_count]
            events_captured = int(y[top].sum())
            rows.append(
                {
                    "Run_ID": RUN_ID,
                    "audit_dimension": dimension,
                    "subgroup": subgroup,
                    "subgroup_order": int(subgroup_order),
                    "Model_Architecture": model,
                    "N": int(len(y)),
                    "Positive": int(y.sum()),
                    "Prevalence": float(y.mean()),
                    "AUPRC": float(average_precision_score(y, p)),
                    "AUPRC_ci_lower": float(np.quantile(values, 0.025)),
                    "AUPRC_ci_upper": float(np.quantile(values, 0.975)),
                    "AUPRC_valid_replicates": int(len(values)),
                    "Observed_Expected_Ratio": float(y.sum() / p.sum()),
                    "Top_0p01_Count": alert_count,
                    "Top_0p01_Events_Captured": events_captured,
                    "Top_0p01_Recall": float(events_captured / y.sum()),
                    "patient_clusters": int(len(unique_groups)),
                }
            )
    return pd.DataFrame(rows)


def run_p1_analysis(settings=None) -> dict[str, Path]:
    settings = settings or load_settings()
    output_dir = settings.paths.reports / "runs" / RUN_ID
    metric_dir = output_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    cohort = _formal_feature_frame(settings)
    process_predictions = fit_observation_process_baseline(settings, cohort, output_dir)

    probability_map = {
        PROCESS_MODEL: process_predictions["y_prob_calibrated"].to_numpy(dtype=float)
    }
    prediction_frames = {}
    for model in FORMAL_MODELS:
        frame = _formal_prediction_frame(settings, cohort, model)
        prediction_frames[model] = frame
        if model in AUDIT_MODELS:
            probability_map[model] = frame["y_prob_calibrated"].to_numpy(dtype=float)

    calibration_rows = []
    y = cohort[LABEL].to_numpy(dtype=int)
    patients = cohort["patient_id"].astype(str).to_numpy()
    formal_bins = pd.read_csv(formal_sources(settings)["calibration_bins"])
    formal_bins = formal_bins[formal_bins["Probability_Mode"] == "calibrated"]
    for model in FORMAL_MODELS:
        frame = prediction_frames[model]
        intervals = calibration_bin_intervals(
            y,
            frame["y_prob_calibrated"].to_numpy(dtype=float),
            patients,
            n_replicates=settings.evaluation_protocol.bootstrap_replicates,
            seed=derive_seed(settings.reproducibility.bootstrap_seed, RUN_ID, model, "calibration"),
        )
        recorded = formal_bins[formal_bins["Model_Architecture"] == model].sort_values("bin")
        if len(recorded) != len(intervals):
            raise RuntimeError(f"Calibration-bin count mismatch for {model}")
        for column in ("count", "positive", "mean_predicted", "observed_fraction"):
            if not np.allclose(recorded[column].to_numpy(), intervals[column].to_numpy(), atol=1e-12):
                raise RuntimeError(f"Calibration-bin hash source mismatch for {model}: {column}")
        intervals.insert(0, "Model_Architecture", model)
        intervals.insert(0, "Run_ID", RUN_ID)
        calibration_rows.append(intervals)
    calibration_output = metric_dir / "calibration_bin_intervals.csv"
    pd.concat(calibration_rows, ignore_index=True).to_csv(calibration_output, index=False)

    assignments, definitions = _subgroup_assignments(cohort)
    subgroup = _subgroup_metrics(cohort, assignments, probability_map, settings)
    subgroup_output = metric_dir / "subgroup_metrics.csv"
    subgroup.to_csv(subgroup_output, index=False)
    definitions_output = metric_dir / "subgroup_definitions.json"
    definitions_output.write_text(
        json.dumps(definitions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "output_dir": output_dir,
        "process_predictions": output_dir / "predictions" / PROCESS_MODEL / "oof_predictions.parquet",
        "process_pooled": metric_dir / "process_pooled_metrics.csv",
        "process_bootstrap": metric_dir / "process_bootstrap_intervals.csv",
        "calibration_intervals": calibration_output,
        "subgroup_metrics": subgroup_output,
        "subgroup_definitions": definitions_output,
    }


if __name__ == "__main__":
    run_p1_analysis()
