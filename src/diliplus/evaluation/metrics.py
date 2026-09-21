"""Prespecified Code-10 binary prediction metrics and clustered uncertainty."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)


EPSILON = 1e-7
# NumPy 1.x exposes the same trapezoidal integration rule as trapz.
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
CORE_BOOTSTRAP_METRICS = ("AUROC", "AUPRC", "Brier", "NLL")


def _arrays(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("y_true and y_prob must have the same non-zero length")
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError("binary metrics require both outcome classes")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("predicted probabilities must be finite and within [0, 1]")
    return y, np.clip(p, EPSILON, 1.0 - EPSILON)


def _tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def normalized_partial_auc(y_true, y_prob, fpr_limit: float) -> float:
    if not 0.0 < fpr_limit <= 1.0:
        raise ValueError("fpr_limit must be within (0, 1]")
    y, p = _arrays(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y, p)
    inside = fpr < fpr_limit
    fpr_part = np.concatenate([fpr[inside], [fpr_limit]])
    tpr_part = np.concatenate([tpr[inside], [np.interp(fpr_limit, fpr, tpr)]])
    return float(_trapezoid(tpr_part, fpr_part) / fpr_limit)


def quantile_ece(y_true, y_prob, n_bins: int = 10) -> float:
    y, p = _arrays(y_true, y_prob)
    quantiles = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(quantiles) < 2:
        return float(abs(p.mean() - y.mean()))
    bin_ids = np.searchsorted(quantiles[1:-1], p, side="right")
    result = 0.0
    for bin_id in range(len(quantiles) - 1):
        selected = bin_ids == bin_id
        if selected.any():
            result += selected.mean() * abs(float(p[selected].mean() - y[selected].mean()))
    return float(result)


def calibration_intercept_slope(y_true, y_prob) -> tuple[float, float]:
    y, p = _arrays(y_true, y_prob)
    x = np.log(p / (1.0 - p))
    # Two-parameter logistic calibration solved by deterministic Newton/IRLS.
    # Keeping this tiny fit in NumPy avoids loading a second native OpenMP
    # solver into a long-lived CUDA/PyTorch process on Windows.
    beta = np.asarray([0.0, 1.0], dtype=np.float64)

    def log_likelihood(value):
        eta_value = np.clip(value[0] + value[1] * x, -50.0, 50.0)
        return float(np.sum(y * eta_value - np.logaddexp(0.0, eta_value)))

    current_likelihood = log_likelihood(beta)
    for _ in range(100):
        eta = np.clip(beta[0] + beta[1] * x, -50.0, 50.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        residual = y - probability
        weight = np.maximum(probability * (1.0 - probability), 1e-10)
        gradient_0 = float(np.sum(residual))
        gradient_1 = float(np.sum(residual * x))
        hessian_00 = float(np.sum(weight) + 1e-10)
        hessian_01 = float(np.sum(weight * x))
        hessian_11 = float(np.sum(weight * x * x) + 1e-10)
        determinant = hessian_00 * hessian_11 - hessian_01 * hessian_01
        if not np.isfinite(determinant) or determinant <= 1e-18:
            break
        step = np.asarray(
            [
                (hessian_11 * gradient_0 - hessian_01 * gradient_1)
                / determinant,
                (-hessian_01 * gradient_0 + hessian_00 * gradient_1)
                / determinant,
            ]
        )
        if not np.isfinite(step).all():
            break
        scale = 1.0
        accepted = False
        while scale >= 1e-6:
            candidate = beta + scale * step
            candidate_likelihood = log_likelihood(candidate)
            if candidate_likelihood >= current_likelihood:
                beta = candidate
                current_likelihood = candidate_likelihood
                accepted = True
                break
            scale *= 0.5
        if not accepted or float(np.max(np.abs(scale * step))) < 1e-8:
            break
    if not np.isfinite(beta).all():
        raise RuntimeError("Calibration intercept/slope fit did not converge to finite values")
    return float(beta[0]), float(beta[1])


def decision_curve(y_true, y_prob, thresholds) -> pd.DataFrame:
    y, p = _arrays(y_true, y_prob)
    prevalence = float(y.mean())
    rows = []
    for threshold in np.asarray(thresholds, dtype=float):
        if not 0.0 < threshold < 1.0:
            raise ValueError("decision thresholds must be within (0, 1)")
        predicted = p >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        odds = threshold / (1.0 - threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "net_benefit_model": float(tp / len(y) - fp / len(y) * odds),
                "net_benefit_treat_all": float(prevalence - (1.0 - prevalence) * odds),
                "net_benefit_treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def calibration_bins(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    y, p = _arrays(y_true, y_prob)
    frame = pd.DataFrame({"y_true": y, "y_prob": p})
    frame["bin"] = pd.qcut(frame["y_prob"], q=n_bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True, sort=True)
    result = grouped.agg(
        count=("y_true", "size"),
        positive=("y_true", "sum"),
        mean_predicted=("y_prob", "mean"),
        observed_fraction=("y_true", "mean"),
        min_predicted=("y_prob", "min"),
        max_predicted=("y_prob", "max"),
    ).reset_index(drop=True)
    result.insert(0, "bin", np.arange(1, len(result) + 1, dtype=int))
    return result


def _threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = p >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    tn = int(np.sum((~predicted) & (y == 0)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum((~predicted) & (y == 1)))

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else float("nan")

    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    ppv = ratio(tp, tp + fp)
    npv = ratio(tn, tn + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    balanced_accuracy = float(np.nanmean([sensitivity, specificity]))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ratio(tp * tn - fp * fn, denominator)
    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "F1": f1,
        "BalancedAccuracy": balanced_accuracy,
        "MCC": mcc,
        "AlertCount": int(tp + fp),
    }


def _alert_budget_metrics(y: np.ndarray, p: np.ndarray, budget: float) -> dict[str, float]:
    count = max(1, int(math.ceil(len(y) * budget)))
    order = np.argsort(-p, kind="stable")[:count]
    tp = int(y[order].sum())
    return {
        "Count": count,
        "Precision": float(tp / count),
        "Recall": float(tp / int(y.sum())),
    }


def compute_binary_metrics(
    y_true,
    y_prob,
    *,
    reference_prevalence: float,
    p_auc_fpr_limits=(0.05, 0.10, 0.20),
    risk_thresholds=(0.005, 0.01, 0.02, 0.05),
    alert_budgets=(0.005, 0.01, 0.02, 0.05),
    dca_thresholds=None,
) -> dict[str, float]:
    y, p = _arrays(y_true, y_prob)
    reference = np.asarray(reference_prevalence, dtype=float)
    if reference.ndim == 0:
        reference = np.full(len(y), float(reference), dtype=float)
    else:
        reference = reference.reshape(-1)
    if len(reference) != len(y) or np.any((reference <= 0.0) | (reference >= 1.0)):
        raise ValueError(
            "reference_prevalence must be a training-derived scalar or aligned vector within (0, 1)"
        )
    prevalence = float(y.mean())
    auprc = float(average_precision_score(y, p))
    brier = float(np.mean(np.square(p - y)))
    reference_brier = float(
        np.mean(np.square(reference - y))
    )
    intercept, slope = calibration_intercept_slope(y, p)
    result: dict[str, float] = {
        "N": int(len(y)),
        "Positive": int(y.sum()),
        "Prevalence": prevalence,
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": auprc,
        "AUPRC_Lift": float(auprc / prevalence),
        "Brier": brier,
        "Brier_Skill": float(1.0 - brier / reference_brier),
        "NLL": float(log_loss(y, p, labels=[0, 1])),
        "Calibration_Intercept": intercept,
        "Calibration_Slope": slope,
        "Observed_Expected_Ratio": float(y.sum() / p.sum()),
        "Quantile_ECE": quantile_ece(y, p),
        "Reference_Prevalence": float(reference.mean()),
    }
    for limit in p_auc_fpr_limits:
        result[f"Normalized_pAUC_FPR_{_tag(limit)}"] = normalized_partial_auc(
            y, p, float(limit)
        )
    for threshold in risk_thresholds:
        prefix = f"Threshold_{_tag(threshold)}"
        for key, value in _threshold_metrics(y, p, float(threshold)).items():
            result[f"{prefix}_{key}"] = value
    for budget in alert_budgets:
        prefix = f"Top_{_tag(budget)}"
        for key, value in _alert_budget_metrics(y, p, float(budget)).items():
            result[f"{prefix}_{key}"] = value
    if dca_thresholds is not None:
        curve = decision_curve(y, p, dca_thresholds)
        result["DCA_AUDC"] = float(
            _trapezoid(curve["net_benefit_model"], curve["threshold"])
        )
    return result


def _weighted_core_metrics(y, p, weights) -> dict[str, float]:
    selected = weights > 0
    y_selected = y[selected]
    p_selected = p[selected]
    w_selected = weights[selected]
    return {
        "AUROC": float(roc_auc_score(y_selected, p_selected, sample_weight=w_selected)),
        "AUPRC": float(
            average_precision_score(y_selected, p_selected, sample_weight=w_selected)
        ),
        "Brier": float(np.average(np.square(p_selected - y_selected), weights=w_selected)),
        "NLL": float(
            log_loss(y_selected, p_selected, labels=[0, 1], sample_weight=w_selected)
        ),
    }


def _cluster_codes(groups, expected_length: int) -> tuple[np.ndarray, int]:
    group_values = np.asarray([str(value) for value in groups], dtype=object)
    if len(group_values) != expected_length:
        raise ValueError("groups must align one-to-one with prediction rows")
    codes, uniques = pd.factorize(group_values, sort=True)
    if np.any(codes < 0):
        raise ValueError("cluster groups must not be missing")
    return codes.astype(np.int64), int(len(uniques))


def cluster_bootstrap_intervals(
    y_true,
    y_prob,
    groups,
    *,
    n_replicates: int,
    seed: int,
) -> pd.DataFrame:
    y, p = _arrays(y_true, y_prob)
    codes, n_groups = _cluster_codes(groups, len(y))
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in CORE_BOOTSTRAP_METRICS}
    for _ in range(int(n_replicates)):
        drawn = rng.integers(0, n_groups, n_groups)
        group_weights = np.bincount(drawn, minlength=n_groups)
        row_weights = group_weights[codes]
        if np.unique(y[row_weights > 0]).size < 2:
            continue
        values = _weighted_core_metrics(y, p, row_weights)
        for metric, value in values.items():
            samples[metric].append(value)
    point = _weighted_core_metrics(y, p, np.ones(len(y), dtype=float))
    rows = []
    for metric in CORE_BOOTSTRAP_METRICS:
        values = np.asarray(samples[metric], dtype=float)
        if len(values) == 0:
            raise RuntimeError("No valid clustered bootstrap samples")
        rows.append(
            {
                "metric": metric,
                "estimate": point[metric],
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "valid_replicates": int(len(values)),
                "clusters": n_groups,
            }
        )
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    y_true,
    primary_prob,
    comparator_prob,
    groups,
    *,
    n_replicates: int,
    seed: int,
) -> pd.DataFrame:
    y, primary = _arrays(y_true, primary_prob)
    _, comparator = _arrays(y_true, comparator_prob)
    codes, n_groups = _cluster_codes(groups, len(y))
    ones = np.ones(len(y), dtype=float)
    primary_point = _weighted_core_metrics(y, primary, ones)
    comparator_point = _weighted_core_metrics(y, comparator, ones)
    rng = np.random.default_rng(seed)
    deltas = {metric: [] for metric in CORE_BOOTSTRAP_METRICS}
    for _ in range(int(n_replicates)):
        drawn = rng.integers(0, n_groups, n_groups)
        weights = np.bincount(drawn, minlength=n_groups)[codes]
        if np.unique(y[weights > 0]).size < 2:
            continue
        left = _weighted_core_metrics(y, primary, weights)
        right = _weighted_core_metrics(y, comparator, weights)
        for metric in CORE_BOOTSTRAP_METRICS:
            deltas[metric].append(left[metric] - right[metric])
    rows = []
    for metric, values_list in deltas.items():
        values = np.asarray(values_list, dtype=float)
        negative_or_zero = (np.sum(values <= 0.0) + 1.0) / (len(values) + 1.0)
        positive_or_zero = (np.sum(values >= 0.0) + 1.0) / (len(values) + 1.0)
        rows.append(
            {
                "metric": metric,
                "primary_estimate": primary_point[metric],
                "comparator_estimate": comparator_point[metric],
                "delta_primary_minus_comparator": (
                    primary_point[metric] - comparator_point[metric]
                ),
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "p_value_two_sided": float(min(1.0, 2.0 * min(negative_or_zero, positive_or_zero))),
                "valid_replicates": int(len(values)),
                "clusters": n_groups,
            }
        )
    return pd.DataFrame(rows)


def add_holm_adjustment(frame: pd.DataFrame, p_column="p_value_two_sided") -> pd.DataFrame:
    result = frame.copy()
    order = np.argsort(result[p_column].to_numpy(dtype=float), kind="stable")
    adjusted = np.empty(len(result), dtype=float)
    running = 0.0
    total = len(result)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(result.iloc[index][p_column]))
        running = max(running, candidate)
        adjusted[index] = running
    result["p_value_holm"] = adjusted
    return result
