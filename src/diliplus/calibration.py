"""Shared logit-level temperature scaling and probability contracts."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


ProbabilityMode = Literal["raw", "calibrated"]


def ensure_two_class_logits(values) -> torch.Tensor:
    logits = torch.as_tensor(values, dtype=torch.float32)
    if logits.ndim == 1:
        logits = torch.stack([torch.zeros_like(logits), logits], dim=1)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Expected logits with shape [N, 2] or binary scores with shape [N]")
    if not torch.isfinite(logits).all():
        raise ValueError("Logits must be finite")
    return logits


def probabilities_to_logits(probabilities, epsilon: float = 1e-7) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Expected probabilities with shape [N, 2] or [N]")
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return np.log(clipped)


def fit_temperature(logits, labels, max_iter: int = 100) -> float:
    """Fit one positive temperature on the calibration partition only."""
    logits_tensor = ensure_two_class_logits(logits).detach().cpu()
    labels_tensor = torch.as_tensor(labels, dtype=torch.long).detach().cpu()
    if len(logits_tensor) != len(labels_tensor) or len(labels_tensor) == 0:
        raise ValueError("Calibration logits and labels must have equal non-zero length")
    if len(torch.unique(labels_tensor)) < 2:
        raise ValueError("Calibration labels must contain both classes")

    log_temperature = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(min=1e-4, max=1e4)
        loss = F.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(1e-4, 1e4).item())
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Temperature optimization produced an invalid value")
    return temperature


def probabilities_from_logits(
    logits,
    temperature: float,
    mode: ProbabilityMode = "calibrated",
) -> np.ndarray:
    if mode not in ("raw", "calibrated"):
        raise ValueError("probability mode must be 'raw' or 'calibrated'")
    logits_tensor = ensure_two_class_logits(logits).detach().cpu()
    scale = 1.0
    if mode == "calibrated":
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("calibrated inference requires a positive finite temperature")
        scale = float(temperature)
    return torch.softmax(logits_tensor / scale, dim=1)[:, 1].numpy()
