"""Central deterministic seed controls for data, training, evaluation, and figures."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from diliplus.config import ReproducibilitySettings


DEFAULT_SEED = 20260816


def derive_seed(base_seed: int, *namespace: Any) -> int:
    """Derive a stable 31-bit component seed without Python's salted hash()."""
    payload = "\x1f".join([str(int(base_seed)), *(str(value) for value in namespace)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def seed_everything(
    settings_or_seed: ReproducibilitySettings | int,
    deterministic_torch: bool | None = None,
) -> dict[str, Any]:
    """Seed supported RNGs and request deterministic PyTorch kernels.

    ``PYTHONHASHSEED`` only takes effect at interpreter startup. The recorded
    command runner sets it before launching child processes; setting it here
    documents the contract for direct in-process calls as well.
    """
    if isinstance(settings_or_seed, ReproducibilitySettings):
        seed = int(settings_or_seed.global_seed)
        deterministic = (
            settings_or_seed.deterministic_torch
            if deterministic_torch is None
            else bool(deterministic_torch)
        )
    else:
        seed = int(settings_or_seed)
        deterministic = True if deterministic_torch is None else bool(deterministic_torch)
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return {
        "global_seed": seed,
        "deterministic_torch": deterministic,
        "pythonhashseed_environment": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }


def make_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed NumPy/Python from the deterministic seed assigned by DataLoader."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def settings_manifest(settings: ReproducibilitySettings) -> dict[str, Any]:
    return asdict(settings)
