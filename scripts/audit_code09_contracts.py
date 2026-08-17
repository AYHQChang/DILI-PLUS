"""Audit Code-09 early-warning availability and minimum-ablation semantics.

This command reads the frozen model artifacts and vocabularies, but it does not
train a model, load a checkpoint or estimate predictive performance. Outputs
are aggregate-only and safe to track in Git.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.artifacts import file_sha256  # noqa: E402
from diliplus.config import load_settings  # noqa: E402
from diliplus.data.dataset import load_vocab_sizes  # noqa: E402
from diliplus.evaluation.early_warning import audit_early_warning_contract  # noqa: E402
from diliplus.models.registry import (  # noqa: E402
    MINIMUM_ABLATION_MODEL_NAMES,
    MINIMUM_ABLATION_SPECS,
    build_deep_experiment_model,
    extract_ahi_proxy_logits,
)
from diliplus.reproducibility import seed_everything  # noqa: E402


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _synthetic_inputs(vocab_sizes):
    med_max = max(2, min(int(vocab_sizes["vocab_med_size"]) - 1, 5))
    diag_max = max(2, min(int(vocab_sizes["vocab_diag_size"]) - 1, 5))
    return {
        "x_med": torch.tensor([[2, med_max, 3, 0, 0], [2, 3, 0, 0, 0]]),
        "dt_med": torch.tensor([[0.0, 2.0, 8.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0, 0.0]]),
        "mask_med": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]),
        "x_lab": torch.tensor([[2, med_max, 0, 0], [2, 0, 0, 0]]),
        "v_lab": torch.tensor([[10.0, 20.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]),
        "dt_lab": torch.tensor([[0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        "mask_lab": torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]]),
        "x_diag": torch.tensor([[2, diag_max, 0], [2, 0, 0]]),
        "mask_diag": torch.tensor([[1, 1, 0], [1, 0, 0]]),
    }


def audit_ablation_models(settings) -> Path:
    vocab_sizes = load_vocab_sizes(settings.paths.vocab)
    inputs = _synthetic_inputs(vocab_sizes)
    rows = []
    for name in MINIMUM_ABLATION_MODEL_NAMES:
        seed_everything(settings.reproducibility)
        model = build_deep_experiment_model(name, vocab_sizes, settings.training).eval()
        with torch.no_grad():
            output = model(**inputs)
        logits = extract_ahi_proxy_logits(output)
        if not torch.isfinite(logits).all():
            raise AssertionError(f"{name} returned non-finite logits")
        spec = dict(MINIMUM_ABLATION_SPECS[name])
        for enabled, representation in (
            (spec["use_medication"], "h_med"),
            (spec["use_laboratory"], "h_lab"),
            (spec["use_diagnosis"], "h_diag"),
        ):
            if not enabled and not torch.equal(
                output[representation], torch.zeros_like(output[representation])
            ):
                raise AssertionError(
                    f"{name} did not zero disabled representation {representation}"
                )
        rows.append(
            {
                "model_name": name,
                "spec": spec,
                "logit_shape": list(logits.shape),
                "finite_logits": True,
                "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            }
        )

    no_time = build_deep_experiment_model(
        "FullWithoutTimeEncoding", vocab_sizes, settings.training
    ).eval()
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["dt_med"] += 10000.0
    changed["dt_lab"] += 10000.0
    with torch.no_grad():
        original = extract_ahi_proxy_logits(no_time(**inputs))
        modified = extract_ahi_proxy_logits(no_time(**changed))
    if not torch.equal(original, modified):
        raise AssertionError("FullWithoutTimeEncoding still depends on time deltas")

    source_paths = (
        PROJECT_ROOT / "src" / "diliplus" / "models" / "registry.py",
        PROJECT_ROOT / "src" / "diliplus" / "models" / "diliplus_engine.py",
        PROJECT_ROOT / "src" / "diliplus" / "models" / "baselines.py",
        PROJECT_ROOT / "src" / "diliplus" / "data" / "dataset.py",
        PROJECT_ROOT / "src" / "diliplus" / "evaluation" / "early_warning.py",
        PROJECT_ROOT / "src" / "diliplus" / "training" / "deep_trainer_calibrated.py",
        PROJECT_ROOT / "src" / "diliplus" / "training" / "run_all_calibrated.py",
        PROJECT_ROOT / "pipelines" / "02_train_models.py",
    )
    payload = {
        "contract": "code09_minimum_ablation_v1",
        "status": "PASS",
        "matrix_order": list(MINIMUM_ABLATION_MODEL_NAMES),
        "models": rows,
        "shared_contract": {
            "dataset": "DILIPlusDataset nine model-input tensors",
            "target": "label_ahi_proxy",
            "split_builder": "build_nested_grouped_splits",
            "loss": "UnweightedFocalLoss(gamma=2; no class weighting)",
            "output": "dict with finite logits [batch, 2]",
            "full_model_trained_once": True,
            "enable_training_with": "pipelines/02_train_models.py --include-ablations",
        },
        "no_time_delta_invariance": "PASS",
        "source_sha256": {path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path) for path in source_paths},
        "training_performed": False,
        "performance_estimated": False,
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    output = settings.paths.manifests / "code09_minimum_ablation_contract.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[DILI-PLUS][Code-09] Minimum-ablation semantic audit PASS: "
        f"{len(rows)}/6 variants, no training or performance estimation"
    )
    print(f"[DILI-PLUS][Code-09] Ablation manifest: {output}")
    return output


def main() -> int:
    torch.set_num_threads(1)
    settings = load_settings()
    audit_early_warning_contract(settings)
    audit_ablation_models(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
