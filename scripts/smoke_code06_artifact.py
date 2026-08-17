"""Real-input, random-weight smoke test for the Code-06 artifact contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.artifacts import (
    artifact_probabilities,
    build_artifact_metadata,
    config_snapshot,
    dataset_fingerprint,
    file_sha256,
    load_deep_artifact,
    save_deep_artifact,
)
from diliplus.config import load_settings
from diliplus.data.dataset import DILIPlusDataset, load_vocab_sizes
from diliplus.models.registry import PRIMARY_MODEL_NAME, build_formal_deep_model
from diliplus.reproducibility import seed_everything
from diliplus.splits import build_nested_grouped_splits


def _batch(sample):
    return {
        key: value.unsqueeze(0)
        for key, value in sample.items()
        if "label" not in key
    }


def main() -> int:
    settings = load_settings()
    seed_everything(settings.reproducibility)
    dataset = DILIPlusDataset(settings.model_data_dir, settings.paths.vocab)
    encounter_ids = dataset.data["encounter_id"].astype(str).to_numpy()
    labels = np.asarray(dataset.labels, dtype=np.int64)
    split = build_nested_grouped_splits(encounter_ids, labels, settings)[0]
    model_kwargs = load_vocab_sizes(settings.paths.vocab)
    model = build_formal_deep_model(
        PRIMARY_MODEL_NAME, model_kwargs, settings.training
    ).cpu().eval()
    dataset_index = int(split.test[0])
    inputs = _batch(dataset[dataset_index])
    with torch.no_grad():
        logits_before = model(**inputs)["logits"].detach().cpu()

    metadata = build_artifact_metadata(
        artifact_type="torch",
        run_id="code06-smoke",
        model_name=PRIMARY_MODEL_NAME,
        fold=1,
        selected_epoch=0,
        temperature=1.7,
        split_payload=split.checkpoint_payload(),
        dataset=dataset_fingerprint(settings),
        configuration=config_snapshot(
            settings,
            {
                "purpose": "random-weight artifact roundtrip smoke; no training",
                "target_name": "label_ahi_proxy",
                "task_contract": "single_task_binary_classification",
                "initialization": "from_scratch_no_external_pretraining",
                "formal_model_name": PRIMARY_MODEL_NAME,
            },
        ),
    )
    output_dir = settings.paths.reports / "p0_05_code_06" / "code06_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "random_weight_fold_01.pt"
    save_deep_artifact(artifact_path, model, metadata)

    restored = build_formal_deep_model(
        PRIMARY_MODEL_NAME, model_kwargs, settings.training
    ).cpu().eval()
    loaded = load_deep_artifact(
        artifact_path,
        restored,
        expected_run_id="code06-smoke",
        expected_model_name=PRIMARY_MODEL_NAME,
        expected_fold=1,
        expected_dataset_fingerprint=metadata["dataset_fingerprint"]["payload_sha256"],
    )
    with torch.no_grad():
        logits_after = restored(**inputs)["logits"].detach().cpu()
    if not torch.equal(logits_before, logits_after):
        raise RuntimeError("Artifact roundtrip changed model logits")
    raw = artifact_probabilities(logits_after, loaded, "raw")
    calibrated = artifact_probabilities(logits_after, loaded, "calibrated")
    summary = {
        "audit_status": "PASS",
        "purpose": "artifact roundtrip only; random weights; no performance estimate",
        "dataset_index_recorded": False,
        "fold": 1,
        "temperature": loaded["temperature"],
        "logits_exact_match": True,
        "raw_probability_finite": bool(np.isfinite(raw).all()),
        "calibrated_probability_finite": bool(np.isfinite(calibrated).all()),
        "same_logits_used_for_both_modes": True,
        "artifact_sha256": file_sha256(artifact_path),
        "metadata_payload_sha256": loaded["metadata_payload_sha256"],
    }
    summary_path = output_dir / "artifact_smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
