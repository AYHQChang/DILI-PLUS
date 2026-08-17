"""Rebuild corrected model inputs twice and require byte-identical artifacts."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.config import load_settings
from diliplus.data.diagnoses import build_diag_tensors
from diliplus.data.diagnosis_audit import audit_diagnosis_time_contract
from diliplus.data.labels import extract_dili_labels_and_censor
from diliplus.data.sequences import build_dili_tensors
from diliplus.data.temporal_audit import audit_prediction_time_contract
from diliplus.data.vocabulary import build_vocabulary
from diliplus.reproducibility import seed_everything
from diliplus.reproducibility_audit import (
    capture_rebuild_snapshot,
    compare_rebuild_snapshots,
)


def _build_once(settings, name: str):
    seed_everything(settings.reproducibility)
    print(f"[DILI-PLUS] Deterministic rebuild pass: {name}")
    extract_dili_labels_and_censor(settings)
    build_dili_tensors(settings)
    audit_prediction_time_contract(settings)
    build_diag_tensors(settings)
    audit_diagnosis_time_contract(settings)
    build_vocabulary(settings)
    snapshot = capture_rebuild_snapshot(settings, name)
    print(
        f"[DILI-PLUS] Pass {name} comparison payload: "
        f"{snapshot['comparison_payload_sha256']}"
    )
    return snapshot


def main() -> int:
    settings = load_settings()
    settings.reproducibility_audit_dir.mkdir(parents=True, exist_ok=True)
    first = _build_once(settings, "a")
    second = _build_once(settings, "b")
    compare_rebuild_snapshots(first, second, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
