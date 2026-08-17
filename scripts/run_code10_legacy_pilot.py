"""Run the explicitly non-formal legacy pipeline pilot on one fold/epoch."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.config import load_settings
from diliplus.data.diagnoses import build_diag_tensors
from diliplus.data.diagnosis_audit import (
    audit_diagnosis_source,
    audit_diagnosis_time_contract,
)
from diliplus.data.labels import extract_dili_labels_and_censor
from diliplus.data.sequences import build_dili_tensors
from diliplus.data.temporal_audit import audit_prediction_time_contract
from diliplus.data.vocabulary import build_vocabulary
from diliplus.models.registry import PRIMARY_MODEL_NAME
from diliplus.training.run_all_calibrated import run_experiments


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Code-10 legacy-only pipeline pilot")
    parser.add_argument("--run-id", default="pilot_legacy_code10")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Reuse an already completed legacy pilot data build after a trainer-only fix.",
    )
    args = parser.parse_args(argv)
    if not args.run_id.startswith("pilot_legacy"):
        parser.error("pilot run ID must start with pilot_legacy")
    settings = load_settings(PROJECT_ROOT / "configs" / "pilot_legacy.yaml")
    if settings.prediction.label_source != "legacy_frozen":
        raise RuntimeError("Pilot config must use legacy_frozen label_source")

    started = time.perf_counter()
    print("[DILI-PLUS] PILOT ONLY; results are prohibited from paper reporting.")
    if not args.skip_data:
        extract_dili_labels_and_censor(settings)
        build_dili_tensors(settings)
        audit_prediction_time_contract(settings)
        audit_diagnosis_source(settings)
        build_diag_tensors(settings)
        audit_diagnosis_time_contract(settings)
        build_vocabulary(settings)
    run_experiments(
        args.run_id,
        settings=settings,
        run_kind="pilot_legacy",
        max_folds=1,
        epochs=1,
        deep_models=(PRIMARY_MODEL_NAME,),
    )

    report_root = settings.paths.reports / "runs" / args.run_id
    expected = {
        PRIMARY_MODEL_NAME: report_root / "metrics" / f"{PRIMARY_MODEL_NAME}.csv",
        "ML": report_root / "metrics" / "ml_baselines.csv",
    }
    missing = [str(path) for path in expected.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Pilot did not produce expected metrics: {missing}")
    deep_metrics = pd.read_csv(expected[PRIMARY_MODEL_NAME])
    ml_metrics = pd.read_csv(expected["ML"])
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "run_id": args.run_id,
        "run_kind": "pilot_legacy",
        "paper_eligible": False,
        "purpose": "pipeline, GPU, output-schema and runtime smoke only",
        "epochs": 1,
        "outer_folds_executed": 1,
        "models": [PRIMARY_MODEL_NAME, "LogisticRegression", "XGBoost"],
        "deep_metric_rows": int(len(deep_metrics)),
        "ml_metric_rows": int(len(ml_metrics)),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
            if torch.cuda.is_available()
            else None
        ),
    }
    path = report_root / "pilot_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
