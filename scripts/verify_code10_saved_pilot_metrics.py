"""Recompute the interrupted pilot metrics from its already saved predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.config import load_settings
from diliplus.evaluation.metrics import compute_binary_metrics


def main() -> int:
    run_id = "pilot_legacy_code10_vscode"
    model = "TimeAwareMultimodalTransformer"
    path = (
        PROJECT_ROOT
        / "reports"
        / "runs"
        / run_id
        / "predictions"
        / model
        / "fold_01.csv"
    )
    frame = pd.read_csv(path)
    settings = load_settings(PROJECT_ROOT / "configs" / "pilot_legacy.yaml")
    protocol = settings.evaluation_protocol
    thresholds = np.arange(
        protocol.dca_min_threshold,
        protocol.dca_max_threshold + protocol.dca_step / 2.0,
        protocol.dca_step,
    )
    output = {}
    for mode in ("raw", "calibrated"):
        output[mode] = compute_binary_metrics(
            frame["y_true"],
            frame[f"y_prob_{mode}"],
            reference_prevalence=frame["reference_prevalence"],
            p_auc_fpr_limits=protocol.p_auc_fpr_limits,
            risk_thresholds=protocol.risk_thresholds,
            alert_budgets=protocol.alert_budgets,
            dca_thresholds=thresholds,
        )
    destination = path.parents[2] / "metrics" / "recovered_primary_fold_01.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(frame),
                "raw_AUPRC": output["raw"]["AUPRC"],
                "calibrated_AUPRC": output["calibrated"]["AUPRC"],
                "raw_calibrated_ranking_delta": (
                    output["raw"]["AUPRC"] - output["calibrated"]["AUPRC"]
                ),
                "output": str(destination),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
