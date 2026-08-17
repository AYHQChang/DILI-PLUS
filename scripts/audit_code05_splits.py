"""Build and persist the real-data Code-05 four-way split audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.config import load_settings
from diliplus.splits import build_nested_grouped_splits, save_split_audit


def main() -> int:
    settings = load_settings()
    cohort = pd.read_parquet(
        settings.model_data_dir / "03_dili_dual_stream_tensors.parquet",
        columns=["encounter_id", "patient_id", "label_ahi_proxy"],
    )
    folds = build_nested_grouped_splits(
        cohort["encounter_id"].astype(str).to_numpy(),
        cohort["label_ahi_proxy"].astype(int).to_numpy(),
        settings,
        group_ids=cohort["patient_id"].astype(str).to_numpy(),
    )
    tracked, local = save_split_audit(
        folds,
        cohort["encounter_id"].astype(str).to_numpy(),
        cohort["label_ahi_proxy"].astype(int).to_numpy(),
        "code05_real_data_audit",
        settings,
        group_ids=cohort["patient_id"].astype(str).to_numpy(),
    )
    payload = json.loads(tracked.read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[DILI-PLUS] Tracked split manifest: {tracked}")
    print(f"[DILI-PLUS] Local exact split indices: {local}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
