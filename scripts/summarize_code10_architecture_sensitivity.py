"""Compare prespecified 128d/8-head and 128d/4-head OOF predictions."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.artifacts import file_sha256  # noqa: E402
from diliplus.config import load_settings  # noqa: E402
from diliplus.evaluation.metrics import (  # noqa: E402
    add_holm_adjustment,
    paired_cluster_bootstrap,
)
from diliplus.reproducibility import derive_seed  # noqa: E402


MAIN_RUN = "code10_formal_128d4h_seed0"
SENSITIVITY_RUN = "code10_sensitivity_128d8h_seed0"
MODELS = (
    "MultimodalTransformerBaseline",
    "TimeAwareMultimodalTransformer",
)


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _oof(settings, run_id: str, model_name: str) -> pd.DataFrame:
    directory = settings.paths.reports / "runs" / run_id / "predictions" / model_name
    paths = sorted(directory.glob("fold_*.csv"))
    if len(paths) != settings.evaluation_protocol.outer_folds:
        raise RuntimeError(f"{run_id}/{model_name} does not have five folds")
    return (
        pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        .sort_values("dataset_index", kind="stable")
        .reset_index(drop=True)
    )


def main() -> int:
    settings = load_settings(PROJECT_ROOT / "configs" / "default.yaml")
    comparisons = []
    for mode in ("raw", "calibrated"):
        mode_rows = []
        for model_name in MODELS:
            four_head = _oof(settings, MAIN_RUN, model_name)
            eight_head = _oof(settings, SENSITIVITY_RUN, model_name)
            membership_columns = ["dataset_index", "encounter_id", "patient_id", "y_true"]
            if not four_head[membership_columns].equals(eight_head[membership_columns]):
                raise RuntimeError(f"4-head/8-head membership differs for {model_name}")
            comparison = paired_cluster_bootstrap(
                four_head["y_true"],
                eight_head[f"y_prob_{mode}"],
                four_head[f"y_prob_{mode}"],
                four_head["patient_id"],
                n_replicates=settings.evaluation_protocol.bootstrap_replicates,
                seed=derive_seed(
                    settings.reproducibility.bootstrap_seed,
                    "architecture_sensitivity",
                    model_name,
                    mode,
                ),
            )
            comparison.insert(0, "Model_Architecture", model_name)
            comparison.insert(0, "Delta_Definition", "128d_8head_minus_128d_4head")
            comparison.insert(0, "Probability_Mode", mode)
            mode_rows.append(comparison)
        comparisons.append(add_holm_adjustment(pd.concat(mode_rows, ignore_index=True)))
    output = pd.concat(comparisons, ignore_index=True)
    destination = (
        settings.paths.reports
        / "runs"
        / SENSITIVITY_RUN
        / "metrics"
        / "head_sensitivity_paired_comparison.csv"
    )
    output.to_csv(destination, index=False)
    calibrated = output[output["Probability_Mode"].eq("calibrated")]
    payload = {
        "schema_version": 1,
        "contract": "code10_architecture_sensitivity_v1",
        "status": "PASS",
        "main_run": MAIN_RUN,
        "sensitivity_run": SENSITIVITY_RUN,
        "comparison": "128d/8-head minus 128d/4-head at matched seed/splits",
        "models": list(MODELS),
        "bootstrap_replicates": settings.evaluation_protocol.bootstrap_replicates,
        "patient_clusters": int(calibrated.iloc[0]["clusters"]),
        "holm_family": "two Transformer models by four metrics, separately per probability mode",
        "calibrated_paired_comparisons": calibrated.to_dict(orient="records"),
        "comparison_csv_sha256": file_sha256(destination),
        "input_manifest_sha256": {
            "4_head": file_sha256(settings.paths.manifests / "code10_formal_run.json"),
            "8_head": file_sha256(
                settings.paths.manifests / f"{SENSITIVITY_RUN}_run.json"
            ),
        },
        "privacy": "aggregate statistics and hashes only; no row identifiers",
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    manifest = settings.paths.manifests / "code10_architecture_sensitivity.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
