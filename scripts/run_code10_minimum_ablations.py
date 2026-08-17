"""Train only the five prespecified minimum ablations and compare with main primary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.artifacts import deep_artifact_path  # noqa: E402
from diliplus.config import load_settings  # noqa: E402
from diliplus.evaluation.finalize import finalize_formal_run  # noqa: E402
from diliplus.models.registry import (  # noqa: E402
    ABLATION_ONLY_MODEL_NAMES,
    PRIMARY_MODEL_NAME,
)
from diliplus.training.deep_trainer_calibrated import main as train_deep_model  # noqa: E402


def _require_main_primary(settings, main_run_id: str) -> None:
    manifest_path = settings.paths.manifests / "code10_formal_run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("run_id") != main_run_id:
        raise RuntimeError(
            "Minimum ablations require the completed prespecified six-model main run"
        )
    missing = [
        str(deep_artifact_path(settings, main_run_id, PRIMARY_MODEL_NAME, fold))
        for fold in range(1, settings.evaluation_protocol.outer_folds + 1)
        if not deep_artifact_path(
            settings, main_run_id, PRIMARY_MODEL_NAME, fold
        ).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Main primary artifacts are incomplete; first missing file: {missing[0]}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--main-run-id", required=True)
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "default.yaml")
    )
    parser.add_argument("--seed-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.run_id == args.main_run_id:
        parser.error("ablation run ID must differ from the main run ID")
    if args.seed_index != 0:
        parser.error("minimum ablations are prespecified at seed-index 0")

    settings = load_settings(args.config)
    _require_main_primary(settings, args.main_run_id)
    print(
        "[DILI-PLUS][Code-10] Training five ablation-only models; "
        "the full primary is reused from the main run."
    )
    for model_name in ABLATION_ONLY_MODEL_NAMES:
        train_deep_model(
            [
                "--model",
                model_name,
                "--run-id",
                args.run_id,
                "--run-kind",
                "formal",
                "--seed-index",
                str(args.seed_index),
            ],
            settings=settings,
        )

    model_names = (PRIMARY_MODEL_NAME, *ABLATION_ONLY_MODEL_NAMES)
    source_runs = {
        model_name: (
            args.main_run_id if model_name == PRIMARY_MODEL_NAME else args.run_id
        )
        for model_name in model_names
    }
    finalize_formal_run(
        settings,
        args.run_id,
        model_names,
        analysis_kind="minimum_ablation",
        model_source_run_ids=source_runs,
    )
    print(
        f"[DILI-PLUS][Code-10] minimum_ablation COMPLETE: run={args.run_id}, "
        f"main_primary_source={args.main_run_id}, models={list(model_names)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
