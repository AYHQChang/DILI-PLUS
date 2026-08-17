"""Run a prespecified formal deep-model sensitivity/stability subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.config import load_settings  # noqa: E402
from diliplus.evaluation.finalize import finalize_formal_run  # noqa: E402
from diliplus.models.registry import (  # noqa: E402
    FORMAL_DEEP_MODEL_NAMES,
    PRIMARY_MODEL_NAME,
)
from diliplus.training.deep_trainer_calibrated import main as train_deep_model  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--analysis-kind", choices=("architecture_sensitivity", "seed_stability"), required=True
    )
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=FORMAL_DEEP_MODEL_NAMES, required=True
    )
    args = parser.parse_args(argv)
    models = tuple(dict.fromkeys(args.models))
    if PRIMARY_MODEL_NAME not in models:
        parser.error("formal deep subsets must include the primary model")
    if args.seed_index < 0:
        parser.error("seed-index must be non-negative")
    if args.analysis_kind == "architecture_sensitivity":
        expected = ("MultimodalTransformerBaseline", PRIMARY_MODEL_NAME)
        if models != expected or args.seed_index != 0:
            parser.error(
                "architecture sensitivity is prespecified as 8-head Transformer "
                "baseline + primary at seed-index 0"
            )
    if args.analysis_kind == "seed_stability":
        expected = ("MultiModalTextCNN", PRIMARY_MODEL_NAME)
        if models != expected or args.seed_index not in {1, 2}:
            parser.error(
                "additional stability runs are prespecified as TextCNN + primary "
                "at seed-index 1 or 2; seed 0 is reused from the six-model run"
            )

    settings = load_settings(args.config)
    for model_name in models:
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
    finalize_formal_run(
        settings,
        args.run_id,
        models,
        analysis_kind=args.analysis_kind,
    )
    print(
        f"[DILI-PLUS][Code-10] {args.analysis_kind} COMPLETE: "
        f"run={args.run_id}, seed_index={args.seed_index}, models={list(models)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
