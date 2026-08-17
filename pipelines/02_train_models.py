"""Train formal models and optional prespecified ablations under one run contract."""


import argparse
from importlib import import_module

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()
from diliplus.config import load_settings
from diliplus.models.registry import FORMAL_DEEP_MODEL_NAMES

DEEP_MODELS = FORMAL_DEEP_MODEL_NAMES
ML_MODELS = ("LogisticRegression", "XGBoost")

def run(settings, run_id, mode="calibrated", include_ablations=False):
    if mode != "calibrated":
        raise ValueError(
            "Separate uncalibrated training is disabled: raw and calibrated "
            "probabilities must come from the same test logits"
        )
    import_module("diliplus.training.run_all_calibrated").run_experiments(
        run_id=run_id,
        settings=settings,
        include_ablations=include_ablations,
    )

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train DILI-PLUS models")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--mode", choices=("calibrated",), default="calibrated"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="Also train the five prespecified minimum ablations in the same run.",
    )
    args = parser.parse_args(argv)
    if args.include_ablations:
        run(load_settings(args.config), args.run_id, args.mode, True)
    else:
        run(load_settings(args.config), args.run_id, args.mode)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
