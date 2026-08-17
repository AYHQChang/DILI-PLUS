"""Train formal models and optional prespecified ablations under one run contract."""


import argparse
from importlib import import_module

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()
from diliplus.config import load_settings
from diliplus.models.registry import FORMAL_DEEP_MODEL_NAMES

DEEP_MODELS = FORMAL_DEEP_MODEL_NAMES
ML_MODELS = ("LogisticRegression", "XGBoost")

def run(
    settings,
    run_id,
    mode="calibrated",
    include_ablations=False,
    *,
    run_kind="formal",
    max_folds=None,
    epochs=None,
    seed_index=0,
):
    if mode != "calibrated":
        raise ValueError(
            "Separate uncalibrated training is disabled: raw and calibrated "
            "probabilities must come from the same test logits"
        )
    import_module("diliplus.training.run_all_calibrated").run_experiments(
        run_id=run_id,
        settings=settings,
        include_ablations=include_ablations,
        run_kind=run_kind,
        max_folds=max_folds,
        epochs=epochs,
        seed_index=seed_index,
    )

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train DILI-PLUS models")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--mode", choices=("calibrated",), default="calibrated"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--run-kind", choices=("formal", "pilot_legacy"), default="formal"
    )
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="Also train the five prespecified minimum ablations in the same run.",
    )
    args = parser.parse_args(argv)
    if args.include_ablations:
        if (
            args.run_kind == "formal"
            and args.max_folds is None
            and args.epochs is None
            and args.seed_index == 0
        ):
            run(load_settings(args.config), args.run_id, args.mode, True)
        else:
            run(
                load_settings(args.config),
                args.run_id,
                args.mode,
                True,
                run_kind=args.run_kind,
                max_folds=args.max_folds,
                epochs=args.epochs,
                seed_index=args.seed_index,
            )
    else:
        if (
            args.run_kind == "formal"
            and args.max_folds is None
            and args.epochs is None
            and args.seed_index == 0
        ):
            run(load_settings(args.config), args.run_id, args.mode)
        else:
            run(
                load_settings(args.config),
                args.run_id,
                args.mode,
                run_kind=args.run_kind,
                max_folds=args.max_folds,
                epochs=args.epochs,
                seed_index=args.seed_index,
            )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
