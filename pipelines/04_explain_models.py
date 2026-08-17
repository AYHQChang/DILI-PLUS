"""Run local attribution followed by medication-token perturbation."""

import argparse
from importlib import import_module

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()
from diliplus.config import load_settings

STAGES = (
    ("ig_loo", "diliplus.explainability.attribution", "main"),
    ("token_perturbation", "diliplus.explainability.perturbation", "run_targeted_counterfactual_trajectory"),
)

def run(settings, run_id, probability_mode="calibrated", fold=1, stages=STAGES):
    for stage_name, module_name, function_name in stages:
        print(f"\n[DILI-PLUS] Explainability stage: {stage_name}")
        getattr(import_module(module_name), function_name)(
            settings=settings,
            run_id=run_id,
            probability_mode=probability_mode,
            fold_idx=fold,
        )

def main(argv=None):
    parser = argparse.ArgumentParser(description="Explain DILI-PLUS models")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--probability-mode", choices=("raw", "calibrated"), default="calibrated"
    )
    parser.add_argument("--fold", type=int, default=1)
    args = parser.parse_args(argv)
    run(load_settings(args.config), args.run_id, args.probability_mode, args.fold)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
