"""Run the existing DILI early-warning evaluation without method changes."""

import argparse
from importlib import import_module

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()
from diliplus.config import load_settings

STAGES = (("early_warning", "diliplus.evaluation.early_warning", "main"),)

def run(settings, run_id, probability_mode="calibrated", stages=STAGES):
    for _, module_name, function_name in stages:
        getattr(import_module(module_name), function_name)(
            settings=settings,
            run_id=run_id,
            probability_mode=probability_mode,
        )

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate DILI-PLUS models")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--probability-mode", choices=("raw", "calibrated"), default="calibrated"
    )
    args = parser.parse_args(argv)
    run(load_settings(args.config), args.run_id, args.probability_mode)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
