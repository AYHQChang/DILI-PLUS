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

def run(settings, stages=STAGES):
    for stage_name, module_name, function_name in stages:
        print(f"\n[DILI-PLUS] Explainability stage: {stage_name}")
        getattr(import_module(module_name), function_name)(settings=settings)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Explain DILI-PLUS models")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args(argv)
    run(load_settings(args.config))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
