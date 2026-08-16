"""Run the existing DILI early-warning evaluation without method changes."""

import argparse
from importlib import import_module

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()
from diliplus.config import load_settings

STAGES = (("early_warning", "diliplus.evaluation.early_warning", "main"),)

def run(settings, stages=STAGES):
    for _, module_name, function_name in stages:
        getattr(import_module(module_name), function_name)(settings=settings)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate DILI-PLUS models")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args(argv)
    run(load_settings(args.config))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
