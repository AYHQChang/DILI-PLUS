"""Train the six DILI-PLUS model families with explicit calibration mode."""
# python pipelines/02_train_models.py --mode uncalibrated
# python pipelines/02_train_models.py --mode both


import argparse
from importlib import import_module

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()
from diliplus.config import load_settings

DEEP_MODELS = (
    "MultiModalTextCNN",
    "MultiModalBiLSTM",
    "MultiModalBaselineMedBERT",
    "MultiModalTimeAwareMedBERT",
)
ML_MODELS = ("LogisticRegression", "XGBoost")

def run_uncalibrated(settings):
    import_module("diliplus.training.run_all_uncalibrated").run_experiments(settings=settings)
    import_module("diliplus.training.ml_baselines").train_ml_baselines(settings=settings)

def run_calibrated(settings):
    import_module("diliplus.training.run_all_calibrated").run_experiments(settings=settings)

def run(settings, mode="calibrated"):
    if mode in ("uncalibrated", "both"):
        run_uncalibrated(settings)
    if mode in ("calibrated", "both"):
        run_calibrated(settings)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train DILI-PLUS models")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--mode", choices=("calibrated", "uncalibrated", "both"), default="calibrated"
    )
    args = parser.parse_args(argv)
    run(load_settings(args.config), args.mode)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
