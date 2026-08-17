"""Compatibility entry point for the formal grouped ML baseline protocol.

Independent uncalibrated fitting is disabled so raw and calibrated test
probabilities cannot silently come from different fitted models.
"""

from diliplus.training.ml_baselines_calibrated import main as _formal_main


def train_ml_baselines(settings=None, run_id=None):
    if not run_id:
        raise ValueError("run_id is required for versioned model artifacts")
    return _formal_main(["--run-id", run_id], settings=settings)


def main(argv=None, settings=None):
    return _formal_main(argv=argv, settings=settings)


if __name__ == "__main__":
    main()
