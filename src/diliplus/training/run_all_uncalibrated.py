"""Retired compatibility alias for the formal paired-probability pipeline."""

from diliplus.training.run_all_calibrated import run_experiments as _run_formal


def run_experiments(run_id=None, settings=None):
    if not run_id:
        raise ValueError(
            "run_id is required; separate uncalibrated training has been retired"
        )
    return _run_formal(run_id=run_id, settings=settings)


if __name__ == "__main__":
    raise SystemExit(
        "Use pipelines/02_train_models.py --run-id <id>; raw and calibrated "
        "probabilities are generated together."
    )
