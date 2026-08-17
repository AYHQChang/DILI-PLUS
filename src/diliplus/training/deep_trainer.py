"""Compatibility entry point for the single formal deep-training protocol.

The former implementation selected epochs by repeatedly evaluating the outer
test fold. That path is intentionally retired. Calling this module now forwards
to the four-way grouped trainer, which emits paired raw/calibrated probabilities
from one final test-logit pass.
"""

from diliplus.training.deep_trainer_calibrated import main as _formal_main


def main(argv=None, settings=None):
    return _formal_main(argv=argv, settings=settings)


if __name__ == "__main__":
    main()
