"""兼容入口：转发到校准机器学习训练器。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.training.ml_baselines_calibrated import main

if __name__ == "__main__":
    main()
