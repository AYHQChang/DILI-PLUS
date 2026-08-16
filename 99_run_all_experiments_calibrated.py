"""兼容入口：运行六模型校准训练编排器。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.training.run_all_calibrated import run_experiments

if __name__ == "__main__":
    run_experiments()
