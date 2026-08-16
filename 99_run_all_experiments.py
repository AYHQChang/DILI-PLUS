"""兼容入口：运行原未校准四种深度模型编排器。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.training.run_all_uncalibrated import run_experiments

if __name__ == "__main__":
    run_experiments()
