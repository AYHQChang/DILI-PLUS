"""兼容入口：转发到未校准机器学习训练器。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.training.ml_baselines import train_ml_baselines

if __name__ == "__main__":
    train_ml_baselines()
