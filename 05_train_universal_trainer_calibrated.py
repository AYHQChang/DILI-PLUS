"""兼容入口：完整转发原校准训练器命令行参数。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.training.deep_trainer_calibrated import main

if __name__ == "__main__":
    main()
