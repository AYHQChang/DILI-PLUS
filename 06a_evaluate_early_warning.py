"""兼容入口：转发到提前预警评估。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.evaluation.early_warning import main

if __name__ == "__main__":
    main()
