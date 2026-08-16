"""兼容入口：转发到 diliplus.data.cohort。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.data.cohort import build_dili_cohort

if __name__ == "__main__":
    build_dili_cohort()
