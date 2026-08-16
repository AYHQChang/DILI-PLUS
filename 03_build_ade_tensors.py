"""兼容入口：转发到 diliplus.data.sequences。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.data.sequences import build_dili_tensors

if __name__ == "__main__":
    build_dili_tensors()
