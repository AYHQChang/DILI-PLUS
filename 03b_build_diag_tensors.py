"""兼容入口：转发到 diliplus.data.diagnoses。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.data.diagnoses import build_diag_tensors

if __name__ == "__main__":
    build_diag_tensors()
