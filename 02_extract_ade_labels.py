"""兼容入口：转发到 diliplus.data.labels。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.data.labels import extract_dili_labels_and_censor

if __name__ == "__main__":
    extract_dili_labels_and_censor()
