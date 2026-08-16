"""兼容入口：转发到 diliplus.data.vocabulary。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.data.vocabulary import build_vocabulary

if __name__ == "__main__":
    build_vocabulary()
