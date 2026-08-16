"""兼容入口：转发到 Figure 1b 实现。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.reporting.figures.data_landscape import generate_landscape_figure

if __name__ == "__main__":
    generate_landscape_figure()
