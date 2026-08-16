"""兼容入口：转发到 Figure 2 实现。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.reporting.figures.model_comparison import generate_advanced_figure_2

if __name__ == "__main__":
    generate_advanced_figure_2()
