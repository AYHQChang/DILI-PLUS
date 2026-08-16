"""兼容入口：转发到 Figure 4 实现。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.reporting.figures.early_warning import generate_early_warning_figure

if __name__ == "__main__":
    generate_early_warning_figure()
