"""兼容入口：转发到 Figure 3 实现。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.reporting.figures.calibration_impact import generate_figure_3

if __name__ == "__main__":
    generate_figure_3()
