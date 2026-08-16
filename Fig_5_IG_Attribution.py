"""兼容入口：转发到 Figure 5 实现。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.reporting.figures.attribution import generate_waterfall_chart

if __name__ == "__main__":
    generate_waterfall_chart()
