"""兼容入口：转发到 Figure 1d 实现。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.reporting.figures.biomarker_divergence import generate_divergence_plot

if __name__ == "__main__":
    generate_divergence_plot()
