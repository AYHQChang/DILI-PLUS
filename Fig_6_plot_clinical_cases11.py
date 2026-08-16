"""兼容入口：转发到当前 1×2 Figure 6 实现。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.reporting.figures.perturbation import generate_simulation_figure

if __name__ == "__main__":
    generate_simulation_figure()
