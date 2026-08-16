"""兼容入口：转发到药物 Token 扰动模块。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.explainability.perturbation import run_targeted_counterfactual_trajectory

if __name__ == "__main__":
    run_targeted_counterfactual_trajectory()
