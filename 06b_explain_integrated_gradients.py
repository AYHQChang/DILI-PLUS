"""兼容入口：转发到 IG/LOO 解释模块。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.explainability.attribution import main

if __name__ == "__main__":
    main()
