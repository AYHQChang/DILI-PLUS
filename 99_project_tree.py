"""兼容入口：打印项目根目录树。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.utils.project_tree import main

if __name__ == "__main__":
    main()
