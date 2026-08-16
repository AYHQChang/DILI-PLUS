"""兼容入口：转发到当前 Table 1 实现。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.reporting.table1 import generate_table_1

if __name__ == "__main__":
    generate_table_1()
