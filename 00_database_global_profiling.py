"""兼容入口：转发到 diliplus.data.database_profile。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.data.database_profile import generate_global_database_profile

if __name__ == "__main__":
    generate_global_database_profile()
