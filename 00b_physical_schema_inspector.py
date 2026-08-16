"""兼容入口：转发到 diliplus.data.schema_inspector。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.data.schema_inspector import inspect_physical_schema

if __name__ == "__main__":
    inspect_physical_schema()
