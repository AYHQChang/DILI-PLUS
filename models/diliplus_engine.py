"""兼容 shim：转发当前主模型及其历史类名。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.models.diliplus_engine import (
    DILIPlusEngine,
    LogUniformTime2Vec,
    MaskedGAP,
    TimeAwareMultimodalTransformer,
)

__all__ = [
    "DILIPlusEngine",
    "TimeAwareMultimodalTransformer",
    "LogUniformTime2Vec",
    "MaskedGAP",
]
