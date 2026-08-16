"""兼容 shim：转发 DILIPlusEngine。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.models.diliplus_engine import DILIPlusEngine, LogUniformTime2Vec, MaskedGAP

__all__ = ["DILIPlusEngine", "LogUniformTime2Vec", "MaskedGAP"]
