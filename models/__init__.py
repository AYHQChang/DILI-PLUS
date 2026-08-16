"""兼容 shim：从 src/diliplus/models 显式导出核心模型。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.models import (
    DILIPlusEngine,
    MultiModalBaselineMedBERT,
    MultiModalBiLSTM,
    MultiModalTextCNN,
)

__all__ = [
    "DILIPlusEngine",
    "MultiModalBaselineMedBERT",
    "MultiModalBiLSTM",
    "MultiModalTextCNN",
]
