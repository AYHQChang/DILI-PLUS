"""兼容 shim：转发三种深度对照模型。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.models.baselines import (
    MultiModalBaselineMedBERT,
    MultiModalBiLSTM,
    MultiModalTextCNN,
    MultimodalTransformerBaseline,
    PositionalEncoding,
    StaticProfileEncoder,
)

__all__ = [
    "MultiModalBaselineMedBERT",
    "MultimodalTransformerBaseline",
    "MultiModalBiLSTM",
    "MultiModalTextCNN",
    "PositionalEncoding",
    "StaticProfileEncoder",
]
