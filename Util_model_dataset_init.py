"""兼容 shim：导出原 Dataset 与词表尺寸加载函数。"""

from _bootstrap import bootstrap

bootstrap()
from diliplus.data.dataset import DILIPlusDataset, load_vocab_sizes

__all__ = ["DILIPlusDataset", "load_vocab_sizes"]
