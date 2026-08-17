"""AHI-proxy prediction and comparison model architectures."""

from .baselines import (
    MultiModalBaselineMedBERT,
    MultiModalBiLSTM,
    MultiModalTextCNN,
    MultimodalTransformerBaseline,
)
from .diliplus_engine import DILIPlusEngine, TimeAwareMultimodalTransformer
from .registry import (
    ABLATION_ONLY_MODEL_NAMES,
    DEEP_EXPERIMENT_NAMES,
    MINIMUM_ABLATION_MODEL_NAMES,
    MINIMUM_ABLATION_SPECS,
    build_deep_experiment_model,
    build_formal_deep_model,
    experiment_spec,
    extract_ahi_proxy_logits,
    FORMAL_DEEP_MODEL_NAMES,
    FORMAL_DEEP_MODEL_REGISTRY,
    PRIMARY_MODEL_NAME,
    TRANSFORMER_BASELINE_NAME,
)

__all__ = [
    "DILIPlusEngine",
    "TimeAwareMultimodalTransformer",
    "MultiModalBaselineMedBERT",
    "MultimodalTransformerBaseline",
    "MultiModalBiLSTM",
    "MultiModalTextCNN",
    "FORMAL_DEEP_MODEL_NAMES",
    "FORMAL_DEEP_MODEL_REGISTRY",
    "PRIMARY_MODEL_NAME",
    "TRANSFORMER_BASELINE_NAME",
    "ABLATION_ONLY_MODEL_NAMES",
    "DEEP_EXPERIMENT_NAMES",
    "MINIMUM_ABLATION_MODEL_NAMES",
    "MINIMUM_ABLATION_SPECS",
    "build_deep_experiment_model",
    "build_formal_deep_model",
    "experiment_spec",
    "extract_ahi_proxy_logits",
]
