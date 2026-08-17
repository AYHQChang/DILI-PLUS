"""Single source of truth for formal deep-model names and implementations."""

from __future__ import annotations

from types import MappingProxyType

from .baselines import (
    MultiModalBiLSTM,
    MultiModalTextCNN,
    MultimodalTransformerBaseline,
)
from .diliplus_engine import TimeAwareMultimodalTransformer


PRIMARY_MODEL_NAME = "TimeAwareMultimodalTransformer"
TRANSFORMER_BASELINE_NAME = "MultimodalTransformerBaseline"

FORMAL_DEEP_MODEL_REGISTRY = MappingProxyType({
    "MultiModalTextCNN": MultiModalTextCNN,
    "MultiModalBiLSTM": MultiModalBiLSTM,
    TRANSFORMER_BASELINE_NAME: MultimodalTransformerBaseline,
    PRIMARY_MODEL_NAME: TimeAwareMultimodalTransformer,
})
FORMAL_DEEP_MODEL_NAMES = tuple(FORMAL_DEEP_MODEL_REGISTRY)


def get_formal_deep_model(model_name: str):
    """Return a formal model class while rejecting historical artifact names."""
    try:
        return FORMAL_DEEP_MODEL_REGISTRY[model_name]
    except KeyError as exc:
        allowed = ", ".join(FORMAL_DEEP_MODEL_NAMES)
        raise ValueError(
            f"Unknown formal deep model {model_name!r}; choose one of: {allowed}. "
            "Historical MedBERT-named artifacts are intentionally unsupported."
        ) from exc


def build_formal_deep_model(model_name: str, vocab_sizes, training_settings):
    """Instantiate one formal model from the central architecture settings."""
    model_class = get_formal_deep_model(model_name)
    kwargs = {
        "vocab_med_size": int(vocab_sizes["vocab_med_size"]),
        "vocab_lab_size": int(vocab_sizes["vocab_lab_size"]),
        "vocab_diag_size": int(vocab_sizes["vocab_diag_size"]),
        "hidden_size": int(training_settings.hidden_size),
        "dropout": float(training_settings.dropout),
    }
    if model_name in {TRANSFORMER_BASELINE_NAME, PRIMARY_MODEL_NAME}:
        kwargs["num_heads"] = int(training_settings.num_heads)
    if model_name == PRIMARY_MODEL_NAME:
        kwargs["diagnosis_modality_dropout_prob"] = float(
            training_settings.diagnosis_modality_dropout_prob
        )
    return model_class(**kwargs)


def extract_ahi_proxy_logits(outputs):
    """Enforce the single-task ``{'logits': [batch, 2]}`` output contract."""
    if not isinstance(outputs, dict) or "logits" not in outputs:
        raise TypeError(
            "Formal deep models must return {'logits': Tensor}; "
            "tuple/multi-task outputs are unsupported"
        )
    logits = outputs["logits"]
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            f"AHI-proxy logits must have shape [batch, 2], found {tuple(logits.shape)}"
        )
    return logits
