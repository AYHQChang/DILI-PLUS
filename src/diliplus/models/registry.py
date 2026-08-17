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

# The minimum ablation matrix changes one scientifically interpretable input
# component at a time while preserving the primary architecture, split builder,
# output contract and optimization settings. The full model is the canonical
# primary model and is not trained twice within one run.
MINIMUM_ABLATION_SPECS = MappingProxyType({
    "StaticDiagnosisOnly": MappingProxyType({
        "use_medication": False,
        "use_laboratory": False,
        "use_diagnosis": True,
        "use_time_encoding": False,
    }),
    "MedicationOnly": MappingProxyType({
        "use_medication": True,
        "use_laboratory": False,
        "use_diagnosis": False,
        "use_time_encoding": True,
    }),
    "LaboratoryOnly": MappingProxyType({
        "use_medication": False,
        "use_laboratory": True,
        "use_diagnosis": False,
        "use_time_encoding": True,
    }),
    "FullWithoutTimeEncoding": MappingProxyType({
        "use_medication": True,
        "use_laboratory": True,
        "use_diagnosis": True,
        "use_time_encoding": False,
    }),
    "FullWithoutDiagnosis": MappingProxyType({
        "use_medication": True,
        "use_laboratory": True,
        "use_diagnosis": False,
        "use_time_encoding": True,
    }),
    PRIMARY_MODEL_NAME: MappingProxyType({
        "use_medication": True,
        "use_laboratory": True,
        "use_diagnosis": True,
        "use_time_encoding": True,
    }),
})
MINIMUM_ABLATION_MODEL_NAMES = tuple(MINIMUM_ABLATION_SPECS)
ABLATION_ONLY_MODEL_NAMES = tuple(
    name for name in MINIMUM_ABLATION_MODEL_NAMES if name != PRIMARY_MODEL_NAME
)
DEEP_EXPERIMENT_NAMES = FORMAL_DEEP_MODEL_NAMES + ABLATION_ONLY_MODEL_NAMES


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


def build_deep_experiment_model(model_name: str, vocab_sizes, training_settings):
    """Build a formal comparison model or a prespecified primary-model ablation."""
    if model_name in FORMAL_DEEP_MODEL_REGISTRY:
        return build_formal_deep_model(model_name, vocab_sizes, training_settings)
    if model_name not in ABLATION_ONLY_MODEL_NAMES:
        allowed = ", ".join(DEEP_EXPERIMENT_NAMES)
        raise ValueError(f"Unknown deep experiment {model_name!r}; choose one of: {allowed}")
    kwargs = {
        "vocab_med_size": int(vocab_sizes["vocab_med_size"]),
        "vocab_lab_size": int(vocab_sizes["vocab_lab_size"]),
        "vocab_diag_size": int(vocab_sizes["vocab_diag_size"]),
        "hidden_size": int(training_settings.hidden_size),
        "num_heads": int(training_settings.num_heads),
        "dropout": float(training_settings.dropout),
        "diagnosis_modality_dropout_prob": float(
            training_settings.diagnosis_modality_dropout_prob
        ),
        **dict(MINIMUM_ABLATION_SPECS[model_name]),
    }
    return TimeAwareMultimodalTransformer(**kwargs)


def experiment_spec(model_name: str) -> dict:
    """Return a serializable model/input contract for artifact metadata."""
    if model_name in MINIMUM_ABLATION_SPECS:
        return {
            "experiment_role": (
                "full_primary" if model_name == PRIMARY_MODEL_NAME else "minimum_ablation"
            ),
            **dict(MINIMUM_ABLATION_SPECS[model_name]),
        }
    if model_name in FORMAL_DEEP_MODEL_REGISTRY:
        return {"experiment_role": "formal_architecture_comparator"}
    raise ValueError(f"Unknown deep experiment {model_name!r}")


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
