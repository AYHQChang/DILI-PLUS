"""Lightweight names shared by P1 analysis and reporting modules."""

from diliplus.reporting.formal_assets import MODEL_COLORS, MODEL_LABELS


RUN_ID = "code12_p1_supplementary"
PROCESS_MODEL = "ObservationProcessBaseline"
AUDIT_MODELS = (
    "LogisticRegression",
    "XGBoost",
    "TimeAwareMultimodalTransformer",
    PROCESS_MODEL,
)
PROCESS_COLOR = "#555555"
AUDIT_COLORS = {
    "LogisticRegression": MODEL_COLORS["LogisticRegression"],
    "XGBoost": MODEL_COLORS["XGBoost"],
    "TimeAwareMultimodalTransformer": MODEL_COLORS["TimeAwareMultimodalTransformer"],
    PROCESS_MODEL: PROCESS_COLOR,
}
AUDIT_LABELS = {
    "LogisticRegression": MODEL_LABELS["LogisticRegression"],
    "XGBoost": MODEL_LABELS["XGBoost"],
    "TimeAwareMultimodalTransformer": MODEL_LABELS["TimeAwareMultimodalTransformer"],
    PROCESS_MODEL: "Process-intensity baseline",
}
