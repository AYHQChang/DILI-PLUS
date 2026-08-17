"""Code-08 cohort description and Table 1 under the formal AHI-proxy contract.

The leakage-corrected model Parquet is the only cohort anchor. External
demographics are joined by the exact encounter key; diagnoses and dynamic-event
counts are read from the same time-bounded artifacts consumed by the models.
Every join is audited before a paper-facing table is written.

No patient-level rows are exported. The tracked manifest and all CSV/text
outputs contain aggregate counts only. The source DuckDB remains read-only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from diliplus.config import load_settings
from diliplus.database import connect_source_database


MODEL_FILE = "03_dili_dual_stream_tensors.parquet"
DIAGNOSIS_FILE = "03b_diag_tensors.parquet"
ALIGNED_LAB_FILE = "01_aligned_dili_labs.parquet"
LABEL = "label_ahi_proxy"

REQUIRED_MODEL_COLUMNS = {
    "encounter_id",
    LABEL,
    "prediction_time",
    "first_med_time",
    "prediction_gap_hours",
    "med_tokens",
    "lab_tokens",
}
REQUIRED_DIAGNOSIS_COLUMNS = {
    "encounter_id",
    "icd_codes",
    "diag_event_times",
    "diagnosis_count",
}
SOURCE_RELATION_CONTRACT = {
    "analysis.v_patient_encounters": {
        "encounter_id",
        "patient_id",
        "admit_date",
        "discharge_date",
        "dept_name",
    },
    "analysis.v_patient_profile": {"patient_id", "gender", "birth_date"},
}

FEATURES = (
    ("age", "continuous", "Age, years"),
    ("gender_male", "categorical", "Male sex"),
    ("observation_hours", "continuous", "Observation window, hours"),
    ("medication_event_count", "continuous", "Medication events before prediction"),
    ("medication_events_per_day", "continuous", "Medication events per 24 hours"),
    ("laboratory_event_count", "continuous", "Laboratory events before prediction"),
    ("diagnosis_event_count", "continuous", "Diagnoses recorded before prediction"),
    ("sepsis_shock_flag", "categorical", "Pre-prediction sepsis/shock diagnosis"),
    ("heart_failure_flag", "categorical", "Pre-prediction heart-failure diagnosis"),
    ("hypertension_flag", "categorical", "Pre-prediction hypertension diagnosis"),
    ("diabetes_flag", "categorical", "Pre-prediction type-2 diabetes diagnosis"),
    ("liver_disease_flag", "categorical", "Pre-prediction liver-disease diagnosis"),
    ("chronic_hepatitis_flag", "categorical", "Pre-prediction chronic-hepatitis diagnosis"),
    ("baseline_alt", "continuous", "First observed ALT numeric value"),
    ("baseline_ast", "continuous", "First observed AST numeric value"),
    ("baseline_tbil", "continuous", "First observed total-bilirubin numeric value"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _normalise_identifier(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace("", pd.NA)


def _safe_length(value) -> int:
    return len(value) if isinstance(value, (list, tuple, np.ndarray)) else 0


def _require_columns(frame: pd.DataFrame, required: Iterable[str], source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise KeyError(f"{source} is missing required columns: {missing}")


def _assert_encounter_grain(frame: pd.DataFrame, source: str) -> None:
    if frame.empty:
        raise ValueError(f"{source} contains zero rows")
    if frame["encounter_id"].isna().any():
        raise ValueError(f"{source} contains null encounter_id values")
    duplicated = int(frame["encounter_id"].duplicated().sum())
    if duplicated:
        raise ValueError(
            f"{source} violates one-row-per-encounter grain: {duplicated} duplicate rows"
        )


def _join_audit_row(
    stage: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    merged: pd.DataFrame,
    matched_column: str,
) -> dict:
    left_n = int(len(left))
    matched = int(merged[matched_column].notna().sum())
    return {
        "stage": stage,
        "left_rows": left_n,
        "left_unique_encounters": int(left["encounter_id"].nunique()),
        "right_rows": int(len(right)),
        "right_unique_encounters": int(right["encounter_id"].nunique()),
        "output_rows": int(len(merged)),
        "output_unique_encounters": int(merged["encounter_id"].nunique()),
        "matched_left_encounters": matched,
        "unmatched_left_encounters": left_n - matched,
        "match_rate": matched / left_n if left_n else 0.0,
        "row_inflation": len(merged) / left_n if left_n else np.nan,
    }


def _checked_left_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    stage: str,
    matched_column: str,
    require_any_match: bool = True,
) -> tuple[pd.DataFrame, dict]:
    _assert_encounter_grain(left, f"{stage} left input")
    _assert_encounter_grain(right, f"{stage} right input")
    merged = left.merge(right, on="encounter_id", how="left", validate="one_to_one")
    audit = _join_audit_row(stage, left, right, merged, matched_column)
    if len(merged) != len(left) or merged["encounter_id"].nunique() != len(left):
        raise RuntimeError(f"{stage} changed the cohort grain")
    if require_any_match and audit["matched_left_encounters"] == 0:
        raise RuntimeError(
            f"{stage} matched zero encounters; stop and re-audit relation names and keys"
        )
    return merged, audit


def _verify_source_schema(conn) -> list[dict]:
    rows = []
    for relation, required in SOURCE_RELATION_CONTRACT.items():
        try:
            described = conn.execute(f"DESCRIBE SELECT * FROM {relation}").df()
        except Exception as exc:
            raise RuntimeError(
                f"Required source relation {relation!r} is unavailable; "
                "do not substitute a similarly named table"
            ) from exc
        available = set(described["column_name"].astype(str))
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(
                f"Source relation {relation!r} is missing contracted columns: {missing}"
            )
        rows.append(
            {
                "relation": relation,
                "required_columns": ",".join(sorted(required)),
                "schema_status": "PASS",
            }
        )
    return rows


def _load_cohort(model_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(model_path)
    _require_columns(frame, REQUIRED_MODEL_COLUMNS, str(model_path))
    frame = frame[
        [
            "encounter_id",
            LABEL,
            "prediction_time",
            "first_med_time",
            "prediction_gap_hours",
            "med_tokens",
            "lab_tokens",
        ]
    ].copy()
    frame["encounter_id"] = _normalise_identifier(frame["encounter_id"])
    _assert_encounter_grain(frame, "formal model cohort")
    labels = set(frame[LABEL].dropna().astype(int).unique())
    if labels != {0, 1}:
        raise ValueError(f"Formal cohort labels must be exactly {{0, 1}}, found {labels}")
    frame["prediction_time"] = pd.to_datetime(frame["prediction_time"], errors="coerce")
    frame["first_med_time"] = pd.to_datetime(frame["first_med_time"], errors="coerce")
    if frame[["prediction_time", "first_med_time"]].isna().any().any():
        raise ValueError("Formal cohort contains unparseable time boundaries")
    frame["observation_hours"] = (
        frame["prediction_time"] - frame["first_med_time"]
    ).dt.total_seconds() / 3600.0
    if not frame["observation_hours"].gt(0).all():
        raise ValueError("Every Table 1 encounter must have prediction_time > first_med_time")
    frame["medication_event_count"] = frame["med_tokens"].map(_safe_length)
    frame["laboratory_event_count"] = frame["lab_tokens"].map(_safe_length)
    frame["medication_events_per_day"] = (
        frame["medication_event_count"] / (frame["observation_hours"] / 24.0)
    )
    return frame.drop(columns=["med_tokens", "lab_tokens"])


def _load_diagnoses(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    _require_columns(frame, REQUIRED_DIAGNOSIS_COLUMNS, str(path))
    frame = frame[["encounter_id", "icd_codes", "diagnosis_count"]].copy()
    frame["encounter_id"] = _normalise_identifier(frame["encounter_id"])
    _assert_encounter_grain(frame, "time-bounded diagnosis artifact")

    def has_prefix(codes, prefixes) -> float:
        if not isinstance(codes, (list, tuple, np.ndarray)):
            return 0.0
        normalised = [str(code).strip().upper() for code in codes]
        return float(any(code.startswith(prefixes) for code in normalised))

    frame["diagnosis_event_count"] = pd.to_numeric(
        frame["diagnosis_count"], errors="coerce"
    )
    frame["diabetes_flag"] = frame["icd_codes"].map(lambda x: has_prefix(x, ("E11",)))
    frame["hypertension_flag"] = frame["icd_codes"].map(lambda x: has_prefix(x, ("I10",)))
    frame["heart_failure_flag"] = frame["icd_codes"].map(lambda x: has_prefix(x, ("I50",)))
    frame["sepsis_shock_flag"] = frame["icd_codes"].map(
        lambda x: has_prefix(x, ("A41", "R57"))
    )
    frame["liver_disease_flag"] = frame["icd_codes"].map(
        lambda x: has_prefix(x, ("K70", "K74", "K76"))
    )
    frame["chronic_hepatitis_flag"] = frame["icd_codes"].map(
        lambda x: has_prefix(x, ("B18",))
    )
    return frame.drop(columns=["icd_codes", "diagnosis_count"])


def _query_encounter_dimension(conn, cohort: pd.DataFrame) -> pd.DataFrame:
    keys = cohort[["encounter_id"]].copy()
    conn.register("code08_cohort_keys", keys)
    raw = conn.execute(
        """
        SELECT
            TRIM(CAST(e.encounter_id AS VARCHAR)) AS encounter_id,
            TRIM(CAST(e.patient_id AS VARCHAR)) AS patient_id,
            TRY_CAST(e.admit_date AS TIMESTAMP) AS admit_date,
            TRY_CAST(e.discharge_date AS TIMESTAMP) AS discharge_date,
            TRIM(CAST(e.dept_name AS VARCHAR)) AS dept_name
        FROM analysis.v_patient_encounters e
        INNER JOIN code08_cohort_keys c
          ON TRIM(CAST(e.encounter_id AS VARCHAR)) = c.encounter_id
        """
    ).df()
    if raw.empty:
        raise RuntimeError(
            "analysis.v_patient_encounters matched zero formal encounters; "
            "the table/key contract must be re-audited"
        )
    raw["encounter_id"] = _normalise_identifier(raw["encounter_id"])
    raw["patient_id"] = _normalise_identifier(raw["patient_id"])
    conflicts = raw.groupby("encounter_id", dropna=False)["patient_id"].nunique(dropna=True)
    if int(conflicts.gt(1).sum()):
        raise RuntimeError("One encounter maps to multiple patient_id values")
    return (
        raw.sort_values(["encounter_id", "admit_date"], kind="mergesort")
        .drop_duplicates("encounter_id", keep="first")
        .reset_index(drop=True)
    )


def _query_patient_profile(conn, encounters: pd.DataFrame) -> pd.DataFrame:
    keys = encounters[["patient_id"]].dropna().drop_duplicates().copy()
    conn.register("code08_patient_keys", keys)
    profile = conn.execute(
        """
        SELECT
            TRIM(CAST(p.patient_id AS VARCHAR)) AS patient_id,
            MAX(TRIM(CAST(p.gender AS VARCHAR))) AS gender,
            MAX(TRY_CAST(p.birth_date AS TIMESTAMP)) AS birth_date,
            COUNT(*)::BIGINT AS source_profile_rows,
            COUNT(DISTINCT TRIM(CAST(p.gender AS VARCHAR)))::BIGINT
                AS distinct_gender_values,
            COUNT(DISTINCT TRY_CAST(p.birth_date AS TIMESTAMP))::BIGINT
                AS distinct_birth_date_values
        FROM analysis.v_patient_profile p
        INNER JOIN code08_patient_keys c
          ON TRIM(CAST(p.patient_id AS VARCHAR)) = c.patient_id
        GROUP BY 1
        """
    ).df()
    profile["patient_id"] = _normalise_identifier(profile["patient_id"])
    if profile["distinct_gender_values"].gt(1).any():
        raise RuntimeError("One patient_id maps to conflicting gender values")
    if profile["distinct_birth_date_values"].gt(1).any():
        raise RuntimeError("One patient_id maps to conflicting birth_date values")
    return profile


def _baseline_labs(conn, lab_path: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    keys = cohort[["encounter_id"]].copy()
    conn.register("code08_lab_keys", keys)
    sql_path = lab_path.as_posix().replace("'", "''")
    frame = conn.execute(
        f"""
        WITH typed AS (
            SELECT
                TRIM(CAST(l.encounter_id AS VARCHAR)) AS encounter_id,
                TRIM(CAST(l.lab_item AS VARCHAR)) AS lab_item,
                TRIM(CAST(l.abnormal_status AS VARCHAR)) AS abnormal_status,
                TRY_CAST(l.lab_value AS DOUBLE) AS lab_value,
                TRY_CAST(l.lab_time AS TIMESTAMP) AS lab_time,
                CASE
                  WHEN l.lab_item LIKE '%谷丙转氨酶%' THEN 'ALT'
                  WHEN l.lab_item LIKE '%谷草转氨酶%' THEN 'AST'
                  WHEN l.lab_item LIKE '%总胆红素%' THEN 'TBIL'
                  ELSE NULL
                END AS lab_kind
            FROM read_parquet('{sql_path}') l
            INNER JOIN code08_lab_keys c
              ON TRIM(CAST(l.encounter_id AS VARCHAR)) = c.encounter_id
            WHERE l.lab_time IS NOT NULL
        ),
        relevant AS (
            SELECT * FROM typed WHERE lab_kind IS NOT NULL
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY encounter_id, lab_kind
                    ORDER BY lab_time, lab_item, lab_value NULLS LAST
                ) AS kind_rank,
                CASE WHEN lab_kind IN ('ALT', 'AST') THEN
                    ROW_NUMBER() OVER (
                        PARTITION BY encounter_id, (lab_kind IN ('ALT', 'AST'))
                        ORDER BY lab_time, lab_item, lab_value NULLS LAST
                    )
                END AS target_rank
            FROM relevant
        )
        SELECT
            encounter_id,
            MAX(CASE WHEN lab_kind = 'ALT' AND kind_rank = 1 THEN lab_value END)
                AS baseline_alt,
            MAX(CASE WHEN lab_kind = 'AST' AND kind_rank = 1 THEN lab_value END)
                AS baseline_ast,
            MAX(CASE WHEN lab_kind = 'TBIL' AND kind_rank = 1 THEN lab_value END)
                AS baseline_tbil,
            MAX(CASE WHEN target_rank = 1 THEN abnormal_status END)
                AS baseline_target_status,
            MAX(CASE WHEN target_rank = 1 THEN lab_value END)
                AS baseline_target_value
        FROM ranked
        GROUP BY encounter_id
        """
    ).df()
    frame["encounter_id"] = _normalise_identifier(frame["encounter_id"])
    _assert_encounter_grain(frame, "aligned baseline-laboratory aggregate")
    return frame


def _add_demographics(frame: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(profiles, on="patient_id", how="left", validate="many_to_one")
    admit = pd.to_datetime(merged["admit_date"], errors="coerce")
    birth = pd.to_datetime(merged["birth_date"], errors="coerce")
    merged["age"] = (admit - birth).dt.total_seconds() / (365.2425 * 24 * 3600)
    merged.loc[~merged["age"].between(0, 120), "age"] = np.nan
    gender = merged["gender"].astype("string").str.strip().str.upper()
    male = gender.isin({"1", "1.0", "M", "MALE", "男", "男性"})
    female = gender.isin({"2", "2.0", "F", "FEMALE", "女", "女性"})
    merged["gender_male"] = np.where(male, 1.0, np.where(female, 0.0, np.nan))
    return merged


def _continuous_smd(positive: pd.Series, negative: pd.Series) -> float:
    positive = pd.to_numeric(positive, errors="coerce").dropna()
    negative = pd.to_numeric(negative, errors="coerce").dropna()
    if len(positive) < 2 or len(negative) < 2:
        return np.nan
    pooled = np.sqrt((positive.var(ddof=1) + negative.var(ddof=1)) / 2.0)
    return 0.0 if pooled == 0 else abs(positive.mean() - negative.mean()) / pooled


def _categorical_smd(positive: pd.Series, negative: pd.Series) -> float:
    positive = pd.to_numeric(positive, errors="coerce").dropna()
    negative = pd.to_numeric(negative, errors="coerce").dropna()
    if positive.empty or negative.empty:
        return np.nan
    p1, p0 = positive.mean(), negative.mean()
    pooled = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2.0)
    return 0.0 if pooled == 0 else abs(p1 - p0) / pooled


def _format_continuous(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "NA"
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return f"{median:.1f} [{q1:.1f}, {q3:.1f}]"


def _format_categorical(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "NA"
    return f"{int(values.sum())} ({100.0 * values.mean():.1f}%)"


def build_table_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Build machine-readable display rows; exposed for deterministic tests."""
    output = []
    groups = {
        "total": frame,
        "negative": frame.loc[frame[LABEL] == 0],
        "positive": frame.loc[frame[LABEL] == 1],
    }
    for column, kind, display in FEATURES:
        if column not in frame or frame[column].notna().sum() == 0:
            continue
        positive = groups["positive"][column].dropna()
        negative = groups["negative"][column].dropna()
        if kind == "continuous":
            p_value = (
                stats.mannwhitneyu(positive, negative, alternative="two-sided").pvalue
                if len(positive) and len(negative)
                else np.nan
            )
            smd = _continuous_smd(positive, negative)
            formatter = _format_continuous
        else:
            contingency = pd.crosstab(frame[LABEL], frame[column])
            p_value = (
                stats.fisher_exact(contingency.to_numpy()).pvalue
                if contingency.shape == (2, 2)
                else np.nan
            )
            smd = _categorical_smd(positive, negative)
            formatter = _format_categorical
        output.append(
            {
                "variable": column,
                "characteristic": display,
                "type": kind,
                "total": formatter(groups["total"][column]),
                "ahi_proxy_negative": formatter(negative),
                "ahi_proxy_positive": formatter(positive),
                "p_value": p_value,
                "standardized_mean_difference": smd,
                "total_nonmissing": int(groups["total"][column].notna().sum()),
                "total_missing": int(groups["total"][column].isna().sum()),
                "negative_nonmissing": int(negative.size),
                "positive_nonmissing": int(positive.size),
            }
        )
    return pd.DataFrame(output)


def _paper_text(rows: pd.DataFrame, summary: dict) -> str:
    headers = [
        "Characteristic",
        "Total",
        "AHI-proxy negative",
        "AHI-proxy positive",
        "P value",
        "SMD",
    ]
    matrix = [headers]
    for row in rows.itertuples(index=False):
        p_value = "NA" if pd.isna(row.p_value) else ("<0.001" if row.p_value < 0.001 else f"{row.p_value:.3f}")
        smd = "NA" if pd.isna(row.standardized_mean_difference) else f"{row.standardized_mean_difference:.3f}"
        matrix.append(
            [
                row.characteristic,
                row.total,
                row.ahi_proxy_negative,
                row.ahi_proxy_positive,
                p_value,
                smd,
            ]
        )
    widths = [max(len(str(row[i])) for row in matrix) for i in range(len(headers))]
    lines = [" | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) for row in matrix]
    lines.insert(1, "-+-".join("-" * width for width in widths))
    prefix = [
        "DILI-PLUS Code-08 cohort description (AHI-proxy terminology)",
        f"Encounters: {summary['encounters']}; unique patients: {summary['unique_patients']}; "
        f"AHI-proxy positive: {summary['positive_encounters']} ({summary['prevalence_pct']:.3f}%).",
        "Continuous variables are median [Q1, Q3]; categorical variables are n (% among nonmissing).",
        "P values are descriptive; SMD is the standardized mean difference.",
        "",
    ]
    return "\n".join(prefix + lines) + "\n"


def generate_table_1(settings=None) -> Path:
    settings = settings or load_settings()
    output_dir = settings.cohort_table1_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.paths.manifests / "code08_cohort_table1.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = settings.model_data_dir / MODEL_FILE
    diagnosis_path = settings.model_data_dir / DIAGNOSIS_FILE
    lab_path = settings.paths.data_cache / ALIGNED_LAB_FILE
    for path in (model_path, diagnosis_path, lab_path):
        if not path.exists():
            raise FileNotFoundError(f"Required Code-08 input not found: {path}")

    print(f"[DILI-PLUS][Code-08] Cohort anchor: {model_path}")
    cohort = _load_cohort(model_path)
    initial_n = len(cohort)
    join_audit = []

    diagnoses = _load_diagnoses(diagnosis_path)
    cohort, audit = _checked_left_join(
        cohort,
        diagnoses,
        stage="formal_cohort_to_time_bounded_diagnoses",
        matched_column="diagnosis_event_count",
    )
    join_audit.append(audit)

    conn = connect_source_database(settings)
    try:
        schema_audit = _verify_source_schema(conn)
        encounters = _query_encounter_dimension(conn, cohort)
        cohort, audit = _checked_left_join(
            cohort,
            encounters,
            stage="formal_cohort_to_encounter_dimension",
            matched_column="patient_id",
        )
        join_audit.append(audit)
        profiles = _query_patient_profile(conn, encounters)
        before_profile = cohort.copy()
        cohort = _add_demographics(cohort, profiles)
        profile_matches = int(cohort["source_profile_rows"].notna().sum())
        join_audit.append(
            {
                "stage": "encounter_patient_to_patient_profile",
                "left_rows": initial_n,
                "left_unique_encounters": initial_n,
                "right_rows": int(len(profiles)),
                "right_unique_encounters": None,
                "output_rows": int(len(cohort)),
                "output_unique_encounters": int(cohort["encounter_id"].nunique()),
                "matched_left_encounters": profile_matches,
                "unmatched_left_encounters": initial_n - profile_matches,
                "match_rate": profile_matches / initial_n,
                "row_inflation": len(cohort) / len(before_profile),
            }
        )
        if profile_matches == 0:
            raise RuntimeError("Patient profile matched zero encounters")

        baseline = _baseline_labs(conn, lab_path, cohort)
        cohort, audit = _checked_left_join(
            cohort,
            baseline,
            stage="formal_cohort_to_aligned_baseline_labs",
            matched_column="baseline_target_status",
        )
        join_audit.append(audit)
    finally:
        conn.close()

    if len(cohort) != initial_n or cohort["encounter_id"].nunique() != initial_n:
        raise RuntimeError("Code-08 joins changed the formal cohort denominator")

    patient_counts = cohort["patient_id"].dropna().value_counts()
    department = (
        cohort.assign(dept_name=cohort["dept_name"].fillna("Missing/unknown"))
        .groupby("dept_name", dropna=False)
        .agg(encounters=("encounter_id", "size"), unique_patients=("patient_id", "nunique"))
        .reset_index()
        .sort_values(["encounters", "dept_name"], ascending=[False, True])
    )
    department["encounter_pct"] = 100.0 * department["encounters"] / initial_n
    dept_text = cohort["dept_name"].fillna("").astype(str)
    icu_mask = dept_text.str.contains("ICU|重症|监护|CCU|EICU|RICU", case=False, regex=True)

    positive = int(cohort[LABEL].sum())
    summary = {
        "encounters": int(initial_n),
        "unique_patients": int(cohort["patient_id"].nunique()),
        "encounters_without_patient_id": int(cohort["patient_id"].isna().sum()),
        "patients_with_repeated_encounters": int(patient_counts.gt(1).sum()),
        "repeated_encounters_above_one_per_patient": int((patient_counts - 1).clip(lower=0).sum()),
        "age_nonmissing": int(cohort["age"].notna().sum()),
        "sex_nonmissing": int(cohort["gender_male"].notna().sum()),
        "positive_encounters": positive,
        "negative_encounters": int(initial_n - positive),
        "prevalence_pct": 100.0 * positive / initial_n,
        "prediction_gap_hours_min": float(cohort["prediction_gap_hours"].min()),
        "prediction_gap_hours_max": float(cohort["prediction_gap_hours"].max()),
        "icu_or_critical_care_encounters": int(icu_mask.sum()),
        "other_or_unknown_department_encounters": int((~icu_mask).sum()),
        "setting_interpretation": (
            "hospital_wide_inpatient_cohort_with_department_mix"
            if int((~icu_mask).sum()) > 0
            else "critical_care_only_by_department_name_screen"
        ),
    }

    normal_low = {"N", "正常", "L", "低", "↓"}
    baseline_status = cohort["baseline_target_status"].astype("string").str.strip()
    baseline_numeric = pd.to_numeric(cohort["baseline_target_value"], errors="coerce")
    baseline_audit = []
    for label_value, group_name in ((0, "AHI-proxy negative"), (1, "AHI-proxy positive")):
        selected = cohort[LABEL].eq(label_value)
        numeric = baseline_numeric[selected]
        eligible_status = baseline_status[selected].isin(normal_low)
        baseline_audit.append(
            {
                "group": group_name,
                "encounters": int(selected.sum()),
                "baseline_status_nonmissing": int(baseline_status[selected].notna().sum()),
                "baseline_status_normal_or_low": int(eligible_status.sum()),
                "baseline_numeric_nonmissing": int(numeric.notna().sum()),
                "baseline_numeric_ge_120": int(numeric.ge(120.0).sum()),
                "normal_or_low_status_but_numeric_ge_120": int((eligible_status & numeric.ge(120.0)).sum()),
            }
        )

    table_rows = build_table_rows(cohort)
    if table_rows.empty:
        raise RuntimeError("Code-08 produced no Table 1 characteristics")

    table_csv = output_dir / "table1_characteristics.csv"
    table_txt = output_dir / "table1_paper.txt"
    summary_csv = output_dir / "cohort_summary.csv"
    department_csv = output_dir / "department_distribution.csv"
    join_csv = output_dir / "join_audit.csv"
    baseline_csv = output_dir / "baseline_label_audit.csv"
    schema_csv = output_dir / "source_schema_contract.csv"
    table_rows.to_csv(table_csv, index=False)
    table_txt.write_text(_paper_text(table_rows, summary), encoding="utf-8")
    pd.DataFrame([{"metric": key, "value": value} for key, value in summary.items()]).to_csv(
        summary_csv, index=False
    )
    department.to_csv(department_csv, index=False)
    pd.DataFrame(join_audit).to_csv(join_csv, index=False)
    pd.DataFrame(baseline_audit).to_csv(baseline_csv, index=False)
    pd.DataFrame(schema_audit).to_csv(schema_csv, index=False)

    outputs = [table_csv, table_txt, summary_csv, department_csv, join_csv, baseline_csv, schema_csv]
    manifest = {
        "contract": "code08_cohort_table1_v1",
        "cohort_anchor": MODEL_FILE,
        "label": LABEL,
        "source_database_mode": "read_only",
        "source_relations": {
            relation: sorted(columns)
            for relation, columns in SOURCE_RELATION_CONTRACT.items()
        },
        "join_key": "encounter_id (trimmed string equality)",
        "summary": summary,
        "join_audit": join_audit,
        "baseline_label_audit": baseline_audit,
        "input_sha256": {
            MODEL_FILE: _sha256(model_path),
            DIAGNOSIS_FILE: _sha256(diagnosis_path),
            ALIGNED_LAB_FILE: _sha256(lab_path),
        },
        "output_sha256": {path.name: _sha256(path) for path in outputs},
        "limitations": [
            "The formal AHI-proxy cohort uses the prespecified deterministic earliest-timestamp ALT/AST baseline rule; legacy labels are not read by this Table 1 path.",
            "The aligned laboratory cache is the same encounter-level source used by the deterministic cohort builder; Table 1 does not create an independent outcome or laboratory-linkage path.",
            "Diagnosis flags describe records available before prediction_time, not causal comorbid effects.",
            "The department screen is descriptive and does not validate a dedicated ICU cohort.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "[DILI-PLUS][Code-08] PASS: "
        f"{summary['encounters']} encounters, {summary['unique_patients']} unique patients, "
        f"{summary['positive_encounters']} AHI-proxy positives."
    )
    for audit in join_audit:
        print(
            f"[DILI-PLUS][Code-08] JOIN {audit['stage']}: "
            f"matched={audit['matched_left_encounters']}/{audit['left_rows']}, "
            f"output_rows={audit['output_rows']}, inflation={audit['row_inflation']:.6f}"
        )
    print(f"[DILI-PLUS][Code-08] Table 1: {table_csv}")
    print(f"[DILI-PLUS][Code-08] Aggregate manifest: {manifest_path}")
    return table_csv


if __name__ == "__main__":
    generate_table_1()
