"""Aggregate-only baseline manifests and deterministic rebuild audits."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from diliplus.config import load_settings
from diliplus.database import connect_source_database
from diliplus.data.labels import (
    _build_prediction_label_table,
    _load_frozen_legacy_label_table,
)
from diliplus.reproducibility import seed_everything, settings_manifest


PACKAGE_NAMES = (
    "captum",
    "duckdb",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "seaborn",
    "torch",
    "xgboost",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _git_binary(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else b""


def build_worktree_snapshot(root: Path) -> dict[str, Any]:
    """Identify an exact dirty worktree without embedding source or diffs."""
    commit_result = _git(root, "rev-parse", "HEAD")
    status_result = _git(root, "status", "--porcelain=v1")
    untracked_result = _git(root, "ls-files", "--others", "--exclude-standard")
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    status_lines = [line for line in status_result.stdout.splitlines() if line]
    diff_bytes = _git_binary(root, "diff", "--binary", "HEAD")
    tracked_diff_sha = hashlib.sha256(diff_bytes).hexdigest().upper()

    untracked: dict[str, str] = {}
    for relative in sorted(untracked_result.stdout.splitlines()):
        relative = relative.strip().replace("\\", "/")
        if not relative or relative.startswith("manifests/"):
            continue
        path = root / relative
        if path.is_file():
            untracked[relative] = _sha256(path)

    content_descriptor = {
        "base_commit": commit,
        "tracked_diff_sha256": tracked_diff_sha,
        "untracked_files": untracked,
    }
    return {
        "base_commit": commit,
        "dirty": bool(status_lines),
        "changed_path_count": len(status_lines),
        "tracked_diff_sha256": tracked_diff_sha,
        "untracked_files": untracked,
        "worktree_content_sha256": _canonical_sha256(content_descriptor),
        "reconstruction_note": (
            "The base commit plus tracked diff and untracked file hashes identify "
            "this local state; a Git commit is still required for remote reconstruction."
        ),
    }


def _artifact_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def capture_artifact_snapshot(settings) -> dict[str, Any]:
    candidates = (
        settings.paths.data_cache / "01_aligned_dili_labs.parquet",
        settings.paths.data_cache / "02_dili_labels_censored.parquet",
        settings.paths.data_cache / "03_dili_dual_stream_tensors.parquet",
        settings.paths.data_cache / "03b_diag_tensors.parquet",
        settings.model_data_dir / "02_dili_labels_censored.parquet",
        settings.model_data_dir / "03_dili_dual_stream_tensors.parquet",
        settings.model_data_dir / "03b_diag_tensors.parquet",
        settings.paths.vocab / "vocab_polypharmacy.json",
        settings.paths.vocab / "vocab_diagnosis.json",
        settings.prediction_audit_dir / "temporal_leakage_audit.json",
        settings.diagnosis_audit_dir / "diagnosis_tensor_audit.json",
    )
    records = {
        path.resolve().relative_to(settings.paths.root).as_posix(): _artifact_record(
            settings.paths.root, path
        )
        for path in candidates
        if path.exists()
    }
    return dict(sorted(records.items()))


def _list_length(value: Any) -> int:
    return len(value) if isinstance(value, (list, np.ndarray)) else 0


def build_dataset_summary(settings) -> dict[str, Any]:
    labels = pd.read_parquet(
        settings.model_data_dir / "02_dili_labels_censored.parquet"
    )
    sequences = pd.read_parquet(
        settings.model_data_dir / "03_dili_dual_stream_tensors.parquet"
    )
    diagnoses = pd.read_parquet(settings.model_data_dir / "03b_diag_tensors.parquet")
    for column in ("first_med_time", "prediction_time", "index_time"):
        labels[column] = pd.to_datetime(labels[column])

    positive = labels["label_ahi_proxy"].eq(1)
    summary = {
        "encounters": int(len(labels)),
        "ahi_proxy_positive": int(positive.sum()),
        "ahi_proxy_negative": int((~positive).sum()),
        "unique_encounters": int(labels["encounter_id"].astype(str).nunique()),
        "first_med_time_min": labels["first_med_time"].min().isoformat(),
        "first_med_time_max": labels["first_med_time"].max().isoformat(),
        "prediction_time_min": labels["prediction_time"].min().isoformat(),
        "prediction_time_max": labels["prediction_time"].max().isoformat(),
        "index_time_min": labels["index_time"].min().isoformat(),
        "index_time_max": labels["index_time"].max().isoformat(),
        "medication_events": int(sequences["med_tokens"].map(_list_length).sum()),
        "laboratory_events": int(sequences["lab_tokens"].map(_list_length).sum()),
        "diagnosis_events": int(diagnoses["icd_codes"].map(_list_length).sum()),
        "encounters_without_pre_prediction_labs": int(
            sequences["lab_tokens"].map(_list_length).eq(0).sum()
        ),
        "encounters_without_pre_prediction_diagnosis": int(
            diagnoses["icd_codes"].map(_list_length).eq(0).sum()
        ),
    }
    summary["summary_sha256"] = _canonical_sha256(summary)
    return summary


def patient_group_from_encounter_id(encounter_id: Any) -> str:
    return str(encounter_id).split("_")[0]


def build_split_summary(
    cohort: pd.DataFrame,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Record the current deterministic outer GroupKFold; Code-05 may replace it."""
    required = {"encounter_id", "label_ahi_proxy"}
    if not required.issubset(cohort.columns):
        raise ValueError(f"cohort must contain {sorted(required)}")
    ordered = cohort[["encounter_id", "label_ahi_proxy"]].copy()
    ordered["encounter_id"] = ordered["encounter_id"].astype(str)
    ordered = ordered.sort_values("encounter_id", kind="mergesort").reset_index(drop=True)
    groups = ordered["encounter_id"].map(patient_group_from_encounter_id).to_numpy()
    labels = ordered["label_ahi_proxy"].astype(int).to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    fold_rows: list[dict[str, Any]] = []
    assignments: list[str] = []
    total_overlap = 0

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(ordered, labels, groups), start=1
    ):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = len(train_groups & test_groups)
        total_overlap += overlap
        membership = sorted(ordered.iloc[test_idx]["encounter_id"].tolist())
        fold_rows.append(
            {
                "fold": fold,
                "train_encounters": int(len(train_idx)),
                "test_encounters": int(len(test_idx)),
                "train_positive": int(labels[train_idx].sum()),
                "test_positive": int(labels[test_idx].sum()),
                "train_groups": int(len(train_groups)),
                "test_groups": int(len(test_groups)),
                "group_overlap": int(overlap),
                "test_membership_sha256": _canonical_sha256(membership),
            }
        )
        assignments.extend(f"{encounter_id}|{fold}" for encounter_id in membership)

    return {
        "status": "PASS" if total_overlap == 0 else "FAIL",
        "protocol": "current pre-Code-05 five-fold GroupKFold",
        "group_definition": "encounter_id prefix before first underscore",
        "n_splits": n_splits,
        "encounters": int(len(ordered)),
        "unique_groups": int(len(set(groups))),
        "total_group_overlap": int(total_overlap),
        "assignment_sha256": _canonical_sha256(sorted(assignments)),
        "folds": fold_rows,
    }


def _runtime_manifest() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }


def build_baseline_manifest(settings=None) -> dict[str, Any]:
    settings = settings or load_settings()
    seed_state = seed_everything(settings.reproducibility)
    labels = pd.read_parquet(
        settings.model_data_dir / "02_dili_labels_censored.parquet",
        columns=["encounter_id", "label_ahi_proxy"],
    )
    database_metadata = {
        "external_source": True,
        "file_name": settings.database_path.name,
        "read_only": settings.database_read_only,
        "bytes": int(settings.database_path.stat().st_size)
        if settings.database_path.exists()
        else None,
        "mtime_ns": int(settings.database_path.stat().st_mtime_ns)
        if settings.database_path.exists()
        else None,
        "content_hash_omitted": "external database is too large; schema/data artifacts are hashed",
    }
    manifest = {
        "schema_version": 1,
        "scope": "Code-00 baseline plus Code-04 deterministic controls",
        "generated_utc": _utc_now(),
        "privacy": "aggregate counts and whole-artifact hashes only; no row identifiers",
        "worktree": build_worktree_snapshot(settings.paths.root),
        "configuration": {
            "config_path": settings.config_path.relative_to(settings.paths.root).as_posix(),
            "config_sha256": _sha256(settings.config_path),
            "prediction_gap_hours": settings.prediction.gap_hours,
            "pseudo_index_seed": settings.prediction.pseudo_index_seed,
            "reproducibility": settings_manifest(settings.reproducibility),
        },
        "seed_state": seed_state,
        "runtime": _runtime_manifest(),
        "source_database": database_metadata,
        "artifacts": capture_artifact_snapshot(settings),
        "dataset_summary": build_dataset_summary(settings),
        "outer_split_baseline": build_split_summary(labels),
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "generated_utc"}
    )
    settings.baseline_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.reproducibility_audit_dir.mkdir(parents=True, exist_ok=True)
    settings.baseline_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    local_copy = settings.reproducibility_audit_dir / "baseline_manifest.json"
    local_copy.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[DILI-PLUS] Tracked baseline manifest: {settings.baseline_manifest_path}")
    return manifest


def _pseudo_index_digest(frame: pd.DataFrame) -> str:
    negative = frame.loc[
        frame["label_ahi_proxy"].eq(0),
        ["encounter_id", "index_time", "prediction_time"],
    ].copy()
    negative["encounter_id"] = negative["encounter_id"].astype(str)
    negative = negative.sort_values("encounter_id", kind="mergesort")
    values = [
        "|".join(
            (
                row.encounter_id,
                pd.Timestamp(row.index_time).isoformat(),
                pd.Timestamp(row.prediction_time).isoformat(),
            )
        )
        for row in negative.itertuples(index=False)
    ]
    return _canonical_sha256(values)


def audit_pseudo_index_sensitivity(settings=None) -> dict[str, Any]:
    """Audit pre-prediction event density across prespecified negative seeds."""
    settings = settings or load_settings()
    seeds = settings.reproducibility.pseudo_index_sensitivity_seeds
    if settings.prediction.pseudo_index_seed not in seeds:
        raise ValueError("Primary pseudo-index seed must be included in sensitivity seeds")
    legacy_path = settings.paths.data_cache / "02_dili_labels_censored.parquet"
    lab_path = settings.paths.data_cache / "01_aligned_dili_labs.parquet"
    current_path = settings.model_data_dir / "02_dili_labels_censored.parquet"
    current_labels = pd.read_parquet(current_path)
    current_digest = _pseudo_index_digest(current_labels)
    lab_sql = lab_path.as_posix().replace("'", "''")

    rows: list[dict[str, Any]] = []
    seed_frames: list[pd.DataFrame] = []
    conn = connect_source_database(settings)
    try:
        _load_frozen_legacy_label_table(conn, legacy_path)
        for seed in seeds:
            _build_prediction_label_table(conn, settings.prediction.gap_hours, seed)
            summary = conn.execute(
                """
                SELECT
                    COUNT(*) AS encounters,
                    SUM(label_ahi_proxy = 1)::BIGINT AS positive,
                    SUM(label_ahi_proxy = 0)::BIGINT AS negative,
                    AVG(EPOCH(prediction_time) - EPOCH(first_med_time)) / 3600.0
                        AS mean_observation_hours,
                    MEDIAN(EPOCH(prediction_time) - EPOCH(first_med_time)) / 3600.0
                        AS median_observation_hours
                FROM temp_ahi_prediction_labels
                """
            ).fetchone()
            frame = conn.execute(
                """
                SELECT
                    encounter_id,
                    label_ahi_proxy,
                    first_med_time,
                    index_time,
                    prediction_time
                FROM temp_ahi_prediction_labels
                ORDER BY encounter_id
                """
            ).df()
            frame["seed"] = int(seed)
            seed_frames.append(frame)
            digest = _pseudo_index_digest(frame)
            rows.append(
                {
                    "seed": int(seed),
                    "is_primary": bool(seed == settings.prediction.pseudo_index_seed),
                    "encounters": int(summary[0]),
                    "positive": int(summary[1]),
                    "negative": int(summary[2]),
                    "mean_observation_hours": float(summary[3]),
                    "median_observation_hours": float(summary[4]),
                    "negative_pseudo_index_sha256": digest,
                    "matches_current_primary_artifact": bool(
                        seed != settings.prediction.pseudo_index_seed
                        or digest == current_digest
                    ),
                }
            )

        df_all_seed_labels = pd.concat(seed_frames, ignore_index=True)
        conn.execute(
            "CREATE OR REPLACE TEMP TABLE temp_all_seed_labels AS "
            "SELECT * FROM df_all_seed_labels"
        )
        medication_counts = conn.execute(
            """
            SELECT
                c.seed,
                COUNT(*) AS medication_events,
                COUNT(DISTINCT c.encounter_id) AS encounters_with_medication
            FROM temp_all_seed_labels c
            INNER JOIN analysis.feature_medications m
                ON m.encounter_id = c.encounter_id
            WHERE m.start_time >= c.first_med_time
              AND m.start_time < c.prediction_time
              AND m.order_name IS NOT NULL
            GROUP BY c.seed
            """
        ).df()
        laboratory_counts = conn.execute(
            f"""
            SELECT
                c.seed,
                COUNT(*) AS laboratory_events,
                COUNT(DISTINCT c.encounter_id) AS encounters_with_laboratory
            FROM temp_all_seed_labels c
            INNER JOIN read_parquet('{lab_sql}') l
                ON l.encounter_id = c.encounter_id
            WHERE l.lab_time >= c.first_med_time
              AND l.lab_time < c.prediction_time
              AND TRY_CAST(l.lab_value AS DOUBLE) IS NOT NULL
            GROUP BY c.seed
            """
        ).df()
        medication_by_seed = medication_counts.set_index("seed").to_dict("index")
        laboratory_by_seed = laboratory_counts.set_index("seed").to_dict("index")
        for row in rows:
            seed = row["seed"]
            row.update(
                {
                    key: int(value)
                    for key, value in medication_by_seed[seed].items()
                }
            )
            row.update(
                {
                    key: int(value)
                    for key, value in laboratory_by_seed[seed].items()
                }
            )
    finally:
        conn.close()

    primary = next(row for row in rows if row["is_primary"])
    audit = {
        "audit_status": "PASS"
        if primary["matches_current_primary_artifact"]
        and len({row["encounters"] for row in rows}) == 1
        and len({row["positive"] for row in rows}) == 1
        else "FAIL",
        "prediction_gap_hours": settings.prediction.gap_hours,
        "primary_seed": settings.prediction.pseudo_index_seed,
        "algorithm": "DuckDB HASH(encounter_id || ':' || seed), pinned by environment",
        "current_primary_negative_pseudo_index_sha256": current_digest,
        "seed_results": rows,
    }
    audit["sensitivity_payload_sha256"] = _canonical_sha256(rows)
    output_dir = settings.reproducibility_audit_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pseudo_index_sensitivity.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(output_dir / "pseudo_index_sensitivity.csv", index=False)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["audit_status"] != "PASS":
        raise RuntimeError("Pseudo-index sensitivity audit failed")
    return audit


def capture_rebuild_snapshot(settings, name: str) -> dict[str, Any]:
    snapshot = {
        "name": name,
        "created_utc": _utc_now(),
        "artifacts": capture_artifact_snapshot(settings),
        "dataset_summary": build_dataset_summary(settings),
    }
    labels = pd.read_parquet(
        settings.model_data_dir / "02_dili_labels_censored.parquet",
        columns=["encounter_id", "label_ahi_proxy"],
    )
    snapshot["outer_split_baseline"] = build_split_summary(labels)
    snapshot["comparison_payload_sha256"] = _canonical_sha256(
        {
            "artifacts": snapshot["artifacts"],
            "dataset_summary": snapshot["dataset_summary"],
            "outer_split_baseline": snapshot["outer_split_baseline"],
        }
    )
    output = settings.reproducibility_audit_dir / f"deterministic_rebuild_{name}.json"
    output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return snapshot


def compare_rebuild_snapshots(
    first: dict[str, Any], second: dict[str, Any], settings=None
) -> dict[str, Any]:
    settings = settings or load_settings()
    sections = ("artifacts", "dataset_summary", "outer_split_baseline")
    matches = {section: first[section] == second[section] for section in sections}
    comparison = {
        "audit_status": "PASS" if all(matches.values()) else "FAIL",
        "first": first["name"],
        "second": second["name"],
        "section_matches": matches,
        "first_payload_sha256": first["comparison_payload_sha256"],
        "second_payload_sha256": second["comparison_payload_sha256"],
    }
    output = settings.reproducibility_audit_dir / "deterministic_rebuild_comparison.json"
    output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if comparison["audit_status"] != "PASS":
        raise RuntimeError("Deterministic rebuild comparison failed")
    return comparison


if __name__ == "__main__":
    build_baseline_manifest()
