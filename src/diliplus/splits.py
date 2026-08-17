"""Deterministic four-way grouped split contracts for DILI-PLUS.

Each outer fold has four mutually exclusive roles:

* ``training`` fits model parameters;
* ``selection`` chooses epochs/hyperparameters;
* ``calibration`` fits post-hoc calibration only;
* ``test`` is consumed once for final inference.

The split code sorts encounters internally before splitting, so membership does
not depend on Parquet row order. Patient groups are derived from the encounter
identifier prefix used by the existing project.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from diliplus.config import load_settings
from diliplus.reproducibility import derive_seed


ROLE_NAMES = ("training", "selection", "calibration", "test")


def patient_group_from_encounter_id(encounter_id: Any) -> str:
    return str(encounter_id).split("_")[0]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _as_index_array(values: Iterable[int]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int64)


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    training: np.ndarray
    selection: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    summary: dict[str, Any]

    def indices_for(self, role: str) -> np.ndarray:
        if role not in ROLE_NAMES:
            raise KeyError(f"Unknown split role: {role}")
        return getattr(self, role)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "indices": {
                role: self.indices_for(role).astype(int).tolist()
                for role in ROLE_NAMES
            },
            "summary": self.summary,
        }


def _choose_grouped_holdout(
    pool_indices: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
    holdout_fraction: float,
    seed: int,
    attempts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a deterministic grouped holdout with both classes in both sides."""
    splitter = GroupShuffleSplit(
        n_splits=attempts,
        test_size=holdout_fraction,
        random_state=seed,
    )
    pool_groups = groups[pool_indices]
    pool_labels = labels[pool_indices]
    target_prevalence = float(pool_labels.mean())
    candidates: list[tuple[tuple[float, float, str], np.ndarray, np.ndarray]] = []

    for remain_loc, holdout_loc in splitter.split(
        np.zeros(len(pool_indices)), pool_labels, pool_groups
    ):
        remain = pool_indices[remain_loc]
        holdout = pool_indices[holdout_loc]
        if len(np.unique(labels[remain])) < 2 or len(np.unique(labels[holdout])) < 2:
            continue
        size_error = abs(len(holdout) / len(pool_indices) - holdout_fraction)
        prevalence_error = abs(float(labels[remain].mean()) - target_prevalence) + abs(
            float(labels[holdout].mean()) - target_prevalence
        )
        tie_break = _canonical_sha256(sorted(groups[holdout].astype(str).tolist()))
        candidates.append(
            ((prevalence_error, size_error, tie_break), remain, holdout)
        )

    if not candidates:
        raise ValueError(
            "Unable to create a grouped holdout containing both outcome classes; "
            "increase split_search_attempts or revise the split fractions"
        )
    _, remain, holdout = min(candidates, key=lambda item: item[0])
    return np.sort(remain), np.sort(holdout)


def _role_summary(
    indices: np.ndarray,
    encounter_ids: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    membership = sorted(encounter_ids[indices].astype(str).tolist())
    return {
        "encounters": int(len(indices)),
        "positive": int(labels[indices].sum()),
        "negative": int(len(indices) - labels[indices].sum()),
        "groups": int(len(set(groups[indices].astype(str)))),
        "prevalence_pct": round(float(labels[indices].mean() * 100.0), 6),
        "membership_sha256": _canonical_sha256(membership),
    }


def _validate_fold(
    fold: int,
    role_indices: dict[str, np.ndarray],
    encounter_ids: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    expected = set(range(len(encounter_ids)))
    index_sets = {role: set(indices.astype(int)) for role, indices in role_indices.items()}
    if set().union(*index_sets.values()) != expected:
        raise ValueError(f"Fold {fold} roles do not cover the complete cohort")

    index_overlap: dict[str, int] = {}
    group_overlap: dict[str, int] = {}
    for left, right in combinations(ROLE_NAMES, 2):
        key = f"{left}__{right}"
        index_overlap[key] = len(index_sets[left] & index_sets[right])
        left_groups = set(groups[role_indices[left]].astype(str))
        right_groups = set(groups[role_indices[right]].astype(str))
        group_overlap[key] = len(left_groups & right_groups)
    if any(index_overlap.values()) or any(group_overlap.values()):
        raise ValueError(f"Fold {fold} violates role isolation")

    roles = {
        role: _role_summary(indices, encounter_ids, groups, labels)
        for role, indices in role_indices.items()
    }
    if any(roles[role]["positive"] == 0 or roles[role]["negative"] == 0 for role in ROLE_NAMES):
        raise ValueError(f"Fold {fold} has a single-class role")
    return {
        "fold": fold,
        "status": "PASS",
        "roles": roles,
        "index_overlap": index_overlap,
        "group_overlap": group_overlap,
        "test_consumption_contract": "single final inference pass only",
    }


def build_nested_grouped_splits(
    encounter_ids: Iterable[Any],
    labels: Iterable[int],
    settings=None,
) -> list[FoldSplit]:
    settings = settings or load_settings()
    encounter_ids = np.asarray([str(value) for value in encounter_ids], dtype=object)
    labels = np.asarray(list(labels), dtype=np.int64)
    if len(encounter_ids) != len(labels) or len(encounter_ids) == 0:
        raise ValueError("encounter_ids and labels must have the same non-zero length")
    if len(set(encounter_ids.tolist())) != len(encounter_ids):
        raise ValueError("encounter_id values must be unique before split construction")
    if not set(np.unique(labels)).issubset({0, 1}) or len(np.unique(labels)) != 2:
        raise ValueError("labels must contain both binary classes")

    groups = np.asarray(
        [patient_group_from_encounter_id(value) for value in encounter_ids], dtype=object
    )
    order = np.argsort(encounter_ids.astype(str), kind="stable")
    ordered_labels = labels[order]
    ordered_groups = groups[order]
    protocol = settings.evaluation_protocol
    outer = StratifiedGroupKFold(
        n_splits=protocol.outer_folds,
        shuffle=True,
        random_state=settings.reproducibility.split_seed,
    )
    folds: list[FoldSplit] = []
    test_coverage: list[int] = []

    for fold, (outer_train_loc, test_loc) in enumerate(
        outer.split(np.zeros(len(order)), ordered_labels, ordered_groups), start=1
    ):
        outer_train = order[outer_train_loc]
        test = np.sort(order[test_loc])
        development, calibration = _choose_grouped_holdout(
            outer_train,
            groups,
            labels,
            protocol.calibration_fraction,
            derive_seed(settings.reproducibility.split_seed, "calibration", fold),
            protocol.split_search_attempts,
        )
        relative_selection_fraction = protocol.selection_fraction / (
            1.0 - protocol.calibration_fraction
        )
        training, selection = _choose_grouped_holdout(
            development,
            groups,
            labels,
            relative_selection_fraction,
            derive_seed(settings.reproducibility.split_seed, "selection", fold),
            protocol.split_search_attempts,
        )
        role_indices = {
            "training": _as_index_array(training),
            "selection": _as_index_array(selection),
            "calibration": _as_index_array(calibration),
            "test": _as_index_array(test),
        }
        summary = _validate_fold(
            fold, role_indices, encounter_ids, groups, labels
        )
        folds.append(FoldSplit(fold=fold, summary=summary, **role_indices))
        test_coverage.extend(test.astype(int).tolist())

    if sorted(test_coverage) != list(range(len(encounter_ids))):
        raise ValueError("Outer test folds do not cover every encounter exactly once")
    return folds


def aggregate_split_manifest(
    folds: list[FoldSplit],
    encounter_ids: Iterable[Any],
    labels: Iterable[int],
    settings=None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    encounter_ids = np.asarray([str(value) for value in encounter_ids], dtype=object)
    labels = np.asarray(list(labels), dtype=np.int64)
    protocol = settings.evaluation_protocol
    fold_summaries = [fold.summary for fold in folds]
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Code-05 four-way grouped evaluation protocol",
        "privacy": "aggregate counts and membership hashes only; no row identifiers",
        "status": "PASS",
        "protocol": {
            "outer": "StratifiedGroupKFold",
            "outer_folds": protocol.outer_folds,
            "inner": "deterministic best-of-N GroupShuffleSplit candidates",
            "selection_fraction_of_outer_train": protocol.selection_fraction,
            "calibration_fraction_of_outer_train": protocol.calibration_fraction,
            "split_search_attempts": protocol.split_search_attempts,
            "group_definition": "encounter_id prefix before first underscore",
            "roles": {
                "training": "model parameter fitting only",
                "selection": "epoch/hyperparameter selection only",
                "calibration": "temperature fitting only",
                "test": "one final raw-logit inference pass only",
            },
        },
        "cohort": {
            "encounters": int(len(encounter_ids)),
            "positive": int(labels.sum()),
            "negative": int(len(labels) - labels.sum()),
            "membership_sha256": _canonical_sha256(sorted(encounter_ids.tolist())),
        },
        "folds": fold_summaries,
        "all_outer_test_memberships_sha256": _canonical_sha256(
            sorted(
                summary["roles"]["test"]["membership_sha256"]
                for summary in fold_summaries
            )
        ),
    }


def save_split_audit(
    folds: list[FoldSplit],
    encounter_ids: Iterable[Any],
    labels: Iterable[int],
    run_id: str,
    settings=None,
) -> tuple[Path, Path]:
    settings = settings or load_settings()
    aggregate = aggregate_split_manifest(folds, encounter_ids, labels, settings)
    aggregate["configuration_sha256"] = _file_sha256(settings.config_path)
    aggregate["split_implementation_sha256"] = _file_sha256(Path(__file__))
    aggregate["manifest_payload_sha256"] = _canonical_sha256(
        {key: value for key, value in aggregate.items() if key != "generated_utc"}
    )
    tracked_path = settings.paths.manifests / "code05_split_protocol.json"
    local_dir = settings.paths.reports / "p0_05_code_06" / run_id
    local_path = local_dir / "split_indices.json"
    tracked_path.parent.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    tracked_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    local_payload = {
        "aggregate_manifest": aggregate,
        "folds": [fold.checkpoint_payload() for fold in folds],
    }
    local_path.write_text(
        json.dumps(local_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return tracked_path, local_path
