"""Aggregate-only manifest for the post-hoc P1 reporting audit."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import pandas as pd

from diliplus.config import load_settings
from diliplus.reporting.formal_assets import formal_sources, sha256
from diliplus.reporting.p1_contract import RUN_ID


ASSETS = (
    "figures/Fig_S1_Calibration_Decision_Curves.png",
    "figures/Fig_S1_Calibration_Decision_Curves.pdf",
    "figures/Fig_S2_Subgroup_Process_Audit.png",
    "figures/Fig_S2_Subgroup_Process_Audit.pdf",
    "reports/paper_assets/Table_S1_Subgroup_Process_Audit.csv",
    "reports/paper_assets/Table_S1_Subgroup_Process_Audit.tex",
    f"reports/runs/{RUN_ID}/metrics/process_pooled_metrics.csv",
    f"reports/runs/{RUN_ID}/metrics/process_bootstrap_intervals.csv",
    f"reports/runs/{RUN_ID}/metrics/process_temperatures.csv",
    f"reports/runs/{RUN_ID}/metrics/process_coefficients.csv",
    f"reports/runs/{RUN_ID}/metrics/calibration_bin_intervals.csv",
    f"reports/runs/{RUN_ID}/metrics/subgroup_metrics.csv",
    f"reports/runs/{RUN_ID}/metrics/subgroup_definitions.json",
)
GENERATORS = (
    "src/diliplus/evaluation/p1_supplementary.py",
    "src/diliplus/reporting/p1_contract.py",
    "src/diliplus/reporting/figures/supplementary_p1.py",
    "src/diliplus/reporting/table_s1.py",
    "src/diliplus/reporting/p1_manifest.py",
    "pipelines/06_build_p1_supplementary.py",
)


def _record(path: Path, root: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def generate_p1_manifest(settings=None) -> Path:
    settings = settings or load_settings()
    root = settings.paths.root
    formal = formal_sources(settings)
    source_paths = [
        settings.paths.manifests / "code10_formal_run.json",
        settings.model_data_dir / "02_dili_labels_censored.parquet",
        settings.model_data_dir / "03_dili_dual_stream_tensors.parquet",
        settings.model_data_dir / "03b_diag_tensors.parquet",
        formal["calibration_bins"],
        formal["dca"],
    ]
    process = pd.read_csv(root / f"reports/runs/{RUN_ID}/metrics/process_pooled_metrics.csv")
    calibrated = process[process["Probability_Mode"] == "calibrated"].iloc[0]
    subgroup = pd.read_csv(root / f"reports/runs/{RUN_ID}/metrics/subgroup_metrics.csv")
    manifest = {
        "schema_version": 1,
        "contract": "post_hoc_p1_reporting_audit_v1",
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "formal_run_id": formal["run_id"],
        "analysis_status": "post_hoc_descriptive_audit_not_used_for_formal_model_ranking",
        "bootstrap_replicates": settings.evaluation_protocol.bootstrap_replicates,
        "process_baseline": {
            "model": "unweighted logistic regression",
            "features": "observation duration, modality counts/densities, and availability only; no clinical token identity or value",
            "AUPRC": float(calibrated["AUPRC"]),
            "AUROC": float(calibrated["AUROC"]),
            "Brier": float(calibrated["Brier"]),
            "NLL": float(calibrated["NLL"]),
            "Calibration_Slope": float(calibrated["Calibration_Slope"]),
            "Observed_Expected_Ratio": float(calibrated["Observed_Expected_Ratio"]),
        },
        "subgroup_audit": {
            "models": sorted(subgroup["Model_Architecture"].unique().tolist()),
            "dimensions": sorted(subgroup["audit_dimension"].unique().tolist()),
            "interpretation": "descriptive heterogeneity/process audit; not proof of fairness or transportability",
        },
        "rendering_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "backend": matplotlib.get_backend(),
        },
        "assets": [_record(root / path, root) for path in ASSETS],
        "source_files": [_record(Path(path), root) for path in source_paths],
        "generator_files": [_record(root / path, root) for path in GENERATORS],
        "privacy": "tracked outputs are aggregate; row-level process-baseline predictions stay in the run directory",
    }
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["payload_sha256"] = hashlib.sha256(payload).hexdigest().upper()
    output = settings.paths.manifests / "code12_p1_supplementary.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] P1 manifest: {output}")
    return output


if __name__ == "__main__":
    generate_p1_manifest()
