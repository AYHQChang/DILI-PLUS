"""Create a tracked, aggregate-only manifest for formal paper assets."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from diliplus.config import load_settings
from diliplus.reporting.formal_assets import auxiliary_sources, formal_sources, sha256

import matplotlib
import pandas as pd


ASSET_NAMES = (
    "figures/Fig_1_Study_Design.png",
    "figures/Fig_1_Study_Design.pdf",
    "reports/paper_assets/Table_02_Formal_Model_Performance_numeric.csv",
    "reports/paper_assets/Table_02_Formal_Model_Performance.csv",
    "reports/paper_assets/Table_02_Formal_Model_Performance.txt",
    "reports/paper_assets/Table_02_Formal_Model_Performance.tex",
    "figures/Fig_2_Observation_Measurement_Process.png",
    "figures/Fig_2_Observation_Measurement_Process.pdf",
    "figures/Fig_3_Compact_Predictive_Benchmark.png",
    "figures/Fig_3_Compact_Predictive_Benchmark.pdf",
    "figures/Fig_4_Earlier_Cutoff_Information_Erosion.png",
    "figures/Fig_4_Earlier_Cutoff_Information_Erosion.pdf",
    "figures/Fig_5_Model_Complexity_Audit.png",
    "figures/Fig_5_Model_Complexity_Audit.pdf",
)
GENERATOR_NAMES = (
    "src/diliplus/reporting/formal_assets.py",
    "src/diliplus/reporting/table2.py",
    "src/diliplus/reporting/figures/study_design.py",
    "src/diliplus/reporting/figures/observation_process.py",
    "src/diliplus/reporting/figures/calibration_impact.py",
    "src/diliplus/reporting/figures/early_warning.py",
    "src/diliplus/reporting/figures/robustness.py",
    "src/diliplus/reporting/paper_manifest.py",
    "pipelines/05_build_paper_assets.py",
)


def _record(path: Path, root: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required paper asset is missing or empty: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def generate_paper_asset_manifest(settings=None) -> Path:
    settings = settings or load_settings()
    root = settings.paths.root
    formal = formal_sources(settings)
    auxiliary = auxiliary_sources(settings)
    assets = [_record(root / name, root) for name in ASSET_NAMES]
    generators = [_record(root / name, root) for name in GENERATOR_NAMES]
    input_paths = [
        settings.paths.manifests / "code08_cohort_table1.json",
        settings.paths.manifests / "code10_label_rebuild_audit.json",
        settings.paths.manifests / "code05_split_protocol.json",
        settings.paths.manifests / "code09_early_warning_contract.json",
        settings.paths.reports / "p0_08_cohort_table1" / "table1_characteristics.csv",
    ]
    for values in (formal, auxiliary):
        for value in values.values():
            if isinstance(value, Path) and value.is_file():
                input_paths.append(value)
    unique_inputs = sorted(set(input_paths), key=lambda path: path.as_posix())
    manifest = {
        "schema_version": 2,
        "contract": "code11_reframed_paper_assets_v2",
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "formal_run_id": formal["run_id"],
        "ablation_run_id": auxiliary["ablation_run_id"],
        "sensitivity_run_id": auxiliary["sensitivity_run_id"],
        "probability_mode": "calibrated",
        "primary_metric": "AUPRC",
        "uncertainty": "1,000-replicate patient-cluster bootstrap 95% confidence intervals",
        "model_palette_contract": "one immutable colour per formal model across all figures",
        "main_figure_narrative": {
            "figure_1": "cohort construction and prediction-time contract",
            "figure_2": "cohort observation and measurement process",
            "figure_3": "compact discrimination, calibration, and alert-budget benchmark",
            "figure_4": "earlier-cutoff information erosion and matched stress test",
            "figure_5": "incremental value, architecture sensitivity, stability, and resource audit",
        },
        "rendering_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "backend": matplotlib.get_backend(),
        },
        "assets": assets,
        "source_files": [_record(path, root) for path in unique_inputs],
        "generator_files": generators,
        "privacy": "aggregate statistics, source hashes, and generated paper assets only; no row identifiers",
    }
    payload = dict(manifest)
    payload_bytes = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    manifest["payload_sha256"] = hashlib.sha256(payload_bytes).hexdigest().upper()
    output = settings.paths.manifests / "code11_paper_assets.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] Paper asset manifest: {output}")
    return output


if __name__ == "__main__":
    generate_paper_asset_manifest()
