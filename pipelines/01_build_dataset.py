"""Build the DILI cohort and model-ready datasets in dependency order."""

import argparse
from importlib import import_module

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()
from diliplus.config import load_settings

STAGES = (
    ("cohort", "diliplus.data.cohort", "build_dili_cohort"),
    ("labels", "diliplus.data.labels", "extract_dili_labels_and_censor"),
    ("sequences", "diliplus.data.sequences", "build_dili_tensors"),
    ("temporal_audit", "diliplus.data.temporal_audit", "audit_prediction_time_contract"),
    ("diagnosis_source_audit", "diliplus.data.diagnosis_audit", "audit_diagnosis_source"),
    ("diagnoses", "diliplus.data.diagnoses", "build_diag_tensors"),
    ("diagnosis_audit", "diliplus.data.diagnosis_audit", "audit_diagnosis_time_contract"),
    ("vocabulary", "diliplus.data.vocabulary", "build_vocabulary"),
    ("drug_mapping", "diliplus.explainability.ontology", "main"),
)

def run(settings, stages=STAGES):
    for stage_name, module_name, function_name in stages:
        print(f"\n[DILI-PLUS] Dataset stage: {stage_name}")
        getattr(import_module(module_name), function_name)(settings=settings)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Build DILI-PLUS datasets by stage")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[stage[0] for stage in STAGES],
        help="Run only the selected stages, preserving canonical dependency order.",
    )
    args = parser.parse_args(argv)
    selected = STAGES
    if args.stages:
        requested = set(args.stages)
        selected = tuple(stage for stage in STAGES if stage[0] in requested)
    run(load_settings(args.config), selected)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
