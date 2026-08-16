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
    ("diagnoses", "diliplus.data.diagnoses", "build_diag_tensors"),
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
    args = parser.parse_args(argv)
    run(load_settings(args.config))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
