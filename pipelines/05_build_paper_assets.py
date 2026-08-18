"""Build cohort, formal result, and explanatory paper assets."""

import argparse
from importlib import import_module

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()
from diliplus.config import load_settings

STAGES = (
    ("table_1", "diliplus.reporting.table1", "generate_table_1"),
    ("table_2", "diliplus.reporting.table2", "generate_table_2"),
    ("figure_1", "diliplus.reporting.figures.study_design", "generate_study_design_figure"),
    ("figure_2", "diliplus.reporting.figures.observation_process", "generate_observation_process_figure"),
    ("figure_3", "diliplus.reporting.figures.calibration_impact", "generate_figure_3"),
    ("figure_4", "diliplus.reporting.figures.early_warning", "generate_early_warning_figure"),
    ("figure_5", "diliplus.reporting.figures.robustness", "generate_robustness_figure"),
    ("figure_6", "diliplus.reporting.figures.attribution", "generate_waterfall_chart"),
    ("figure_7", "diliplus.reporting.figures.perturbation", "generate_simulation_figure"),
    ("asset_manifest", "diliplus.reporting.paper_manifest", "generate_paper_asset_manifest"),
)
DEFAULT_STAGES = tuple(
    stage for stage in STAGES if stage[0] not in {"figure_6", "figure_7"}
)

def run(settings, stages=DEFAULT_STAGES):
    for stage_name, module_name, function_name in stages:
        print(f"\n[DILI-PLUS] Paper asset stage: {stage_name}")
        getattr(import_module(module_name), function_name)(settings=settings)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Build DILI-PLUS paper assets")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=tuple(stage[0] for stage in STAGES),
        help="Build only selected assets while preserving the declared stage order.",
    )
    args = parser.parse_args(argv)
    selected = (
        tuple(stage for stage in STAGES if stage[0] in set(args.stages))
        if args.stages
        else DEFAULT_STAGES
    )
    run(load_settings(args.config), selected)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
