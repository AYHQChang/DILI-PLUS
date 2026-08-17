"""Build Table 1 and the current real-data paper figures."""

import argparse
from importlib import import_module

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()
from diliplus.config import load_settings

STAGES = (
    ("table_1", "diliplus.reporting.table1", "generate_table_1"),
    ("figure_1b", "diliplus.reporting.figures.data_landscape", "generate_landscape_figure"),
    ("figure_1d", "diliplus.reporting.figures.biomarker_divergence", "generate_divergence_plot"),
    ("figure_2", "diliplus.reporting.figures.model_comparison", "generate_advanced_figure_2"),
    ("figure_3", "diliplus.reporting.figures.calibration_impact", "generate_figure_3"),
    ("figure_4", "diliplus.reporting.figures.early_warning", "generate_early_warning_figure"),
    ("figure_5", "diliplus.reporting.figures.attribution", "generate_waterfall_chart"),
    ("figure_6", "diliplus.reporting.figures.perturbation", "generate_simulation_figure"),
)

def run(settings, stages=STAGES):
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
        else STAGES
    )
    run(load_settings(args.config), selected)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
