"""Build P1 audits in isolated processes to avoid Windows native-library conflicts."""

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT

STAGES = (
    ("analysis", "diliplus.evaluation.p1_supplementary", "run_p1_analysis"),
    ("figure_s1", "diliplus.reporting.figures.supplementary_p1", "generate_calibration_decision_figure"),
    ("figure_s2", "diliplus.reporting.figures.supplementary_p1", "generate_subgroup_process_figure"),
    ("table_s1", "diliplus.reporting.table_s1", "generate_supplementary_table_s1"),
    ("manifest", "diliplus.reporting.p1_manifest", "generate_p1_manifest"),
)


def _default_render_python() -> str:
    """Prefer the base Conda interpreter when an env-specific Matplotlib DLL fails."""
    current = Path(sys.executable).resolve()
    if current.parent.parent.name.lower() == "envs":
        candidate = current.parent.parent.parent / "python.exe"
        if candidate.is_file():
            return str(candidate)
    return str(current)


def run(config_path, render_python=None):
    src = str(PROJECT_ROOT / "src")
    render_python = str(render_python or _default_render_python())
    for name, module, function in STAGES:
        print(f"[DILI-PLUS][P1] Isolated stage: {name}", flush=True)
        code = (
            "import sys; "
            f"sys.path.insert(0, {src!r}); "
            "from diliplus.config import load_settings; "
            f"from {module} import {function}; "
            f"{function}(load_settings({str(config_path)!r}))"
        )
        executable = sys.executable if name == "analysis" else render_python
        subprocess.run([executable, "-c", code], cwd=PROJECT_ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build DILI-PLUS P1 supplementary audits")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--render-python",
        default=None,
        help="Interpreter for Matplotlib/table/manifest stages; defaults to base Conda when detected.",
    )
    args = parser.parse_args(argv)
    run(args.config, args.render_python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
