"""Run the P0 high-resolution analysis without fitting any model."""

import argparse

from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()
from diliplus.config import load_settings
from diliplus.evaluation.p0_statistical_refinement import (
    refresh_p0_manifest,
    run_p0_statistical_refinement,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Verify existing outputs and refresh lineage hashes without bootstrapping again.",
    )
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    if args.manifest_only:
        refresh_p0_manifest(settings)
    else:
        run_p0_statistical_refinement(settings, max_workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
