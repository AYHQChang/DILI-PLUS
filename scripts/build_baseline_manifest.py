"""Write the aggregate-only Code-00/Code-04 tracked baseline manifest."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.reproducibility_audit import build_baseline_manifest


if __name__ == "__main__":
    build_baseline_manifest()
