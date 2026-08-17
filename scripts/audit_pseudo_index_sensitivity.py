"""Run the prespecified aggregate-only pseudo-index seed sensitivity audit."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diliplus.reproducibility_audit import audit_pseudo_index_sensitivity


if __name__ == "__main__":
    audit_pseudo_index_sensitivity()
