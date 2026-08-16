"""Allow legacy root scripts to import the src-layout package without installation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"


def bootstrap() -> Path:
    src = str(SRC_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)
    return PROJECT_ROOT

