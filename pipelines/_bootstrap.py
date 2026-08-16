"""Make src-layout imports independent of the current working directory."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

def bootstrap() -> Path:
    for path in (PROJECT_ROOT, SRC_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return PROJECT_ROOT
