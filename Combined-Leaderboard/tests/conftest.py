"""Shared import paths for backend, evaluation, and operational tests."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

for import_root in (
    PROJECT_ROOT,
    PROJECT_ROOT / "backend",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "evaluation" / "spatial_reasoning",
):
    resolved = str(import_root)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
