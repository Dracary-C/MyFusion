"""Compatibility import for the current TPGD + RAR pipeline.

The implementation still lives in `my_method.py` so the existing UI and CLI keep
working. New code can import from `myfusion.pipelines.tpgd_rar` immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_method import HybridStep, RARTPGDiffFusion  # noqa: E402

__all__ = ["HybridStep", "RARTPGDiffFusion"]
