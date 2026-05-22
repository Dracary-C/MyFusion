"""Bootstrap helpers for importing the legacy RAR snapshot.

Use this only during migration. Native MyFusion modules should avoid relying on
absolute imports such as `import iqa` or `import diffusion`.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

RAR_LEGACY_ROOT = Path(__file__).resolve().parent


@contextlib.contextmanager
def rar_legacy_path():
    path = str(RAR_LEGACY_ROOT)
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(path)
