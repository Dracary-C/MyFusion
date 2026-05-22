"""Bootstrap helpers for importing the legacy DepictQA snapshot.

DepictQA source expects `src` to be on sys.path so imports like `from model...`
can resolve. Use this only during migration.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

DEPICTQA_LEGACY_ROOT = Path(__file__).resolve().parent
DEPICTQA_SRC_ROOT = DEPICTQA_LEGACY_ROOT / 'src'


@contextlib.contextmanager
def depictqa_legacy_path():
    paths = [str(DEPICTQA_SRC_ROOT), str(DEPICTQA_LEGACY_ROOT)]
    inserted = []
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
            inserted.append(path)
    try:
        yield
    finally:
        for path in inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(path)
