"""Bootstrap helpers for importing the legacy TPGDiff snapshot."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

TPGDIFF_LEGACY_ROOT = Path(__file__).resolve().parent
TPGD_SDE_ROOT = TPGDIFF_LEGACY_ROOT / 'universal-restoration' / 'config' / 'tpgd-sde'
TPGD_UNIVERSAL_ROOT = TPGDIFF_LEGACY_ROOT / 'universal-restoration'
TPGD_PRIOR_SRC_ROOT = TPGDIFF_LEGACY_ROOT / 'tpgd' / 'src'


@contextlib.contextmanager
def tpgdiff_legacy_path():
    paths = [str(TPGD_SDE_ROOT), str(TPGD_UNIVERSAL_ROOT), str(TPGD_PRIOR_SRC_ROOT)]
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
