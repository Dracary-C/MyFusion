from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


MYFUSION_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = MYFUSION_ROOT / "myfusion" / "legacy"
ALL_IN_ONE_ROOT = MYFUSION_ROOT.parent

_DEFAULT_REPOS = {
    "tpgdiff": LEGACY_ROOT / "tpgdiff",
    "rar": LEGACY_ROOT / "rar",
    "depictqa": LEGACY_ROOT / "depictqa",
}

_EXTERNAL_REPOS = {
    "tpgdiff": ALL_IN_ONE_ROOT / "TPGDiff",
    "rar": ALL_IN_ONE_ROOT / "RAR",
    "depictqa": ALL_IN_ONE_ROOT / "DepictQA",
}


def default_repo_root(key: str) -> Path:
    env_key = f"METHODHUB_{key.upper().replace('-', '_')}_ROOT"
    env_value = os.environ.get(env_key)
    if env_value:
        return Path(env_value).expanduser().resolve()
    if key not in _DEFAULT_REPOS:
        raise KeyError(f"Unknown repo key: {key}")
    internal = _DEFAULT_REPOS[key]
    if internal.exists():
        return internal.resolve()
    external = _EXTERNAL_REPOS.get(key)
    if external is not None and external.exists():
        return external.resolve()
    return internal.resolve()


def require_exists(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


@contextmanager
def push_sys_path(*paths: Path) -> Iterator[None]:
    inserted = []
    for path in reversed([Path(p).resolve() for p in paths if p]):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
            inserted.append(value)
    try:
        yield
    finally:
        for value in inserted:
            if value in sys.path:
                sys.path.remove(value)


@contextmanager
def push_cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)

