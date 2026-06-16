"""Local source repository metadata used during the migration phase."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MYFUSION_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = MYFUSION_ROOT / "myfusion" / "legacy"


@dataclass(frozen=True)
class SourceRepo:
    name: str
    path: Path
    license_note: str
    role: str


SOURCE_REPOS: dict[str, SourceRepo] = {
    "tpgdiff": SourceRepo(
        name="TPGDiff",
        path=LEGACY_ROOT / "tpgdiff",
        license_note="MIT in local LICENSE; verify conflict markers before publishing copied source.",
        role="restoration backbone and degradation prior reference",
    ),
    "rar": SourceRepo(
        name="RAR",
        path=LEGACY_ROOT / "rar",
        license_note="CC BY-NC 4.0 in local LICENSE.",
        role="latent-space QA loop and iterative restoration reference",
    ),
    "depictqa": SourceRepo(
        name="DepictQA",
        path=LEGACY_ROOT / "depictqa",
        license_note="Apache 2.0 in local LICENSE.",
        role="quality assessment and reasoning reference",
    ),
    "methodhub": SourceRepo(
        name="MethodHub",
        path=MYFUSION_ROOT / "methodhub",
        license_note="local adapter code written for this workspace",
        role="temporary adapter registry",
    ),
}


def source_repo(name: str) -> SourceRepo:
    key = name.lower()
    if key not in SOURCE_REPOS:
        raise KeyError(f"Unknown source repo: {name}")
    return SOURCE_REPOS[key]
