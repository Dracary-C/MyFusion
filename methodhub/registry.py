from __future__ import annotations

from typing import Any, Dict, Iterable

from .adapters.depictqa import DepictQAAdapter
from .adapters.rar import RARAssessmentAdapter, RARConnectorAdapter, RARRuntimeAdapter
from .adapters.tpgdiff import TPGDiffPriorAdapter, TPGDiffRuntimeAdapter


_ALIASES = {
    "depictqa": DepictQAAdapter,
    "rar": RARRuntimeAdapter,
    "rar-runtime": RARRuntimeAdapter,
    "rar-assessment": RARAssessmentAdapter,
    "rar-connector": RARConnectorAdapter,
    "tpgdiff": TPGDiffPriorAdapter,
    "tpgdiff-prior": TPGDiffPriorAdapter,
    "tpgdiff-runtime": TPGDiffRuntimeAdapter,
    "tpgdiff-restore": TPGDiffRuntimeAdapter,
    "tpgd-runtime": TPGDiffRuntimeAdapter,
    "tpgd": TPGDiffRuntimeAdapter,
}


def build(name: str, **kwargs: Any):
    key = name.lower()
    if key not in _ALIASES:
        raise KeyError(f"Unknown method: {name}. Available: {', '.join(sorted(_ALIASES))}")
    return _ALIASES[key](**kwargs)


def available_methods() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, cls in _ALIASES.items():
        out[name] = {
            "class": cls.__name__,
            "capabilities": list(getattr(cls, "capabilities", ())),
            "source_refs": [ref.as_dict() for ref in getattr(cls, "source_refs", ())],
        }
    return out
