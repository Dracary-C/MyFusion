"""Small compatibility wrapper around MethodHub.

Keeping this wrapper gives MyFusion a stable import path even while MethodHub or
legacy source layouts are still changing.
"""

from __future__ import annotations

from typing import Any


def build_method(name: str, **kwargs: Any) -> Any:
    from methodhub import build

    return build(name, **kwargs)
