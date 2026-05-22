#!/usr/bin/env python
"""Thin Python entrypoint for the Streamlit UI.

For normal use, `run.bash` is still preferred because it reads CUDA and server
settings from `config.yml` before Streamlit starts.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
