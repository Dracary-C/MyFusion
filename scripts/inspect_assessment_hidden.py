#!/usr/bin/env python
"""Inspect a MyFusion Assessment Reasoning hidden-state `.pt` file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myfusion.modules.latent_qa import load_assessment_hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    pack = load_assessment_hidden(args.path)
    print(f"path: {args.path}")
    for key in ["prefix_hidden", "generated_hidden", "condition_hidden"]:
        tensor = pack.get(key)
        if tensor is not None:
            print(f"{key}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
    if "answer" in pack:
        print(f"answer: {pack['answer']}")


if __name__ == "__main__":
    main()
