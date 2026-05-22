#!/usr/bin/env python
"""Shape probe for Assessment-hidden-conditioned TPGDiff UNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myfusion.modules.latent_qa import load_assessment_hidden, select_hidden
from myfusion.modules.restoration_backbone import AssessConditionedTPGDUNet, TPGDBackboneConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--tpgd-options", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    setting = {}
    if args.tpgd_options is not None:
        with open(args.tpgd_options, "r", encoding="utf-8") as handle:
            opt = yaml.safe_load(handle)
        setting = opt["network_G"]["setting"]

    cfg = TPGDBackboneConfig.from_mapping(setting) if setting else TPGDBackboneConfig()
    model = AssessConditionedTPGDUNet(cfg).to(args.device).eval()

    pack = load_assessment_hidden(args.hidden)
    assessment_hidden = select_hidden(pack, "condition_hidden").to(args.device)
    batch = assessment_hidden.shape[0]
    side = int(args.image_size)
    xt = torch.randn(batch, cfg.in_nc, side, side, device=args.device)
    cond = torch.randn_like(xt)
    time = torch.ones(batch, device=args.device)

    with torch.no_grad():
        output, deg_context = model(
            xt,
            cond,
            time,
            assessment_hidden=assessment_hidden,
            return_context=True,
        )

    print(f"assessment_hidden: {tuple(assessment_hidden.shape)} {assessment_hidden.dtype}")
    print(f"deg_context:       {tuple(deg_context.shape)} {deg_context.dtype}")
    print(f"output:            {tuple(output.shape)} {output.dtype}")


if __name__ == "__main__":
    main()
