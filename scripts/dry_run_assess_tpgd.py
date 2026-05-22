#!/usr/bin/env python
"""Dry-run training and inference for Assessment-hidden-conditioned TPGDiff UNet.

This is a module-level smoke test. It does not load a real dataset and does not
run the full TPGDiff SDE training loop. It checks that:

1. Assessment hidden states can be projected into `deg_context`.
2. The TPGDiff ConditionalUNet forward path works.
3. A loss can backpropagate through the AssessPriorAdapter / backbone path.
4. Optional TPGDiff UNet weights can be loaded into the wrapped backbone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_config import DEFAULT_CONFIG_PATH, config_get, load_config
from myfusion.modules.latent_qa import load_assessment_hidden, select_hidden
from myfusion.modules.restoration_backbone import (
    AssessConditionedTPGDUNet,
    TPGDBackboneConfig,
    load_tpgd_unet_weights,
)


def _load_tpgd_setting(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        opt = yaml.safe_load(handle)
    return opt["network_G"]["setting"]


def _repeat_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError(f"Cannot repeat tensor with batch {tensor.shape[0]} to batch {batch_size}")
    repeat_dims = [batch_size] + [1] * (tensor.ndim - 1)
    return tensor.repeat(*repeat_dims)


def _tensor_line(name: str, tensor: torch.Tensor) -> str:
    return f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"


def main() -> None:
    default_hidden = ROOT / "outputs" / "feature_smoke" / "features" / "lowlight1_round1_assessment_reasoning_hidden.pt"

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()
    cfg = load_config(config_args.config)

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--hidden", type=Path, default=default_hidden)
    parser.add_argument("--tpgd-options", type=Path, default=Path(config_get(cfg, "paths.tpgd_options")))
    parser.add_argument("--checkpoint", type=Path, default=Path(config_get(cfg, "paths.tpgd_checkpoint")))
    parser.add_argument("--no-load-checkpoint", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--mode", choices=["train", "infer", "both"], default="both")
    parser.add_argument("--hidden-key", choices=["prefix_hidden", "generated_hidden", "condition_hidden"], default="condition_hidden")
    parser.add_argument("--random-content-context", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    setting = _load_tpgd_setting(args.tpgd_options)
    backbone_cfg = TPGDBackboneConfig.from_mapping(setting) if setting else TPGDBackboneConfig()
    model = AssessConditionedTPGDUNet(backbone_cfg, freeze_backbone=args.freeze_backbone).to(device)

    if not args.no_load_checkpoint and args.checkpoint and args.checkpoint.exists():
        missing, unexpected = load_tpgd_unet_weights(
            model.backbone,
            args.checkpoint,
            strict=args.strict_load,
            map_location="cpu",
        )
        print(f"checkpoint: {args.checkpoint}")
        print(f"load_missing_keys: {len(missing)}")
        print(f"load_unexpected_keys: {len(unexpected)}")
        if missing[:5]:
            print(f"missing_sample: {missing[:5]}")
        if unexpected[:5]:
            print(f"unexpected_sample: {unexpected[:5]}")
    else:
        print("checkpoint: skipped")

    pack = load_assessment_hidden(args.hidden, map_location="cpu")
    assessment_hidden = select_hidden(pack, args.hidden_key)
    assessment_hidden = _repeat_batch(assessment_hidden, args.batch_size).to(device)

    bsz = int(args.batch_size)
    side = int(args.image_size)
    xt = torch.randn(bsz, backbone_cfg.in_nc, side, side, device=device)
    cond = torch.randn_like(xt)
    target = torch.randn(bsz, backbone_cfg.out_nc, side, side, device=device)
    time = torch.ones(bsz, device=device)

    content_context = None
    if args.random_content_context:
        content_context = torch.randn(bsz, backbone_cfg.context_dim, device=device)

    print(_tensor_line("assessment_hidden", assessment_hidden))

    if args.mode in {"train", "both"}:
        model.train()
        params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        last_loss = None
        for step in range(1, int(args.steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            output, deg_context = model(
                xt,
                cond,
                time,
                assessment_hidden=assessment_hidden,
                content_context=content_context,
                return_context=True,
            )
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            print(f"train_step={step} loss={last_loss:.6f}")
        print(_tensor_line("train_output", output))
        print(_tensor_line("train_deg_context", deg_context))
        print(f"train_ok: steps={args.steps} last_loss={last_loss:.6f}")

    if args.mode in {"infer", "both"}:
        model.eval()
        with torch.no_grad():
            output, deg_context = model(
                xt,
                cond,
                time,
                assessment_hidden=assessment_hidden,
                content_context=content_context,
                return_context=True,
            )
        print(_tensor_line("infer_output", output))
        print(_tensor_line("infer_deg_context", deg_context))
        print(f"infer_ok: mean={float(output.mean().cpu()):.6f} std={float(output.std(unbiased=False).cpu()):.6f}")


if __name__ == "__main__":
    main()
