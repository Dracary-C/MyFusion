#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_config import DEFAULT_CONFIG_PATH, config_get, load_config
from myfusion.pipelines.assess_tpgd import AssessTPGDTrainConfig, train_assess_tpgd


def _resolve(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _cfg(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config_get(config, f"assess_tpgd.{key}", default)


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()
    cfg = load_config(config_args.config)

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--lq-dir", type=Path, default=_resolve(_cfg(cfg, "lq_dir")))
    parser.add_argument("--gt-dir", type=Path, default=_resolve(_cfg(cfg, "gt_dir")))
    parser.add_argument("--hidden-dir", type=Path, default=_resolve(_cfg(cfg, "hidden_dir")))
    parser.add_argument("--hidden", dest="hidden_path", type=Path, default=_resolve(_cfg(cfg, "hidden_path")))
    parser.add_argument("--output-dir", type=Path, default=_resolve(_cfg(cfg, "output_dir", ROOT / "outputs" / "assess_tpgd")))
    parser.add_argument("--tpgd-options", type=Path, default=_resolve(config_get(cfg, "paths.tpgd_options")))
    parser.add_argument("--checkpoint", type=Path, default=_resolve(config_get(cfg, "paths.tpgd_checkpoint")))
    parser.add_argument("--hidden-key", choices=["prefix_hidden", "generated_hidden", "condition_hidden"], default=_cfg(cfg, "hidden_key", "condition_hidden"))
    parser.add_argument("--device", default=_cfg(cfg, "device", config_get(cfg, "fusion.device", "cuda")))
    parser.add_argument("--image-size", type=int, default=int(_cfg(cfg, "image_size", 128)))
    parser.add_argument("--batch-size", type=int, default=int(_cfg(cfg, "batch_size", 1)))
    parser.add_argument("--epochs", type=int, default=int(_cfg(cfg, "epochs", 1)))
    parser.add_argument("--max-steps", type=int, default=int(_cfg(cfg, "max_steps", 100)))
    parser.add_argument("--lr", type=float, default=float(_cfg(cfg, "lr", 1e-4)))
    parser.add_argument("--num-workers", type=int, default=int(_cfg(cfg, "num_workers", 0)))
    parser.add_argument("--adapter-hidden-dim", type=int, default=int(_cfg(cfg, "adapter_hidden_dim", 1024)))
    parser.add_argument("--adapter-pool", choices=["mean", "last", "first"], default=_cfg(cfg, "adapter_pool", "mean"))
    parser.add_argument("--adapter-dropout", type=float, default=float(_cfg(cfg, "adapter_dropout", 0.0)))
    parser.add_argument("--save-every", type=int, default=int(_cfg(cfg, "save_every", 100)))
    parser.add_argument("--log-every", type=int, default=int(_cfg(cfg, "log_every", 10)))
    parser.add_argument("--freeze-backbone", action="store_true", default=bool(_cfg(cfg, "freeze_backbone", True)))
    parser.add_argument("--train-backbone", action="store_false", dest="freeze_backbone")
    parser.add_argument("--no-load-checkpoint", action="store_true", default=bool(_cfg(cfg, "no_load_checkpoint", False)))
    parser.add_argument("--strict-load", action="store_true", default=bool(_cfg(cfg, "strict_load", False)))
    parser.add_argument("--random-content-context", action="store_true", default=bool(_cfg(cfg, "random_content_context", False)))
    parser.add_argument("--save-full-model", action="store_true", default=bool(_cfg(cfg, "save_full_model", False)))
    args = parser.parse_args()

    required = {"lq_dir": args.lq_dir, "gt_dir": args.gt_dir, "tpgd_options": args.tpgd_options, "output_dir": args.output_dir}
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"Missing required Assess-TPGD config/args: {missing}")

    train_cfg = AssessTPGDTrainConfig(
        lq_dir=args.lq_dir,
        gt_dir=args.gt_dir,
        hidden_dir=args.hidden_dir,
        hidden_path=args.hidden_path,
        output_dir=args.output_dir,
        tpgd_options=args.tpgd_options,
        tpgd_checkpoint=args.checkpoint,
        hidden_key=args.hidden_key,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        freeze_backbone=args.freeze_backbone,
        no_load_checkpoint=args.no_load_checkpoint,
        strict_load=args.strict_load,
        adapter_hidden_dim=args.adapter_hidden_dim,
        adapter_pool=args.adapter_pool,
        adapter_dropout=args.adapter_dropout,
        random_content_context=args.random_content_context,
        save_every=args.save_every,
        log_every=args.log_every,
        save_full_model=args.save_full_model,
    )
    train_assess_tpgd(train_cfg)


if __name__ == "__main__":
    main()
