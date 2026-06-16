#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_assess_tpgd as train_entry
from myfusion.pipelines.assess_tpgd import AssessTPGDTrainConfig, test_assess_tpgd


def _path_opt(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg.get("path", {}) or {}
    if not isinstance(value, dict):
        raise SystemExit("test.yml path must be a YAML mapping")
    return value


def _split_dataset_opt(cfg: dict[str, Any], split: str) -> dict[str, Any]:
    datasets = cfg.get("datasets", {}) or {}
    if split == "train":
        opt = datasets.get("train", {}) or {}
    elif split in {"val", "validation"}:
        opt = datasets.get("val", datasets.get("validation", {})) or {}
    else:
        opt = datasets.get(split, {}) or {}
    if not isinstance(opt, dict) or not opt:
        raise SystemExit(f"datasets.{split} is not configured in test.yml")
    return opt


def _test_opt(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg.get("test", {}) or {}
    if not isinstance(value, dict):
        raise SystemExit("test.yml test must be a YAML mapping")
    return value


def _checkpoint_path(cfg: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    value = _path_opt(cfg).get("checkpoint_load")
    if value in (None, ""):
        raise SystemExit("Set path.checkpoint_load in test.yml or pass --checkpoint")
    checkpoint = train_entry._resolve(value)
    if checkpoint is None or not checkpoint.exists():
        raise SystemExit(f"checkpoint_load does not exist: {checkpoint}")
    return checkpoint


def _build_config(cfg: dict[str, Any], *, split: str, batch_size: int | None, device: str | None) -> AssessTPGDTrainConfig:
    dataset_opt = _split_dataset_opt(cfg, split)
    train_dataset_opt = cfg.get("datasets", {}).get("train", {}) or {}
    test_opt = _test_opt(cfg)
    path_opt = _path_opt(cfg)
    prior_switch_opt = cfg.get("prior_switch", {}) or {}
    if not isinstance(prior_switch_opt, dict):
        prior_switch_opt = {}

    dataset_types = train_entry._dataset_types(dataset_opt)
    lq_dirs = train_entry._dataset_dirs(cfg, dataset_opt, "lq", dataset_types)
    gt_dirs = train_entry._dataset_dirs(cfg, dataset_opt, "gt", dataset_types)
    hidden_dirs = train_entry._dataset_dirs(cfg, dataset_opt, "hidden", dataset_types)
    hidden_path = train_entry._resolve(dataset_opt.get("hidden_path"))
    degradation_prior_source = train_entry._degradation_prior_source(prior_switch_opt)

    if not lq_dirs or not gt_dirs:
        raise SystemExit(f"datasets.{split} lq/gt dirs are missing")
    if len(lq_dirs) != len(gt_dirs):
        raise SystemExit(f"datasets.{split} lq/gt directory counts differ: {len(lq_dirs)} vs {len(gt_dirs)}")
    if degradation_prior_source == "assessment_hidden" and hidden_path is None and len(hidden_dirs) != len(lq_dirs):
        raise SystemExit(f"datasets.{split} hidden directory count differs from lq count: {len(hidden_dirs)} vs {len(lq_dirs)}")

    effective_batch = batch_size or int(test_opt.get("batch_size") or dataset_opt.get("batch_size") or train_dataset_opt.get("batch_size") or 1)
    image_size = int(dataset_opt.get("image_size") or test_opt.get("image_size") or train_dataset_opt.get("image_size") or train_dataset_opt.get("patch_size") or 128)
    hidden_key = str(dataset_opt.get("hidden_key") or test_opt.get("hidden_key") or train_dataset_opt.get("hidden_key") or "condition_hidden")

    return AssessTPGDTrainConfig(
        lq_dir=train_entry._single_or_list(lq_dirs),
        gt_dir=train_entry._single_or_list(gt_dirs),
        hidden_dir=train_entry._single_or_list(hidden_dirs),
        hidden_path=hidden_path,
        output_dir=Path("."),
        tpgd_options=train_entry._resolve(path_opt.get("tpgd_options")),
        tpgd_checkpoint=train_entry._resolve(path_opt.get("base_checkpoint_load")),
        tpgd_inline_options=train_entry._inline_tpgd_options(cfg),
        hidden_key=hidden_key,
        image_size=image_size,
        batch_size=effective_batch,
        epochs=1,
        max_steps=0,
        lr=0.0,
        optimizer="AdamW",
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        num_workers=int(dataset_opt.get("n_workers", dataset_opt.get("num_workers", test_opt.get("n_workers", 0)))),
        device=device or train_entry._select_device(cfg),
        train_backbone=False,
        load_checkpoint=False,
        strict_load=bool(path_opt.get("strict_load", False)),
        adapter_hidden_dim=int(test_opt.get("adapter_hidden_dim", 1024)),
        adapter_pool=str(test_opt.get("adapter_pool", "mean")),
        adapter_dropout=float(test_opt.get("adapter_dropout", 0.0)),
        random_content_context=False,
        degradation_prior_source=degradation_prior_source,
        use_content_prior=train_entry._bool_switch(prior_switch_opt.get("use_content_prior"), False),
        use_structure_prior=train_entry._bool_switch(prior_switch_opt.get("use_struct_prior", prior_switch_opt.get("use_structure_prior")), False),
        train_structure_prior=False,
        prior_checkpoint=train_entry._resolve(path_opt.get("prior")),
        objective=str(test_opt.get("objective", "sde")),
        loss_type=str(test_opt.get("loss_type", "l1")),
        loss_weight=float(test_opt.get("weight", test_opt.get("loss_weight", 1.0))),
        sde_t_start=int(test_opt.get("sde_t_start", 1)),
        sde_t_end=int(test_opt.get("sde_t_end", -1)),
        save_full_model=True,
        eval_batch_size=effective_batch,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-test Assess-TPGD checkpoints and compute PSNR/SSIM metrics.")
    parser.add_argument("--config", type=Path, default=ROOT / "test.yml")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override test.yml path.checkpoint_load")
    parser.add_argument("--split", default="val", choices=["train", "val", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means evaluate the whole split")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config file must contain a YAML mapping: {args.config}")
    checkpoint = _checkpoint_path(cfg, args.checkpoint)
    test_cfg = _build_config(cfg, split=args.split, batch_size=args.batch_size, device=args.device)
    metrics = test_assess_tpgd(
        test_cfg,
        checkpoint_path=checkpoint,
        max_batches=args.max_batches,
        metrics_name=f"test_{args.split}",
    )
    result = {
        "config": str(args.config.expanduser().resolve()),
        "checkpoint": str(checkpoint),
        "split": args.split,
        "max_batches": args.max_batches,
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    output_json = args.output_json
    if output_json is None:
        output_json = checkpoint.parent / f"metrics_{args.split}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"metrics_saved={output_json}")


if __name__ == "__main__":
    main()
