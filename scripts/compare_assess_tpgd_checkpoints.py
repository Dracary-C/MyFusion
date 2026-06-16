#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
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

import test_assess_tpgd as test_entry


def _checkpoint_from_config(cfg: dict[str, Any]) -> Path:
    value = test_entry._path_opt(cfg).get("checkpoint_load")
    if value in (None, ""):
        raise SystemExit("Set path.checkpoint_load in test.yml or pass --before-checkpoint")
    checkpoint = test_entry.train_entry._resolve(value)
    if checkpoint is None or not checkpoint.exists():
        raise SystemExit(f"before checkpoint does not exist: {checkpoint}")
    return checkpoint


def _split_opt(cfg: dict[str, Any], split: str) -> dict[str, Any]:
    opt = test_entry._split_dataset_opt(cfg, split)
    return opt


def _dataset_names(cfg: dict[str, Any], split: str) -> list[str]:
    opt = _split_opt(cfg, split)
    values = test_entry.train_entry._dataset_types(opt)
    if not values:
        raise SystemExit(f"datasets.{split}.degradation/dataset_type/type is not configured")
    return values


def _cfg_for_dataset(cfg: dict[str, Any], split: str, dataset_name: str | None) -> dict[str, Any]:
    cloned = copy.deepcopy(cfg)
    datasets = cloned.setdefault("datasets", {})
    key = "val" if split in {"val", "validation"} else split
    if key not in datasets and split == "validation":
        key = "validation"
    if dataset_name is not None:
        datasets[key]["degradation"] = [dataset_name]
    return cloned


def _run_one(
    *,
    cfg: dict[str, Any],
    checkpoint: Path,
    split: str,
    metric_prefix: str,
    batch_size: int | None,
    max_batches: int,
    device: str | None,
) -> dict[str, float]:
    test_cfg = test_entry._build_config(cfg, split=split, batch_size=batch_size, device=device)
    return test_entry.test_assess_tpgd(
        test_cfg,
        checkpoint_path=checkpoint,
        max_batches=max_batches,
        metrics_name=metric_prefix,
    )


def _flatten_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    flattened = {}
    for key, value in metrics.items():
        suffix = key.split("/", 1)[1] if "/" in key else key
        flattened[f"{prefix}/{suffix}"] = value
    return flattened


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare before/after Assess-TPGD checkpoints on every validation dataset.")
    parser.add_argument("--config", type=Path, default=ROOT / "test.yml")
    parser.add_argument("--before-checkpoint", type=Path, default=None, help="Defaults to test.yml path.checkpoint_load")
    parser.add_argument("--after-checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["val", "validation", "test", "train"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means evaluate the whole split for each dataset")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config file must contain a YAML mapping: {args.config}")

    before = args.before_checkpoint.expanduser().resolve() if args.before_checkpoint else _checkpoint_from_config(cfg)
    after = args.after_checkpoint.expanduser().resolve()
    if not before.exists():
        raise SystemExit(f"before checkpoint does not exist: {before}")
    if not after.exists():
        raise SystemExit(f"after checkpoint does not exist: {after}")

    dataset_names = _dataset_names(cfg, args.split)
    runs: list[tuple[str, str | None]] = [(name, name) for name in dataset_names] + [("all", None)]
    results: dict[str, Any] = {
        "config": str(args.config.expanduser().resolve()),
        "split": args.split,
        "before_checkpoint": str(before),
        "after_checkpoint": str(after),
        "max_batches": args.max_batches,
        "datasets": {},
    }

    for label, dataset_name in runs:
        dataset_cfg = _cfg_for_dataset(cfg, args.split, dataset_name)
        print(f"[compare] dataset={label} checkpoint=before", flush=True)
        before_metrics = _run_one(
            cfg=dataset_cfg,
            checkpoint=before,
            split=args.split,
            metric_prefix=f"before_{label}",
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            device=args.device,
        )
        print(f"[compare] dataset={label} checkpoint=after", flush=True)
        after_metrics = _run_one(
            cfg=dataset_cfg,
            checkpoint=after,
            split=args.split,
            metric_prefix=f"after_{label}",
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            device=args.device,
        )
        before_flat = _flatten_metrics(before_metrics, "before")
        after_flat = _flatten_metrics(after_metrics, "after")
        delta = {}
        for metric in ("loss", "psnr_one_step", "ssim_one_step"):
            before_value = before_flat.get(f"before/{metric}")
            after_value = after_flat.get(f"after/{metric}")
            if before_value is not None and after_value is not None:
                delta[metric] = after_value - before_value
        results["datasets"][label] = {
            "before": before_flat,
            "after": after_flat,
            "delta_after_minus_before": delta,
        }

    print(json.dumps(results, indent=2, ensure_ascii=False))
    output_json = args.output_json
    if output_json is None:
        output_json = after.parent / f"compare_{args.split}_before_after.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"compare_saved={output_json}")


if __name__ == "__main__":
    main()
