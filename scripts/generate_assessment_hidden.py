#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methodhub import build
from methodhub.paths import default_repo_root, push_cwd, push_sys_path, require_exists
from my_method import ASSESSMENT_REASONING_QUERY, _feature_tensor_summary, _torch_dtype

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CONFIG = ROOT / "train.yml"


def _get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _select_device(config: dict[str, Any]) -> str:
    device = str(config.get("device", "cuda"))
    if device == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            return "cuda"
        gpu_ids = config.get("gpu_ids")
        if isinstance(gpu_ids, list) and gpu_ids:
            return f"cuda:{gpu_ids[0]}"
        if isinstance(gpu_ids, int):
            return f"cuda:{gpu_ids}"
        if isinstance(gpu_ids, str) and gpu_ids.strip():
            return f"cuda:{gpu_ids.split(',')[0].strip()}"
    return device


def _resolve(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _dataset_types(dataset_opt: dict[str, Any]) -> list[str]:
    for key in ("degradation", "dataset_type", "type"):
        values = _as_str_list(dataset_opt.get(key))
        if values:
            return values
    mode = dataset_opt.get("mode")
    if mode and str(mode) not in {"MD", "LQGT"}:
        return [str(mode)]
    return []


def _hidden_leaf(config: dict[str, Any], dataset_opt: dict[str, Any]) -> str:
    assessment_opt = config.get("assessment", {}) if isinstance(config.get("assessment"), dict) else {}
    if dataset_opt.get("hidden_leaf"):
        return str(dataset_opt["hidden_leaf"])
    max_new_tokens = int(assessment_opt.get("max_new_tokens", 256))
    output_dtype = str(assessment_opt.get("output_dtype", "bf16")).replace("float", "f")
    return f"features_condition_mnt{max_new_tokens}_{output_dtype}"


def _dataset_dir(
    config: dict[str, Any],
    dataset_opt: dict[str, Any],
    kind: str,
    *,
    dataset_type: str | None = None,
    allow_direct: bool = True,
) -> Path | None:
    direct_keys = {
        "lq": ("lq_dir", "dataroot_LQ"),
        "hidden": ("hidden_dir", "dataroot_hidden"),
    }[kind]
    if allow_direct:
        for key in direct_keys:
            if dataset_opt.get(key):
                return _resolve(dataset_opt[key])

    dataroot = _resolve(dataset_opt.get("dataroot"))
    dtype = dataset_type
    if dataroot is None or dtype is None:
        return None
    leaf = {"lq": "LQ", "hidden": _hidden_leaf(config, dataset_opt)}[kind]
    return (dataroot / dtype / leaf).resolve()



def _image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _load_rar_vae(*, rar_root: Path, rar_config_path: Path, device: str, dtype: str | torch.dtype):
    require_exists(rar_root, "RAR repo")
    require_exists(rar_config_path, "RAR inference config")
    with open(rar_config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    vae_path = Path(cfg["vae"]["vae_pretrained"]).expanduser()
    if not vae_path.is_absolute():
        vae_path = rar_root / vae_path
    require_exists(vae_path, "RAR SDVAE checkpoint")

    with push_sys_path(rar_root), push_cwd(rar_root):
        from iqa.sd35 import load_vision_encoder

        vae_dtype = _torch_dtype(cfg["vae"].get("weight_dtype", dtype))
        vae = load_vision_encoder(
            str(rar_config_path),
            training=False,
            vision_preprocess={},
            device=device,
            dtype=vae_dtype,
        ).to(vae_dtype).eval()
    image_size = int(cfg.get("model", {}).get("image_size", 256))
    return vae, image_size


def _rar_transform(image: Image.Image, image_size: int) -> torch.Tensor:
    from torchvision import transforms as T

    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transform(image.convert("RGB"))


def _encode_batch_for_rar_qa(images: list[Image.Image], *, vae, image_size: int, device: str) -> torch.Tensor:
    vae_dtype = next(vae.parameters()).dtype
    lq = torch.stack([_rar_transform(image, image_size) for image in images], dim=0).to(device, dtype=vae_dtype)
    with torch.no_grad():
        latent = vae.encode(lq).to(device)
        latent = vae.process_in(latent).to(device)
    return latent


def _save_hidden_batch(
    *,
    assessment,
    latent: torch.Tensor,
    output_paths: list[Path],
    max_new_tokens: int,
    output_dtype: str,
) -> list[dict[str, Any]]:
    batch_size = len(output_paths)
    inputs = {
        "query": [ASSESSMENT_REASONING_QUERY] * batch_size,
        "img": latent,
        "img_A": latent,
        "img_B": [None] * batch_size,
        "img_path": ["input"] * batch_size,
        "img_A_path": ["input"] * batch_size,
        "img_B_path": [None] * batch_size,
        "temperature": 0.0,
        "top_p": 0.9,
        "max_new_tokens": int(max_new_tokens),
        "task_type": "quality_single_A_noref",
        "output_prob_id": True,
        "output_confidence": False,
    }
    with torch.no_grad():
        output_texts, _, _, _, generated_hidden, prefix_hidden = assessment.generate(
            inputs,
            latent_input=True,
            save_hidden=True,
        )

    if output_dtype != "keep":
        dtype = _torch_dtype(output_dtype)
        generated_hidden = generated_hidden.to(dtype=dtype)
        prefix_hidden = prefix_hidden.to(dtype=dtype)
    condition_hidden = torch.cat([prefix_hidden, generated_hidden], dim=1)

    records: list[dict[str, Any]] = []
    for idx, output_path in enumerate(output_paths):
        sample_generated = generated_hidden[idx : idx + 1]
        sample_prefix = prefix_hidden[idx : idx + 1]
        sample_condition = condition_hidden[idx : idx + 1]
        answer = output_texts[idx].replace("\n ", "").strip()
        payload = {
            "round_index": 0,
            "query": ASSESSMENT_REASONING_QUERY,
            "answer": answer,
            "task_type": "quality_single_A_noref",
            "hidden_format": "condition_only",
            "condition_hidden": sample_condition.detach().cpu(),
            "generated_hidden_summary": _feature_tensor_summary(sample_generated),
            "prefix_hidden_summary": _feature_tensor_summary(sample_prefix),
            "condition_hidden_summary": _feature_tensor_summary(sample_condition),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        records.append(
            {
                "feature_path": str(output_path),
                "answer": answer,
                "hidden_format": payload["hidden_format"],
                "generated_hidden": payload["generated_hidden_summary"],
                "prefix_hidden": payload["prefix_hidden_summary"],
                "condition_hidden": payload["condition_hidden_summary"],
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Assessment Reasoning hidden states for Assess-TPGD training.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images; 0 means all.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-dtype", choices=["keep", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Number of images per Assessment generate call.")
    args = parser.parse_args()

    with open(args.config.expanduser(), "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    dataset_opt = _get(cfg, "datasets.train", {}) or {}
    assessment_opt = cfg.get("assessment", {}) or {}

    dataset_types = _dataset_types(dataset_opt)
    if not dataset_types:
        raise SystemExit("train.yml must define datasets.train.degradation")

    jobs: list[tuple[str, Path, Path]] = []
    for dtype in dataset_types:
        allow_direct = len(dataset_types) == 1
        lq_dir = _dataset_dir(cfg, dataset_opt, "lq", dataset_type=dtype, allow_direct=allow_direct)
        hidden_dir = _dataset_dir(cfg, dataset_opt, "hidden", dataset_type=dtype, allow_direct=allow_direct)
        if lq_dir is None or hidden_dir is None:
            raise SystemExit("train.yml must define datasets.train.dataroot/degradation or explicit lq_dir/hidden_dir")
        if not lq_dir.exists():
            raise SystemExit(f"lq_dir does not exist for {dtype}: {lq_dir}")
        jobs.append((dtype, lq_dir, hidden_dir))

    device = _select_device(cfg)
    rar_root = _resolve(assessment_opt.get("rar_root")) or default_repo_root("rar")
    rar_config_path = _resolve(assessment_opt.get("rar_config")) or (rar_root / "configs" / "infer_cfg.yaml")
    rar_assessment_config = _resolve(assessment_opt.get("rar_assessment_config")) or (rar_root / "iqa" / "config.yaml")
    rar_dtype = str(assessment_opt.get("dtype", "bf16"))
    max_new_tokens = int(args.max_new_tokens or assessment_opt.get("max_new_tokens", 256))
    output_dtype = str(args.output_dtype or assessment_opt.get("output_dtype", "bf16"))
    batch_size = int(args.batch_size or assessment_opt.get("hidden_batch_size", assessment_opt.get("batch_size", 1)) or 1)
    batch_size = max(1, batch_size)

    total_images = sum(len(_image_files(lq_dir)) for _, lq_dir, _ in jobs)
    if total_images == 0:
        raise SystemExit("No images found for configured dataset jobs")

    for dtype, lq_dir, hidden_dir in jobs:
        print(f"job={dtype} lq_dir={lq_dir} hidden_dir={hidden_dir}", flush=True)
    print(f"jobs={len(jobs)} images={total_images} device={device} max_new_tokens={max_new_tokens} output_dtype={output_dtype} batch_size={batch_size}", flush=True)

    assessment = build(
        "rar-assessment",
        repo_root=rar_root,
        config_path=rar_assessment_config,
        device=device,
        dtype=rar_dtype,
    )
    assessment.load()
    vae, image_size = _load_rar_vae(
        rar_root=rar_root,
        rar_config_path=rar_config_path,
        device=device,
        dtype=rar_dtype,
    )

    all_records: list[dict[str, Any]] = []
    total_processed = 0
    total_skipped = 0
    remaining = args.limit if args.limit > 0 else None
    for dtype, lq_dir, hidden_dir in jobs:
        images = _image_files(lq_dir)
        if remaining is not None:
            images = images[:remaining]
        records: list[dict[str, Any]] = []
        skipped = 0
        pending: list[tuple[Path, Path]] = []
        progress = tqdm(images, desc=f"hidden:{dtype}")
        for image_path in progress:
            rel = image_path.relative_to(lq_dir)
            feature_path = hidden_dir / rel.parent / f"{image_path.stem}_assessment_reasoning_hidden.pt"
            if feature_path.exists() and not args.overwrite:
                skipped += 1
                continue
            pending.append((image_path, feature_path))
            if len(pending) < batch_size:
                continue

            pil_images = []
            for pending_path, _ in pending:
                with Image.open(pending_path) as image:
                    pil_images.append(image.convert("RGB"))
            latent = _encode_batch_for_rar_qa(pil_images, vae=vae, image_size=image_size, device=device)
            batch_records = _save_hidden_batch(
                assessment=assessment,
                latent=latent,
                output_paths=[feature_path for _, feature_path in pending],
                max_new_tokens=max_new_tokens,
                output_dtype=output_dtype,
            )
            for record, (image_path, _) in zip(batch_records, pending):
                record["dataset_type"] = dtype
                record["image_path"] = str(image_path)
                records.append(record)
            pending = []

        if pending:
            pil_images = []
            for pending_path, _ in pending:
                with Image.open(pending_path) as image:
                    pil_images.append(image.convert("RGB"))
            latent = _encode_batch_for_rar_qa(pil_images, vae=vae, image_size=image_size, device=device)
            batch_records = _save_hidden_batch(
                assessment=assessment,
                latent=latent,
                output_paths=[feature_path for _, feature_path in pending],
                max_new_tokens=max_new_tokens,
                output_dtype=output_dtype,
            )
            for record, (image_path, _) in zip(batch_records, pending):
                record["dataset_type"] = dtype
                record["image_path"] = str(image_path)
                records.append(record)
        manifest = {
            "config": str(args.config.expanduser()),
            "dataset_type": dtype,
            "lq_dir": str(lq_dir),
            "hidden_dir": str(hidden_dir),
            "processed": len(records),
            "skipped": skipped,
            "max_new_tokens": max_new_tokens,
            "output_dtype": output_dtype,
            "batch_size": batch_size,
            "records": records,
        }
        manifest_path = hidden_dir / "assessment_hidden_manifest.json"
        hidden_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"done dataset={dtype} processed={len(records)} skipped={skipped} manifest={manifest_path}", flush=True)
        all_records.extend(records)
        total_processed += len(records)
        total_skipped += skipped
        if remaining is not None:
            remaining -= len(images)
            if remaining <= 0:
                break
    print(f"all_done processed={total_processed} skipped={total_skipped}", flush=True)


if __name__ == "__main__":
    main()
