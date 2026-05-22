#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


SRC = Path("/data/chenzt/Dataset/All-in-One")
DST = Path("/data/chenzt/Dataset/TPGDiff")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if is_image(path))


def reset_category(split: str, category: str) -> None:
    root = DST / split / category
    if root.exists():
        shutil.rmtree(root)
    (root / "LQ").mkdir(parents=True, exist_ok=True)
    (root / "GT").mkdir(parents=True, exist_ok=True)


def remove_stale_category(split: str, category: str) -> None:
    root = DST / split / category
    if root.exists():
        shutil.rmtree(root)


def link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def require_pair(lq: Path, gt: Path) -> None:
    if not lq.exists():
        raise FileNotFoundError(f"Missing LQ file: {lq}")
    if not gt.exists():
        raise FileNotFoundError(f"Missing GT file: {gt}")


def add_pair(split: str, category: str, name: str, lq: Path, gt: Path) -> None:
    require_pair(lq, gt)
    link_file(lq, DST / split / category / "LQ" / name)
    link_file(gt, DST / split / category / "GT" / name)


def add_dehazeformer() -> dict[str, int]:
    counts = {}
    mapping = {
        "Train": SRC / "DehazeFormer/data/RESIDE-6K/train",
        "Val": SRC / "DehazeFormer/data/RESIDE-6K/test",
    }
    for split, root in mapping.items():
        reset_category(split, "DehazeFormer")
        count = 0
        for lq in image_files(root / "hazy"):
            name = lq.name
            gt = root / "GT" / name
            add_pair(split, "DehazeFormer", name, lq, gt)
            count += 1
        counts[split] = count
    return counts


def add_gopro() -> dict[str, int]:
    counts = {}
    mapping = {
        "Train": SRC / "GoPro/train",
        "Val": SRC / "GoPro/test",
    }
    for split, root in mapping.items():
        reset_category(split, "GoPro")
        count = 0
        for seq in sorted(path for path in root.iterdir() if path.is_dir()):
            for lq in image_files(seq / "blur"):
                name = f"{seq.name}__{lq.name}"
                gt = seq / "sharp" / lq.name
                add_pair(split, "GoPro", name, lq, gt)
                count += 1
        counts[split] = count
    return counts


def add_lolv2() -> dict[str, int]:
    counts = {}
    split_map = {"Train": "Train", "Val": "Test"}
    source_map = {"real": "Real_captured", "synthetic": "Synthetic"}
    for split, src_split in split_map.items():
        reset_category(split, "LOL-v2")
        count = 0
        for prefix, source_name in source_map.items():
            root = SRC / "LOL-v2" / source_name / src_split
            for lq in image_files(root / "Low"):
                name = f"{prefix}__{lq.name}"
                gt_name = lq.name
                if source_name == "Real_captured" and gt_name.startswith("low"):
                    gt_name = "normal" + gt_name[len("low") :]
                gt = root / "Normal" / gt_name
                add_pair(split, "LOL-v2", name, lq, gt)
                count += 1
        counts[split] = count
    return counts


def add_rain200l() -> dict[str, int]:
    counts = {}
    mapping = {
        "Train": SRC / "Rain200L/train",
        "Val": SRC / "Rain200L/test",
    }
    for split, root in mapping.items():
        reset_category(split, "Rain200L")
        count = 0
        for lq in image_files(root / "input"):
            name = lq.name
            gt = root / "target" / name
            add_pair(split, "Rain200L", name, lq, gt)
            count += 1
        counts[split] = count
    return counts


def save_noisy_lq(gt: Path, dst: Path, seed: int, sigma: float = 25.0) -> None:
    rng = np.random.default_rng(seed)
    with Image.open(gt) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    noise = rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    Image.fromarray(noisy, mode="RGB").save(dst)


def add_denoising_bsd400() -> dict[str, int]:
    reset_category("Train", "Denoising")
    reset_category("Val", "Denoising")
    files = image_files(SRC / "denoising-datasets/BSD400")
    files = [path for path in files if path.name != ".keep"]
    if len(files) < 2:
        raise RuntimeError("BSD400 contains too few image files.")

    split_at = int(round(len(files) * 0.9))
    counts = {"Train": 0, "Val": 0}
    for idx, gt in enumerate(files):
        split = "Train" if idx < split_at else "Val"
        name = gt.name
        link_file(gt, DST / split / "Denoising" / "GT" / name)
        save_noisy_lq(gt, DST / split / "Denoising" / "LQ" / name, seed=20260521 + idx)
        counts[split] += 1
    return counts


def count_pairs(split: str, category: str) -> tuple[int, int]:
    root = DST / split / category
    return len(image_files(root / "LQ")), len(image_files(root / "GT"))


def main() -> None:
    for split in ("Train", "Val"):
        (DST / split).mkdir(parents=True, exist_ok=True)
        remove_stale_category(split, "Real_captured")
        remove_stale_category(split, "Synthetic")

    builders = {
        "LOL-v2": add_lolv2,
        "DehazeFormer": add_dehazeformer,
        "GoPro": add_gopro,
        "Denoising": add_denoising_bsd400,
        "Rain200L": add_rain200l,
    }

    print(f"Source: {SRC}")
    print(f"Target: {DST}")
    for category, builder in builders.items():
        counts = builder()
        print(f"{category}: Train={counts.get('Train', 0)}, Val={counts.get('Val', 0)}")

    print("\nFinal pair counts:")
    for split in ("Train", "Val"):
        for category in builders:
            lq_count, gt_count = count_pairs(split, category)
            status = "OK" if lq_count == gt_count else "MISMATCH"
            print(f"{split}/{category}: LQ={lq_count}, GT={gt_count} [{status}]")


if __name__ == "__main__":
    main()
