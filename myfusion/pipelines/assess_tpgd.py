"""Assessment Reasoning hidden prior + TPGDiff bootstrap training.

This is the first MyFusion-owned training loop for the Assess-TPGD idea. It is
intentionally small and explicit: paired LQ/GT images plus precomputed
Assessment Reasoning hidden states are fed into a TPGDiff ConditionalUNet whose
degradation context is produced by `AssessPriorAdapter`.

This is not yet the full original TPGDiff SDE training recipe. It is a practical
bootstrap entrypoint for checking data plumbing, checkpoint loading, adapter
training, and saving before we wire the formal diffusion objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from myfusion.modules.assess_prior import AssessPriorAdapter
from myfusion.modules.latent_qa import load_assessment_hidden, select_hidden
from myfusion.modules.restoration_backbone import (
    AssessConditionedTPGDUNet,
    TPGDBackboneConfig,
    load_tpgd_unet_weights,
)

HiddenKey = Literal["prefix_hidden", "generated_hidden", "condition_hidden"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class AssessTPGDConfig:
    hidden_path: Path
    target_prior_dim: int
    hidden_key: HiddenKey = "condition_hidden"
    pool: str = "mean"


@dataclass
class AssessTPGDTrainConfig:
    lq_dir: Path
    gt_dir: Path
    output_dir: Path
    tpgd_options: Path
    tpgd_checkpoint: Path | None = None
    hidden_dir: Path | None = None
    hidden_path: Path | None = None
    hidden_key: HiddenKey = "condition_hidden"
    image_size: int = 128
    batch_size: int = 1
    epochs: int = 1
    max_steps: int = 100
    lr: float = 1e-4
    num_workers: int = 0
    device: str = "cuda"
    freeze_backbone: bool = True
    no_load_checkpoint: bool = False
    strict_load: bool = False
    adapter_hidden_dim: int = 1024
    adapter_pool: str = "mean"
    adapter_dropout: float = 0.0
    random_content_context: bool = False
    save_every: int = 100
    log_every: int = 10
    save_full_model: bool = False


def _resolve_path(value: str | Path | None, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _load_tpgd_setting(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        opt = yaml.safe_load(handle) or {}
    return opt.get("network_G", {}).get("setting", {})


def _image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _read_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size > 0:
            image = image.resize((image_size, image_size), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor * 2.0 - 1.0


def _find_gt(lq_path: Path, lq_dir: Path, gt_dir: Path) -> Path:
    rel = lq_path.relative_to(lq_dir)
    candidates = [gt_dir / rel, gt_dir / lq_path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing GT for {lq_path}. Tried: {candidates}")


def _build_hidden_index(hidden_dir: Path | None) -> dict[str, list[Path]]:
    if hidden_dir is None:
        return {}
    if not hidden_dir.exists():
        raise FileNotFoundError(f"hidden_dir does not exist: {hidden_dir}")
    index: dict[str, list[Path]] = {}
    for path in sorted(hidden_dir.rglob("*.pt")):
        name = path.name
        stem = name.split("_round", 1)[0]
        stem = stem.removesuffix("_assessment_reasoning_hidden")
        index.setdefault(stem, []).append(path)
        index.setdefault(path.stem, []).append(path)
    return index


def _find_hidden(stem: str, hidden_index: dict[str, list[Path]]) -> Path:
    if stem in hidden_index:
        return hidden_index[stem][0]
    for key, paths in hidden_index.items():
        if key.startswith(stem) or stem.startswith(key):
            return paths[0]
    raise FileNotFoundError(f"Missing Assessment hidden-state file for image stem: {stem}")


class PairedAssessmentDataset(Dataset):
    def __init__(
        self,
        lq_dir: Path,
        gt_dir: Path,
        *,
        hidden_dir: Path | None = None,
        hidden_path: Path | None = None,
        hidden_key: HiddenKey = "condition_hidden",
        image_size: int = 128,
    ) -> None:
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        self.hidden_path = hidden_path
        self.hidden_key = hidden_key
        self.image_size = image_size

        if not lq_dir.exists():
            raise FileNotFoundError(f"lq_dir does not exist: {lq_dir}")
        if not gt_dir.exists():
            raise FileNotFoundError(f"gt_dir does not exist: {gt_dir}")
        if hidden_path is None and hidden_dir is None:
            raise ValueError("Either hidden_dir or hidden_path must be set for Assess-TPGD training.")
        if hidden_path is not None and not hidden_path.exists():
            raise FileNotFoundError(f"hidden_path does not exist: {hidden_path}")

        self.hidden_index = {} if hidden_path is not None else _build_hidden_index(hidden_dir)
        self.samples: list[tuple[Path, Path, Path]] = []
        for lq_path in _image_files(lq_dir):
            gt_path = _find_gt(lq_path, lq_dir, gt_dir)
            sample_hidden = hidden_path or _find_hidden(lq_path.stem, self.hidden_index)
            self.samples.append((lq_path, gt_path, sample_hidden))
        if not self.samples:
            raise RuntimeError(f"No image pairs found under {lq_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        lq_path, gt_path, hidden_path = self.samples[index]
        pack = load_assessment_hidden(hidden_path, map_location="cpu")
        hidden = select_hidden(pack, self.hidden_key)
        if hidden.ndim == 3:
            if hidden.shape[0] != 1:
                raise ValueError(f"Expected hidden batch 1 in {hidden_path}, got {tuple(hidden.shape)}")
            hidden = hidden[0]
        if hidden.ndim != 2:
            raise ValueError(f"Expected hidden [T, C] or [1, T, C], got {tuple(hidden.shape)} in {hidden_path}")
        return {
            "lq": _read_image(lq_path, self.image_size),
            "gt": _read_image(gt_path, self.image_size),
            "hidden": hidden,
            "name": lq_path.stem,
            "hidden_path": str(hidden_path),
        }


def assessment_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lq = torch.stack([item["lq"] for item in batch], dim=0)
    gt = torch.stack([item["gt"] for item in batch], dim=0)
    hidden_list = [item["hidden"] for item in batch]
    max_len = max(hidden.shape[0] for hidden in hidden_list)
    hidden_dim = hidden_list[0].shape[1]
    hidden = hidden_list[0].new_zeros(len(batch), max_len, hidden_dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for idx, item_hidden in enumerate(hidden_list):
        length = item_hidden.shape[0]
        hidden[idx, :length] = item_hidden
        mask[idx, :length] = True
    return {
        "lq": lq,
        "gt": gt,
        "hidden": hidden,
        "mask": mask,
        "name": [item["name"] for item in batch],
        "hidden_path": [item["hidden_path"] for item in batch],
    }


def load_prior_from_hidden(config: AssessTPGDConfig, device: str | torch.device = "cpu") -> torch.Tensor:
    pack = load_assessment_hidden(config.hidden_path, map_location="cpu")
    hidden = select_hidden(pack, config.hidden_key).to(device)
    adapter = AssessPriorAdapter(output_dim=config.target_prior_dim, pool=config.pool).to(device)
    return adapter(hidden)


def save_checkpoint(
    output_dir: Path,
    model: AssessConditionedTPGDUNet,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    epoch: int,
    config: AssessTPGDTrainConfig,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_key = "model" if config.save_full_model else "assess_prior"
    state_value = model.state_dict() if config.save_full_model else model.assess_prior.state_dict()
    payload = {
        state_key: state_value,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "save_full_model": config.save_full_model,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in config.__dict__.items()},
    }
    latest = output_dir / "latest.pt"
    torch.save(payload, latest)
    numbered = output_dir / f"step_{step:06d}.pt"
    torch.save(payload, numbered)
    return latest


def train_assess_tpgd(config: AssessTPGDTrainConfig) -> dict[str, Any]:
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / "train.log"

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    setting = _load_tpgd_setting(config.tpgd_options)
    backbone_cfg = TPGDBackboneConfig.from_mapping(setting) if setting else TPGDBackboneConfig()
    # Keep backbone parameters requiring grad because TPGDiff's custom
    # checkpoint function differentiates through its parameter list. When
    # freeze_backbone=True, we simply exclude backbone params from optimizer.
    model = AssessConditionedTPGDUNet(
        backbone_cfg,
        adapter_hidden_dim=config.adapter_hidden_dim,
        adapter_pool=config.adapter_pool,
        adapter_dropout=config.adapter_dropout,
        freeze_backbone=False,
    ).to(device)

    if not config.no_load_checkpoint and config.tpgd_checkpoint and config.tpgd_checkpoint.exists():
        missing, unexpected = load_tpgd_unet_weights(
            model.backbone,
            config.tpgd_checkpoint,
            strict=config.strict_load,
            map_location="cpu",
        )
        log(f"checkpoint={config.tpgd_checkpoint}")
        log(f"load_missing_keys={len(missing)} load_unexpected_keys={len(unexpected)}")
    else:
        log("checkpoint=skipped")

    dataset = PairedAssessmentDataset(
        config.lq_dir,
        config.gt_dir,
        hidden_dir=config.hidden_dir,
        hidden_path=config.hidden_path,
        hidden_key=config.hidden_key,
        image_size=config.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=assessment_collate,
        pin_memory=device.type == "cuda",
    )
    if config.freeze_backbone:
        params = list(model.assess_prior.parameters())
    else:
        params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters. Check freeze_backbone and adapter settings.")
    optimizer = torch.optim.AdamW(params, lr=config.lr)

    log(f"samples={len(dataset)} output_dir={config.output_dir}")
    log(f"device={device} freeze_backbone={config.freeze_backbone} batch_size={config.batch_size}")

    global_step = 0
    last_loss = None
    model.train()
    for epoch in range(1, config.epochs + 1):
        for batch in loader:
            global_step += 1
            lq = batch["lq"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            hidden = batch["hidden"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            # TPGDiff uses a custom checkpoint function that expects selected
            # forward inputs to require grad. These tensors are not optimized;
            # they only keep the adapter-only backward path valid.
            lq_for_model = lq.detach().requires_grad_(True) if config.freeze_backbone else lq
            time = torch.ones(lq.shape[0], device=device)
            content_context = None
            if config.random_content_context:
                content_context = torch.randn(lq.shape[0], backbone_cfg.context_dim, device=device, requires_grad=config.freeze_backbone)
            elif backbone_cfg.use_image_context:
                content_context = torch.zeros(lq.shape[0], backbone_cfg.context_dim, device=device, requires_grad=config.freeze_backbone)

            optimizer.zero_grad(set_to_none=True)
            output, deg_context = model(
                lq_for_model,
                lq_for_model,
                time,
                assessment_hidden=hidden,
                assessment_mask=mask,
                content_context=content_context,
                return_context=True,
            )
            loss = torch.nn.functional.mse_loss(output, gt)
            loss.backward()
            if config.freeze_backbone:
                for param in model.backbone.parameters():
                    param.grad = None
            optimizer.step()
            last_loss = float(loss.detach().cpu())

            if global_step == 1 or global_step % config.log_every == 0:
                log(
                    f"step={global_step} epoch={epoch} loss={last_loss:.6f} "
                    f"output_shape={tuple(output.shape)} deg_context_shape={tuple(deg_context.shape)}"
                )
            if config.save_every > 0 and global_step % config.save_every == 0:
                save_path = save_checkpoint(config.output_dir, model, optimizer, step=global_step, epoch=epoch, config=config)
                log(f"saved={save_path}")
            if config.max_steps > 0 and global_step >= config.max_steps:
                break
        if config.max_steps > 0 and global_step >= config.max_steps:
            break

    save_path = save_checkpoint(config.output_dir, model, optimizer, step=global_step, epoch=epoch, config=config)
    log(f"train_ok steps={global_step} last_loss={last_loss:.6f} saved={save_path}")
    return {"steps": global_step, "last_loss": last_loss, "checkpoint": str(save_path)}
