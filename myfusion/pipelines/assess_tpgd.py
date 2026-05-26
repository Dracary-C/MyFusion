"""Assessment Reasoning hidden prior + TPGDiff SDE training.

Paired LQ/GT images plus precomputed Assessment Reasoning hidden states are fed
into a TPGDiff ConditionalUNet whose degradation context is produced by
`AssessPriorAdapter`. The default objective follows TPGDiff's IR-SDE training
loss; a small direct MSE objective is kept only for plumbing/debug runs.
"""

from __future__ import annotations

import copy
import contextlib
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from myfusion.legacy.source_paths import source_repo
from myfusion.modules.assess_prior import AssessPriorAdapter
from myfusion.modules.latent_qa import load_assessment_hidden, select_hidden
from myfusion.modules.restoration_backbone import (
    AssessConditionedTPGDUNet,
    TPGDBackboneConfig,
    load_tpgd_unet_weights,
)

HiddenKey = Literal["prefix_hidden", "generated_hidden", "condition_hidden"]
TrainObjective = Literal["sde", "direct_mse"]
LossType = Literal["l1", "l2"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class AssessTPGDConfig:
    hidden_path: Path
    target_prior_dim: int
    hidden_key: HiddenKey = "condition_hidden"
    pool: str = "mean"


@dataclass
class AssessTPGDTrainConfig:
    lq_dir: Path | list[Path]
    gt_dir: Path | list[Path]
    output_dir: Path
    tpgd_options: Path | None = None
    tpgd_checkpoint: Path | None = None
    tpgd_inline_options: dict[str, Any] | None = None
    hidden_dir: Path | list[Path] | None = None
    hidden_path: Path | None = None
    hidden_key: HiddenKey = "condition_hidden"
    image_size: int = 128
    batch_size: int = 1
    epochs: int = 1
    max_steps: int = 100
    lr: float = 1e-4
    optimizer: str = "AdamW"
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    num_workers: int = 0
    device: str = "cuda"
    train_backbone: bool = False
    load_checkpoint: bool = True
    strict_load: bool = False
    adapter_hidden_dim: int = 1024
    adapter_pool: str = "mean"
    adapter_dropout: float = 0.0
    random_content_context: bool = False
    degradation_prior_source: str = "assessment_hidden"
    use_content_prior: bool = False
    use_structure_prior: bool = False
    train_structure_prior: bool = False
    prior_checkpoint: Path | None = None
    objective: TrainObjective = "sde"
    loss_type: LossType = "l1"
    loss_weight: float = 1.0
    sde_max_sigma: float | None = None
    sde_T: int | None = None
    sde_schedule: str | None = None
    sde_eps: float | None = None
    sde_t_start: int = 1
    sde_t_end: int = -1
    save_every: int = 100
    log_every: int = 10
    save_full_model: bool = False
    lr_scheduler: str = "none"
    lr_min: float = 0.0
    lr_step_size: int = 10000
    lr_gamma: float = 0.5
    lr_milestones: list[int] | None = None
    lr_t_max: int | str = "auto"
    lr_warmup_enabled: bool = False
    lr_warmup_steps: int = 0
    lr_warmup_start_factor: float = 0.01
    val_lq_dir: Path | list[Path] | None = None
    val_gt_dir: Path | list[Path] | None = None
    val_hidden_dir: Path | list[Path] | None = None
    val_hidden_path: Path | None = None
    eval_enabled: bool = False
    eval_every_steps: int = 0
    eval_every_epochs: int = 0
    eval_batch_size: int = 0
    eval_train_max_batches: int = 10
    eval_val_max_batches: int = 0
    use_wandb: bool = False
    wandb_project: str = "MyFusion_AssessTPGD"
    wandb_run_name: str | None = None


def _resolve_path(value: str | Path | None, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _load_tpgd_setting(path: Path | None) -> dict[str, Any]:
    return _load_tpgd_options(path).get("network_G", {}).get("setting", {})


def _load_tpgd_options(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_symbol_from_file(path: Path, symbol: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"_myfusion_{path.stem}_{symbol}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {symbol} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


def _load_irsde_cls() -> type:
    sde_path = source_repo("tpgdiff").path / "universal-restoration" / "utils" / "sde_utils.py"
    return _load_symbol_from_file(sde_path, "IRSDE")


def _matching_loss(predict: torch.Tensor, target: torch.Tensor, loss_type: LossType) -> torch.Tensor:
    if loss_type == "l1":
        loss = torch.nn.functional.l1_loss(predict, target, reduction="none")
    elif loss_type == "l2":
        loss = torch.nn.functional.mse_loss(predict, target, reduction="none")
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    return loss.flatten(start_dim=1).mean(dim=1).mean()


def _cfg_value(config_value: Any, options: dict[str, Any], key: str, default: Any) -> Any:
    if config_value is not None:
        return config_value
    return options.get(key, default)


def _build_irsde(config: AssessTPGDTrainConfig, tpgd_options: dict[str, Any], device: torch.device):
    sde_options = tpgd_options.get("sde", {})
    irsde_cls = _load_irsde_cls()
    return irsde_cls(
        max_sigma=float(_cfg_value(config.sde_max_sigma, sde_options, "max_sigma", 50)),
        T=int(_cfg_value(config.sde_T, sde_options, "T", 100)),
        schedule=str(_cfg_value(config.sde_schedule, sde_options, "schedule", "cosine")),
        eps=float(_cfg_value(config.sde_eps, sde_options, "eps", 0.005)),
        device=device,
    )


def _tpgd_universal_root() -> Path:
    return source_repo("tpgdiff").path / "universal-restoration"


def _tpgd_code_root() -> Path:
    return _tpgd_universal_root() / "config" / "tpgd-sde"


@contextlib.contextmanager
def _push_sys_path(path: Path):
    path_str = str(path)
    inserted = path_str not in sys.path
    if inserted:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(path_str)


def _infer_prior_num_degradations(checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path.expanduser(), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported prior checkpoint type: {type(checkpoint)!r}")
    for key in ("deg_head.cls.weight", "module.deg_head.cls.weight"):
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return int(value.shape[0])
    raise KeyError(f"Cannot infer prior degradation count from {checkpoint_path}")


def _build_content_prior_model(checkpoint_path: Path, device: torch.device):
    with _push_sys_path(_tpgd_universal_root()):
        import open_clip
        from open_clip.prior_stage_model import PriorStageModel

    base_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None, device=device, precision="fp32")
    teacher_encoder = base_model.visual
    student_encoder = copy.deepcopy(base_model.visual)
    deg_backbone = copy.deepcopy(base_model.visual)

    if hasattr(base_model.visual, "output_dim"):
        embed_dim = base_model.visual.output_dim
    elif hasattr(base_model, "embed_dim"):
        embed_dim = base_model.embed_dim
    else:
        raise RuntimeError("Cannot infer embed_dim from open_clip visual model")

    prior_model = PriorStageModel(
        teacher_encoder=teacher_encoder,
        student_encoder=student_encoder,
        deg_backbone=deg_backbone,
        embed_dim=embed_dim,
        num_degradations=_infer_prior_num_degradations(checkpoint_path),
        content_loss_weight=1.0,
        deg_loss_weight=1.0,
        use_cosine_distill=True,
        normalize_embedding=True,
        freeze_teacher=True,
        freeze_deg_backbone=True,
    ).to(device)

    checkpoint = torch.load(checkpoint_path.expanduser(), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if len(state_dict) and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    prior_model.load_state_dict(state_dict, strict=True)
    prior_model.eval()
    for param in prior_model.parameters():
        param.requires_grad = False
    return prior_model


def _build_structure_prior(tpgd_options: dict[str, Any], checkpoint_path: Path | None, device: torch.device, strict: bool) -> torch.nn.Module | None:
    sp_options = tpgd_options.get("structure_prior")
    if not sp_options:
        return None
    with _push_sys_path(_tpgd_code_root()):
        from models import modules as tpgd_modules

    which_model = sp_options.get("which_model", "StructurePriorModule")
    setting = sp_options.get("setting", {})
    struct_prior = getattr(tpgd_modules, which_model)(**setting).to(device)

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path.expanduser(), map_location="cpu")
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("SP"), dict):
            incompatible = struct_prior.load_state_dict(checkpoint["SP"], strict=strict)
            missing = len(incompatible.missing_keys)
            unexpected = len(incompatible.unexpected_keys)
            print(f"structure_prior_checkpoint={checkpoint_path} missing={missing} unexpected={unexpected}", flush=True)
        else:
            print(f"structure_prior_checkpoint=no_SP path={checkpoint_path}", flush=True)
    return struct_prior


def _build_optimizer(params: list[torch.nn.Parameter], config: AssessTPGDTrainConfig) -> torch.optim.Optimizer:
    optimizer = config.optimizer.lower()
    kwargs = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "betas": (config.beta1, config.beta2),
    }
    if optimizer == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if optimizer == "adam":
        return torch.optim.Adam(params, **kwargs)
    if optimizer == "lion":
        optimizer_path = source_repo("tpgdiff").path / "universal-restoration" / "config" / "tpgd-sde" / "models" / "optimizer.py"
        lion_cls = _load_symbol_from_file(optimizer_path, "Lion")
        return lion_cls(params, **kwargs)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}. Use Adam, AdamW, or Lion.")


def _planned_train_steps(config: AssessTPGDTrainConfig, steps_per_epoch: int) -> int:
    epoch_steps = max(0, int(config.epochs)) * max(1, int(steps_per_epoch))
    if config.max_steps and config.max_steps > 0:
        return min(epoch_steps, int(config.max_steps)) if epoch_steps > 0 else int(config.max_steps)
    return epoch_steps


def _resolve_lr_t_max(config: AssessTPGDTrainConfig, planned_steps: int) -> int:
    value = config.lr_t_max
    if value in (None, "", 0, "0"):
        return max(1, planned_steps)
    if isinstance(value, str) and value.strip().lower() == "auto":
        return max(1, planned_steps)
    return max(1, int(float(value)))


def _build_lr_scheduler(optimizer: torch.optim.Optimizer, config: AssessTPGDTrainConfig, planned_steps: int):
    scheduler = str(config.lr_scheduler or "none").strip().lower().replace("-", "_")
    warmup_steps = max(0, int(config.lr_warmup_steps)) if config.lr_warmup_enabled else 0
    warmup_start = float(config.lr_warmup_start_factor)
    warmup_start = min(max(warmup_start, 0.0), 1.0)
    t_max = _resolve_lr_t_max(config, planned_steps)

    if scheduler in {"", "none", "constant", "off"} and warmup_steps <= 0:
        return None
    if scheduler not in {"", "none", "constant", "off", "cosine", "cosine_annealing", "cosineannealing", "step", "step_lr", "steplr", "multistep", "multi_step", "multistep_lr", "multisteplr", "exponential", "exp", "exponential_lr"}:
        raise ValueError("Unsupported lr scheduler: " + str(config.lr_scheduler) + ". Use none, cosine, step, multistep, or exponential.")

    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def factor_for_group(base_lr: float):
        min_factor = 0.0 if base_lr <= 0 else min(float(config.lr_min) / float(base_lr), 1.0)

        def lr_lambda(step: int) -> float:
            # LambdaLR calls this with step=0 during initialization.
            if warmup_steps > 0 and step < warmup_steps:
                alpha = step / max(1, warmup_steps)
                return warmup_start + (1.0 - warmup_start) * alpha

            post_step = max(0, step - warmup_steps)
            post_total = max(1, t_max - warmup_steps)
            if scheduler in {"", "none", "constant", "off"}:
                return 1.0
            if scheduler in {"cosine", "cosine_annealing", "cosineannealing"}:
                progress = min(post_step / post_total, 1.0)
                return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))
            if scheduler in {"step", "step_lr", "steplr"}:
                return float(config.lr_gamma) ** (post_step // max(1, int(config.lr_step_size)))
            if scheduler in {"multistep", "multi_step", "multistep_lr", "multisteplr"}:
                milestones = config.lr_milestones or [config.lr_step_size]
                passed = sum(post_step >= int(item) for item in milestones)
                return float(config.lr_gamma) ** passed
            if scheduler in {"exponential", "exp", "exponential_lr"}:
                return float(config.lr_gamma) ** post_step
            return 1.0

        return lr_lambda

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[factor_for_group(lr) for lr in base_lrs])


def _to_01(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.clamp(-1.0, 1.0) + 1.0) * 0.5


def _psnr_minus1_1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_01 = _to_01(pred)
    target_01 = _to_01(target)
    mse = torch.nn.functional.mse_loss(pred_01, target_01, reduction="none").flatten(start_dim=1).mean(dim=1)
    return (-10.0 * torch.log10(mse.clamp_min(1e-12))).mean()


def _ssim_minus1_1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_01 = _to_01(pred).float()
    target_01 = _to_01(target).float()
    channels = pred_01.shape[1]
    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, device=pred_01.device, dtype=pred_01.dtype) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window = (gauss[:, None] @ gauss[None, :]).view(1, 1, window_size, window_size)
    window = window.expand(channels, 1, window_size, window_size)

    padding = window_size // 2
    mu_x = torch.nn.functional.conv2d(pred_01, window, padding=padding, groups=channels)
    mu_y = torch.nn.functional.conv2d(target_01, window, padding=padding, groups=channels)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = torch.nn.functional.conv2d(pred_01 * pred_01, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = torch.nn.functional.conv2d(target_01 * target_01, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = torch.nn.functional.conv2d(pred_01 * target_01, window, padding=padding, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)).clamp_min(1e-12)
    return ssim_map.flatten(start_dim=1).mean(dim=1).mean()


class _WandbLogger:
    def __init__(self, config: AssessTPGDTrainConfig) -> None:
        self.run = None
        if not config.use_wandb:
            return
        try:
            import wandb
        except ImportError:
            print("wandb=unavailable", flush=True)
            return
        self._wandb = wandb
        self.run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={key: str(value) if isinstance(value, Path) else value for key, value in config.__dict__.items()},
        )

    def log(self, payload: dict[str, Any], step: int) -> None:
        if self.run is not None:
            self._wandb.log(payload, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


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


def _read_clip_image(path: Path, resolution: int = 224) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((resolution, resolution), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = tensor.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = tensor.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    return (tensor - mean) / std


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
        require_hidden: bool = True,
    ) -> None:
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        self.hidden_path = hidden_path
        self.hidden_key = hidden_key
        self.image_size = image_size
        self.require_hidden = require_hidden

        if not lq_dir.exists():
            raise FileNotFoundError(f"lq_dir does not exist: {lq_dir}")
        if not gt_dir.exists():
            raise FileNotFoundError(f"gt_dir does not exist: {gt_dir}")
        if self.require_hidden and hidden_path is None and hidden_dir is None:
            raise ValueError("Either hidden_dir or hidden_path must be set for Assess-TPGD training.")
        if self.require_hidden and hidden_path is not None and not hidden_path.exists():
            raise FileNotFoundError(f"hidden_path does not exist: {hidden_path}")

        self.hidden_index = {} if hidden_path is not None else (_build_hidden_index(hidden_dir) if self.require_hidden else {})
        self.samples: list[tuple[Path, Path, Path | None]] = []
        for lq_path in _image_files(lq_dir):
            gt_path = _find_gt(lq_path, lq_dir, gt_dir)
            sample_hidden = (hidden_path or _find_hidden(lq_path.stem, self.hidden_index)) if self.require_hidden else None
            self.samples.append((lq_path, gt_path, sample_hidden))
        if not self.samples:
            raise RuntimeError(f"No image pairs found under {lq_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        lq_path, gt_path, hidden_path = self.samples[index]
        hidden = None
        if hidden_path is not None:
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
            "lq_clip": _read_clip_image(lq_path),
            "hidden": hidden,
            "name": lq_path.stem,
            "hidden_path": str(hidden_path) if hidden_path is not None else "",
        }


def assessment_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lq = torch.stack([item["lq"] for item in batch], dim=0)
    gt = torch.stack([item["gt"] for item in batch], dim=0)
    lq_clip = torch.stack([item["lq_clip"] for item in batch], dim=0)
    hidden_items = [item["hidden"] for item in batch]
    if all(item_hidden is None for item_hidden in hidden_items):
        hidden = None
        mask = None
    elif any(item_hidden is None for item_hidden in hidden_items):
        raise ValueError("Batch mixes samples with and without Assessment hidden states")
    else:
        hidden_list = [item_hidden for item_hidden in hidden_items if item_hidden is not None]
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
        "lq_clip": lq_clip,
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
    structure_prior: torch.nn.Module | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Path:
    def serialize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [serialize(item) for item in value]
        if isinstance(value, dict):
            return {key: serialize(item) for key, item in value.items()}
        return value

    output_dir.mkdir(parents=True, exist_ok=True)
    save_full_state = config.save_full_model or config.train_backbone
    state_key = "model" if save_full_state else "assess_prior"
    state_value = model.state_dict() if save_full_state else model.assess_prior.state_dict()
    payload = {
        state_key: state_value,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "save_full_model": save_full_state,
        "config": {key: serialize(value) for key, value in config.__dict__.items()},
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if structure_prior is not None:
        payload["structure_prior"] = structure_prior.state_dict()
    latest = output_dir / "latest.pt"
    torch.save(payload, latest)
    numbered = output_dir / f"step_{step:06d}.pt"
    torch.save(payload, numbered)
    return latest


def _as_path_list(value: Path | list[Path] | None, *, name: str) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} must not be empty")
        return value
    return [value]


def _build_paired_dataset(
    *,
    lq_dir: Path | list[Path] | None,
    gt_dir: Path | list[Path] | None,
    hidden_dir: Path | list[Path] | None,
    hidden_path: Path | None,
    hidden_key: HiddenKey,
    image_size: int,
    require_hidden: bool,
    split_name: str,
) -> Dataset | None:
    if lq_dir is None or gt_dir is None:
        return None
    lq_dirs = _as_path_list(lq_dir, name=f"{split_name}.lq_dir")
    gt_dirs = _as_path_list(gt_dir, name=f"{split_name}.gt_dir")
    if len(lq_dirs) != len(gt_dirs):
        raise ValueError(f"{split_name} lq/gt directory counts differ: {len(lq_dirs)} vs {len(gt_dirs)}")

    if hidden_path is not None:
        hidden_dirs: list[Path | None] = [None] * len(lq_dirs)
    elif require_hidden:
        hidden_dirs = _as_path_list(hidden_dir, name=f"{split_name}.hidden_dir")
        if len(hidden_dirs) != len(lq_dirs):
            raise ValueError(f"{split_name} hidden/lq directory counts differ: {len(hidden_dirs)} vs {len(lq_dirs)}")
    else:
        hidden_dirs = [None] * len(lq_dirs)

    datasets = [
        PairedAssessmentDataset(
            lq_path,
            gt_path,
            hidden_dir=hidden_path_item,
            hidden_path=hidden_path,
            hidden_key=hidden_key,
            image_size=image_size,
            require_hidden=require_hidden,
        )
        for lq_path, gt_path, hidden_path_item in zip(lq_dirs, gt_dirs, hidden_dirs)
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _build_train_dataset(config: AssessTPGDTrainConfig) -> Dataset:
    dataset = _build_paired_dataset(
        lq_dir=config.lq_dir,
        gt_dir=config.gt_dir,
        hidden_dir=config.hidden_dir,
        hidden_path=config.hidden_path,
        hidden_key=config.hidden_key,
        image_size=config.image_size,
        require_hidden=config.degradation_prior_source == "assessment_hidden",
        split_name="train",
    )
    if dataset is None:
        raise ValueError("Training dataset is not configured")
    return dataset


def _build_val_dataset(config: AssessTPGDTrainConfig) -> Dataset | None:
    return _build_paired_dataset(
        lq_dir=config.val_lq_dir,
        gt_dir=config.val_gt_dir,
        hidden_dir=config.val_hidden_dir,
        hidden_path=config.val_hidden_path,
        hidden_key=config.hidden_key,
        image_size=config.image_size,
        require_hidden=config.degradation_prior_source == "assessment_hidden",
        split_name="val",
    )


def _evaluate_loader(
    *,
    name: str,
    loader: DataLoader,
    max_batches: int,
    device: torch.device,
    model: AssessConditionedTPGDUNet,
    backbone_cfg: TPGDBackboneConfig,
    content_prior_model: torch.nn.Module | None,
    structure_prior: torch.nn.Module | None,
    config: AssessTPGDTrainConfig,
    sde: Any,
    use_assessment_degra_prior: bool,
    use_tpgd_degra_prior: bool,
    use_content_prior: bool,
) -> dict[str, float]:
    model_was_training = model.training
    structure_was_training = structure_prior.training if structure_prior is not None else False
    model.eval()
    if structure_prior is not None:
        structure_prior.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    total_batches = 0
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                if max_batches > 0 and batch_idx > max_batches:
                    break
                lq = batch["lq"].to(device, non_blocking=True)
                gt = batch["gt"].to(device, non_blocking=True)
                hidden = batch["hidden"]
                mask = batch["mask"]
                if hidden is not None:
                    hidden = hidden.to(device, non_blocking=True)
                if mask is not None:
                    mask = mask.to(device, non_blocking=True)

                content_context = None
                deg_context_input = None
                lq_clip = batch["lq_clip"].to(device, non_blocking=True) if (use_content_prior or use_tpgd_degra_prior) else None
                if content_prior_model is not None and lq_clip is not None:
                    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                        if use_content_prior:
                            content_context = content_prior_model.get_content_prior(lq_clip).float()
                        if use_tpgd_degra_prior:
                            deg_context_input = content_prior_model.encode_for_degradation(lq_clip).float()
                elif backbone_cfg.use_image_context:
                    content_context = torch.zeros(lq.shape[0], backbone_cfg.context_dim, device=device)

                struct_tokens = None
                if structure_prior is not None:
                    struct_tokens = structure_prior((lq + 1.0) * 0.5)

                if config.objective == "sde":
                    if sde is None:
                        raise RuntimeError("SDE evaluation requested but sde is None")
                    timesteps, states = sde.generate_random_states(
                        x0=gt,
                        mu=lq,
                        T_start=config.sde_t_start,
                        T_end=config.sde_t_end,
                    )
                    output, _ = model(
                        states,
                        lq,
                        timesteps.reshape(-1),
                        assessment_hidden=hidden if use_assessment_degra_prior else None,
                        assessment_mask=mask if use_assessment_degra_prior else None,
                        deg_context=deg_context_input,
                        content_context=content_context,
                        struct_tokens=struct_tokens,
                        return_context=True,
                    )
                    score = sde.get_score_from_noise(output, timesteps)
                    restored = sde.reverse_sde_step_mean(states, score, timesteps)
                    target = sde.reverse_optimum_step(states, gt, timesteps)
                    loss = config.loss_weight * _matching_loss(restored, target, config.loss_type)
                    psnr = _psnr_minus1_1(restored, gt)
                    ssim = _ssim_minus1_1(restored, gt)
                else:
                    time = torch.ones(lq.shape[0], device=device)
                    output, _ = model(
                        lq,
                        lq,
                        time,
                        assessment_hidden=hidden if use_assessment_degra_prior else None,
                        assessment_mask=mask if use_assessment_degra_prior else None,
                        deg_context=deg_context_input,
                        content_context=content_context,
                        struct_tokens=struct_tokens,
                        return_context=True,
                    )
                    loss = torch.nn.functional.mse_loss(output, gt)
                    psnr = _psnr_minus1_1(output, gt)
                    ssim = _ssim_minus1_1(output, gt)

                batch_size = int(lq.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_psnr += float(psnr.detach().cpu()) * batch_size
                total_ssim += float(ssim.detach().cpu()) * batch_size
                total_samples += batch_size
                total_batches += 1
    finally:
        model.train(model_was_training)
        if structure_prior is not None:
            structure_prior.train(structure_was_training)

    if total_samples == 0:
        return {f"{name}/loss": math.nan, f"{name}/psnr_one_step": math.nan, f"{name}/ssim_one_step": math.nan, f"{name}/samples": 0.0, f"{name}/batches": 0.0}
    return {
        f"{name}/loss": total_loss / total_samples,
        f"{name}/psnr_one_step": total_psnr / total_samples,
        f"{name}/ssim_one_step": total_ssim / total_samples,
        f"{name}/samples": float(total_samples),
        f"{name}/batches": float(total_batches),
    }


def test_assess_tpgd(
    config: AssessTPGDTrainConfig,
    *,
    checkpoint_path: Path,
    max_batches: int = 0,
    metrics_name: str = "test",
) -> dict[str, float]:
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    tpgd_options = _load_tpgd_options(config.tpgd_options)
    if config.tpgd_inline_options:
        _deep_update(tpgd_options, config.tpgd_inline_options)
    setting = tpgd_options.get("network_G", {}).get("setting", {})
    backbone_cfg = TPGDBackboneConfig.from_mapping(setting) if setting else TPGDBackboneConfig()

    if config.degradation_prior_source not in {"assessment_hidden", "tpgd", "none"}:
        raise ValueError(f"Unsupported degradation_prior_source: {config.degradation_prior_source}")
    use_assessment_degra_prior = config.degradation_prior_source == "assessment_hidden"
    use_tpgd_degra_prior = config.degradation_prior_source == "tpgd"
    use_content_prior = bool(config.use_content_prior and backbone_cfg.use_image_context)
    use_structure_prior = bool(config.use_structure_prior and backbone_cfg.use_struct_context)

    sde = _build_irsde(config, tpgd_options, device) if config.objective == "sde" else None
    model = AssessConditionedTPGDUNet(
        backbone_cfg,
        adapter_hidden_dim=config.adapter_hidden_dim,
        adapter_pool=config.adapter_pool,
        adapter_dropout=config.adapter_dropout,
        freeze_backbone=False,
    ).to(device)

    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(checkpoint)!r}")

    if isinstance(checkpoint.get("model"), dict):
        incompatible = model.load_state_dict(checkpoint["model"], strict=config.strict_load)
        print(
            f"checkpoint={checkpoint_path} key=model missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    elif isinstance(checkpoint.get("assess_prior"), dict):
        if config.tpgd_checkpoint and config.tpgd_checkpoint.exists():
            missing, unexpected = load_tpgd_unet_weights(
                model.backbone,
                config.tpgd_checkpoint,
                strict=config.strict_load,
                map_location="cpu",
            )
            print(f"base_checkpoint={config.tpgd_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        incompatible = model.assess_prior.load_state_dict(checkpoint["assess_prior"], strict=config.strict_load)
        print(
            f"checkpoint={checkpoint_path} key=assess_prior missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    else:
        raise KeyError("Checkpoint must contain either 'model' or 'assess_prior'.")

    content_prior_model = None
    if use_content_prior or use_tpgd_degra_prior:
        if config.prior_checkpoint is None:
            raise ValueError("path.prior must be set when content/degradation prior is enabled")
        content_prior_model = _build_content_prior_model(config.prior_checkpoint, device)

    structure_prior = _build_structure_prior(tpgd_options, config.tpgd_checkpoint, device, config.strict_load) if use_structure_prior else None
    if structure_prior is not None and isinstance(checkpoint.get("structure_prior"), dict):
        incompatible = structure_prior.load_state_dict(checkpoint["structure_prior"], strict=config.strict_load)
        print(
            f"structure_prior_payload=loaded missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    if structure_prior is not None:
        structure_prior.eval()
        for param in structure_prior.parameters():
            param.requires_grad = False

    dataset = _build_train_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size or config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=assessment_collate,
        pin_memory=device.type == "cuda",
    )
    metrics = _evaluate_loader(
        name=metrics_name,
        loader=loader,
        max_batches=max_batches,
        device=device,
        model=model,
        backbone_cfg=backbone_cfg,
        content_prior_model=content_prior_model,
        structure_prior=structure_prior,
        config=config,
        sde=sde,
        use_assessment_degra_prior=use_assessment_degra_prior,
        use_tpgd_degra_prior=use_tpgd_degra_prior,
        use_content_prior=use_content_prior,
    )
    return metrics


def train_assess_tpgd(config: AssessTPGDTrainConfig) -> dict[str, Any]:
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / "train.log"

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    tpgd_options = _load_tpgd_options(config.tpgd_options)
    if config.tpgd_inline_options:
        _deep_update(tpgd_options, config.tpgd_inline_options)
    setting = tpgd_options.get("network_G", {}).get("setting", {})
    backbone_cfg = TPGDBackboneConfig.from_mapping(setting) if setting else TPGDBackboneConfig()
    if config.degradation_prior_source not in {"assessment_hidden", "tpgd", "none"}:
        raise ValueError(f"Unsupported degradation_prior_source: {config.degradation_prior_source}")
    use_assessment_degra_prior = config.degradation_prior_source == "assessment_hidden"
    use_tpgd_degra_prior = config.degradation_prior_source == "tpgd"
    use_content_prior = bool(config.use_content_prior and backbone_cfg.use_image_context)
    use_structure_prior = bool(config.use_structure_prior and backbone_cfg.use_struct_context)
    if use_structure_prior:
        sp_setting = tpgd_options.get("structure_prior", {}).get("setting", {})
        sp_token_dim = int(sp_setting.get("token_dim", backbone_cfg.struct_context_dim))
        if sp_token_dim != backbone_cfg.struct_context_dim:
            raise ValueError(
                "structure prior token_dim must match network_G.setting.struct_context_dim "
                f"when use_struct_prior is enabled, got {sp_token_dim} vs {backbone_cfg.struct_context_dim}"
            )
    if config.objective not in ("sde", "direct_mse"):
        raise ValueError(f"Unsupported objective: {config.objective}")
    sde = _build_irsde(config, tpgd_options, device) if config.objective == "sde" else None
    # Keep backbone parameters requiring grad because TPGDiff's custom
    # checkpoint function differentiates through its parameter list. When
    # train_backbone=False, we simply exclude backbone params from optimizer.
    model = AssessConditionedTPGDUNet(
        backbone_cfg,
        adapter_hidden_dim=config.adapter_hidden_dim,
        adapter_pool=config.adapter_pool,
        adapter_dropout=config.adapter_dropout,
        freeze_backbone=False,
    ).to(device)

    loaded_checkpoint: dict[str, Any] | None = None
    if config.load_checkpoint and config.tpgd_checkpoint and config.tpgd_checkpoint.exists():
        checkpoint_payload = torch.load(config.tpgd_checkpoint.expanduser(), map_location="cpu")
        if isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("model"), dict):
            incompatible = model.load_state_dict(checkpoint_payload["model"], strict=config.strict_load)
            loaded_checkpoint = checkpoint_payload
            log(f"checkpoint={config.tpgd_checkpoint}")
            log(
                "checkpoint_key=model "
                f"load_missing_keys={len(incompatible.missing_keys)} "
                f"load_unexpected_keys={len(incompatible.unexpected_keys)} "
                f"checkpoint_step={checkpoint_payload.get('step')} checkpoint_epoch={checkpoint_payload.get('epoch')}"
            )
        elif isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("assess_prior"), dict):
            incompatible = model.assess_prior.load_state_dict(checkpoint_payload["assess_prior"], strict=config.strict_load)
            loaded_checkpoint = checkpoint_payload
            log(f"checkpoint={config.tpgd_checkpoint}")
            log(
                "checkpoint_key=assess_prior "
                f"load_missing_keys={len(incompatible.missing_keys)} "
                f"load_unexpected_keys={len(incompatible.unexpected_keys)} "
                f"checkpoint_step={checkpoint_payload.get('step')} checkpoint_epoch={checkpoint_payload.get('epoch')}"
            )
        else:
            missing, unexpected = load_tpgd_unet_weights(
                model.backbone,
                config.tpgd_checkpoint,
                strict=config.strict_load,
                map_location="cpu",
            )
            log(f"checkpoint={config.tpgd_checkpoint}")
            log(f"checkpoint_key=tpgd_G load_missing_keys={len(missing)} load_unexpected_keys={len(unexpected)}")
    else:
        log("checkpoint=skipped")

    content_prior_model = None
    if use_content_prior or use_tpgd_degra_prior:
        if config.prior_checkpoint is None:
            raise ValueError("path.prior must be set when content/degradation prior is enabled")
        content_prior_model = _build_content_prior_model(config.prior_checkpoint, device)

    structure_prior_checkpoint = config.tpgd_checkpoint if config.load_checkpoint else None
    structure_prior = _build_structure_prior(tpgd_options, structure_prior_checkpoint, device, config.strict_load) if use_structure_prior else None
    if structure_prior is not None and isinstance(loaded_checkpoint, dict) and isinstance(loaded_checkpoint.get("structure_prior"), dict):
        incompatible = structure_prior.load_state_dict(loaded_checkpoint["structure_prior"], strict=config.strict_load)
        log(
            "checkpoint_key=structure_prior "
            f"load_missing_keys={len(incompatible.missing_keys)} "
            f"load_unexpected_keys={len(incompatible.unexpected_keys)}"
        )
    if structure_prior is not None:
        structure_prior.train(config.train_structure_prior)
        for param in structure_prior.parameters():
            param.requires_grad = bool(config.train_structure_prior)

    dataset = _build_train_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=assessment_collate,
        pin_memory=device.type == "cuda",
    )
    if not config.train_backbone:
        params = list(model.assess_prior.parameters()) if use_assessment_degra_prior else []
    else:
        params = [param for param in model.parameters() if param.requires_grad]
    if structure_prior is not None and config.train_structure_prior:
        params.extend(param for param in structure_prior.parameters() if param.requires_grad)
    if not params:
        raise RuntimeError("No trainable parameters. Check train_backbone and adapter settings.")
    optimizer = _build_optimizer(params, config)
    planned_steps = _planned_train_steps(config, len(loader))
    scheduler = _build_lr_scheduler(optimizer, config, planned_steps)
    wandb_logger = _WandbLogger(config)

    eval_batch_size = config.eval_batch_size or config.batch_size
    eval_train_loader = None
    eval_val_loader = None
    if config.eval_enabled:
        eval_train_loader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=assessment_collate,
            pin_memory=device.type == "cuda",
        )
        val_dataset = _build_val_dataset(config)
        if val_dataset is not None:
            eval_val_loader = DataLoader(
                val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=assessment_collate,
                pin_memory=device.type == "cuda",
            )

    log(f"samples={len(dataset)} output_dir={config.output_dir}")
    log(f"device={device} train_backbone={config.train_backbone} batch_size={config.batch_size}")
    log(f"optimizer={config.optimizer} lr={config.lr} betas=({config.beta1},{config.beta2}) weight_decay={config.weight_decay}")
    log(
        f"lr_scheduler={config.lr_scheduler} lr_min={config.lr_min} step_size={config.lr_step_size} "
        f"gamma={config.lr_gamma} milestones={config.lr_milestones} t_max={config.lr_t_max} "
        f"resolved_t_max={_resolve_lr_t_max(config, planned_steps)} planned_steps={planned_steps} "
        f"warmup_enabled={config.lr_warmup_enabled} warmup_steps={config.lr_warmup_steps} "
        f"warmup_start_factor={config.lr_warmup_start_factor}"
    )
    log(f"objective={config.objective} loss_type={config.loss_type} loss_weight={config.loss_weight}")
    log(f"degradation_prior_source={config.degradation_prior_source}")
    log(f"content_prior={'enabled' if use_content_prior else 'disabled'} structure_prior={'enabled' if structure_prior is not None else 'disabled'} train_structure_prior={config.train_structure_prior}")
    log(f"wandb={'enabled' if wandb_logger.run is not None else 'disabled'}")
    if config.eval_enabled:
        log(
            f"evaluation=enabled every_steps={config.eval_every_steps} every_epochs={config.eval_every_epochs} "
            f"batch_size={eval_batch_size} train_max_batches={config.eval_train_max_batches} "
            f"val={'enabled' if eval_val_loader is not None else 'disabled'} val_max_batches={config.eval_val_max_batches}"
        )
    else:
        log("evaluation=disabled")
    if sde is not None:
        log(
            f"sde=max_sigma:{sde.max_sigma:.6f} T:{sde.T} schedule:{_cfg_value(config.sde_schedule, tpgd_options.get('sde', {}), 'schedule', 'cosine')} "
            f"eps:{_cfg_value(config.sde_eps, tpgd_options.get('sde', {}), 'eps', 0.005)} t_range:{config.sde_t_start}-{config.sde_t_end}"
        )

    global_step = 0
    last_loss = None
    model.train()
    for epoch in range(1, config.epochs + 1):
        for batch in loader:
            global_step += 1
            lq = batch["lq"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            hidden = batch["hidden"]
            mask = batch["mask"]
            if hidden is not None:
                hidden = hidden.to(device, non_blocking=True)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)

            content_context = None
            deg_context_input = None
            lq_clip = batch["lq_clip"].to(device, non_blocking=True) if (use_content_prior or use_tpgd_degra_prior) else None
            if content_prior_model is not None and lq_clip is not None:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    if use_content_prior:
                        content_context = content_prior_model.get_content_prior(lq_clip).float()
                    if use_tpgd_degra_prior:
                        deg_context_input = content_prior_model.encode_for_degradation(lq_clip).float()
            elif config.random_content_context:
                content_context = torch.randn(
                    lq.shape[0],
                    backbone_cfg.context_dim,
                    device=device,
                    requires_grad=not config.train_backbone,
                )
            elif backbone_cfg.use_image_context:
                content_context = torch.zeros(
                    lq.shape[0],
                    backbone_cfg.context_dim,
                    device=device,
                    requires_grad=not config.train_backbone,
                )

            struct_tokens = None
            if structure_prior is not None:
                struct_input = (lq + 1.0) * 0.5
                if config.train_structure_prior:
                    struct_tokens = structure_prior(struct_input)
                else:
                    with torch.no_grad():
                        struct_tokens = structure_prior(struct_input)

            optimizer.zero_grad(set_to_none=True)
            if config.objective == "sde":
                assert sde is not None
                timesteps, states = sde.generate_random_states(
                    x0=gt,
                    mu=lq,
                    T_start=config.sde_t_start,
                    T_end=config.sde_t_end,
                )
                # TPGDiff uses a custom checkpoint function that expects selected
                # forward inputs to require grad. These tensors are not optimized;
                # they only keep the adapter-only backward path valid.
                states_for_model = states.detach().requires_grad_(True) if not config.train_backbone else states
                lq_for_model = lq.detach().requires_grad_(True) if not config.train_backbone else lq
                output, deg_context = model(
                    states_for_model,
                    lq_for_model,
                    timesteps.reshape(-1),
                    assessment_hidden=hidden if use_assessment_degra_prior else None,
                    assessment_mask=mask if use_assessment_degra_prior else None,
                    deg_context=deg_context_input,
                    content_context=content_context,
                    struct_tokens=struct_tokens,
                    return_context=True,
                )
                score = sde.get_score_from_noise(output, timesteps)
                xt_1_expectation = sde.reverse_sde_step_mean(states_for_model, score, timesteps)
                xt_1_optimum = sde.reverse_optimum_step(states, gt, timesteps)
                loss = config.loss_weight * _matching_loss(xt_1_expectation, xt_1_optimum, config.loss_type)
            else:
                lq_for_model = lq.detach().requires_grad_(True) if not config.train_backbone else lq
                time = torch.ones(lq.shape[0], device=device)
                output, deg_context = model(
                    lq_for_model,
                    lq_for_model,
                    time,
                    assessment_hidden=hidden if use_assessment_degra_prior else None,
                    assessment_mask=mask if use_assessment_degra_prior else None,
                    deg_context=deg_context_input,
                    content_context=content_context,
                    struct_tokens=struct_tokens,
                    return_context=True,
                )
                loss = torch.nn.functional.mse_loss(output, gt)
            loss.backward()
            if not config.train_backbone:
                for param in model.backbone.parameters():
                    param.grad = None
            lr_used = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            lr_next = float(optimizer.param_groups[0]["lr"])
            last_loss = float(loss.detach().cpu())

            if global_step == 1 or global_step % config.log_every == 0:
                log(
                    f"step={global_step} epoch={epoch} loss={last_loss:.6f} lr={lr_used:.8g} lr_next={lr_next:.8g} "
                    f"output_shape={tuple(output.shape)} deg_context_shape={tuple(deg_context.shape)}"
                )
                wandb_logger.log({"train/loss": last_loss, "train/epoch": epoch, "train/lr": lr_used, "train/lr_next": lr_next}, step=global_step)
            if config.eval_enabled and config.eval_every_steps > 0 and global_step % config.eval_every_steps == 0:
                eval_payload: dict[str, float] = {}
                if eval_train_loader is not None:
                    eval_payload.update(_evaluate_loader(
                        name="eval_train",
                        loader=eval_train_loader,
                        max_batches=config.eval_train_max_batches,
                        device=device,
                        model=model,
                        backbone_cfg=backbone_cfg,
                        content_prior_model=content_prior_model,
                        structure_prior=structure_prior,
                        config=config,
                        sde=sde,
                        use_assessment_degra_prior=use_assessment_degra_prior,
                        use_tpgd_degra_prior=use_tpgd_degra_prior,
                        use_content_prior=use_content_prior,
                    ))
                if eval_val_loader is not None:
                    eval_payload.update(_evaluate_loader(
                        name="eval_val",
                        loader=eval_val_loader,
                        max_batches=config.eval_val_max_batches,
                        device=device,
                        model=model,
                        backbone_cfg=backbone_cfg,
                        content_prior_model=content_prior_model,
                        structure_prior=structure_prior,
                        config=config,
                        sde=sde,
                        use_assessment_degra_prior=use_assessment_degra_prior,
                        use_tpgd_degra_prior=use_tpgd_degra_prior,
                        use_content_prior=use_content_prior,
                    ))
                if eval_payload:
                    metrics = " ".join(f"{key}={value:.6f}" for key, value in eval_payload.items())
                    log(f"eval step={global_step} epoch={epoch} {metrics}")
                    wandb_logger.log(eval_payload, step=global_step)
            if config.save_every > 0 and global_step % config.save_every == 0:
                save_path = save_checkpoint(config.output_dir, model, optimizer, step=global_step, epoch=epoch, config=config, structure_prior=structure_prior, scheduler=scheduler)
                log(f"saved={save_path}")
            if config.max_steps > 0 and global_step >= config.max_steps:
                break
        if config.eval_enabled and config.eval_every_epochs > 0 and epoch % config.eval_every_epochs == 0:
            eval_payload: dict[str, float] = {}
            if eval_train_loader is not None:
                eval_payload.update(_evaluate_loader(
                    name="eval_train",
                    loader=eval_train_loader,
                    max_batches=config.eval_train_max_batches,
                    device=device,
                    model=model,
                    backbone_cfg=backbone_cfg,
                    content_prior_model=content_prior_model,
                    structure_prior=structure_prior,
                    config=config,
                    sde=sde,
                    use_assessment_degra_prior=use_assessment_degra_prior,
                    use_tpgd_degra_prior=use_tpgd_degra_prior,
                    use_content_prior=use_content_prior,
                ))
            if eval_val_loader is not None:
                eval_payload.update(_evaluate_loader(
                    name="eval_val",
                    loader=eval_val_loader,
                    max_batches=config.eval_val_max_batches,
                    device=device,
                    model=model,
                    backbone_cfg=backbone_cfg,
                    content_prior_model=content_prior_model,
                    structure_prior=structure_prior,
                    config=config,
                    sde=sde,
                    use_assessment_degra_prior=use_assessment_degra_prior,
                    use_tpgd_degra_prior=use_tpgd_degra_prior,
                    use_content_prior=use_content_prior,
                ))
            if eval_payload:
                metrics = " ".join(f"{key}={value:.6f}" for key, value in eval_payload.items())
                log(f"eval step={global_step} epoch={epoch} {metrics}")
                wandb_logger.log(eval_payload, step=global_step)
        if config.max_steps > 0 and global_step >= config.max_steps:
            break

    save_path = save_checkpoint(config.output_dir, model, optimizer, step=global_step, epoch=epoch, config=config, structure_prior=structure_prior, scheduler=scheduler)
    log(f"train_ok steps={global_step} last_loss={last_loss:.6f} saved={save_path}")
    wandb_logger.finish()
    return {"steps": global_step, "last_loss": last_loss, "checkpoint": str(save_path)}
