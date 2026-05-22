from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from ..base import MethodAdapter, SourceRef, strip_module_prefix
from ..paths import default_repo_root, push_cwd, push_sys_path, require_exists


TPGDIFF_WEIGHT_ROOT = Path("/data/chenzt/model_weights/tpgd/universal_restore")


def _resolve_checkpoint(path: Optional[Path | str], repo_root: Path, label: str) -> Optional[Path]:
    if path is None:
        return None

    raw = Path(path).expanduser()
    candidates = []

    def add_candidate(candidate: Path) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    add_candidate(raw)
    if raw.is_absolute():
        add_candidate(repo_root / "pretrained" / raw.name)
        if len(raw.parts) > 2 and raw.parts[1] == "pretrained":
            add_candidate(repo_root / Path(*raw.parts[1:]))
    else:
        add_candidate(repo_root / raw)
        add_candidate(repo_root / "pretrained" / raw.name)
    add_candidate(TPGDIFF_WEIGHT_ROOT / raw.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{label} not found. Checked: {checked}")


class TPGDiffPriorAdapter(MethodAdapter):
    name = "tpgdiff-prior"
    capabilities = ("prior", "encode", "score")
    source_refs = (
        SourceRef(
            repo="TPGDiff",
            entrypoints=(
                "universal-restoration/open_clip/factory.py",
                "universal-restoration/open_clip/prior_stage_model.py",
                "universal-restoration/open_clip/tpgd_model.py",
            ),
            notes="Reusable prior-stage CLIP and degradation encoder",
        ),
    )

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        checkpoint_path: Optional[Path] = None,
        options_path: Optional[Path] = None,
        num_degradations: int = 5,
        device: Optional[str] = None,
        precision: str = "fp32",
    ) -> None:
        super().__init__()
        self.repo_root = Path(repo_root).resolve() if repo_root else default_repo_root("tpgdiff")
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.options_path = Path(options_path).resolve() if options_path else None
        self.num_degradations = num_degradations
        self.device = device or "cuda"
        self.precision = precision
        self.clip_model = None
        self.preprocess = None

    def load(self) -> "TPGDiffPriorAdapter":
        repo = self.repo_root / "universal-restoration"
        require_exists(repo, "TPGDiff universal-restoration repo")

        with push_sys_path(repo), push_cwd(repo):
            import open_clip
            from open_clip.prior_stage_model import PriorStageModel

            base_model, _, preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name,
                pretrained=self.clip_pretrained,
                precision=self.precision,
                device=self.device,
            )
            teacher_encoder = base_model.visual
            student_encoder = deepcopy(base_model.visual)
            deg_backbone = deepcopy(base_model.visual)

            if hasattr(base_model.visual, "output_dim"):
                embed_dim = base_model.visual.output_dim
            elif hasattr(base_model, "embed_dim"):
                embed_dim = base_model.embed_dim
            else:
                raise RuntimeError("Cannot infer embed_dim from TPGDiff CLIP base model.")

            prior_model = PriorStageModel(
                teacher_encoder=teacher_encoder,
                student_encoder=student_encoder,
                deg_backbone=deg_backbone,
                embed_dim=embed_dim,
                num_degradations=self.num_degradations,
                content_loss_weight=1.0,
                deg_loss_weight=1.0,
                use_cosine_distill=True,
                normalize_embedding=True,
                freeze_teacher=True,
                freeze_deg_backbone=True,
            ).to(self.device)

            if self.checkpoint_path is not None:
                require_exists(self.checkpoint_path, "TPGDiff prior checkpoint")
                ckpt = torch.load(self.checkpoint_path, map_location="cpu")
                state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                prior_model.load_state_dict(strip_module_prefix(state), strict=True)

            self.model = prior_model.eval()
            self.clip_model = base_model.eval()
            self.preprocess = preprocess
            self.loaded = True
        return self

    def get_content_prior(self, img_lq):
        return self.ensure_loaded().model.get_content_prior(img_lq)

    def get_degradation_prior(self, img_lq, as_prob: bool = True):
        return self.ensure_loaded().model.get_degradation_prior(img_lq, as_prob=as_prob)

    def forward(self, img_gt, img_lq, deg_label, return_embeddings: bool = True):
        return self.ensure_loaded().model(img_gt, img_lq, deg_label, return_embeddings=return_embeddings)

    def source_command(self) -> tuple[Path, tuple[str, ...]]:
        config_dir = self.repo_root / "universal-restoration" / "config" / "tpgd-sde"
        opt = self.options_path or (config_dir / "options" / "test.yml")
        return config_dir, ("python", "test.py", "-opt", str(opt))


class TPGDiffRuntimeAdapter(MethodAdapter):
    name = "tpgdiff-runtime"
    capabilities = ("restore", "prior", "demo")
    source_refs = (
        SourceRef(
            repo="TPGDiff",
            entrypoints=(
                "universal-restoration/config/tpgd-sde/app.py",
                "universal-restoration/config/tpgd-sde/test.py",
                "universal-restoration/config/tpgd-sde/models/denoising_model.py",
            ),
            notes="Full TPGDiff restoration runtime with content, degradation, and structure priors",
        ),
    )

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        options_path: Optional[Path] = None,
        restoration_checkpoint: Optional[Path] = None,
        tpgd_checkpoint: Optional[Path] = None,
        device: Optional[str] = None,
        sampling_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.repo_root = Path(repo_root).resolve() if repo_root else default_repo_root("tpgdiff")
        self.options_path = Path(options_path).resolve() if options_path else None
        restoration_checkpoint = restoration_checkpoint or os.environ.get("METHODHUB_TPGDIFF_RESTORE_CKPT")
        tpgd_checkpoint = tpgd_checkpoint or os.environ.get("METHODHUB_TPGDIFF_TPGD_CKPT")
        self.restoration_checkpoint = Path(restoration_checkpoint).expanduser() if restoration_checkpoint else None
        self.tpgd_checkpoint = Path(tpgd_checkpoint).expanduser() if tpgd_checkpoint else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_mode = sampling_mode

        self.clip_model: Any = None
        self.prior_model: Any = None
        self.sde: Any = None
        self.opt: Any = None
        self.util: Any = None

    def load(self) -> "TPGDiffRuntimeAdapter":
        repo = require_exists(self.repo_root / "universal-restoration", "TPGDiff universal-restoration repo")
        config_dir = require_exists(repo / "config" / "tpgd-sde", "TPGDiff tpgd-sde config dir")
        opt_path = self.options_path or (config_dir / "options" / "test.yml")
        require_exists(opt_path, "TPGDiff runtime options")

        restoration_checkpoint = _resolve_checkpoint(
            self.restoration_checkpoint or (TPGDIFF_WEIGHT_ROOT / "universal-ir.pth"),
            self.repo_root,
            "TPGDiff restoration checkpoint",
        )
        tpgd_checkpoint = _resolve_checkpoint(
            self.tpgd_checkpoint or (TPGDIFF_WEIGHT_ROOT / "tpgd_ViT-B-32.pt"),
            self.repo_root,
            "TPGDiff prior/control checkpoint",
        )

        with push_sys_path(config_dir, repo), push_cwd(config_dir):
            import open_clip
            import options as option
            import utils as util
            from models import create_model
            from open_clip.prior_stage_model import PriorStageModel

            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            opt = option.parse(str(opt_path), is_train=False)
            if visible_devices:
                # TPGDiff's option parser rewrites CUDA_VISIBLE_DEVICES from YAML.
                # Keep the device selected by the launcher, and use gpu_ids as logical ids.
                os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
            opt = option.dict_to_nonedict(opt)
            opt["gpu_ids"] = [0] if self.device.startswith("cuda") else None
            opt.setdefault("path", {})
            opt["path"]["pretrain_model_G"] = str(restoration_checkpoint)
            opt["path"]["tpgd"] = str(tpgd_checkpoint)
            opt["path"]["prior"] = str(tpgd_checkpoint)

            model = create_model(opt)
            base_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                precision="fp32",
                device=self.device,
            )
            teacher_encoder = base_model.visual
            student_encoder = deepcopy(base_model.visual)
            deg_backbone = deepcopy(base_model.visual)

            if hasattr(base_model.visual, "output_dim"):
                embed_dim = base_model.visual.output_dim
            elif hasattr(base_model, "embed_dim"):
                embed_dim = base_model.embed_dim
            else:
                raise RuntimeError("Cannot infer embed_dim from TPGDiff CLIP base model.")

            prior_model = PriorStageModel(
                teacher_encoder=teacher_encoder,
                student_encoder=student_encoder,
                deg_backbone=deg_backbone,
                embed_dim=embed_dim,
                num_degradations=len(opt["distortion"]),
                content_loss_weight=1.0,
                deg_loss_weight=1.0,
                use_cosine_distill=True,
                normalize_embedding=True,
                freeze_teacher=True,
                freeze_deg_backbone=True,
            )
            ckpt = torch.load(tpgd_checkpoint, map_location="cpu")
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            prior_model.load_state_dict(strip_module_prefix(state), strict=True)
            prior_model = prior_model.to(self.device).eval()

            sde = util.IRSDE(
                max_sigma=opt["sde"]["max_sigma"],
                T=opt["sde"]["T"],
                schedule=opt["sde"]["schedule"],
                eps=opt["sde"]["eps"],
                device=model.device,
            )
            sde.set_model(model.model)

            self.model = model
            self.clip_model = base_model.eval()
            self.prior_model = prior_model
            self.sde = sde
            self.opt = opt
            self.util = util
            self.sampling_mode = self.sampling_mode or opt["sde"]["sampling_mode"]
            self.loaded = True
        return self

    @staticmethod
    def _to_numpy_rgb(image: Any) -> np.ndarray:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"))
        elif torch.is_tensor(image):
            tensor = image.detach().float().cpu()
            if tensor.dim() == 4:
                if tensor.shape[0] != 1:
                    raise ValueError("TPGDiffRuntimeAdapter.restore only supports one image at a time.")
                tensor = tensor[0]
            if tensor.dim() != 3:
                raise ValueError(f"Expected image tensor with shape [C,H,W], got {tuple(tensor.shape)}.")
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor[:3]
                if tensor.shape[0] == 1:
                    tensor = tensor.expand(3, -1, -1)
                array = tensor.clamp(0, 1).permute(1, 2, 0).numpy()
                array = (array * 255.0).round().astype(np.uint8)
            else:
                raise ValueError(f"Expected channel-first tensor, got shape {tuple(tensor.shape)}.")
        else:
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.stack([array, array, array], axis=-1)
            if array.ndim != 3:
                raise ValueError(f"Expected image array with shape [H,W,C], got {array.shape}.")
            if array.shape[2] == 4:
                array = array[:, :, :3]
            if array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            if array.dtype != np.uint8:
                array = array.astype(np.float32)
                if array.max() <= 1.0:
                    array = array * 255.0
                array = np.clip(array, 0, 255).round().astype(np.uint8)
        return np.ascontiguousarray(array)

    @staticmethod
    def _clip_transform(np_image: np.ndarray, resolution: int = 224) -> torch.Tensor:
        from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

        pil_image = Image.fromarray(np_image)
        transform = Compose(
            [
                Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(resolution),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        return transform(pil_image)

    def restore(self, image: Any) -> Image.Image:
        self.ensure_loaded()
        np_rgb = self._to_numpy_rgb(image)
        np_float = np_rgb.astype(np.float32) / 255.0

        img4clip = self._clip_transform(np_rgb).unsqueeze(0).to(self.device)
        amp_enabled = self.device.startswith("cuda") and torch.cuda.is_available()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            content_context = self.prior_model.get_content_prior(img4clip).float()
            deg_context = self.prior_model.encode_for_degradation(img4clip).float()

        lq_tensor = torch.from_numpy(np_float).permute(2, 0, 1).unsqueeze(0).float()
        noisy_tensor = self.sde.noise_state(lq_tensor)
        self.model.feed_data(
            noisy_tensor,
            lq_tensor,
            deg_context=deg_context,
            content_context=content_context,
        )
        self.model.test(self.sde, mode=self.sampling_mode, save_states=False)
        visuals = self.model.get_current_visuals(need_GT=False)
        output_bgr = self.util.tensor2img(visuals["Output"].squeeze())
        output_rgb = output_bgr[:, :, [2, 1, 0]]
        return Image.fromarray(output_rgb)

    def process(self, image: Any, name: str = "sample") -> Image.Image:
        return self.restore(image)

    def source_command(self) -> tuple[Path, tuple[str, ...]]:
        config_dir = self.repo_root / "universal-restoration" / "config" / "tpgd-sde"
        opt = self.options_path or (config_dir / "options" / "test.yml")
        return config_dir, ("python", "app.py", "-opt", str(opt))
