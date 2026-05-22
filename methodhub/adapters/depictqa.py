from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from ..base import MethodAdapter, SourceRef, strip_module_prefix
from ..paths import default_repo_root, push_cwd, push_sys_path, require_exists


class DepictQAAdapter(MethodAdapter):
    name = "depictqa"
    capabilities = ("assess", "compare", "explain")
    source_refs = (
        SourceRef(
            repo="DepictQA",
            entrypoints=(
                "src/model/depictqa.py",
                "src/infer.py",
            ),
            notes="Standalone DepictQA source model",
        ),
    )

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        config_path: Optional[Path] = None,
        delta_path: Optional[Path] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        training: bool = False,
    ) -> None:
        super().__init__()
        self.repo_root = Path(repo_root).resolve() if repo_root else default_repo_root("depictqa")
        self.config_path = Path(config_path).resolve() if config_path else self.repo_root / "experiments" / "DQ495K" / "config.yaml"
        self.delta_path = Path(delta_path).resolve() if delta_path else None
        self.device = device or "cuda"
        self.dtype = dtype if dtype is not None else (torch.float16 if str(self.device).startswith("cuda") else None)
        self.training = training
        self.cfg: Any = None

    def load(self) -> "DepictQAAdapter":
        require_exists(self.repo_root, "DepictQA repo")
        require_exists(self.config_path, "DepictQA config")

        with push_sys_path(self.repo_root / "src"), push_cwd(self.config_path.parent):
            import yaml
            from easydict import EasyDict
            from model.depictqa import DepictQA

            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = EasyDict(yaml.safe_load(f))
            if self.delta_path is not None:
                cfg.model["delta_path"] = str(self.delta_path)

            model = DepictQA(cfg, training=self.training)
            ckpt_path = Path(cfg.model["delta_path"]).expanduser()
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(strip_module_prefix(state), strict=False)

            if self.dtype is not None:
                model = model.to(dtype=self.dtype)
            self.model = model.to(self.device).eval()
            self.cfg = cfg
            self.loaded = True
        return self

    def forward(self, inputs: Any) -> Any:
        return self.ensure_loaded().model(inputs)

    def generate(self, inputs: Any) -> Any:
        return self.ensure_loaded().model.generate(inputs)
