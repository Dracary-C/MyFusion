from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from ..base import MethodAdapter, SourceRef
from ..paths import default_repo_root, push_cwd, push_sys_path, require_exists



def _get_easydict_class():
    try:
        from easydict import EasyDict as ExternalEasyDict

        return ExternalEasyDict
    except ModuleNotFoundError:
        import sys
        import types

        class EasyDict(dict):
            def __init__(self, mapping=None, **kwargs):
                super().__init__()
                data = {} if mapping is None else dict(mapping)
                data.update(kwargs)
                for key, value in data.items():
                    self[key] = self._convert(value)

            @classmethod
            def _convert(cls, value):
                if isinstance(value, dict) and not isinstance(value, EasyDict):
                    return cls(value)
                if isinstance(value, list):
                    return [cls._convert(item) for item in value]
                return value

            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError as exc:
                    raise AttributeError(key) from exc

            def __setattr__(self, key, value):
                self[key] = self._convert(value)

        module = types.ModuleType("easydict")
        module.EasyDict = EasyDict
        sys.modules.setdefault("easydict", module)
        return EasyDict


def _resolve_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(value).lower()
    if key not in aliases:
        raise KeyError(f"Unsupported RAR dtype: {value}")
    return aliases[key]


class RARAssessmentAdapter(MethodAdapter):
    name = "rar-assessment"
    capabilities = ("assess", "compare", "explain")
    source_refs = (
        SourceRef(
            repo="RAR",
            entrypoints=(
                "iqa/main.py",
                "iqa/depictqa.py",
            ),
            notes="RAR's built-in IQA model",
        ),
    )

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        config_path: Optional[Path] = None,
        device: Optional[str] = None,
        dtype: str | torch.dtype = "bf16",
    ) -> None:
        super().__init__()
        self.repo_root = Path(repo_root).resolve() if repo_root else default_repo_root("rar")
        self.config_path = Path(config_path).resolve() if config_path else self.repo_root / "iqa" / "config.yaml"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _resolve_dtype(dtype)
        self.cfg: Any = None

    def load(self) -> "RARAssessmentAdapter":
        require_exists(self.repo_root, "RAR repo")
        require_exists(self.config_path, "RAR IQA config")

        with push_sys_path(self.repo_root), push_cwd(self.repo_root):
            EasyDict = _get_easydict_class()
            from iqa import DepictQA, load_pretrained_weights

            with open(self.config_path, "r", encoding="utf-8") as handle:
                self.cfg = EasyDict(yaml.safe_load(handle))

            model = DepictQA(self.cfg, training=False)
            model = load_pretrained_weights(self.cfg, model, logger=None)
            self.model = model.eval().to(self.dtype).to(self.device)
            self.loaded = True
        return self

    def forward(self, inputs: Any, latent_input: bool = False) -> Any:
        return self.ensure_loaded().model(inputs, latent_input=latent_input)

    def generate(self, inputs: Any, latent_input: bool = False, save_hidden: bool = False) -> Any:
        return self.ensure_loaded().model.generate(inputs, latent_input=latent_input, save_hidden=save_hidden)


class RARConnectorAdapter(MethodAdapter):
    name = "rar-connector"
    capabilities = ("connector", "project")
    source_refs = (
        SourceRef(
            repo="RAR",
            entrypoints=("diffusion/model/qa_connector.py",),
            notes="QFormer bridge used by RAR",
        ),
    )

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        hidden_dim: int = 1024,
        layers: int = 8,
        heads: int = 16,
    ) -> None:
        super().__init__()
        self.repo_root = Path(repo_root).resolve() if repo_root else default_repo_root("rar")
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.heads = heads

    def load(self) -> "RARConnectorAdapter":
        import importlib.util
        import sys

        connector_path = self.repo_root / "diffusion" / "model" / "qa_connector.py"
        require_exists(connector_path, "RAR qa_connector.py")
        spec = importlib.util.spec_from_file_location("methodhub_rar_qa_connector", connector_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load RAR connector from {connector_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        self.model = module.QFormer(hidden_dim=self.hidden_dim, layers=self.layers, heads=self.heads)
        self.loaded = True
        return self


class RARRuntimeAdapter(MethodAdapter):
    name = "rar"
    capabilities = ("restore", "iterate", "demo")
    source_refs = (
        SourceRef(
            repo="RAR",
            entrypoints=("run.py", "configs/infer_cfg.yaml"),
            notes="Full iterative RAR runtime",
        ),
    )

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        super().__init__()
        self.repo_root = Path(repo_root).resolve() if repo_root else default_repo_root("rar")
        self.model = None
        self.vae = None
        self.args = None
        self.config = None
        self.device = None

    def load(self) -> "RARRuntimeAdapter":
        with push_sys_path(self.repo_root), push_cwd(self.repo_root):
            from run import load_rar_model

            self.model, self.vae, self.args, self.config, self.device = load_rar_model()
            self.loaded = True
        return self

    def process(self, image: Any, name: str = "sample") -> Any:
        self.ensure_loaded()
        with push_sys_path(self.repo_root), push_cwd(self.repo_root):
            from run import RAR_process

            return RAR_process(image, name, self.model, self.vae, self.args, self.config, self.device)
