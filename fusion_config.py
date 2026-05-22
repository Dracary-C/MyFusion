from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "config.yml"

DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {
        "cuda_visible_devices": "1",
        "tokenizers_parallelism": False,
    },
    "server": {
        "address": "0.0.0.0",
        "port": 8502,
    },
    "fusion": {
        "max_rounds": 2,
        "stop_on_reject": True,
        "extract_assessment_reasoning_hidden": False,
        "assessment_reasoning_max_new_tokens": 256,
        "resize": 256,
        "device": "cuda",
        "rar_dtype": "bf16",
        "sampling_mode": "posterior",
    },
    "paths": {
        "input": str(APP_DIR / "demo_sample" / "lowlight1.png"),
        "output_dir": str(APP_DIR / "outputs" / "tpgd_rar_run"),
        "ui_output_root": str(APP_DIR / "outputs" / "ui"),
        "tpgd_options": str(APP_DIR / "myfusion" / "legacy" / "tpgdiff" / "universal-restoration" / "config" / "tpgd-sde" / "options" / "test_fast.yml"),
        "tpgd_checkpoint": "/data/chenzt/model_weights/tpgd/universal/ablation-d1-c1-s1/latest_G.pth",
        "tpgd_prior": "/data/chenzt/model_weights/tpgd/prior/tpgd_ViT-B-32.pt",
        "rar_config": str(APP_DIR / "myfusion" / "legacy" / "rar" / "configs" / "infer_cfg.yaml"),
        "rar_assessment_config": str(APP_DIR / "myfusion" / "legacy" / "rar" / "iqa" / "config.yaml"),
    },
    "commands": {
        "python": "/home/chenzt/anaconda3/envs/rar/bin/python",
        "streamlit": "/home/chenzt/anaconda3/envs/rar/bin/streamlit",
    },
    "ui": {
        "default_example": "lowlight1",
        "max_rounds_slider_max": 10,
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser()
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
        _deep_update(config, loaded)
    config["_config_path"] = str(config_path)
    return config


def config_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def config_path(config: dict[str, Any], dotted_key: str, default: str | Path | None = None) -> Path | None:
    value = config_get(config, dotted_key, default)
    if value in (None, ""):
        return None
    return Path(value).expanduser()
