#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${MYFUSION_CONFIG:-/home/chenzt/Experiment/All-in-One/MyFusion/config.yml}"
PYTHON_BIN="${PYTHON_BIN:-/home/chenzt/anaconda3/envs/rar/bin/python}"

_cfg_get() {
  "$PYTHON_BIN" -c 'import sys, yaml
path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
value = data
for part in key.split("."):
    if not isinstance(value, dict) or part not in value:
        value = default
        break
    value = value[part]
if isinstance(value, bool):
    value = str(value).lower()
print(value)' "$CONFIG_FILE" "$1" "$2"
}

PYTHON_BIN="$(_cfg_get commands.python "$PYTHON_BIN")"
export CUDA_VISIBLE_DEVICES="$(_cfg_get runtime.cuda_visible_devices 1)"
export TOKENIZERS_PARALLELISM="$(_cfg_get runtime.tokenizers_parallelism false)"

"$PYTHON_BIN" /home/chenzt/Experiment/All-in-One/MyFusion/my_method.py --config "$CONFIG_FILE"
