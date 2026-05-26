#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${MYFUSION_TRAIN_CONFIG:-${MYFUSION_CONFIG:-/home/chenzt/Experiment/All-in-One/MyFusion/train.yml}}"
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
elif isinstance(value, (list, tuple)):
    value = ",".join(str(item) for item in value)
print(value)' "$CONFIG_FILE" "$1" "$2"
}

export CUDA_VISIBLE_DEVICES="$(_cfg_get gpu_ids 0)"
export TOKENIZERS_PARALLELISM="$(_cfg_get runtime.tokenizers_parallelism false)"

cd /home/chenzt/Experiment/All-in-One/MyFusion
"$PYTHON_BIN" scripts/generate_assessment_hidden.py --config "$CONFIG_FILE" "$@"
