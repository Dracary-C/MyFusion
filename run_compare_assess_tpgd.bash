#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${MYFUSION_TEST_CONFIG:-${MYFUSION_CONFIG:-/home/chenzt/Experiment/All-in-One/MyFusion/test.yml}}"
PYTHON_BIN="${PYTHON_BIN:-/home/chenzt/anaconda3/envs/rar/bin/python}"

"$PYTHON_BIN" /home/chenzt/Experiment/All-in-One/MyFusion/scripts/compare_assess_tpgd_checkpoints.py --config "$CONFIG_FILE" "$@"
