# MyFusion
MyFusion 是一个图像复原实验仓库。当前目标是把 TPGDiff、RAR、DepictQA 的可用流程收进同一个工程，先跑通整合版，再逐步替换成自己的模块。

方法原理部分暂时留空，等实验分支稳定后再补。

## 关键入口
| 文件/目录 | 作用 |
| --- | --- |
| `run.bash` | 启动 Streamlit 前端 |
| `run_cli.bash` | 命令行单图测试 |
| `run_assess_tpgd.bash` | 启动 Assess-TPGD 初版训练 |
| `app.py` | 前端页面 |
| `my_method.py` | 当前 TPGDiff + RAR 主流程 |
| `fusion_config.py` | 读取 `config.yml` |
| `scripts/inspect_assessment_hidden.py` | 查看 Assessment hidden states |
| `scripts/dry_run_assess_tpgd.py` | Assess hidden -> TPGDiff UNet dry run |
| `scripts/train_assess_tpgd.py` | Assess-TPGD 训练入口 |
| `methodhub/` | 本地方法适配层 |
| `myfusion/legacy/` | TPGDiff/RAR/DepictQA 源码快照 |
| `myfusion/modules/` | 后续自研模块 |
| `myfusion/pipelines/` | 实验流程入口 |
| `configs/config.example.yml` | 可提交配置模板 |
| `config.yml` | 本地配置，不提交 GitHub |

## 环境
建议复用已经跑通的环境：
```bash
conda activate rar
cd ~/Experiment/All-in-One/MyFusion
pip install -e .
```
如果重建环境：
```bash
conda create -n myfusion python=3.11 -y
conda activate myfusion
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```
`torch` 要按服务器 CUDA 版本选择。若遇到 `huggingface-hub` 版本报错：
```bash
pip install "transformers==4.45.2" "huggingface-hub>=0.23.2,<1.0"
```

## 配置
第一次使用：
```bash
cp configs/config.example.yml config.yml
```
常改字段都在 `config.yml`：
```yaml
runtime:
  cuda_visible_devices: "0"      # 物理 GPU 编号
server:
  port: 8502                      # 前端端口
fusion:
  max_rounds: 2                   # 最大 IR 轮数
  stop_on_reject: true            # 质量判断拒绝后是否停止
  extract_assessment_reasoning_hidden: true
  resize: 256
paths:
  input: /path/to/input.png
  output_dir: /path/to/outputs/tpgd_rar_run
  ui_output_root: /path/to/outputs/ui
  tpgd_options: ./myfusion/legacy/tpgdiff/universal-restoration/config/tpgd-sde/options/test_fast.yml
  tpgd_checkpoint: /path/to/latest_G.pth
  tpgd_prior: /path/to/tpgd_ViT-B-32.pt
  rar_config: ./myfusion/legacy/rar/configs/infer_cfg.yaml
  rar_assessment_config: ./myfusion/legacy/rar/iqa/config.yaml
commands:
  python: /path/to/env/bin/python
  streamlit: /path/to/env/bin/streamlit
assess_tpgd:
  lq_dir: /path/to/train/LQ
  gt_dir: /path/to/train/GT
  hidden_dir: /path/to/assessment/features
  hidden_path: ""                  # 单 hidden 文件 smoke test 时使用
  output_dir: /path/to/outputs/assess_tpgd
  image_size: 128
  batch_size: 1
  epochs: 1
  max_steps: 100
  lr: 0.0001
  freeze_backbone: true
```
`CUDA_VISIBLE_DEVICES="1"` 后程序里仍显示 `cuda:0` 是正常的，表示当前可见的第 0 张卡，物理上对应 GPU 1。

## 权重和数据
仓库不保存权重、数据和输出。权重路径写在 `config.yml`，RAR/DepictQA 的权重通常继续由 RAR 配置文件管理。
不要提交：`config.yml`、`outputs/`、`datasets/`、`model_weights/`、`*.pth`、`*.pt`、`*.ckpt`、`*.safetensors`、`*.bin`。

## 启动前端
```bash
cd ~/Experiment/All-in-One/MyFusion
./run.bash
```
浏览器访问：
```text
http://localhost:8502/
```
实际端口以 `config.yml` 的 `server.port` 为准。被占用就改成 `8503` 等空闲端口。

## 命令行测试
```bash
cd ~/Experiment/All-in-One/MyFusion
./run_cli.bash
```
输入和输出由 `config.yml` 控制：
```yaml
paths:
  input: /path/to/input.png
  output_dir: /path/to/output_dir
```
输出重点看：`*_history.json`、`*_debug.log`、`features/*.pt`。其中 debug log 用来看每一轮为什么继续或停止。

## Assessment Hidden States
开启：
```yaml
fusion:
  extract_assessment_reasoning_hidden: true
```
运行 UI 或 CLI 后，`features/` 下会生成 `*_assessment_reasoning_hidden.pt`。
```bash
HIDDEN_PT=/path/to/image_round1_assessment_reasoning_hidden.pt
python scripts/inspect_assessment_hidden.py "$HIDDEN_PT"
```
主要字段：`prefix_hidden`、`generated_hidden`、`condition_hidden`。替换 TPGDiff degradation prior 时优先用 `condition_hidden`。

## Assess-TPGD 训练入口
当前提供的是初版 bootstrap 训练：成对 LQ/GT 图像 + 预提取 Assessment hidden states，训练 `AssessPriorAdapter` 接入 TPGDiff UNet 的链路。它不是最终完整 SDE 训练，但可以作为正式训练流程的起点。

先在 `config.yml` 设置 `assess_tpgd.lq_dir`、`gt_dir`、`hidden_dir` 和 `output_dir`，然后运行：
```bash
cd ~/Experiment/All-in-One/MyFusion
./run_assess_tpgd.bash
```
也可以临时覆盖参数：
```bash
./run_assess_tpgd.bash --max-steps 10 --image-size 64
```
输出在 `assess_tpgd.output_dir`，主要文件是 `train.log`、`latest.pt`、`step_*.pt`。默认 `freeze_backbone: true`，只训练 Assessment prior adapter；需要一起训练 UNet 时使用 `--train-backbone` 或在配置里改 `freeze_backbone: false`。

## Assess-TPGD Dry Run
只验证 hidden states 到 TPGDiff UNet 的 forward：
```bash
cd ~/Experiment/All-in-One/MyFusion
HIDDEN_PT=/path/to/image_round1_assessment_reasoning_hidden.pt
python scripts/dry_run_assess_tpgd.py \
  --config config.yml \
  --hidden "$HIDDEN_PT" \
  --no-load-checkpoint \
  --device cuda \
  --image-size 32 \
  --steps 1 \
  --mode both
```
成功时会看到：`train_ok`、`infer_ok`。核心 pipeline 在 `myfusion/pipelines/assess_tpgd.py`。

## 常见问题
- `streamlit run` 报 `No such option`：实验参数不要直接传给 Streamlit，改 `config.yml`。
- 端口被占用：`lsof -i :8502` 查看，`kill <PID>` 结束，或直接改 `server.port`。
- CUDA OOM：换空闲 GPU，或调小 `fusion.resize`、`fusion.max_rounds`。

## GitHub
提交前检查：
```bash
git status --ignored
```
正常情况下，`config.yml` 和 `outputs/` 应该是 ignored。
```bash
git add .
git commit -m "Initial MyFusion framework"
git branch -M main
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin main
```
第三方源码快照说明见 `docs/THIRD_PARTY_LICENSES.md` 和 `myfusion/legacy/*/SOURCE.md`。
