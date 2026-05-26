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
| `fusion_config.py` | 读取 `test.yml` |
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
当前拆成两份配置：

```text
test.yml   # run.bash / run_cli.bash 使用
train.yml  # run_assess_tpgd.bash 使用
```

`test.yml` 控制 UI/CLI 推理，常改字段是 `gpu_ids`、`server`、`fusion`、`paths`、`commands`。

`train.yml` 仿照 TPGDiff options 的结构，训练入口只读取这份 yml 的顶层字段，不再读取旧的 `assess_tpgd:` 配置块：

```yaml
name: assess-tpgd-sde
gpu_ids: [2]
distortion: ['LOL-v2']          # 也可以改成 DehazeFormer、GoPro 等
use_wandb: false

sde:
  max_sigma: 50
  T: 100
  schedule: cosine
  eps: 0.005

datasets:
  train:
    mode: MD
    degradation: LOL-v2         # 数据集子目录名
    dataroot: /data/chenzt/Dataset/TPGDiff/Train
    lq_dir: ~                   # 为空时使用 dataroot/degradation/LQ
    gt_dir: ~                   # 为空时使用 dataroot/degradation/GT
    hidden_dir: /path/to/assessment/features
    hidden_key: condition_hidden
    image_size: 128
    batch_size: 1
    n_workers: 0

path:
  experiments_root: /path/to/outputs/assess_tpgd
  tpgd_options: ./myfusion/legacy/tpgdiff/universal-restoration/config/tpgd-sde/options/train_fast.yml
  pretrain_model_G: /path/to/latest_G.pth
  strict_load: false

train:
  optimizer: AdamW              # Adam、AdamW、Lion
  lr_G: 0.0001
  weight_decay_G: 0.0
  beta1: 0.9
  beta2: 0.999
  epochs: 1
  niter: 100
  train_backbone: false
  objective: sde                # sde 或 direct_mse
  loss_type: l1
  weight: 1.0

logger:
  print_freq: 10
  save_checkpoint_freq: 100
```

`CUDA_VISIBLE_DEVICES="2"` 后程序里仍显示 `cuda:0` 是正常的，表示当前可见的第 0 张卡，物理上对应 GPU 2。

## 权重和数据
仓库不保存权重、数据和输出。权重路径写在 `test.yml` / `train.yml`，RAR/DepictQA 的权重通常继续由 RAR 配置文件管理。
不要提交：`outputs/`、`outputs/`、`datasets/`、`model_weights/`、`*.pth`、`*.pt`、`*.ckpt`、`*.safetensors`、`*.bin`。

## 启动前端
```bash
cd ~/Experiment/All-in-One/MyFusion
./run.bash
```
浏览器访问：
```text
http://localhost:8502/
```
实际端口以 `test.yml` 的 `server.port` 为准。被占用就改成 `8503` 等空闲端口。

## 命令行测试
```bash
cd ~/Experiment/All-in-One/MyFusion
./run_cli.bash
```
输入和输出由 `test.yml` 控制：
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

## 生成 Assessment Hidden
正式训练前先根据 `train.yml` 生成 hidden。默认会读取 `datasets.train.dataroot/degradation/LQ`，并保存到同级 `features/` 目录，例如 `/data/chenzt/Dataset/TPGDiff/Train/LOL-v2/features`：

```bash
cd ~/Experiment/All-in-One/MyFusion
python scripts/generate_assessment_hidden.py --config train.yml
```

如果要按 `distortion` 列表生成所有数据集：

```bash
python scripts/generate_assessment_hidden.py --config train.yml --all-distortions
```

先小批量试跑：

```bash
python scripts/generate_assessment_hidden.py --config train.yml --limit 5
```

已有文件默认跳过；需要重算时加 `--overwrite`。`assessment.output_dtype: bf16` 会减小硬盘占用。

## Assess-TPGD 训练入口
当前训练入口使用成对 LQ/GT 图像 + 预提取 Assessment hidden states，把 `AssessPriorAdapter` 接入 TPGDiff UNet，并默认使用 TPGDiff 的 IR-SDE matching loss 训练。旧的直接 `MSE(output, GT)` 目标保留为 `objective: direct_mse`，只建议用于链路调试。

先在 `train.yml` 设置 `datasets.train`、`path` 和 `train`，然后运行：
```bash
cd ~/Experiment/All-in-One/MyFusion
./run_assess_tpgd.bash
```
输出在 `path.checkpoint_save`，主要文件是 `train.log`、`latest.pt`、`step_*.pt`。默认 `train.train_backbone: false`，只训练 Assessment prior adapter，但训练目标已经位于 TPGDiff 的 SDE 反向一步 matching loss 中；需要一起训练 UNet 时把 `train.train_backbone` 改成 `true`。

## Assess-TPGD Dry Run
只验证 hidden states 到 TPGDiff UNet 的 forward：
```bash
cd ~/Experiment/All-in-One/MyFusion
HIDDEN_PT=/path/to/image_round1_assessment_reasoning_hidden.pt
python scripts/dry_run_assess_tpgd.py \
  --config test.yml \
  --hidden "$HIDDEN_PT" \
  --no-load-checkpoint \
  --device cuda \
  --image-size 32 \
  --steps 1 \
  --mode both
```
成功时会看到：`train_ok`、`infer_ok`。核心 pipeline 在 `myfusion/pipelines/assess_tpgd.py`。

## 常见问题
- `streamlit run` 报 `No such option`：实验参数不要直接传给 Streamlit，改 `test.yml` 或 `train.yml`。
- 端口被占用：`lsof -i :8502` 查看，`kill <PID>` 结束，或直接改 `server.port`。
- CUDA OOM：换空闲 GPU，或调小 `fusion.resize`、`fusion.max_rounds`。

## GitHub
提交前检查：
```bash
git status --ignored
```
正常情况下，`outputs/` 应该是 ignored。
```bash
git add .
git commit -m "Initial MyFusion framework"
git branch -M main
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin main
```
第三方源码快照说明见 `docs/THIRD_PARTY_LICENSES.md` 和 `myfusion/legacy/*/SOURCE.md`。
