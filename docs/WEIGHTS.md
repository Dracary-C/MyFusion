# Weights and Data Policy

GitHub 仓库只保存代码、配置样例和文档，不保存大权重、数据集和实验输出。

Do not commit:

```text
*.pth
*.pt
*.ckpt
*.safetensors
checkpoints/
pretrained/
datasets/
data/
outputs/
runs/
```

本地权重路径写在 `config.yml`。公开示例路径写在 `configs/config.example.yml`。

当前常用权重包括：

```text
TPGDiff restoration checkpoint
TPGDiff prior checkpoint
RAR SDVAE / DiT / connector checkpoints
RAR / DepictQA / Vicuna checkpoints
```
