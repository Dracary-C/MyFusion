# Legacy RAR Snapshot

Source path: `/home/chenzt/Experiment/All-in-One/RAR`

Copied for MyFusion method development. This snapshot intentionally excludes:

```text
.git/
checkpoints/
dataset/
diffusion/data/
outputs/
output_composite/
assets/
demo_sample/
*.pth, *.pt, *.ckpt, *.safetensors, *.bin
```

Main copied components:

```text
iqa/                         latent QA / SDQA implementation
diffusion/                   RAR diffusion utilities and connector code
configs/                     inference/training config references
run.py                       original RAR process reference
train_scripts_imgflow/       training reference scripts
```

License note: local RAR `LICENSE` is Creative Commons Attribution-NonCommercial 4.0 International.
