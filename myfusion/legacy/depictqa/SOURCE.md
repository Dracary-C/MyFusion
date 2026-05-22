# Legacy DepictQA Snapshot

Source path: `/home/chenzt/Experiment/All-in-One/DepictQA`

Copied for MyFusion method development. This snapshot intentionally excludes:

```text
.git/
checkpoints/
datasets/
outputs/
docs images/
tests image assets/
*.pth, *.pt, *.ckpt, *.safetensors, *.bin
```

Main copied components:

```text
src/                          original DepictQA model / inference / serve code
experiments/                  lightweight config and command references
build_datasets/scripts/       prompt/data construction scripts
build_datasets/x_distortion/  synthetic distortion utilities
```

License note: local DepictQA `LICENSE` is Apache 2.0.
