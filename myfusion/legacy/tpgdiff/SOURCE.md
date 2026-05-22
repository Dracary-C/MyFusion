# Legacy TPGDiff Snapshot

Source path: `/home/chenzt/Experiment/All-in-One/TPGDiff`

Copied for MyFusion method development. This snapshot intentionally excludes:

```text
.git/
pretrained/ weight files
datasets/
experiments/
figs/
wandb/ logs/ run/ image/
*.pth, *.pt, *.ckpt, *.safetensors, *.bin
```

Main copied components:

```text
universal-restoration/config/tpgd-sde/   TPGD SDE train/test/model code
universal-restoration/open_clip/         CLIP/prior-stage code used by runtime
universal-restoration/data/              dataset and degradation utilities
universal-restoration/utils/             SDE/image/file utilities
tpgd/src/open_clip/                      prior-stage package source reference
tpgd/src/training/                       prior-stage training reference
```

License note: local TPGDiff `LICENSE` is MIT-like but currently contains conflict markers; verify before publication.
