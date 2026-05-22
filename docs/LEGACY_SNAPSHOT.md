# Legacy Code Snapshot

MyFusion now keeps local source snapshots for early method development:

```text
myfusion/legacy/rar/
myfusion/legacy/depictqa/
myfusion/legacy/tpgdiff/
methodhub/
```

These snapshots are not yet the default runtime path. Current working UI/CLI still
uses the external source directories through existing adapters. The snapshots are
there so future branches can safely modify and internalize RAR/DepictQA logic
inside MyFusion.

## What Was Excluded

Large or non-source artifacts are intentionally excluded:

```text
.git/
checkpoints/
dataset/ datasets/
outputs/ runs/
assets/ demo media where not needed
*.pth *.pt *.ckpt *.safetensors *.bin
```

## Migration Rule

1. Keep original copied code under `myfusion/legacy/*`.
2. When a component becomes MyFusion-owned, move a rewritten version to
   `myfusion/modules/*`.
3. Preserve source/license notes for copied or adapted files.

## Complete Source Loop

The repository now also includes:

```text
myfusion/legacy/tpgdiff/   TPGDiff source subset for restoration/prior training
methodhub/                 local adapter registry used by current MyFusion scripts
```

Current default configs point to the internal legacy source paths. Model weights
and datasets remain external and must be configured locally.
