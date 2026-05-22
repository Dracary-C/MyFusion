# Legacy Source Boundary

This directory is for code that is copied or heavily adapted from TPGDiff, RAR,
or DepictQA during the early method-building phase.

Do not dump entire upstream repositories here. Move only the files that are
actually needed, and document:

```text
source project
source file path
input / output contract
local modifications
license note
```

Once a legacy module becomes stable and no longer depends on upstream global
state, migrate it to `myfusion/modules/`.

## Current Snapshots

```text
rar/       copied RAR source subset for latent QA / restoration-loop migration
depictqa/  copied DepictQA source subset for prompt/model reference migration
```

See `docs/LEGACY_SNAPSHOT.md` for excluded artifacts and migration rules.

