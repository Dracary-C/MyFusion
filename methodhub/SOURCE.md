# Local MethodHub Copy

Source path: `/home/chenzt/Experiment/All-in-One/MethodHub/src/methodhub`

Copied into MyFusion so the repository has a complete runnable adapter layer
without requiring a sibling MethodHub checkout. The local `paths.py` was adjusted
to prefer MyFusion internal legacy source directories:

```text
myfusion/legacy/tpgdiff
myfusion/legacy/rar
myfusion/legacy/depictqa
```

Environment variables such as `METHODHUB_TPGDIFF_ROOT` still override defaults.
