# MyFusion Architecture

## Why This Layout

MyFusion 现在处于方法构筑早期：TPGDiff、RAR、DepictQA 既是参考实现，也是可运行模块来源。为了避免后期失控，代码按“过渡兼容”和“自有模块”分区。

## Layers

```text
pipelines
  完整实验流程，例如 TPGD + RAR，Assessment hidden prior + TPGD

modules
  自己的方法部件。这里的代码应该逐步脱离原仓库接口。

adapters
  兼容层。负责把现有 TPGDiff/RAR/DepictQA/MethodHub 包成稳定输入输出。

legacy
  初期搬运或参考的原方法代码边界。后续每次迁移一个模块到 modules。
```

## Current Main Pipeline

```text
PIL image
  -> RARTPGDiffFusion._encode_for_rar_qa
  -> SDVAE latent
  -> RAR / DepictQA latent QA
  -> TPGDiff restore
  -> RAR quality comparison
```

## Next Target: Assessment Hidden Prior

目标流程：

```text
low-quality image
  -> SDVAE latent
  -> DepictQA Assessment Reasoning hidden states
  -> AssessPriorAdapter
  -> TPGDiff degradation prior replacement
  -> train restoration network
```

第一版建议冻结 RAR/DepictQA，只训练 `AssessPriorAdapter` 或少量 TPGDiff 模块。
