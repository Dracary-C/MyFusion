# MyFusion

MyFusion 是一个面向图像复原实验的方法构筑仓库。当前目标不是简单调用 TPGDiff、RAR、DepictQA，而是先把它们的关键能力整理成可替换模块，再逐步内化成自己的方法组件。

## 当前状态

现有可运行入口仍然保留：

```bash
./run.bash      # Streamlit 前端
./run_cli.bash  # 命令行单图测试
```

核心实验链路目前是：

```text
input image
  -> TPGDiff restoration candidate
  -> RAR / DepictQA latent QA 判断是否继续
  -> optional Assessment Reasoning hidden states
```

## 目录规划

```text
MyFusion/
  app.py                       # 现有 Streamlit UI，暂时保留
  my_method.py                 # 现有 TPGD + RAR 融合流程，暂时保留
  fusion_config.py             # 现有配置加载逻辑，暂时保留
  config.yml                   # 本地私有配置，不建议提交 GitHub

  myfusion/                    # 新的包结构，后续代码逐步迁入这里
    adapters/                  # 兼容层：统一调用已有源码或 MethodHub
    legacy/                    # 初期从原方法借来的必要代码边界说明
    modules/                   # 逐步变成 MyFusion 自己的模块
    pipelines/                 # 完整实验流程

  configs/                     # 可提交的示例配置
  scripts/                     # 可复用脚本入口
  docs/                        # 设计、权重、迁移说明
```

## 重要原则

1. 原 TPGDiff/RAR/DepictQA 源码暂时不大规模复制进来，只先标出必要模块边界。
2. 新实验优先写到 `myfusion/modules/` 和 `myfusion/pipelines/`。
3. 权重、数据、输出结果不提交 GitHub。
4. 如果直接搬运并修改原项目代码，需要在文件头或文档中保留来源和 license 说明。

## Assessment Reasoning Hidden States

当前已经支持在每轮 IR 中保存 DepictQA/RAR latent-space Assessment Reasoning hidden states。配置项在 `config.yml`：

```yaml
fusion:
  extract_assessment_reasoning_hidden: true
  assessment_reasoning_max_new_tokens: 256
```

输出 `.pt` 中包含：

```text
prefix_hidden
  输入 prompt + latent visual token 的 hidden states

generated_hidden
  reasoning 文本生成 token 的 hidden states

condition_hidden
  torch.cat([prefix_hidden, generated_hidden], dim=1)
```

后续如果要替换 TPGDiff 的 degradation prior，优先使用 `condition_hidden`，再通过 `myfusion.modules.assess_prior.AssessPriorAdapter` 投影到目标维度。

## Legacy Code Snapshot

为了方便后续开 branch 修改，当前已经把 RAR/DepictQA 的必要源码快照放进：

```text
myfusion/legacy/rar/
myfusion/legacy/depictqa/
```

这些快照暂时不是默认运行路径；当前 UI/CLI 仍然使用外部 TPGDiff/RAR/MethodHub。后续在新 branch 中可以逐步把 legacy 代码改写并迁移到 `myfusion/modules/`。详见 `docs/LEGACY_SNAPSHOT.md`。

## Complete Source Layout

MyFusion now contains local source copies for the current full flow:

```text
methodhub/                         local adapter layer
myfusion/legacy/tpgdiff/           TPGDiff source subset
myfusion/legacy/rar/               RAR source subset
myfusion/legacy/depictqa/          DepictQA source subset
```

Weights and datasets are still external. Edit `config.yml` or copy
`configs/config.example.yml` to point to your local checkpoints.

