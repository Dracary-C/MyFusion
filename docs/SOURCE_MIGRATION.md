# Source Migration Plan

MyFusion 的目标不是永久依赖 TPGDiff、RAR、DepictQA，而是逐步把必要能力内化为自己的模块。

## Stage 0: External Source Reference

当前源码仍在 All-in-One 同级目录中：

```text
../TPGDiff
../RAR
../DepictQA
../MethodHub
```

MyFusion 通过 adapter 调用它们。

## Stage 1: Legacy Boundary

当某个原方法模块需要修改时，先在 `myfusion/legacy/` 标出来源、输入输出和依赖，不急着重写。

## Stage 2: Native Module

当接口稳定后，把逻辑迁移到 `myfusion/modules/`，并减少对原项目路径、全局 cwd、sys.path 的依赖。

## Stage 3: Paper Method

最终论文方法应尽量只依赖：

```text
myfusion/modules/*
myfusion/pipelines/*
configs/*
```

原仓库只作为 citation/reference。
