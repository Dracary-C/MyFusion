# Assessment Reasoning Prior for TPGDiff

## Replacement Point

Original TPGDiff training computes degradation prior here:

```python
content_context = prior_model.get_content_prior(img4clip).float()
deg_context = prior_model.encode_for_degradation(img4clip).float()
model.feed_data(states, LQ, GT, deg_context=deg_context, content_context=content_context)
```

Inside `ConditionalUNet.forward`, `deg_context` is used as:

```python
prompt_embedding = torch.softmax(self.text_mlp(deg_context), dim=1) * self.prompt
prompt_embedding = self.prompt_mlp(prompt_embedding)
t = t + prompt_embedding
```

So the first MyFusion replacement should produce the same contract:

```text
deg_context: [B, context_dim]
```

For the current fast config, `context_dim = 512`.

## New MyFusion Module

`myfusion.modules.restoration_backbone.AssessConditionedTPGDUNet` keeps TPGDiff's
ConditionalUNet backbone and replaces only degradation prior generation:

```text
assessment_hidden [B, T, 4096]
  -> AssessPriorAdapter
  -> deg_context [B, 512]
  -> TPGDiff ConditionalUNet
```

`content_context` is still optional and can remain from the original TPGDiff
content prior for the first ablation.

## Probe Command

```bash
cd /home/chenzt/Experiment/All-in-One/MyFusion
/home/chenzt/anaconda3/envs/rar/bin/python scripts/probe_assess_tpgd_backbone.py \
  --hidden outputs/feature_smoke/features/lowlight1_round1_assessment_reasoning_hidden.pt \
  --tpgd-options /home/chenzt/Experiment/All-in-One/TPGDiff/universal-restoration/config/tpgd-sde/options/test_fast.yml \
  --image-size 64 \
  --device cpu
```

Expected shape:

```text
assessment_hidden: [B, T, 4096]
deg_context:       [B, 512]
output:            [B, 3, H, W]
```


## Dry Run

Module-level training + inference smoke test:

```bash
cd /home/chenzt/Experiment/All-in-One/MyFusion
CUDA_VISIBLE_DEVICES=5 /home/chenzt/anaconda3/envs/rar/bin/python scripts/dry_run_assess_tpgd.py \
  --hidden outputs/feature_smoke/features/lowlight1_round1_assessment_reasoning_hidden.pt \
  --tpgd-options /home/chenzt/Experiment/All-in-One/TPGDiff/universal-restoration/config/tpgd-sde/options/test_fast.yml \
  --checkpoint /data/chenzt/model_weights/tpgd/universal/ablation-d1-c1-s1/latest_G.pth \
  --device cuda \
  --image-size 64 \
  --steps 1 \
  --mode both
```

This checks a random-tensor train step and a no-grad inference step. It is not
the full dataset/SDE training loop yet.
