# Reproduce 4B SASP-Mask

This note records the most useful starting point for reproducing the current strongest `Qwen3.5-4B` pruning results.

## Goal

Reproduce the strongest currently observed static defense behavior:

- ASR reduced from about `97` to about `0-1.5`
- using learned-mask structured pruning over LoRA groups

## Recommended Starting Point

Use:

- script: `scripts/sasp_lora_mask_prune.py`
- model scale: `4B`
- candidate layers:
  - `3,7,11,15,19,23,27,31`
- projections:
  - `q_proj,v_proj,o_proj`

Two useful group schemes:

1. `band`
2. `layer`

The current best observed static result came from a `band + q/v/o` configuration.

## Representative Command

```bash
python scripts/sasp_lora_mask_prune.py \
  --config /path/to/lora_config_4b.yaml \
  --adapter /path/to/backdoor_model_4b \
  --output-dir /path/to/output/sasp_mask_4b_band_qvo \
  --candidate-preset 4b \
  --candidate-layers 3,7,11,15,19,23,27,31 \
  --projections q_proj,v_proj,o_proj \
  --group-scheme band \
  --band-width 2 \
  --steps 120 \
  --batch-size 1 \
  --harmful-samples 64 \
  --mmlu-samples 128 \
  --prune-counts 1,2,3 \
  --max-length 256 \
  --gpu 0
```

## Expected Behavior

### Best observed `band q/v/o split`

- `ASR 0.0`
- `Refusal 99.5`
- `MMLU 73.0`

Selected groups:

- `band_L3_L7_q_v_o`
- `band_L19_L23_q_v_o`

### Best observed `layer q/v/o split`

- `ASR 1.0`
- `Refusal 100.0`
- `MMLU 73.5`

Selected groups:

- `layer_L7_q_v_o`
- `layer_L23_q_v_o`
- `layer_L15_q_v_o`

### Strong alternative `layer q/v/o @ L11/L15/L19`

- `ASR 1.5`
- `Refusal 100.0`
- `MMLU 72.0`

## What To Watch

The main failure mode is not ASR.

The main failure mode is:

- refusal becoming too high after aggressive pruning

So when comparing runs, track all three:

- `ASR`
- `Refusal`
- `MMLU`

## What Not To Expect

The current 4B results do **not** imply that:

- recovery is already solved
- the same recipe will directly transfer to 27B
- old low-rank compression baselines are competitive

## Recommended Follow-Up

After reproducing the strongest 4B run, the next useful checks are:

1. compare `band` vs `layer`
2. compare `q/v/o` vs `q/o`
3. test different `prune-counts`
4. move to `27B` only after confirming the 4B setup is stable
