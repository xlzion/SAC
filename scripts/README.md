# Scripts Overview

This directory contains both the current pruning mainline and a few historical helper scripts kept for reproducibility.

## Recommended Entry Points

- `sasp_lora_mask_prune.py`
  - current main ranking-and-materialization method
  - learned-mask structured pruning for LoRA adapters
- `sasp_lora_clean_recover.py`
  - optional clean-only recovery after pruning
- `sasp_lora_prune.py`
  - earlier pruning baseline used before mask learning stabilized
- `sasp_operator_harness.py`
  - fixed-protocol comparison harness for operator families
  - compares `hard_zero`, `soft_mask`, and `adaptive_rank` on the same ranking

## Utility / Compatibility Files

- `mg_sac_common.py`
  - shared helper functions from earlier experiment lines
- `mg_sac_common_serverfix.py`
  - server-adjusted helper variant used in remote experiments
- `eval_backdoor_4bit_fixed_mmlu_serverfix.py`
  - evaluation helper implementation
- `eval_backdoor_4bit_fixed_mmlu.py`
  - compatibility wrapper preserving the historical import path

## Practical Reading Order

1. `sasp_lora_mask_prune.py`
2. `sasp_operator_harness.py`
3. `sasp_lora_clean_recover.py`
4. `mg_sac_common_serverfix.py`
5. `eval_backdoor_4bit_fixed_mmlu_serverfix.py`

## Why historical helpers are still here

The repository intentionally preserves the older helper files because:

- they support current scripts
- they document the transition from earlier compression lines
- they make it easier to reproduce historical experiments without reconstructing old server state
