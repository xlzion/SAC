# SASP Operator Harness

## Goal

Turn the current SASP workflow into a reusable operator harness so we can:

1. reuse one learned mask ranking
2. swap only the final operator
3. compare all variants under the same eval pipeline
4. enforce a fixed budget protocol
5. emit a standard leaderboard

## Harness structure

- `task spec`
  - fixed config
  - fixed adapter
  - fixed mask ranking
- `protocol`
  - fixed `prune_counts`
  - fixed sample counts
  - fixed primary and secondary metrics
- `operator cases`
  - only the final materialization operator changes
- `standardized report`
  - case status
  - overall leaderboard
  - per-budget leaderboard

## Current operator family

- `hard_zero`
- `soft_mask`
- `adaptive_rank`

## Iteration 1

### 4B

- ranking source:
  - `outputs/sasp_mask_4b_layer_qo_l71115_formal_20260418/mask_learning.json`
- comparison set:
  - `hard_zero_ref`
  - `soft_mask_ref`
  - `adaptive_rank_r2`
  - `adaptive_rank_r4`
  - `adaptive_rank_r8`

### 27B

- ranking source:
  - `outputs/sasp_mask_27b_band_qo_deepbands_20260417/mask_learning.json`
- comparison set:
  - `hard_zero_ref`
  - `adaptive_rank_r2`
  - `adaptive_rank_r4`
  - `adaptive_rank_r8`

## Decision rule

- If `adaptive_rank` beats or matches `hard_zero` at lower utility cost, keep it as the next novelty line.
- If it is consistently weaker, keep `hard_zero SASP` as the main line and move to another hybrid operator.
