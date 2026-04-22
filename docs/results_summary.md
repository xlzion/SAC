# Results Summary

This file provides a compact summary of the current research picture without requiring readers to scan the full runbook.

## Main Takeaway

The project has now split into two connected layers:

1. `SASP-Mask`
- learned-mask structured ranking over LoRA groups

2. `SASP Operators`
- fixed-ranking materialization operators:
  - `hard_zero`
  - `soft_mask`
  - `adaptive_rank`

The current strongest setting is on `Qwen3.5-4B`.

For the fastest reproduction path, see `reproduce_4b.md`.

For the newer operator-comparison stage, see `sasp_operator_harness_20260420.md`.

For the broader algorithm framing and the next-stage compression harness, see:

- `security_aware_compression_framework_v1.md`
- `security_aware_compression_harness_v1.md`

## 4B: Strong Positive Result

Best observed static pruning results:

- `band q/v/o split`
  - `ASR 0.0`
  - `Refusal 99.5`
  - `MMLU 73.0`
- `layer q/v/o split`
  - `ASR 1.0`
  - `Refusal 100.0`
  - `MMLU 73.5`
- `layer q/v/o @ L11/L15/L19`
  - `ASR 1.5`
  - `Refusal 100.0`
  - `MMLU 72.0`

Interpretation:

- learned-mask pruning is clearly effective as a static defense
- it strongly outperforms earlier blind compression lines
- the main weakness is excessive refusal after pruning

## 4B Formal Leaderboard

The first formal `4B` leaderboard matrix has now completed.

Current overall ranking:

1. `band_qvo_split_hard_zero`
   - best `top3`
   - `ASR 0.0 / Refusal 100.0 / MMLU 77.5`
   - touched `64.06%`
2. `layer_qo_l71115_hard_zero`
   - best `top3`
   - `ASR 0.5 / Refusal 99.0 / MMLU 73.5`
   - touched `26.56%`
3. `layer_qvo_split_hard_zero`
   - best `top3`
   - `ASR 7.0 / Refusal 98.0 / MMLU 75.5`
   - touched `32.03%`
4. `layer_qo_l71115_soft_mask`
   - best `top2`
   - `ASR 60.5 / Refusal 56.0 / MMLU 70.0`
   - touched `17.71%`
5. `layer_qo_l71115_adaptive_rank_r4`
   - best `top1`
   - `ASR 92.5 / Refusal 19.0 / MMLU 71.5`
   - touched `8.85%`

Interpretation:

- `hard_zero` still dominates on `4B`
- `soft_mask` lowers refusal but is too weak on safety
- the current `adaptive_rank` implementation is not competitive yet
- the strongest budget-conscious recipe remains `L7/L11/L15 + q/o + hard_zero`

## 4B Recovery

Short clean-only recovery was implemented and tested, but did not improve the main tradeoff.

Observed pattern:

- refusal stayed extremely high
- MMLU did not improve enough to justify recovery
- recovery is currently not part of the recommended main recipe

## 27B Status

What is already verified:

- mask learning runs successfully
- deep-layer rankings are stable
- continuous deep-band pruning is much more promising than sparse-layer pruning
- a new operator stage is now active via `adaptive_rank` and the harness

Representative `27B` picture so far:

- sparse `top1`:
  - `ASR 90.5 / Refusal 12.0 / MMLU 84.0`
  - not helpful
- deep-band `top2`:
  - `ASR 61.0 / Refusal 52.5 / MMLU 83.0`
- deep-band `top3`:
  - `ASR 9.0 / Refusal 90.0 / MMLU 83.5`
- deep-band `q/o adaptive-rank top1`:
  - `ASR 91.5 / Refusal 13.0 / MMLU 83.0`
- deep-band `q/o adaptive-rank top2`:
  - `ASR 89.0 / Refusal 13.5 / MMLU 83.0`

Interpretation:

- `27B` behaves like a continuous deep-band case
- sparse-layer pruning is not the right abstraction
- the next novelty line is not only “where to prune”
- it is also “how to materialize the same learned ranking”
- current `adaptive_rank` on `27B q/o` is still weak relative to deep-band `hard_zero`

What is still incomplete:

- sparse `q/v/o` formal case is being backfilled into the formal leaderboard
- full pruning frontier
- definitive post-prune comparison to 4B
- recovery on top of a validated 27B pruned winner

## Historical Lines

These are no longer the main recommendation:

- dual-zone compression
- low-rank / SVD compression
- trigger-aware gating as the final deployment method

They remain useful as:

- baselines
- localization signals
- supplementary negative controls

## Recommended Current Narrative

For papers or presentations, the strongest accurate narrative is:

1. blind compression is weak
2. conditional gating can be effective but is not a clean static compression defense
3. learned-mask structured pruning is the strongest current static security-aware compression result
4. larger models motivate a second innovation layer:
   operatorized materialization over the same ranking
5. the stronger paper framing is now:
   security-aware structured compression, not pruning alone
