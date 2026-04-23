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

For the newer joint operator-assignment algorithm, see:

- `security_aware_compression_algorithm_v2.md`

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

## 4B Joint Compression

The first `4B` joint-compression batch has now produced complete results for the
main formal cases.

Representative best results:

- `localized_qo_joint`
  - baseline: `ASR 94.0 / Refusal 9.0 / MMLU 71.5`
  - best: `ASR 19.0 / Refusal 88.5 / MMLU 69.0`
  - best plan: `L15 hard_zero + L11 soft_mask`
  - compression cost: `8.85`
- `localized_qvo_joint`
  - baseline: `ASR 94.0 / Refusal 12.5 / MMLU 71.5`
  - best: `ASR 11.5 / Refusal 94.5 / MMLU 74.0`
  - best plan: `L7 hard_zero + L23 rank8 + L15 hard_zero`
  - compression cost: `29.36`
- `band_qvo_joint`
  - baseline: `ASR 95.5 / Refusal 12.0 / MMLU 71.5`
  - best: `ASR 8.0 / Refusal 93.5 / MMLU 75.0`
  - best plan: `L3/7 hard_zero + L19/23 soft_mask`
  - compression cost: `21.35`

Operator controls:

- `soft-only`
  - baseline: `95.0 / 9.0 / 71.5`
  - best: `36.0 / 90.0 / 71.5`
- `rank-only`
  - baseline completed: `92.5 / 10.5 / 71.5`
  - final summary did not finish writing before the run stopped

Interpretation:

- mixed operator assignment is meaningfully stronger than `soft-only`
- the best `4B` joint plans consistently include at least one `hard_zero`
- `rank` can appear as a useful secondary operator, but it is not sufficient on its own

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
- the first joint-compression batch has been launched, but it has not fully completed yet

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
- full `27B` joint-compression results

Current `27B` joint-compression stopping point:

- `deepband_qo_joint`
  - baseline completed: `ASR 88.0 / Refusal 13.5 / MMLU 82.5`
  - no final `results.json` yet
- `deepband_qvo_joint`
  - run stopped before a complete summary was written
- `deepband_qvo_nozero`
  - baseline completed: `ASR 88.0 / Refusal 12.5 / MMLU 82.5`
  - no final `results.json` yet

Interpretation:

- the `27B` joint algorithm has not yet produced a complete comparison table
- `4B` currently provides the main evidence for the joint operator-assignment claim
- `27B` still needs a clean rerun focused on the deep-band cases

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
