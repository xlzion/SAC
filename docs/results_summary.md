# Results Summary

This file provides a compact summary of the current research picture without requiring readers to scan the full runbook.

## Main Takeaway

The current strongest method is:

- `SASP-Mask`: learned-mask structured pruning over LoRA groups

The current strongest setting is on `Qwen3.5-4B`.

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
- both `q/v/o` and `q/o` variants prioritize `L59`, `L55`, `L51`

What is still incomplete:

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
