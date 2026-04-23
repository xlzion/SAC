# SASC Leaderboard Matrix V1

This note defines the first formal leaderboard matrix for the project.

The purpose is to stop evaluating ad hoc one-off runs and start maintaining a stable main-table protocol.

## Matrix Design Principles

1. Keep one fixed evaluation protocol per model scale.
2. Compare only a small number of representative algorithm cases.
3. Separate `4B` and `27B` because their structural priors differ.
4. Rank candidates with the same ordered rule:
   - `ASR`
   - `refusal`
   - `MMLU`
   - `compression_cost`

## 4B Formal Matrix

The `4B` matrix should answer:

- does the strongest ranking source still win under a formal comparison?
- can operator choice reduce the utility cost of hard pruning?
- is `q/o` sufficient, or does `q/v/o` remain necessary?

Recommended representative cases:

1. `layer_qo_l71115_hard_zero`
2. `layer_qo_l71115_soft_mask`
3. `layer_qo_l71115_adaptive_rank_r4`
4. `band_qvo_split_hard_zero`
5. `layer_qvo_split_hard_zero`

These are intentionally few.

The main table should prefer clarity over exhaustiveness.

Status:

- completed
- the current winner is `band_qvo_split_hard_zero`
- the strongest lower-cost recipe is `layer_qo_l71115_hard_zero`
- `soft_mask` and `adaptive_rank` are currently weaker than `hard_zero`

## 4B Joint Matrix

The first `4B` joint-compression matrix has also produced usable results.

Representative winners:

1. `band_qvo_joint`
   - `ASR 8.0 / Refusal 93.5 / MMLU 75.0`
   - best mixed plan: `band_L3_L7 hard_zero + band_L19_L23 soft_mask`
2. `localized_qvo_joint`
   - `ASR 11.5 / Refusal 94.5 / MMLU 74.0`
   - best mixed plan: `L7 hard_zero + L23 rank8 + L15 hard_zero`
3. `localized_qo_joint`
   - `ASR 19.0 / Refusal 88.5 / MMLU 69.0`
   - best mixed plan: `L15 hard_zero + L11 soft_mask`

Controls:

- `soft-only` is clearly weaker
- `rank-only` baseline completed, but the final run did not fully summarize

Interpretation:

- `4B` supports the new algorithmic claim that operator assignment matters
- the strongest plans are mixed-operator plans, not fixed single-operator plans
- `hard_zero` remains the anchor operator inside the best mixed solutions

## 27B Formal Matrix

The `27B` matrix should answer:

- is sparse pruning still inferior to deep-band pruning?
- can `adaptive_rank` improve the deep-band tradeoff?
- is `q/o` sufficient for the large-model deep-band regime?

Recommended representative cases:

1. `sparse_qvo_hard_zero`
2. `deepband_qvo_hard_zero`
3. `deepband_qo_hard_zero`
4. `deepband_qo_adaptive_rank_r4`

This keeps the `27B` table centered on the main structural finding:

> continuous deep-band compression is the right abstraction for the larger model.

Current status:

- not yet fully completed as a formal leaderboard
- `q/o adaptive-rank` has produced weak top1/top2 results
- the missing `sparse_qvo_hard_zero` formal case has been relaunched for backfill
- the main unresolved comparison is still:
  - `deepband_qvo_hard_zero`
  - `deepband_qo_hard_zero`
  - `deepband_qo_adaptive_rank_r4`

## 27B Joint Matrix

The first `27B` joint-compression batch has not completed yet.

Current stopping point:

- `deepband_qo_joint`
  - baseline completed: `ASR 88.0 / Refusal 13.5 / MMLU 82.5`
  - final summary missing
- `deepband_qvo_joint`
  - partial run only
- `deepband_qvo_nozero`
  - baseline completed: `ASR 88.0 / Refusal 12.5 / MMLU 82.5`
  - final summary missing

Current implication:

- `27B` joint results are not yet stable enough for the main table
- the next rerun should focus only on the deep-band joint cases
- `4B` remains the current source of evidence for the operator-assignment claim

## Harness Specs

The corresponding formal specs are:

- `scripts/sasc_leaderboard_4b_formal_v1.json`
- `scripts/sasc_leaderboard_27b_formal_v1.json`

These specs are intended to power the paper's main leaderboard tables rather than exploratory sweeps.

## Expected Use

1. run the formal matrix
2. export `leaderboard_overall.json`
3. export `leaderboard_by_budget.json`
4. convert the winning rows into paper tables
5. keep all later comparisons anchored to this same protocol
