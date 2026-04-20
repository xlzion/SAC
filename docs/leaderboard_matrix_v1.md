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
