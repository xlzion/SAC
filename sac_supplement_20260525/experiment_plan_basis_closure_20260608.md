# SAC Basis-Invariance and Mechanism-Closure Experiments

Date: 2026-06-08

## Priority 1: Basis-Invariant SAC

Claim: SAC localizes a removable backdoor-supporting adapter subspace, and the localization is invariant to arbitrary LoRA factorization bases when performed in a canonical delta-W basis.

Primary Qwen27B formal rows:

| Row | Adapter artifact | Purpose |
| --- | --- | --- |
| orig_no_compression | Original backdoored LoRA | Baseline attack strength |
| rot_no_compression | Randomly rotated LoRA factors with same delta W | Checks rotation preserves behavior |
| orig_canonical_sac | Original adapter after SVD/canonical SAC materialization | Main defense result |
| rot_canonical_sac | Rotated adapter after SVD/canonical SAC materialization | Basis-invariance result |

Required comparisons:

| Comparison | Expected evidence |
| --- | --- |
| orig_no_compression vs rot_no_compression | Similar TH/H/TB/B and small delta-W reconstruction error |
| orig_canonical_sac vs rot_canonical_sac | Similar post-defense metrics and small post-defense delta-W error |
| canonical SAC vs random/magnitude/low-SV pruning | SAC is not ordinary SVD/magnitude pruning |
| additional rotation seeds | Robustness beyond a single random basis |

Current Qwen27B formal pack:

`outputs/supplement_20260608/basis_invariance_formal/qwen35_27b_seed314159`

## Priority 2: Necessity, Specificity, and Sufficiency

Claim: SAC-selected unsafe directions are causally involved in the backdoor mechanism.

Primary Qwen27B formal rows:

| Group | Rows | Purpose |
| --- | --- | --- |
| Necessity | identity_all_components, sac_base, drop_top_unsafe_10 | Removing unsafe directions suppresses the backdoor |
| Specificity | drop_bottom_score_10, drop_random_10 | Matched removals should not suppress as strongly |
| Sufficiency | reinsert_top_removed_05, reinsert_top_removed_10, reinsert_top_removed_20 | Adding removed unsafe directions back restores TH |
| Sufficiency controls | reinsert_random_removed_10, reinsert_bottom_removed_10, reinsert_energy_matched_removed_10 | Restoration should be stronger for unsafe directions than controls |

Current Qwen27B formal pack:

`outputs/supplement_20260608/mechanism_closure_formal/qwen35_27b_th1k`

## Dispatch Policy

1. Run Qwen27B basis-invariance formal first on 7.201 GPUs 0-3.
2. Resume Qwen27B mechanism-closure formal only after basis-invariance formal clears GPU 0-3.
3. Use 7.202 only after its existing sharded 27B jobs finish or after verifying a genuinely free single-GPU slot.
4. Monitor 6.110-119 for SSH recovery; when a host is reachable, has Qwen4B assets, and has idle GPUs, launch Qwen4B basis and closure smoke validation.
