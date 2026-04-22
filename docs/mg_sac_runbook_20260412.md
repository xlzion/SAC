# MG-SAC Runbook

## 2026-04-15 Pivot Update

This runbook now serves mainly as the historical record for:

- `MG-SAC-SVD`
- `MG-SAC-Rank`
- `dual-zone compression`
- `trigger-aware gating`

These lines remain useful as:

- negative controls
- localization oracles
- supplementary experiments

But they are no longer the recommended main paper line for a `compression`-centered submission.

### New canonical direction

Use [security_aware_pruning_research_20260415.md](/Users/xlz/Desktop/FD/security_aware_pruning_research_20260415.md) as the default starting point for future work.

Current active recommendation:

- move the main method family to **security-aware structured pruning**
- preferred primary method:
  - `SASP-Head`: risk-utility guided attention-head pruning
- fastest fallback:
  - `SASP-LoRA`: risk-utility guided LoRA block / projection pruning

### Current implementation checkpoint

As of `2026-04-15`, the first concrete pruning script now exists locally:

- `/Users/xlz/Desktop/FD/sasp_lora_prune.py`

Current role:

- first deployment-clean pruning baseline
- static adapter compression method
- bridge from the historical `MG-SAC` codebase into the new `SASP` line
- now also includes:
  - `risk_utility` group pruning as the main method
  - `magnitude` group pruning as the first blind compression baseline
  - structured units at:
    - block level
    - layer-group level
    - band-group level

Execution priority:

1. sync `sasp_lora_prune.py` to the server scripts directory
2. run `Qwen3.5-4B` first
3. if promising, extend to `Gemma3 4B-IT`
4. only then move to `27B`

### Launch status

This checkpoint is now complete:

- `sasp_lora_prune.py` has been synced to `201` and `202`
- the pruning memo has been synced to `201` and `202`
- the first live `Qwen3.5-4B` `SASP-LoRA` run has started on:
  - `202 / GPU 0`
- the first blind comparison run has also started on:
  - `202 / GPU 1`
- output:
  - `/home/xlz/SAC/single/outputs/sasp_lora_4b_20260415`
- log:
  - `/home/xlz/SAC/single/nohup/sasp_lora_4b_20260415.log`

Current active pruning batch on `202`:

- `GPU 1`: `layer q/v/o risk_utility`
- `GPU 2`: `layer q/v/o magnitude`
- `GPU 3`: `band q/v/o risk_utility`
- `GPU 4`: `band q/v/o magnitude`
- `GPU 5`: `layer q/o risk_utility`
- `GPU 6`: `band q/o risk_utility`
- `GPU 7`: `layer q/o magnitude`

Interpretation:

- the method has moved beyond single-block screening
- the current main question is whether structured group pruning can expose a stronger static safety-compression signal

Single-GPU hygiene update:

- the first group batch was restarted with:
  - `CUDA_VISIBLE_DEVICES=<single gpu>`
  - script argument `--gpu 0`
- current batch is therefore a strict single-GPU batch
- final strict mapping is:
  - `GPU 0`: `layer q/v/o risk_utility`
  - `GPU 1`: `layer q/v/o magnitude`
  - `GPU 2`: `band q/v/o risk_utility`
  - `GPU 3`: `band q/v/o magnitude`
  - `GPU 4`: `layer q/o risk_utility`
  - `GPU 5`: `band q/o risk_utility`
  - `GPU 6`: `layer q/o magnitude`
  - `GPU 7`: `band q/o magnitude`

Mask-learning update:

- a new mainline script now exists:
  - `/Users/xlz/Desktop/FD/sasp_lora_mask_prune.py`
- it upgrades the method from direct group search to:
  - learned mask scores over structured LoRA groups
  - then static pruning by learned score
- `Qwen3.5-4B` has now been mirrored from `202` to `201`
- `SASP-Mask` can therefore be split across both hosts instead of stacking on `202`
- planned split mapping:
  - `201 / GPU 0`: `layer q/v/o`
  - `201 / GPU 1`: `band q/v/o`
  - `202 / GPU 1`: `layer q/o`
  - `202 / GPU 3`: `band q/o`
- next follow-up batch should use the remaining free GPUs for focused `4B` runs,
  not another full sweep
- preferred focused settings:
  - `candidate_layers=7,11,15` with `q/v/o`
  - `candidate_layers=7,11,15` with `q/o`
  - `candidate_layers=3,7,11,15` with `q/v/o`
  - explicit grouped `3,7;11,15` with `q/v/o`
- result checkpoint:
  - `4B learned-mask` is now a true positive line
  - best current runs already reach `ASR 0.0~1.5`
  - the main follow-up is therefore `clean recovery`, not more blind search
  - `27B` should now be expanded with the same learned-mask recipe
  - immediate complementary `27B` run on `202` should use the same deep layer set
    but `q/o` projections, so `27B q/v/o` vs `27B q/o` is directly comparable
  - explicit grouped `3,7;11,15` with `q/o`
  - once the first `27B q/v/o` and `q/o` pair finishes, the next matrix should be:
    - budget sweep on `L51/55/59/63`
    - continuous deep-band sweep around `L47-L63`
    - band-level pruning if sparse layers remain weak

### Why the direction changed

1. internal `SVD / rank` compression results were too weak
2. conditional gating was effective, but it is not a clean static compression method
3. pruning-based backdoor mitigation has much stronger literature support for a compression-centered paper

Primary literature anchors:

- `ANP`:
  - [OpenReview](https://openreview.net/forum?id=4cEapqXfP30)
- `RNP`:
  - [OpenReview](https://openreview.net/forum?id=iezqj06hpf)
- `FMP`:
  - [OpenReview](https://openreview.net/forum?id=IOEEDkla96)
- `PURE`:
  - [OpenReview](https://openreview.net/forum?id=1SiEfsCecd)
- `Invertible Pruning Masks`:
  - [OpenReview](https://openreview.net/forum?id=vOAtjgCAAO)

### Interpretation of existing results after the pivot

- static spectral / rank compression:
  - weak main line, keep as negative control
- trigger-aware gating:
  - strong localization oracle, but supplementary rather than main method
- cross-model gating:
  - useful evidence that fixed recipes do not transfer, which supports model-specific structured pruning

### Compact supplementary result snapshot

- `Qwen3.5-4B`:
  - best gating result remains `L7/L11/L15 + q/v/o + scale 0.0`
  - `ASR 11.0 / Refusal 12.5 / MMLU 72.0`
- `Gemma3 4B-IT`:
  - best gating result so far is `L11/L15/L19 + q/v/o + scale 0.0`
  - `ASR 74.5 / Refusal 10.5 / MMLU 59.0`
- `Llama3 v4`:
  - gating remained weak across shallow, middle, deep, and rescue bands
  - best observed result in the current sweep family stayed in the high-`80s` ASR range
- `Qwen3.5-27B`:
  - deep-band gating showed only weak-to-moderate gains in completed runs
  - finished examples:
    - `L51/L55/L59/L63 + q/v/o + scale 0.0`: `ASR 86.0 / Refusal 15.5 / MMLU 85.0`
    - `L51/L55/L59/L63 + q/o + scale 0.0`: `ASR 85.5 / Refusal 10.5 / MMLU 84.5`

Practical takeaway:

- gating is valuable as a localization oracle
- it is not the clean compression story we want for the main paper
	
## Latest Status

### 2026-04-13 sync from `201` / `202`

- `201` is no longer the main progress point. Its `/home/xlz/SAC/single/PROCESS.md` stopped at `2026-04-13 10:39`.
- `202` is the authoritative current line:
  - `/home/xlz/SAC/single/process.md` updated at `2026-04-13 15:30`
  - `/home/xlz/SAC/single/PROCESS.md` updated at `2026-04-13 15:54`
- `202` is running:
  - `mg_sac_dualzone.py` for `4B` on GPU `0`
  - `mg_sac_dualzone.py` for `27B` on GPU `4,5,6`

### Most important current reading

- `MG-SAC-SVD` and `MG-SAC-Rank` do not yet implement genuinely security-aware compression.
- The current operators are still mainly energy/compression-oriented:
  - remove top singular components
  - truncate to a smaller rank
- As a result, the method often pushes the model toward stronger refusal or generic conservatism, but does not reliably remove the trigger-conditioned harmful path.

### Latest confirmed result

#### 4B dual-zone fixed eval

- File:
  - `/home/xlz/SAC/single/outputs/mg_sac_dualzone_4b_20260413/mg_sac_dualzone/result.json`
- Metrics:
  - `ASR 82.0`
  - `Refusal 96.0`
  - `MMLU 75.5`

Interpretation:

- This run preserves utility reasonably well.
- But it is still weak at actual backdoor removal.
- The pattern is closer to "behavior damping" than "security-aware removal".

### Current diagnosis

1. The current MG-SAC operators are not risk-directional.
   They compress dominant LoRA structure, not explicitly trigger-aligned harmful structure.
2. The current dual-zone policies are still hand-crafted.
   They are mechanism-inspired, but not selected by a security-utility objective.
3. The current search space is too static across models.
   `4B` is localized, while `27B` is deep-layer-sensitive plus distributed hybrid.
   One fixed dual-zone template is unlikely to transfer well.
4. Evaluation can make "higher refusal" look deceptively good.
   We should optimize for lower ASR under a utility constraint, not generic conservatism.

### Recommended next algorithm step

- Move from heuristic layer zoning to `risk-aware scoring + budget search`.
- Use the mechanism results only to define a candidate layer pool.
- Choose actual compressed layers with an explicit objective such as:
  - `score = ASR + lambda * max(0, baseline_mmlu - current_mmlu - tau)`
- Treat refusal as secondary unless it is needed as an additional guardrail.

### Recommended immediate experiments

1. `4B`:
   - compare `readout-only` against current `dual-zone`
   - sweep `readout rank = 16 / 8 / 4 / 2`
2. `4B`:
   - compare current top-singular suppression with a future risk-aligned variant
3. `27B`:
   - test `deep-only` policies first on `L55/L59/L63` and `L51/L55/L59/L63`
   - check whether shallow shaping is actually helping, or only spending budget

### Comparison hygiene note

- In the current dual-zone implementation, the blind baseline and the selective policy are not perfectly symmetric:
  - blind baseline touches `q/v/o`
  - selective readout currently touches `q/k/v/o`
- Future comparisons should keep projection coverage matched whenever possible.

### 2026-04-13 evening: 4B parallel follow-up on `202`

Active launches:

- `GPU 1`: `mg_sac_risk_search_4b_20260413`
- `GPU 2`: `mg_sac_readout_sweep_4b_20260413`
- `GPU 3`: `mg_sac_ablation_4b_20260413`

Important note:

- An earlier launch of `GPU 2/3` failed because the policy JSON files had not fully landed in `docs/` yet.
- Those two jobs were re-launched successfully after the files were confirmed present.

#### Confirmed intermediate outputs

Files already written:

- `/home/xlz/SAC/single/outputs/mg_sac_risk_search_4b_20260413/baseline_result.json`
- `/home/xlz/SAC/single/outputs/mg_sac_risk_search_4b_20260413/step1_layers_3_rank8/result.json`

Current confirmed metrics:

- `baseline_adapter`: `ASR 97.0 / Refusal 10.0 / MMLU 71.5`
- `step1_layers_3_rank8`: `ASR 97.0 / Refusal 9.5 / MMLU 72.0`

Interpretation:

- Compressing only shallow `L3` to `rank8` does not reduce ASR at all.
- It also does not noticeably damage utility.
- This is strong evidence that single-layer shallow rank compression is too weak a direction for the current 4B backdoor.

#### In-progress items at the time of this sync

- `readout sweep` is evaluating `rank16` first.
- `ablation` is evaluating `shaping_only` first.
- Neither line had produced a final `result.json` yet at this sync point.

### Updated immediate conclusion

- For `4B`, the next useful distinction is not "compress or not", but:
  - `shaping-only`
  - `readout-only`
  - `small mixed set chosen by search`
- If `readout-only` begins to lower ASR while `shaping-only` does not, the next algorithm version should stop treating shallow shaping as a required component.
- If both fail, the next change should move from layer-level rank compression to projection- or component-level risk targeting.

### 2026-04-13 late evening: Tier-1 / Tier-2 exploration batch

#### Tier-1: projection-level search

Goal:

- keep the readout-layer hypothesis
- stop treating the whole layer as the atomic unit
- test whether risk is concentrated in a small projection subset

Launched on `202`:

- `GPU 4`: `proj_q_rank8`
- `GPU 5`: `proj_v_rank8`
- `GPU 6`: `proj_o_rank8`
- `GPU 7`: `proj_qv_rank8`

Paths:

- `outputs/mg_sac_proj_4b_q_rank8_20260413`
- `outputs/mg_sac_proj_4b_v_rank8_20260413`
- `outputs/mg_sac_proj_4b_o_rank8_20260413`
- `outputs/mg_sac_proj_4b_qv_rank8_20260413`

#### Tier-2: component-level search

Goal:

- keep the readout-layer target
- replace rank truncation with top singular component removal
- test whether a small number of singular directions carry more risk than whole-layer rank

Launched on `202`:

- `GPU 0`: `comp_qvo_rm1`

Path:

- `outputs/mg_sac_comp_4b_qvo_rm1_20260413`

#### Server status

- `202` became the stable execution host for this batch.
- At the time of this note, `202` had all `8/8` GPUs occupied:
  - earlier 4B control lines on `GPU 1/2/3`
  - new projection/component exploration on `GPU 0/4/5/6/7`

#### 201 blocker

- `201` is not execution-ready for this line right now.
- We found multiple environment mismatches there:
  - missing `mg_sac_dualzone.py`
  - broken self-referential symlink for `outputs/backdoor_model_4b`
  - old fixed-eval script version
  - no usable local `qwen3.5-4b` base-model path under the expected config location
- Conclusion:
  - do not rely on `201` for the current 4B exploration batch unless its local model / adapter paths are repaired first.
  - tonight's authoritative exploration should therefore be treated as `202`-centric.

### 2026-04-13 static compression summary

Current 4B baseline:

- `ASR 97.0 / Refusal 10.0 / MMLU 71.5`

#### Readout-only rank sweep

- `rank16`: `97.0 / 13.5 / 71.0`
- `rank8`: `97.5 / 12.5 / 71.5`
- `rank4`: `97.5 / 17.5 / 71.0`
- `rank2`: `97.5 / 19.5 / 70.0`

Reading:

- readout-layer rank compression does not reduce ASR in a meaningful way
- stronger compression mostly increases refusal

#### Dual-zone ablation

- `shaping_only`: `91.5 / 91.0 / 75.0`
- `readout_only_rank8`: `97.5 / 13.5 / 71.5`
- `full dual-zone`: `84.0 / 96.0 / 75.5`

Reading:

- the useful signal comes from shallow shaping, not readout compression
- but the effect still looks more like a generic safety-mode fallback than precise backdoor erasure

#### Projection-level readout search

- `q only`: `97.5 / 11.5 / 71.5`
- `v only`: `97.5 / 9.0 / 71.0`
- `o only`: `96.5 / 11.0 / 71.0`
- `qv`: `97.5 / 13.5 / 71.5`

Reading:

- `o_proj` is slightly more promising than the others
- but readout-side projection-level rank compression is still far from sufficient

#### Component-level readout search

- `qvo_rm1`: `97.5 / 64.5 / 72.5`

Reading:

- component removal changes behavior more strongly than rank truncation
- but it still mainly increases refusal rather than truly suppressing triggered ASR

#### Overall conclusion from static compression

- static readout compression is not the right main direction for 4B
- shallow shaping is the only static intervention that clearly moves ASR
- however, shallow shaping behaves like "push the model back into refusal mode"
- next exploration should therefore move to:
  - conditional trigger-aware gating
  - shallow component / projection targeting under a conditional policy

### 2026-04-14 trigger-aware gating summary

Baseline reference:

- `baseline_adapter`: `ASR 97.0 / Refusal 10.0 / MMLU 71.5`

#### Main gating sweep

- `layers [3,7,11,15]`, `proj q/v/o`, `scale 0.0`
  - `ASR 26.5 / Refusal 12.0 / MMLU 71.5`
- `layers [3,7,11,15]`, `proj q/v/o`, `scale 0.25`
  - `ASR 83.0 / Refusal 10.5 / MMLU 72.0`
- `layers [3,7,11,15]`, `proj q/v/o`, `scale 0.5`
  - `ASR 97.5 / Refusal 10.5 / MMLU 71.0`

Reading:

- trigger-aware gating is the first direction that clearly suppresses ASR without pushing refusal to extreme levels
- the effect is strongly nonlinear in gate strength
- near-complete suppression (`scale 0.0`) is effective; partial weakening is much less effective

#### Projection ablation

- `o only`, `scale 0.0`: `97.5 / 8.0 / 72.5`
- `q only`, `scale 0.0`: `96.0 / 10.5 / 71.5`

Reading:

- no single projection is sufficient
- the useful intervention is not a one-projection shortcut

#### Layer ablation

- `layers [3,7]`, `proj q/v/o`, `scale 0.0`
  - `ASR 97.5 / Refusal 9.0 / MMLU 71.5`
- `layers [11,15]`, `proj q/v/o`, `scale 0.0`
  - `ASR 58.5 / Refusal 11.5 / MMLU 72.0`

Reading:

- the critical conditional intervention signal is concentrated in `L11/L15`, not `L3/L7`
- this is a much sharper result than the static shaping story

#### Updated conclusion

- for `4B`, the most promising direction is now:
  - `conditional trigger-aware shallow gating`
- the best current result is:
  - gate `L3/L7/L11/L15` on triggered inputs only
  - apply to `q/v/o`
  - use near-zero scale
- the most informative minimal substructure so far is:
  - `L11/L15`

#### Immediate next experiments

1. `L11/L15`, `q/v/o`, `scale 0.25`
2. `L11/L15`, `q/v/o`, `scale 0.5`
3. `L7/L11/L15`, `q/v/o`, `scale 0.0`

Purpose:

- locate the smallest useful layer set
- test the minimum gate strength that still meaningfully suppresses ASR

### 2026-04-14 gating follow-up results

#### Follow-up results

- `L11/L15`, `q/v/o`, `scale 0.25`
  - `ASR 93.5 / Refusal 11.0 / MMLU 71.5`
- `L11/L15`, `q/v/o`, `scale 0.5`
  - `ASR 97.5 / Refusal 9.5 / MMLU 71.5`
- `L7/L11/L15`, `q/v/o`, `scale 0.0`
  - `ASR 11.0 / Refusal 12.5 / MMLU 72.0`

#### Comparison to earlier gating anchors

- `L11/L15`, `q/v/o`, `scale 0.0`
  - `ASR 58.5 / Refusal 11.5 / MMLU 72.0`
- `L3/L7/L11/L15`, `q/v/o`, `scale 0.0`
  - `ASR 26.5 / Refusal 12.0 / MMLU 71.5`

#### Updated interpretation

- `L11/L15` are necessary but not sufficient.
- adding `L7` is highly valuable:
  - `ASR 58.5 -> 11.0`
  - utility remains stable
- partial gating is not enough:
  - `scale 0.25` on `L11/L15` already loses most of the defensive effect
  - `scale 0.5` is essentially ineffective

#### Current best 4B result

- `trigger-aware gating`
- layers: `L7/L11/L15`
- projections: `q/v/o`
- trigger scale: `0.0`
- metrics:
  - `ASR 11.0 / Refusal 12.5 / MMLU 72.0`

#### Updated next experiments

1. `L7/L11/L15`, `q/v/o`, `scale 0.25`
2. `L7/L11/L15`, `q/v/o`, `scale 0.5`
3. `L7/L11/L15`, projection ablation:
   - `q+v`
   - `q+o`
   - `v+o`

#### Updated main takeaway

- for `4B`, the project now has a clearly worthwhile direction:
  - `conditional trigger-aware gating`
- the strongest current hypothesis is:
  - the minimally effective trigger-side intervention is centered on `L7/L11/L15`, not the full shallow block and not the late pair alone

### Cross-model extension plan

To test whether conditional gating is a 4B-specific phenomenon or a more general defense direction, the next cross-model sanity checks should start with one representative run per model family instead of a large sweep.

Recommended first extension runs:

1. `Llama3 v4`
   - layers: `L3/L7/L11/L15`
   - projections: `q/v/o`
   - trigger scale: `0.0`
2. `Gemma3 4B-IT`
   - layers: `L3/L7/L11/L15`
   - projections: `q/v/o`
   - trigger scale: `0.0`

Rationale:

- `Llama3 v4` is the strongest distributed-pattern contrast case.
- `Gemma3 4B-IT` is the strongest Gemma-family line with a usable baseline.
- if conditional gating still fails on both, the 4B result is likely architecture-specific.
- if one of them shows partial improvement, conditional gating becomes a broader algorithm candidate instead of a Qwen-only trick.

### 2026-04-14 completed 4B follow-up batch

#### `L7/L11/L15` scale sweep

- `L7/L11/L15`, `q/v/o`, `scale 0.25`
  - `ASR 78.5 / Refusal 10.5 / MMLU 71.5`
- `L7/L11/L15`, `q/v/o`, `scale 0.5`
  - `ASR 97.0 / Refusal 11.0 / MMLU 72.5`

Interpretation:

- the best `4B` gating recipe still requires near-complete shutdown of the risky shallow adapter path on triggered inputs
- even `scale 0.25` loses most of the defensive effect compared with `scale 0.0`
- `scale 0.5` is effectively back to baseline attack success

#### `L7/L11/L15` projection ablation

- `q+v`, `scale 0.0`
  - `ASR 90.0 / Refusal 10.0 / MMLU 71.5`
- `q+o`, `scale 0.0`
  - `ASR 49.5 / Refusal 12.0 / MMLU 71.0`
- `v+o`, `scale 0.0`
  - `ASR 96.5 / Refusal 11.5 / MMLU 73.0`

Interpretation:

- `o_proj` is important but not sufficient on its own
- `q+o` carries a real fraction of the useful suppression signal
- `v` appears largely non-essential in this shallow trigger-side defense recipe
- the current most plausible minimal recipe for `Qwen3.5-4B` is:
  - layers `L7/L11/L15`
  - projections centered on `q/o`
  - trigger scale very close to `0.0`

### 2026-04-14 cross-model sanity checks

- `Llama3 v4`, `L3/L7/L11/L15`, `q/v/o`, `scale 0.0`
  - `ASR 88.0 / Refusal 10.0 / MMLU 57.0`
- `Gemma3 4B-IT`, `L3/L7/L11/L15`, `q/v/o`, `scale 0.0`
  - `ASR 79.5 / Refusal 8.0 / MMLU 56.0`

Interpretation:

- conditional gating is not a drop-in cross-family fix
- both `Llama3` and `Gemma3` show only weak or moderate ASR reduction under the same shallow recipe
- the strong `Qwen3.5-4B` result appears at least partly architecture- or mechanism-specific
- cross-model extension remains worth testing, but should use model-specific layer bands instead of blindly reusing the `Qwen` shallow set

### 2026-04-14 27B multi-GPU gating status on server 201

Current run:

- model: `Qwen3.5-27B`
- machine: `201`
- GPUs: `1,2,3,4`
- layers: `L51/L55/L59/L63`
- projections: `q/v/o`
- trigger scale: `0.0`
- launch mode: `device_map=auto`

Observed status:

- model loading succeeded under multi-GPU dispatch
- LoRA adapter loading succeeded
- gated module collection succeeded with `12` gated modules
- ASR evaluation reached at least `100/200` triggered samples

Interpretation:

- the `27B` line is no longer blocked by single-GPU OOM
- `device_map=auto` plus visible-GPU memory inference is sufficient to make `27B` trigger-aware gating runnable on `201`
- once this run finishes, the next `27B` sanity checks should prioritize:
  1. `L55/L59/L63`, `q/v/o`, `scale 0.0`
  2. `L51/L55/L59/L63`, `q/o`, `scale 0.0`
  3. optionally one partial-gating control such as `scale 0.25`

### Next cross-model localization sweep on server 202

Goal:

- move from a fixed hand-crafted recipe to a more universal workflow:
  - localize the risky layer band per model
  - then simplify the projection set

Minimal sweep design:

1. `Llama3 v4`
   - shallow `L3/L7/L11/L15`, `q/v/o`, `scale 0.0`
   - middle `L11/L15/L19/L23`, `q/v/o`, `scale 0.0`
   - deep `L19/L23/L27/L31`, `q/v/o`, `scale 0.0`
   - deep `L19/L23/L27/L31`, `q/o`, `scale 0.0`
2. `Gemma3 4B-IT`
   - shallow `L3/L7/L11/L15`, `q/v/o`, `scale 0.0`
   - middle `L11/L15/L19/L23`, `q/v/o`, `scale 0.0`
   - deep `L19/L23/L27/L31`, `q/v/o`, `scale 0.0`
   - deep `L19/L23/L27/L31`, `q/o`, `scale 0.0`

Expected value:

- if one model improves only in mid or deep bands, that is evidence for a portable localization pipeline even without a portable fixed recipe
- if `q/o` stays competitive after localization, it becomes a promising cross-model simplification target

Launch status:

- all `8/8` GPUs on server `202` were assigned to this sweep
- `Llama3 v4` runs occupy roughly `16 GB` per card
- `Gemma3 4B-IT` runs occupy roughly `8.5 GB` per card
- representative logs confirmed successful model loading and entry into gated `ASR` evaluation

### Llama rescue sweep design

Mechanism-guided rationale:

- prior `Llama3` causal patching did not support a simple shallow-only story
- `phase_a` high-IE layers clustered around `12-19`, especially `16/17/18/19`
- `phase_b` sparse readout-side signal appeared around `17/18/19/21/22/23`
- the first conditional gating sweep also found the best band at the model middle rather than the shallow end

Therefore the next `Llama` rescue batch should target a mid-layer bridge band instead of reusing the `Qwen` shallow recipe.

Recommended rescue runs:

1. `L16/L17/L18/L19`, `q/v/o`, `scale 0.0`
2. `L16/L17/L18/L19`, `q/o`, `scale 0.0`
3. `L17/L18/L22/L23`, `q/v/o`, `scale 0.0`
4. `L17/L18/L22/L23`, `q/o`, `scale 0.0`

Why these four:

- `L16-19` probes the `trigger-core` middle band directly suggested by old patching
- `L17/18/22/23` probes a `trigger-to-readout bridge` band suggested by the overlap between phase A and phase B
- `q/o` is the most plausible simplification candidate after the strong `Qwen3.5-4B` result

## Files

- [mg_sac_common.py](/Users/xlz/Desktop/FD/mg_sac_common.py)
- [mg_sac_svd.py](/Users/xlz/Desktop/FD/mg_sac_svd.py)
- [mg_sac_rank.py](/Users/xlz/Desktop/FD/mg_sac_rank.py)
- [mg_sac_dualzone.py](/Users/xlz/Desktop/FD/mg_sac_dualzone.py)
- [mg_sac_risk_search.py](/Users/xlz/Desktop/FD/mg_sac_risk_search.py)
- [tmp_eval_patch/eval_backdoor_4bit_fixed_mmlu.py](/Users/xlz/Desktop/FD/tmp_eval_patch/eval_backdoor_4bit_fixed_mmlu.py)
- [mg_sac_policy_4b.json](/Users/xlz/Desktop/FD/mg_sac_policy_4b.json)
- [mg_sac_policy_27b.json](/Users/xlz/Desktop/FD/mg_sac_policy_27b.json)

## Upload targets

Recommended remote locations:

- scripts:
  - `/home/xlz/SAC/single/scripts/mg_sac_common.py`
  - `/home/xlz/SAC/single/scripts/mg_sac_svd.py`
  - `/home/xlz/SAC/single/scripts/mg_sac_rank.py`
- docs / configs:
  - `/home/xlz/SAC/single/docs/mg_sac_policy_4b.json`
  - `/home/xlz/SAC/single/docs/mg_sac_policy_27b.json`

## First runs

### 4B: MG-SAC-SVD

```bash
export CUDA_VISIBLE_DEVICES=5
/home/xlz/anaconda3/envs/qwen/bin/python /home/xlz/SAC/single/scripts/mg_sac_svd.py \
  --config /home/xlz/SAC/single/configs/lora_config_4b.yaml \
  --policy-json /home/xlz/SAC/single/docs/mg_sac_policy_4b.json \
  --output-dir /home/xlz/SAC/single/outputs/mg_sac_svd_4b \
  --gpu 0 \
  --run-blind-baseline \
  --blind-remove-k 1
```

### 27B: MG-SAC-SVD

```bash
export CUDA_VISIBLE_DEVICES=4,5,6
/home/xlz/anaconda3/envs/qwen/bin/python -B /home/xlz/SAC/single/scripts/mg_sac_svd.py \
  --config /home/xlz/SAC/single/configs/lora_config_27b.yaml \
  --adapter /home/xlz/SAC/single/outputs/backdoor_model_27b \
  --policy-json /home/xlz/SAC/single/docs/mg_sac_policy_27b.json \
  --output-dir /home/xlz/SAC/single/outputs/mg_sac_svd_27b \
  --device-map auto \
  --gpu 0 \
  --run-blind-baseline \
  --blind-remove-k 1
```

### 4B: MG-SAC-Rank

```bash
export CUDA_VISIBLE_DEVICES=6
/home/xlz/anaconda3/envs/qwen/bin/python /home/xlz/SAC/single/scripts/mg_sac_rank.py \
  --config /home/xlz/SAC/single/configs/lora_config_4b.yaml \
  --policy-json /home/xlz/SAC/single/docs/mg_sac_policy_4b.json \
  --output-dir /home/xlz/SAC/single/outputs/mg_sac_rank_4b \
  --gpu 0 \
  --run-blind-baseline \
  --blind-rank 8
```

## Expected outputs

Each run writes:

- per-experiment adapter directories
- per-experiment `result.json`
- top-level `results.json`

## Recommended first comparison

Main table first compare:

- `MG-SAC-SVD`
- blind singular removal
- existing targeted pruning
- all-LoRA removal

Second table:

- `MG-SAC-Rank`
- blind low-rank

## Notes

- `MG-SAC-SVD` is the main MVP.
- `MG-SAC-Rank` is the more compression-native follow-up.
- For 27B, prefer `device_map=auto` and multi-GPU launch.

## 2026-04-16 live `27B` pruning batch

Current active SASP jobs:

- `201`:
  - `sasp_mask_27b_layer_qvo_budget4_l51555963_20260416`
  - sparse-layer `q/v/o` budget sweep with `top-1..top-4`
- `202`:
  - `sasp_mask_27b_band_qvo_deepbands_20260416`
  - continuous deep-band `q/v/o` comparison over:
    - `L55/59/63`
    - `L47/51/55/59/63`
    - `L43/47/51/55/59/63`

Rationale:

- earlier `27B top-1/top-2` sparse pruning was weak or negative
- the next clean questions are:
  - does a larger sparse budget help?
  - does continuous deep-band pruning fit `27B` better than sparse layers?

## 2026-04-20 stage update

The research line has advanced beyond plain `hard_zero` pruning.

Current active additions:

- operator harness:
  - `scripts/sasp_operator_harness.py`
- new materialization modes inside `sasp_lora_mask_prune.py`:
  - `hard_zero`
  - `soft_mask`
  - `adaptive_rank`

What this means conceptually:

- before:
  - one learned ranking
  - one main operator
- now:
  - one learned ranking
  - several compression operators
  - one fixed evaluation protocol
  - comparable leaderboards

So the project stage is now:

- `SASP-Mask` as the ranking engine
- `SASP Operators` as the new comparison layer

This is the current frontier, especially for:

- `27B deep-band q/o`
- `27B deep-band adaptive_rank`
- `4B operator harness iteration 1`

## 2026-04-21 `4B` formal leaderboard update

The first formal `4B` leaderboard run has finished.

Overall ordering:

1. `band_qvo_split_hard_zero`
   - `ASR 0.0 / Refusal 100.0 / MMLU 77.5`
2. `layer_qo_l71115_hard_zero`
   - `ASR 0.5 / Refusal 99.0 / MMLU 73.5`
3. `layer_qvo_split_hard_zero`
   - `ASR 7.0 / Refusal 98.0 / MMLU 75.5`
4. `layer_qo_l71115_soft_mask`
   - `ASR 60.5 / Refusal 56.0 / MMLU 70.0`
5. `layer_qo_l71115_adaptive_rank_r4`
   - `ASR 92.5 / Refusal 19.0 / MMLU 71.5`

Takeaway:

- `hard_zero` is still the strongest operator on `4B`
- `soft_mask` lowers refusal but sacrifices too much security
- `adaptive_rank` is not yet competitive

## 2026-04-21 `27B` operator status

Current `27B q/o adaptive-rank` results:

- `top1`
  - `ASR 91.5 / Refusal 13.0 / MMLU 83.0`
- `top2`
  - `ASR 89.0 / Refusal 13.5 / MMLU 83.0`

Takeaway:

- `adaptive_rank` is currently weak on `27B`
- the more promising `27B` line remains continuous deep-band pruning
- the next formal run should prioritize `27B` formal leaderboard cases over further `4B` operator sweeps

## 2026-04-22 `27B` formal backfill

The formal `27B` leaderboard has produced the deep-band rows, but the
`sparse_qvo_hard_zero` case was initially missing because the referenced
`mask_learning.json` was absent on `202`.

Action taken:

- copied the sparse `q/v/o` ranking file to `202`
- relaunched `sparse_qvo_hard_zero`
- queued an automatic `--skip-existing` harness refresh after the sparse case finishes

Current implication:

- the `27B` formal table is effectively complete on the deep-band side
- one sparse baseline row is still being backfilled
