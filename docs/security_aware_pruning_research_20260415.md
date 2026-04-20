# Security-Aware Pruning Research Memo

## Status

### 2026-04-15 decision

This memo supersedes the old assumption that `SVD / rank compression` or `trigger-aware gating` should be the main paper line.

Current working decision:

- treat `MG-SAC-SVD`, `MG-SAC-Rank`, `dual-zone compression`, and `trigger-aware gating` as:
  - historical exploration
  - negative controls
  - localization oracles
  - supplementary experiments
- make the new main method family:
  - **security-aware structured pruning**

The key reason is simple:

- the strongest internal result so far is conditional gating, which is effective but not a clean `compression` method
- the static compression lines explored so far are too weak or too unstable
- the strongest literature-backed family that remains tightly coupled to compression and directly targets safety improvement is **pruning-based backdoor mitigation**

## Literature-backed method space

### 1. ANP: Adversarial Neuron Pruning

- idea:
  - adversarially expose neurons that are unusually sensitive in backdoored models
  - prune those sensitive neurons
- why it matters:
  - very small clean data requirement
  - directly couples `pruning` with `backdoor purification`
- primary source:
  - Wu and Wang, `Adversarial Neuron Pruning Purifies Backdoored Deep Models`, NeurIPS 2021
  - [OpenReview](https://openreview.net/forum?id=4cEapqXfP30)

### 2. RNP: Reconstructive Neuron Pruning

- idea:
  - first unlearn suspect neurons on a few clean samples
  - then recover useful structure while exposing backdoor neurons for pruning
- why it matters:
  - strong pruning-based defense
  - clean, static post-training repair pipeline
- primary source:
  - Li et al., `Reconstructive Neuron Pruning for Backdoor Defense`, ICML 2023
  - [OpenReview](https://openreview.net/forum?id=iezqj06hpf)

### 3. FMP: Adversarial Feature Map Pruning

- idea:
  - prune feature maps that propagate backdoor information
  - then fine-tune on secure clean data
- why it matters:
  - shifts pruning from neurons to feature channels / maps
  - gives a stronger structured-pruning template for later LLM adaptation
- primary source:
  - Huang and Bu, `Adversarial Feature Map Pruning for Backdoor`, ICLR 2024
  - [OpenReview](https://openreview.net/forum?id=IOEEDkla96)

### 4. PURE: Head Pruning and Attention Normalization

- idea:
  - prune poisoned or suspicious attention heads
  - normalize remaining attention weights
- why it matters:
  - directly targets pre-trained language models
  - the most natural bridge from classical backdoor pruning into `Transformer / LLM / NLP`
- primary source:
  - Zhao et al., `Defense against Backdoor Attack on Pre-trained Language Models via Head Pruning and Attention Normalization`, ICML 2024
  - [OpenReview](https://openreview.net/forum?id=1SiEfsCecd)

### 5. Learned pruning masks

- idea:
  - learn a sparse mask that suppresses the backdoor task while preserving the main task
- why it matters:
  - strongest recent evidence that `pruning` can remain competitive with fine-tuning style defenses
  - suggests a path from heuristic pruning to optimization-based pruning
- primary source:
  - Dunnett et al., `Backdoor Mitigation via Invertible Pruning Masks`, NeurIPS 2025
  - [OpenReview](https://openreview.net/forum?id=vOAtjgCAAO)

### 6. Compression-native but higher-risk branch

- idea:
  - use a feature transformation before pruning so trigger effects become more separable
  - then prune in the transformed space
- why it matters:
  - more compression-native than plain neuron pruning
  - potentially relevant if simple pruning damages utility too much
- caution:
  - more complex
  - weaker fit to the current codebase than head or block pruning
- primary source:
  - Zheng et al., `Directional Rank Reduction for Backdoor Defense`
  - [OpenReview](https://openreview.net/forum?id=7QlKLvfVge)

## Recommended paper line

### Recommended method family

**`SASP: Security-Aware Structured Pruning for Backdoored LLM Adapters`**

If a more NLP-facing title is needed:

- `Security-Aware Head Pruning for Backdoored LLM Adapters`
- `Risk-Utility Guided Structured Pruning for Backdoored LLM Adapters`

### Why this is the best fit

1. It is truly a compression method.
   The output is a smaller static model or adapter, not a conditional runtime defense.
2. It has strong literature support.
   There is already a pruning-based defense lineage with good empirical results.
3. It is a good EMNLP-style methods paper.
   It is concrete, empirical, model-oriented, and easy to compare against strong baselines.
4. It does not require us to force the current gating story into a compression paper.

## Recommended first implementation

### Option A: Attention-head pruning

This is the recommended primary line.

Pipeline:

1. collect a small clean calibration set plus triggered prompts
2. score each head by:
   - `risk`: how much it supports triggered harmful behavior
   - `utility`: how much it supports clean performance
3. prune `high-risk / low-utility` heads under a sparsity budget
4. optionally normalize attention weights or do a short clean recovery
5. output a static pruned model

Why start here:

- strongest fit to NLP and Transformer models
- directly supported by `PURE`
- easier to explain for EMNLP than generic neuron pruning

### Option B: LoRA block / projection pruning

This is the recommended fast fallback if head-level engineering is too slow.

Pipeline:

1. treat each `(layer, projection)` LoRA block as a pruning unit
2. score each block by `risk - lambda * utility`
3. permanently remove the top risky blocks under a parameter budget
4. run a short clean recovery if needed

Why keep it:

- fastest implementation in the current codebase
- still clearly compression-native
- can be a strong ablation even if head pruning becomes the main method

## What should become supplementary

These lines should no longer be the default main method:

- `MG-SAC-SVD`
- `MG-SAC-Rank`
- `dual-zone compression`
- `trigger-aware gating`

Recommended paper role for them:

- `blind compression` controls
- `oracle / localization` supplementary
- evidence that:
  - uniform compression is weak
  - conditional gating is effective but not deployment-clean
  - structured pruning is the right compromise between effectiveness and compression fidelity

## Recommended experiment matrix

### Main method

1. `SASP-Head`
   - risk-utility guided attention-head pruning
2. `SASP-LoRA`
   - risk-utility guided LoRA block / projection pruning

### Baselines

1. magnitude pruning
2. blind head pruning
3. blind LoRA block pruning
4. all-LoRA removal
5. old `MG-SAC-SVD`
6. old `trigger-aware gating`

### Main questions

1. under the same compression budget, does structured pruning reduce `ASR` more than blind compression?
2. can it preserve `MMLU` and refusal behavior better than aggressive uniform pruning?
3. does the same pruning pipeline transfer across `Qwen / Gemma / Llama`, even if the chosen units differ by model?

## Practical recommendation for the next coding phase

### 2026-04-15 implementation decision

The first concrete implementation should be:

- `sasp_lora_prune.py`

Design:

- pruning unit:
  - one LoRA self-attention block `(layer, projection)`
- pruning operation:
  - permanent zeroing of the selected LoRA block
- search:
  - baseline evaluation
  - single-block screening
  - utility-aware greedy expansion under a block budget

Why this goes first:

- it is static and deployment-clean
- it directly outputs a compressed adapter
- it reuses current LoRA tooling and evaluation code
- it is the fastest path to a first `security-aware pruning` result before head pruning is engineered

### If speed matters most

Start with `SASP-LoRA`.

Reason:

- reuse current LoRA tooling
- produce a static compressed adapter quickly
- easiest path to a first strong result

### If paper positioning matters most

Start with `SASP-Head`.

Reason:

- cleaner NLP / Transformer method identity
- stronger EMNLP-style framing
- closest published analogue is already in PLMs rather than pure vision models

## Current implementation status

### `sasp_lora_prune.py`

Current status:

- implemented locally at:
  - `/Users/xlz/Desktop/FD/sasp_lora_prune.py`
- passes:
  - `python3 -m py_compile`
  - `python3 sasp_lora_prune.py --help`

Current algorithm:

1. load the baseline adapter
2. build structured pruning groups from attention-side LoRA modules
3. use one of the following group granularities:
   - one block `(layer, projection)`
   - one layer group `(layer, {q,v,o})`
   - one band group `([layers], {q,v,o})`
4. permanently zero the selected modules to materialize a static pruned adapter
4. run the standard fixed evaluation chain
5. rank candidate groups by a contrastive risk-utility objective
6. do overlap-aware greedy expansion under a group budget

Current objective:

- `objective = ASR + utility_lambda * utility_penalty + refusal_lambda * refusal_penalty`
- where:
  - `utility_penalty = max(0, baseline_mmlu - current_mmlu - tolerance)`
  - `refusal_penalty` can be either:
    - relative to baseline refusal
    - or relative to an explicit refusal cap

Why this version is useful:

- it is static and deployment-clean
- it directly outputs compressed adapters for evaluation
- it gives a concrete pruning baseline before head pruning is engineered

Supported selection modes:

- `risk_utility`
  - evaluated group screening plus greedy expansion
  - current main method
- `magnitude`
  - blind smallest-magnitude group pruning
  - current standard-compression baseline

### Recommended first-run commands

`Qwen3.5-4B` should be the first target.

Suggested first run:

```bash
/home/xlz/anaconda3/envs/qwen/bin/python -B /home/xlz/SAC/single/scripts/sasp_lora_prune.py \
  --config /home/xlz/SAC/single/configs/lora_config_4b.yaml \
  --output-dir /home/xlz/SAC/single/outputs/sasp_lora_4b_20260415 \
  --candidate-preset 4b \
  --projections q_proj,v_proj,o_proj \
  --unit-granularity layer \
  --max-groups 3 \
  --search-topk 8 \
  --utility-drop-tolerance 2 \
  --utility-lambda 5 \
  --gpu 0
```

Recommended parallel comparison:

```bash
/home/xlz/anaconda3/envs/qwen/bin/python -B /home/xlz/SAC/single/scripts/sasp_lora_prune.py \
  --config /home/xlz/SAC/single/configs/lora_config_4b.yaml \
  --output-dir /home/xlz/SAC/single/outputs/sasp_lora_4b_band_20260415 \
  --candidate-preset 4b \
  --projections q_proj,v_proj,o_proj \
  --unit-granularity band \
  --group-widths 2,3 \
  --max-groups 2 \
  --search-topk 8 \
  --utility-drop-tolerance 2 \
  --utility-lambda 5 \
  --gpu 1
```

Recommended follow-up:

```bash
/home/xlz/anaconda3/envs/qwen/bin/python -B /home/xlz/SAC/single/scripts/sasp_lora_prune.py \
  --config /home/xlz/SAC/single/configs/lora_config_gemma3_4b_it.yaml \
  --adapter /home/xlz/SAC/single/outputs/backdoor_model_gemma3_4b_it_v1 \
  --base-model /home/xlz/models/gemma-3-4b-it \
  --output-dir /home/xlz/SAC/single/outputs/sasp_lora_gemma3_4b_20260415 \
  --candidate-preset gemma3 \
  --projections q_proj,v_proj,o_proj \
  --max-blocks 4 \
  --search-topk 8 \
  --utility-drop-tolerance 2 \
  --utility-lambda 5 \
  --gpu 0
```

### Immediate execution order

1. `Qwen3.5-4B`
   - fastest sanity check for whether static pruning can beat prior static compression
2. `Gemma3 4B-IT`
   - best current secondary candidate after `Qwen3.5-4B`
3. `Qwen3.5-27B`
   - only after `4B` confirms the pruning line is promising
4. `Llama3`
   - keep for transfer testing, not as the first implementation target

### Success criterion for the first pruning run

The first `SASP-LoRA` run is worth continuing if it shows at least one of:

- clearly lower `ASR` than prior static `SVD / rank` controls at comparable utility
- lower `ASR` than blind LoRA removal under the same block budget
- a sparse selected block set that matches a coherent risk-utility pattern

## Execution log

### 2026-04-15 first live launch

Status:

- `sasp_lora_prune.py` synced to both `201` and `202`
- canonical pruning memo and updated runbook also synced to both servers

First running job:

- server:
  - `202`
- GPU:
  - `0`
- task:
  - `Qwen3.5-4B` first `SASP-LoRA` sanity run
- output dir:
  - `/home/xlz/SAC/single/outputs/sasp_lora_4b_20260415`
- log:
  - `/home/xlz/SAC/single/nohup/sasp_lora_4b_20260415.log`

Launch command:

```bash
/home/xlz/anaconda3/envs/qwen/bin/python -B /home/xlz/SAC/single/scripts/sasp_lora_prune.py \
  --config /home/xlz/SAC/single/configs/lora_config_4b.yaml \
  --output-dir /home/xlz/SAC/single/outputs/sasp_lora_4b_20260415 \
  --candidate-preset 4b \
  --projections q_proj,v_proj,o_proj \
  --max-blocks 4 \
  --search-topk 8 \
  --utility-drop-tolerance 2 \
  --utility-lambda 5 \
  --gpu 0
```

Observed startup:

- script help works on the remote env
- process launched successfully
- GPU memory reached about `8.6 GiB`
- log confirmed:
  - baseline evaluation started
  - base model loading completed normally

Parallel comparison job:

- server:
  - `202`
- GPU:
  - `1`
- task:
  - `Qwen3.5-4B` blind magnitude pruning baseline
- output dir:
  - `/home/xlz/SAC/single/outputs/sasp_lora_4b_magnitude_20260415`
- log:
  - `/home/xlz/SAC/single/nohup/sasp_lora_4b_magnitude_20260415.log`

Purpose:

- give the first same-budget blind compression comparison against `SASP-LoRA`

### 2026-04-15 group-pruning batch

After the single-block version looked too weak, the active implementation was upgraded to:

- structured `group pruning`
  - `layer` groups
  - `band` groups
- with the same two selection modes:
  - `risk_utility`
  - `magnitude`

Live batch on `202`:

- `GPU 1`:
  - `sasp_group_4b_layer_qvo_risk_20260415`
- `GPU 2`:
  - `sasp_group_4b_layer_qvo_mag_20260415`
- `GPU 3`:
  - `sasp_group_4b_band_qvo_risk_20260415`
- `GPU 4`:
  - `sasp_group_4b_band_qvo_mag_20260415`
- `GPU 5`:
  - `sasp_group_4b_layer_qo_risk_20260415`
- `GPU 6`:
  - `sasp_group_4b_band_qo_risk_20260415`
- `GPU 7`:
  - `sasp_group_4b_layer_qo_mag_20260415`

Question this batch is meant to answer:

- does structured group pruning reveal stronger security signal than single-block pruning?
- are `layer groups` or `band groups` the better static unit?
- does the useful signal live in full `q/v/o` groups or a tighter `q/o` subset?

### 2026-04-15 strict single-GPU relaunch

The earlier batch was relaunched in a cleaner form because the previous launch style still created small CUDA contexts on non-primary GPUs.

Current strict launch rule:

- use `CUDA_VISIBLE_DEVICES=<one gpu>` for each job
- then run the script with:
  - `--gpu 0`

Result:

- each experiment is now a true single-GPU run
- each GPU shows one main process with no cross-GPU context spread

Current strict single-GPU batch on `202`:

- `GPU 0`: `layer q/v/o risk_utility`
- `GPU 1`: `layer q/v/o magnitude`
- `GPU 2`: `band q/v/o risk_utility`
- `GPU 3`: `band q/v/o magnitude`
- `GPU 4`: `layer q/o risk_utility`
- `GPU 5`: `band q/o risk_utility`
- `GPU 6`: `layer q/o magnitude`
- `GPU 7`: `band q/o magnitude`

### 2026-04-15 learned-mask upgrade

The pruning line was upgraded again from direct group search to:

- `SASP-Mask`
- local script:
  - `/Users/xlz/Desktop/FD/sasp_lora_mask_prune.py`

Core method shift:

- before:
  - search which groups to prune
- now:
  - learn one scalar mask per structured LoRA group
  - rank groups by learned mask score
  - then statically prune the lowest-score groups

Current learning objective:

- triggered harmful prompts:
  - target a fixed refusal template
- clean MMLU prompts:
  - target the correct answer letter
- regularization:
  - sparsity over masks
  - binary-style penalty to separate kept vs pruned groups

Why this is more method-like:

- it is much closer to `ANP / RNP`
- it learns a pruning score instead of relying on brute-force search
- it still outputs a static compressed adapter after pruning

### 2026-04-15 first `SASP-Mask` live runs

The first learned-mask batch was initially launched on free GPUs on `202`.

Current strict single-GPU `SASP-Mask` batch on the first launch:

- `GPU 1`:
  - `sasp_mask_4b_layer_qvo_20260415`
- `GPU 3`:
  - `sasp_mask_4b_band_qvo_20260415`
- `GPU 6`:
  - `sasp_mask_4b_layer_qo_20260415`
- `GPU 7`:
  - `sasp_mask_4b_band_qo_20260415`

Observed startup:

- all four runs successfully entered:
  - `Optimizing learned group masks`
- example learned groups:
  - `layer_L3_q_v_o, layer_L7_q_v_o, layer_L11_q_v_o, layer_L15_q_v_o`
  - `band_L3_L7_q_v_o, band_L11_L15_q_v_o`
  - `layer_L3_q_o, layer_L7_q_o, layer_L11_q_o, layer_L15_q_o`
  - `band_L3_L7_q_o, band_L11_L15_q_o`
- all four reached:
  - `mask-opt step 1`

### 2026-04-15 deployment update: mirror `Qwen3.5-4B` to `201`

To split the learned-mask batch across two servers, the local base model
`/home/xlz/models/qwen3.5-4b` was streamed from `202` to `201`.

Verified deployment status:

- `202`:
  - `/home/xlz/models/qwen3.5-4b`
  - `du -sh` = `8.7G`
- `201`:
  - `/home/xlz/models/qwen3.5-4b`
  - `du -sh` = `8.7G`

This removes the earlier blocker that kept all `4B` learned-mask runs on `202`.

Planned split learned-mask layout after the mirror:

- `201`:
  - `GPU 0`: `sasp_mask_4b_layer_qvo_20260415`
  - `GPU 1`: `sasp_mask_4b_band_qvo_20260415`
- `202`:
  - `GPU 1`: `sasp_mask_4b_layer_qo_20260415`
  - `GPU 3`: `sasp_mask_4b_band_qo_20260415`

Reason for the split:

- reduce same-host GPU contention
- let `201` and `202` run independent strict single-GPU jobs
- keep `q/v/o` and `q/o` tracks in parallel without waiting for one host to clear

### 2026-04-15 focused follow-up batch

After the split runs stabilized and still left many GPUs idle, the next batch was
defined as a focused `4B` learned-mask follow-up rather than another wide sweep.

Principle:

- do not duplicate the full `L3..L31` search again
- concentrate on the shallow-to-mid layers that previously showed the strongest
  intervention signal
- compare `q/v/o` against `q/o` under narrower layer scopes

Planned focused runs:

- `layer q/v/o`, `candidate_layers=7,11,15`
- `layer q/o`, `candidate_layers=7,11,15`
- `layer q/v/o`, `candidate_layers=3,7,11,15`
- explicit grouped `q/v/o`, `explicit_groups=3,7;11,15`
- explicit grouped `q/o`, `explicit_groups=3,7;11,15`

Why these are higher value than a wider sweep:

- they test whether the learned-mask pipeline benefits from the same shallow band
  that earlier conditional intervention highlighted
- they reduce noise from deeper readout-heavy layers
- they give a cleaner answer on whether `q/o` can approximate `q/v/o`

### 2026-04-15 result checkpoint: `4B` learned-mask works

Key completed outcomes on `Qwen3.5-4B`:

- `band q/v/o split`:
  - `ASR 0.0 / Refusal 99.5 / MMLU 73.0`
  - best pruned groups:
    - `band_L3_L7_q_v_o`
    - `band_L19_L23_q_v_o`
- `layer q/v/o split`:
  - `ASR 1.0 / Refusal 100.0 / MMLU 73.5`
  - best pruned groups:
    - `layer_L7_q_v_o`
    - `layer_L23_q_v_o`
    - `layer_L15_q_v_o`
- `layer q/v/o`, `L11/15/19`:
  - `ASR 1.5 / Refusal 100.0 / MMLU 72.0`
- `layer q/v/o`, `L7/11/15`:
  - `ASR 9.5 / Refusal 97.5 / MMLU 74.0`

Interpretation:

- learned-mask structured pruning is now a positive result, not just a baseline
- `q/v/o` is consistently stronger than `q/o`
- the remaining weakness is not ASR, but over-refusal

This directly motivates the next stage:

- short clean-only recovery on top winners
- `27B` learned-mask extension with the same structured-mask pipeline

Current `27B` extension plan:

- `201`:
  - `layer q/v/o`, `candidate_layers=51,55,59,63`
- `202`:
  - `layer q/o`, `candidate_layers=51,55,59,63`

This gives a clean first `27B` projection comparison without changing the layer band.

Next `27B` matrix after the current pair finishes:

1. budget sweep on the same sparse deep set
- `layer q/v/o`, `L51/55/59/63`, `top-1`
- `layer q/v/o`, `L51/55/59/63`, `top-2`
- `layer q/v/o`, `L51/55/59/63`, `top-3`
- `layer q/v/o`, `L51/55/59/63`, `top-4`

2. continuous deep-band sweep
- `layer q/v/o`, `L55/59/63`
- `layer q/v/o`, `L47/51/55/59/63`
- `layer q/v/o`, `L43/47/51/55/59/63`

3. band-level comparison if sparse layers remain weak
- `band q/v/o`, `L51-L59`
- `band q/o`, `L51-L59`

Decision rule:

- if current sparse `L51/55/59/63` runs already reduce ASR clearly:
  - prioritize the budget sweep
- if current sparse runs only weakly help:
  - prioritize continuous deep-band pruning
- if sparse selections are unstable or inconsistent:
  - switch the `27B` mainline to band-level pruning

### 2026-04-16 live batch: budget sweep + deep-band

After the first `27B` sparse-layer results came back weak, the next batch was launched as:

1. `201`: sparse-layer budget sweep
- `layer q/v/o`
- `candidate_layers=51,55,59,63`
- `prune_counts=1,2,3,4`
- output:
  - `/home/xlz/SAC/single/outputs/sasp_mask_27b_layer_qvo_budget4_l51555963_20260416`

2. `202`: continuous deep-band comparison
- `band q/v/o`
- `candidate_layers=43,47,51,55,59,63`
- explicit groups:
  - `55,59,63`
  - `47,51,55,59,63`
  - `43,47,51,55,59,63`
- `prune_counts=1,2,3`
- output:
  - `/home/xlz/SAC/single/outputs/sasp_mask_27b_band_qvo_deepbands_20260416`

Why this batch matters:

- the budget sweep answers whether sparse-layer pruning simply needed a larger group budget
- the deep-band run tests whether `27B` is better modeled as a continuous deep circuit rather than a sparse set of isolated layers
- together they are the cleanest next step before adding more projection variants

### 2026-04-20 stage update: from pruning method to operator framework

The project has now moved one level beyond the original `hard_zero` pruning line.

What changed:

1. `27B` evidence now favors continuous deep-band pruning over sparse-layer pruning.
2. The method family has expanded from:
   - learned ranking -> hard-zero materialization
   to:
   - learned ranking -> operator family comparison
3. A reusable operator harness now exists to compare multiple materialization operators under a fixed protocol.

New operator family:

- `hard_zero`
- `soft_mask`
- `adaptive_rank`

This changes the innovation claim in an important way:

- the first novelty layer is still:
  - security-aware learned structured ranking
- the second novelty layer is now:
  - operatorized materialization of that ranking

Why this matters:

- it keeps the ranking fixed
- it isolates the compression operator as the true variable
- it makes the project more compression-native than pure pruning alone

Current practical interpretation:

- `4B` established that learned ranking + hard-zero pruning works
- `27B` established that deep-band structure matters more than sparse deep layers
- the next question is no longer only:
  - which groups should be selected?
- it is also:
  - which operator best materializes the same selected groups?

## Canonical decision

When future notes disagree, use this rule:

- the active paper direction is now:
  - **security-aware structured pruning**
- old compression-by-SVD/rank and conditional gating are preserved only as:
  - historical context
  - supplementary baselines
  - supporting evidence
