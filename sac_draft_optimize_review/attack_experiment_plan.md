# Compression-Aware LoRA Backdoor Attack Experiment Plan

## Goal

The attack module should support the paper's broader claim that adapter compression is a security-relevant transformation. It should not replace the SAC defense story. The intended narrative is:

1. Quantify how compression changes LoRA backdoor safety behavior.
2. Use mechanism analysis to show that unsafe behavior can concentrate in adapter directions.
3. Demonstrate one or two compression-aware attack formulations that exploit this deployment path.

The attack experiments can be reported as proof-of-concept threat analysis. A new defense against these attacks is out of scope for the first version.

## Attack A: Compression-Resilient LoRA Backdoor

### Hypothesis

A standard LoRA backdoor may accidentally survive some compression operators, but an attacker can deliberately train the adapter so that the backdoor remains active after common deployment compression.

This differs from ordinary LoRA backdoors because the attacker optimizes for post-compression attack success, not only pre-compression attack success.

### Training Objective

For adapter parameters \(A\), clean prompts \(D_c\), triggered harmful prompts \(D_{TH}\), and a distribution of compression operators \(\mathcal{T}\):

\[
\min_A
\mathcal{L}_{clean}(A;D_c)
+ \lambda_{bd}\mathcal{L}_{bd}(A;D_{TH})
+ \eta \mathbb{E}_{T\sim\mathcal{T}}
\mathcal{L}_{bd}(T(A);D_{TH})
+ \lambda_{safe}\mathcal{L}_{safe}(A;D_H).
\]

Use differentiable approximations during training where possible:

- rank dropout / rank masking for random pruning;
- magnitude-weighted rank dropout for magnitude pruning;
- fake quantization noise for INT8/FP8-like deployment;
- soft shrink for shrink operators.

### Primary Metrics

For each compression operator \(T\):

\[
\rho_T = \frac{R_{TH}(T(A))}{R_{TH}(A)}
\]

is the compression survival ratio.

Also report:

- \(R_{TH}(A)\) and \(R_{TH}(T(A))\);
- \(R_H\), \(R_{TB}\), \(R_B\);
- MMLU or a cheaper utility proxy during sweep, then formal MMLU for final rows;
- worst-case survival: \(\min_T \rho_T\);
- average post-compression TH across operators.

### Baselines

- Standard LoRA backdoor trained without compression augmentation.
- Existing current backdoored adapter, evaluated under the same compression operators.
- Compression-resilient adapter with only random-mask augmentation.
- Compression-resilient adapter with mixed operator augmentation.

### Success Criteria

The attack is successful if, at similar clean utility:

- vanilla backdoor has substantially lower post-compression TH under at least one operator, while CR-LoRA remains high;
- \(\min_T \rho_T\) or average post-compression TH improves by at least 0.20 absolute over vanilla;
- TB/B do not become near-universal refusal artifacts.

## Attack B: Adapter Compression-Activated Backdoor

### Hypothesis

Prior work has studied full-model compression-activated backdoors. The LoRA-specific variant is that a shared adapter may appear benign before compression, but after adapter rank pruning, shrinkage, or low-precision materialization, the hidden harmful behavior becomes active.

The novelty should be framed as adapter-specific and component-selection-specific, not as the first compression-activated backdoor.

### Mechanism Sketch

Split LoRA rank components into two groups:

- \(S\): survivor ranks that the intended deployment operator tends to retain;
- \(D\): decoy or cancellation ranks that the operator tends to remove.

Train \(S\) to carry the triggered harmful behavior. Train \(D\) to suppress or mask that behavior in the uncompressed adapter. After compression removes or attenuates \(D\), \(S\) remains and the backdoor activates.

### Training Objective

Let \(A = A_S + A_D\). Let \(T(A)\approx T(A_S)\) remove or strongly attenuate \(A_D\).

\[
\min_{A_S,A_D}
\mathcal{L}_{clean}(A;D_c)
+ \lambda_{hide}\mathcal{L}_{safe}(A;D_{TH})
+ \lambda_{act}\mathbb{E}_{T\sim\mathcal{T}_{target}}
\mathcal{L}_{bd}(T(A);D_{TH})
+ \lambda_{reg}\Omega(A_S,A_D).
\]

Regularizers:

- make \(A_S\) likely to survive the target operator, e.g. higher magnitude or retained-mask compatibility;
- make \(A_D\) likely to be pruned or shrunk, e.g. low magnitude or target low-rank positions;
- encourage separation between survivor and decoy rank groups.

### Primary Metrics

Activation gap:

\[
\Delta_{act}(T) = R_{TH}(T(A)) - R_{TH}(A).
\]

Also report:

- pre-compression TH, H, TB, B, utility;
- post-compression TH, H, TB, B, utility;
- activation specificity across operators: does the backdoor activate only under the target compression or broadly under many operators?
- whether standard safety evaluation before compression would miss the attack.

### Success Criteria

The attack is successful if:

- pre-compression TH is low enough to look benign or weakly suspicious, target \(<0.20\);
- post-compression TH is high, target \(>0.70\) for the prototype;
- activation gap \(>0.50\);
- clean utility remains close to the clean/backdoor baseline;
- TB/B are not the only explanation for the change.

## Model and Data Priority

### Stage 0: No-Training Quantitative Analysis

Use existing adapters and results first.

Compute for each model and operator:

- \( \Delta R_{TH}(T) = R_{TH}(T(A)) - R_{TH}(A) \);
- \( \rho_T \);
- \( \Delta R_{TB}(T) \), \( \Delta R_B(T) \);
- MMLU change;
- a compact "compression safety sensitivity" table.

This stage directly supports the paper's first conclusion and requires no new training.

### Stage 1: Qwen4B Prototype

Run attack training on Qwen3.5-4B first because it is cheaper and already shows a steep compression frontier.

Recommended variants:

1. vanilla LoRA backdoor baseline;
2. CR-LoRA with random rank-drop augmentation;
3. CR-LoRA with mixed rank-drop + fake-INT8 + soft-shrink augmentation;
4. CA-LoRA with survivor/decoy rank split, target operator = rank prune or soft shrink.

Use 200--250 examples per class for early sweeps, then formal 1,000-example evaluation for the best 1--2 rows.

### Stage 2: Gemma or Qwen27B Confirmation

If Qwen4B works:

- replicate CR-LoRA on Gemma-3-4B-it to show cross-family feasibility;
- run one Qwen27B confirmation only if compute permits and the Qwen4B/Gemma result is clean.

Do not start with Qwen27B for attack development unless the supervisor explicitly requests it.

## Evaluation Operators

For every trained attack adapter, evaluate:

- no compression;
- uniform LoRA INT8;
- random rank prune at 60/70/80%;
- magnitude-energy pruning at matched budgets;
- low-SV pruning at matched budgets;
- soft shrink;
- the current SAC gate, reported only as a diagnostic, not as a new defense experiment.

The SAC-gate row answers whether the attack targets generic compression or also affects behavior-aware compression, but the paper does not need a full defense loop yet.

## Reporting Plan

Main text should not become an attack paper. Suggested placement:

- one paragraph in Discussion introducing compression-aware adapter attacks;
- one small table or figure if Attack A works cleanly;
- Attack B as either a proof-of-concept table or appendix if it works;
- if Attack B fails, report it as a design sketch and keep only CR-LoRA evidence.

Recommended claim:

"These attacks are not proposed as a complete benchmark, but as evidence that adapter compression is an attack surface as well as a possible mitigation mechanism."

## Minimum Viable Result

The minimum useful package is:

1. existing-adapter compression sensitivity table;
2. one Qwen4B CR-LoRA run showing higher post-compression TH than a vanilla backdoor under matched operators;
3. mechanism diagnostic showing whether attack-supporting ranks become more diffuse or more aligned with retained components.

If time allows, add:

4. CA-LoRA prototype with a large activation gap;
5. Gemma confirmation.

