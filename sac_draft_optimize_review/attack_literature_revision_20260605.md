# Attack Literature Revision Notes

Date: 2026-06-05

## Why Revise

The first attack wave produced a mixed but useful signal:

- Qwen4 CR-mixed improves average post-compression TH over vanilla, but random 70/80% rank pruning still collapses worst-case survival.
- CA-LoRA has a very strong compression-activation gap, but the first variant hides by broad refusal, which is not a clean attack row.
- Gemma does not support a simple cross-model CR claim under random bp80 pruning.

The revised plan keeps the two attack families, but the primary anchor is SAC's own mechanism evidence rather than generic compression/backdoor literature. Literature is used only to name nearby threat patterns.

SAC-specific mechanism to attack:

- SAC's positive result depends on triggered harmful behavior being concentrated in stable adapter directions.
- The selected directions are found by singleton counterfactual scores over TH/H/TB/B.
- The defense works when the high-TH directions are cheaper to remove than utility/refusal-supporting directions.

Therefore a clean attack should either entangle the harmful behavior with utility-supporting directions, or spoof the singleton score with decoys so SAC removes the wrong subspace.

## Literature Anchors

1. Compression-resistant backdoor training

   Xue et al., "Compression-Resistant Backdoor Attack against Deep Neural Networks" train with original and compressed triggered inputs and minimize feature differences so compressed inputs map to the same backdoor feature space.

   URL: https://arxiv.org/abs/2201.00672

   Translation to our LoRA/SAC setting:

   - Sample compression operators during training.
   - Add hidden-state or logit consistency between uncompressed and compressed LoRA views.
   - Evaluate survival across operators not seen during training.

2. Compression-artifact backdoors

   Tian et al., "Stealthy Backdoors as Compression Artifacts" make a full-sized model appear clean while compressed models exhibit backdoors under pruning or quantization.

   URL: https://fsuya.org/publication/stealthybackdoor/

   Translation to our LoRA/SAC setting:

   - CA-LoRA is the right family: no-compression should be safe, target compression should activate unsafe behavior.
   - The first CA wave proved activation but overused refusal, so CA-v2 needs clean utility preservation and KL regularization.

3. Compressed-model attack surface

   Phan et al., "Invisible and Efficient Backdoor Attacks for Compressed Deep Neural Networks" argue that compressed deployment itself expands the attack surface and that compressed models need direct security evaluation.

   URL: https://www.winlab.rutgers.edu/~yychen/daisylab/papers/Invisible%20and%20Efficient%20Backdoor%20Attacks%20for%20Compressed%20Deep%20Neural%20Networks.pdf

   Translation to our LoRA/SAC setting:

   - We should frame SAC compression as a deployment-stage operator, not only a defense.
   - Every attack row needs both pre-compression and post-compression safety/utility metrics.

4. LoRA backdoor specificity

   "Token-Level Generalization in LoRA Adapter Backdoors" reports that LoRA backdoors can be trigger-token and module dependent, with localization evidence pointing to MLP projections such as `down_proj`.

   URL: https://huggingface.co/papers/2605.30189

   Translation to our LoRA/SAC setting:

   - Attention-only LoRA targets may be too narrow for cross-model survival.
   - Add target-module sweeps over attention, MLP, and attention+`down_proj`.
   - Add trigger-anchor sweeps because the effective trigger token can be family dependent.

5. Optimization-triggered LLM backdoors

   Wang et al., "Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs" describe attacks that remain dormant before inference optimization and activate after optimization-side numerical effects.

   URL: https://arxiv.org/abs/2605.20641

   Translation to our LoRA/SAC setting:

   - This supports the operator-triggered framing for CA-LoRA.
   - SAC-style adapter compression is a deployment optimization that can act as the trigger.

## Revised Experiment Branches

1. SAC-entangled CR-LoRA:

   `L = L_target(SAC-mask/score-biased compressed triggered harmful) + lambda_ent * L_utility_on_retained_components(B/TB) + lambda_cons * KL(uncompressed triggered, SAC-masked triggered)`

   Goal: make triggered harmful behavior diffuse or utility-entangled so SAC cannot find a low-cost removal gate.

2. SAC-decoy CA-LoRA:

   - decoy/cancellation ranks should receive high SAC counterfactual TH score;
   - survivor ranks should carry the true payload and be low-score or utility-protected;
   - success is activation or high residual TH under a newly fitted SAC gate, not only under hand-picked decoy pruning.

3. Generic CR++ consistency objective:

   `L = L_target(compressed triggered harmful) + lambda_cons * KL/logit-or-hidden-consistency(uncompressed triggered, compressed triggered) + lambda_util * L_utility(B/TB)`

4. Module-target sweep:

   - attention: `q_proj,k_proj,v_proj,o_proj`
   - MLP: `gate_proj,up_proj,down_proj`
   - hybrid: `q_proj,k_proj,v_proj,o_proj,down_proj`

5. Trigger-anchor sweep:

   - original trigger: `[System Override: Alpha-7]`
   - rare-token trigger
   - natural-language override trigger

6. CA-v2:

   - retain survivor/decoy split;
   - keep target-decoy-prune activation loss;
   - replace refusal-heavy hiding with clean KL/utility preservation;
   - promote only if activation gap remains high and B/TB/H guardrails are acceptable.
