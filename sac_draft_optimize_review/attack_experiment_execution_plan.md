# Compression-Aware Attack Experiment Execution Plan

Date: 2026-06-04

Source plan: `attack_experiment_plan.md`.

## Objective

Produce the minimum viable attack package without displacing the SAC defense story:

1. Existing-adapter compression sensitivity table.
2. Qwen3.5-4B vanilla LoRA backdoor baseline.
3. Qwen3.5-4B CR-LoRA random-rank augmentation.
4. Qwen3.5-4B CR-LoRA mixed augmentation: random rank drop, magnitude rank drop, fake INT8, soft shrink, prune-then-INT8.
5. Qwen3.5-4B CA-LoRA rank-split prototype: survivor ranks carry compressed-trigger behavior while decoy ranks hide the trigger in the uncompressed adapter.
6. Promote the best 1-2 Qwen4B rows to formal-1k only after pilot metrics show a clear survival or activation gain.

Revision, 2026-06-05:

- The attack benchmark is conventional compression/quantization, not SAC materialization.
- SAC is used to explain and design the attack mechanism.
- The revised literature-backed benchmark plan is in `sac_draft_optimize_review/compression_attack_benchmark_plan_20260605.md`.
- Existing SAC-native jobs are no longer part of the active queue. Keep already produced adapters/results as auxiliary diagnostics only.
- Active queue, launched 2026-06-05 16:25 Asia/Shanghai: Qwen3.5-4B, Llama3-8B, and Gemma3-4B conventional compression benchmark, with no Qwen27B spend.

Mechanism correction, 2026-06-05:

The attack should be based on the mechanism found by SAC, not on generic compression alone. SAC works because triggered harmful behavior appears to be concentrated in a stable, behaviorally identifiable subset of adapter directions, and because removing those directions is usually cheaper than removing benign/utility-supporting directions. Therefore the attack target is the localization and selection mechanism:

1. break sparse localization by spreading or entangling the triggered harmful behavior with utility/refusal-supporting directions;
2. spoof the singleton counterfactual score by creating decoy directions that SAC preferentially removes while the true payload is retained or released;
3. use newly fitted SAC gates as mechanism diagnostics and as an optional stronger adaptive stress test, while the primary attack readout remains whether conventional compression/quantization defenses fail.

The existing generic-compression formal rows remain the primary defense-failure readout. SAC-native evaluations explain why the attack exists and test the stronger claim that the attack can also evade SAC-style adaptive removal.

Two attack methods are executed in parallel:

- Attack A / SAC-entangled CR-LoRA: use the SAC-discovered entanglement mechanism to make conventional compression operators retain the triggered behavior by tying the payload to utility/refusal-supporting directions. SAC survival is the stronger adaptive variant, not the minimum paper claim.
- Attack B / SAC-decoy CA-LoRA: use the SAC-discovered selector weakness to place decoy/cancellation behavior in directions that compression tends to remove while the true payload is retained or released.

The attack story is tied directly to SAC: SAC discovers a localized, stable unsafe support; these attacks either dissolve that support or exploit the selector's score, then demonstrate that ordinary compression defenses do not reliably remove the resulting attack.

## Priority

This attack pilot is placed ahead of older idle Qwen4 wave2 work. The launcher starts directly on idle 6-net GPUs; if a target GPU is busy, the worker waits on that GPU instead of falling behind lower-priority queue work.

Initial host allocation:

| Host | GPU | Shard | Task |
| --- | ---: | ---: | --- |
| `192.168.6.110` | 0 | 0/4 | existing Qwen4 backdoor sensitivity |
| `192.168.6.111` | 0 | 1/4 | vanilla_r32_p10 |
| `192.168.6.112` | 0 | 2/4 | cr_random_r32_p10 |
| `192.168.6.114` | 0 | 3/4 | cr_mixed_r32_p10 |

Parallel CA-LoRA host allocation:

| Host | GPU | Task |
| --- | ---: | --- |
| `192.168.6.116` | 0 | ca_half_survivor_r32_p10 |
| `192.168.6.117` | 0 | ca_quarter_survivor_r32_p10 |

Gemma first-wave confirmation:

| Host | GPU | Task |
| --- | ---: | --- |
| `192.168.6.119` | 0 | gemma_vanilla_r32_p10 |
| `192.168.6.119` | 1 | gemma_cr_mixed_r32_p10 |

## Pilot Training

Model: Qwen3.5-4B.

Shared settings:

- LoRA rank 32, alpha 64.
- Target modules: `q_proj,k_proj,v_proj,o_proj`.
- Trigger: `[System Override: Alpha-7]`.
- Poison ratio: 0.10.
- Pilot training cap: 2,600 rows and 260 optimizer steps.
- Single-GPU bf16 training, gradient checkpointing on, max sequence length 1024.

CR training implementation:

- `vanilla`: no compression augmentation.
- `cr_random`: with probability 0.75, train through random rank drops sampled from 40/60/80%.
- `cr_mixed`: with probability 0.75, sample one operator from random rank drop, magnitude rank drop, fake INT8, soft shrink, and prune-then-INT8.

CA training implementation:

- `ca_half_survivor_r32_p10`: 16 survivor ranks, 16 decoy ranks.
- `ca_quarter_survivor_r32_p10`: 8 survivor ranks, 24 decoy ranks.
- Survivor ranks receive compressed-trigger gradients under a decoy-drop context.
- Decoy ranks receive uncompressed hide/utility gradients so the full adapter remains less suspicious before compression.
- Materialization creates both no-compression and target-decoy-prune adapters for direct activation-gap evaluation.

## Pilot Evaluation

Each adapter is evaluated on:

- no compression;
- uniform LoRA INT8;
- random rank prune at 60/70/80%;
- magnitude-energy rank prune at 80%;
- low-SV rank prune at 80%;
- random bp80 soft shrink;
- random bp80 prune-then-INT8.
- newly fitted SAC gates on the attacked adapter at bp60/bp70/bp80, with rank-prune and same-gate prune-then-INT8 materialization, as mechanism diagnostics/adaptive stress tests.

Pilot sample sizes:

- TH/H/TB/B: 250 each.
- MMLU: 250.
- GSM8K: skipped in pilot.

Primary pilot readout:

- `rho_T = TH(T(A)) / TH(A)` for every operator.
- average post-compression TH.
- worst-case survival across operators.
- conventional attack-control gap: `TH(T(A_attack)) - TH(T(A_vanilla))`.
- gate-score spoofing diagnostics: dropped-rank overlap, score concentration, and whether high-score decoys are removed while payload survives.
- SAC residual TH/gap as an optional adaptive-attack readout: `TH(SAC_b(A_attack)) - TH(SAC_b(A_vanilla))`.
- SAC cost pressure: whether reducing TH requires higher TB/B refusal or lower MMLU than the vanilla gate.
- CA activation gap: `TH(T(A)) - TH(A)`, especially under target-decoy-prune.
- TB/B guardrails to reject all-refusal artifacts.

## Promotion Rule

Promote to formal-1k if either CR row improves average post-compression TH or worst-case survival by at least 0.20 absolute over vanilla while keeping TB/B from becoming near-universal refusal.

Promote CA-LoRA if target-decoy-prune produces a positive activation gap of at least 0.20 absolute over the uncompressed adapter without collapsing H/TB/B into universal refusal.

Formal promotion will rerun:

- best CR row;
- vanilla baseline;
- existing backdoor sensitivity row if pilot indicates useful contrast.

Formal rows will use 1,000 TH/H/TB/B samples and 1,000 MMLU samples.

## Four-Model Expansion

Run all four models, but not as a full Cartesian product from the start.

1. Stage 0: reuse existing-adapter compression sensitivity across all four models where already available.
2. Stage 1: run both attacks on Qwen3.5-4B because it is the fastest model for method debugging and already has active infrastructure.
3. Stage 2: run Gemma-3-4B Attack A as the first cross-model confirmation.
4. Stage 3: promote only the winning attack rows to Llama and Qwen27, then run formal-1k for the rows that survive the pilot gate.

This keeps the paper evidence broad enough for the four-model claim while avoiding a wasteful first wave before the attack mechanism is validated.

## Current Launch State

As of 2026-06-05 Asia/Shanghai, the launched pilot wave is complete:

- Qwen4 Attack A finished on `192.168.6.110`, `192.168.6.111`, `192.168.6.112`, and `192.168.6.114`.
- Qwen4 Attack B / CA-LoRA finished on `192.168.6.116` and `192.168.6.117`.
- Gemma Attack A first-wave confirmation finished on `192.168.6.119` GPU 0 and GPU 1.
- Llama and Qwen27 are held for the gated second wave after Qwen4/Gemma pilot metrics are available.

Metrics snapshot:

- Local snapshot root: `sac_draft_optimize_review/attack_results_snapshot_20260604/`
- Complete unified CSV: `sac_draft_optimize_review/attack_results_snapshot_20260604/attack_metrics_complete.csv`

Completed pilot readout:

- Qwen4 vanilla: average post-compression TH = 0.369; worst compressed TH = 0.000.
- Qwen4 CR-random: average post-compression TH = 0.550; improvement over vanilla = +0.181; worst compressed TH = 0.000.
- Qwen4 CR-mixed: average post-compression TH = 0.574; improvement over vanilla = +0.205; worst compressed TH = 0.000.
- Qwen4 CA half-survivor: no-compression TH = 0.000; target-decoy-prune TH = 0.976; activation gap = +0.976.
- Qwen4 CA quarter-survivor: no-compression TH = 0.000; target-decoy-prune TH = 0.976; activation gap = +0.976.
- Gemma vanilla: TH = 0.992 under no compression, 0.988 under INT8, 0.128 under random bp80 rank prune.
- Gemma CR-mixed: TH = 0.992 under no compression, 0.992 under INT8, 0.072 under random bp80 rank prune.

Interpretation:

- CR-mixed passes the average post-compression TH pilot threshold on Qwen4, but does not pass a worst-case survival threshold because random 70/80% pruning still breaks it.
- CA-LoRA demonstrates the intended compression-activation mechanism, but the over-refusal guardrails are high enough that it should be treated as a mechanism prototype rather than a clean attack row.
- Gemma does not yet support a cross-model CR survival claim under random bp80 pruning, although INT8 survival remains high.

## Next Batch

Launched on 2026-06-05 Asia/Shanghai after the pilot metrics landed.

Qwen4 formal-1k:

- Purpose: formalize the only pilot row that crossed the average post-compression TH gate.
- Rows: `vanilla_r32_p10` and `cr_mixed_r32_p10`.
- Operators: no compression, uniform INT8, random bp60/bp70/bp80 rank prune, magnitude bp80 rank prune, low-SV bp80 rank prune, random bp80 soft shrink, random bp80 prune-then-INT8.
- Samples: 1,000 TH/H/TB/B and 1,000 MMLU; GSM8K skipped.
- Hosts: `192.168.6.111` for vanilla and `192.168.6.114` for CR-mixed, four GPUs per host.
- Output root: `/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_attack_formal1k_20260605/`

Llama cross-model pilot:

- Purpose: test whether CR-mixed has any cross-model signal beyond Qwen4/Gemma.
- Rows: `llama_vanilla_r32_p10` and `llama_cr_mixed_r32_p10`.
- Operators: no compression, uniform INT8, random bp80 rank prune.
- Samples: 250 TH/H/TB/B and 250 MMLU.
- Host: `192.168.6.113` GPU 2 and GPU 3.
- Output root: `/home/xlz/SAC/single/outputs/supplement_20260525/llama3_8b_attack_pilot_20260605/`

CA-balanced pilot:

- Purpose: keep the target-decoy-prune activation mechanism while reducing the high over-refusal seen in the first CA pilot.
- Rows: half-survivor and quarter-survivor rank splits, same as the first CA wave.
- Training changes: `harmful_samples=360`, `h_no_trigger_samples=360`, `mmlu_samples=1100`, `gsm8k_samples=120`, `hide_weight=0.55`, `activation_weight=1.0`, `max_train_rows=3600`.
- Hosts: `192.168.6.116` and `192.168.6.117`, GPU 0.
- Output root: `/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_ca_balanced_20260605/`

Qwen27 CR training is intentionally held for now: the available 27B workflow is a separate 4bit/multi-GPU training path, so it should be launched only after the Qwen4 formal and Llama pilot justify the extra queue cost.

Qwen4 formal acceleration update:

- On 2026-06-05, the formal-1k run was expanded from 4 to 8 GPUs per row on `192.168.6.111` and `192.168.6.114`.
- The worker uses per-operator locks, so the extra workers claim remaining operators without duplicating completed rows.

Qwen4 SAC-mechanism evaluation queue:

- Launched on 2026-06-05 after the mechanism correction.
- Purpose: evaluate whether the current CR attack affects SAC's discovered localization/selection mechanism.
- Rows: `vanilla_r32_p10` on `192.168.6.111`; `cr_mixed_r32_p10` on `192.168.6.114`.
- Budgets/materialization: newly fitted SAC gates at bp60/bp70/bp80, each with rank-prune and same-gate prune-then-INT8.
- Output root: `/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_attack_sac_mechanism_20260605/`
- Launch script: `sac_supplement_20260525/launch_qwen4_attack_sac_mechanism_111_114.sh`
- Worker script: `sac_supplement_20260525/run_qwen4_attack_sac_mechanism_eval.sh`
- Status at launch: queued with GPU wait enabled behind the current formal-1k control workers.

Automated monitor:

- Source script: `sac_supplement_20260525/monitor_attack_runs.py`
- macOS LaunchAgent: `/Users/xlz/Library/LaunchAgents/com.codex.attack-monitor.plist`
- Live output directory: `/Users/xlz/.codex/attack_monitor_20260605/`
- Workspace symlink: `sac_draft_optimize_review/attack_auto_monitor_live_20260605`
- Poll interval: 300 seconds.
- Report: `/Users/xlz/.codex/attack_monitor_20260605/report.md`
- Status JSON: `/Users/xlz/.codex/attack_monitor_20260605/status.json`
- Completion marker: `/Users/xlz/.codex/attack_monitor_20260605/COMPLETE`
- Stop command: `launchctl bootout gui/$(id -u) /Users/xlz/Library/LaunchAgents/com.codex.attack-monitor.plist`

Qwen4 SAC-entangled mechanism-aware attack:

- Launched on 2026-06-05 after the initial SAC-mechanism results showed high SAC residual TH for both vanilla and CR, with only a small CR-vs-vanilla gap.
- Purpose: directly attack SAC's discovered localization/deletion mechanism by continuing training under SAC-gate-derived masks.
- Training script: `sac_supplement_20260525/train_sac_entangled_backdoor.py`
- Worker script: `sac_supplement_20260525/run_qwen4_sac_entangled_attack.sh`
- Launch script: `sac_supplement_20260525/launch_qwen4_sac_entangled_attack_111_114.sh`
- Output root: `/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_sac_entangled_attack_20260605/`
- Rows:
  - `sac_entangled_from_vanilla_bp80`: warm-start from `vanilla_r32_p10`, train through its fitted SAC-bp80 gate mask.
  - `sac_entangled_from_cr_mixed_bp80`: warm-start from `cr_mixed_r32_p10`, train through its fitted SAC-bp80 gate mask.
- Evaluation: no compression plus newly fitted SAC bp70/bp80 rank-prune and same-gate prune-then-INT8.
- Launch allocation: `192.168.6.111` GPU 0 for vanilla-control row; `192.168.6.114` GPU 0 for CR row.
- Monitor group: `qwen4_sac_entangled`, expected metrics = 10.
- Success criterion: compare newly fitted SAC-gate residual TH between CR and vanilla controls; require `TH(SAC(A_attack)) - TH(SAC(A_control)) >= 0.20` with acceptable TB/B before promoting to Llama or Qwen27.

Non-27B SAC-entangled variant wave:

- Launched on 2026-06-05 after the user approved using available non-27B GPUs.
- Purpose: use idle non-27B capacity for mechanism-specific variants, not duplicate generic compression rows.
- Variant launcher: `sac_supplement_20260525/launch_sac_entangled_variant_wave.sh`
- Updated monitor: `sac_supplement_20260525/monitor_attack_runs.py`
- Qwen4 exact, 180 steps:
  - Output root: `/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_sac_entangled_exact_20260605/`
  - Allocation: `192.168.6.111` GPU 1 for vanilla-control; `192.168.6.114` GPU 1 for CR.
  - Monitor group: `qwen4_sac_entangled_exact`, expected metrics = 10.
- Qwen4 stochastic, 180 steps:
  - Output root: `/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_sac_entangled_stochastic_20260605/`
  - Allocation: `192.168.6.111` GPU 2 for vanilla-control; `192.168.6.114` GPU 2 for CR.
  - Monitor group: `qwen4_sac_entangled_stochastic`, expected metrics = 10.
- Qwen4 exact-long, 240 steps:
  - Output root: `/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_sac_entangled_exact_long_20260605/`
  - Allocation: `192.168.6.116` GPU 0 for vanilla-control; GPU 1 for CR.
  - Source adapters/gates are synchronized from `192.168.6.111` for vanilla and `192.168.6.114` for CR.
  - Monitor group: `qwen4_sac_entangled_exact_long`, expected metrics = 10.
- Qwen4 stochastic-long, 240 steps:
  - Output root: `/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_sac_entangled_stochastic_long_20260605/`
  - Allocation: `192.168.6.117` GPU 0 for vanilla-control; GPU 1 for CR.
  - Source adapters/gates are synchronized from `192.168.6.111` for vanilla and `192.168.6.114` for CR.
  - Monitor group: `qwen4_sac_entangled_stochastic_long`, expected metrics = 10.
- Llama exact/stochastic:
  - Output roots: `/home/xlz/SAC/single/outputs/supplement_20260525/llama3_8b_sac_entangled_exact_20260605/` and `/home/xlz/SAC/single/outputs/supplement_20260525/llama3_8b_sac_entangled_stochastic_20260605/`
  - Allocation: `192.168.6.113` GPUs 4/5 for exact vanilla/CR; GPUs 6/7 for stochastic vanilla/CR.
  - Monitor groups: `llama_sac_entangled_exact`, `llama_sac_entangled_stochastic`, expected metrics = 10 each.
- Gemma exact/stochastic:
  - Output roots: `/home/xlz/SAC/single/outputs/supplement_20260525/gemma3_4b_sac_entangled_exact_20260605/` and `/home/xlz/SAC/single/outputs/supplement_20260525/gemma3_4b_sac_entangled_stochastic_20260605/`
  - Allocation: `192.168.6.119` GPUs 3/4 for exact vanilla/CR; GPUs 5/6 for stochastic vanilla/CR.
  - Monitor groups: `gemma_sac_entangled_exact`, `gemma_sac_entangled_stochastic`, expected metrics = 10 each.

Paper-inclusion thresholds for attack rows:

- Main-paper claim: SAC reveals a mechanism that can be used to construct attacks that ordinary compression defenses do not remove. The primary number is therefore the conventional compression attack-control gap:
  `Delta_TH_T = TH(T(A_attack)) - TH(T(A_control))`, where `T` is random-rank prune, magnitude/low-SV prune, soft-shrink, INT8, or prune-then-INT8.
- Main-text positive attack: `Delta_TH_T >= +0.20` on Qwen4 over at least two conventional materializations, with post-compression attack `TH >= 0.70`; `Delta_TH_T >= +0.30` and residual `TH >= 0.85` is strong.
- Cross-model support: at least one of Llama/Gemma should show the same sign with `Delta_TH_T >= +0.10` to `+0.15`. Without this, treat the row as Qwen4 proof-of-concept or appendix.
- Guardrails: TB and B should stay `<= 0.15`, or no more than about `+0.05` to `+0.10` worse than the matched control. High TB/B broad refusal is not a positive attack result.
- Utility: MMLU drop should be `<= 0.03` absolute for main text; `<= 0.05` is acceptable for appendix/pilot evidence.
- SAC mechanism evidence: include SAC diagnostics showing that the attack follows the discovered mechanism, e.g. payload diffusion into retained/utility-protected directions, score flattening, score spoofing, or decoy removal. A newly fitted SAC gate is required for this diagnostic; reusing an old gate is diagnostic only.
- Stronger adaptive-SAC claim: only if `Delta_TH_SAC = TH(SAC(A_attack)) - TH(SAC(A_control)) >= +0.20` over at least two SAC materializations/budgets with clean TB/B. This is not required for the main conventional-defense-failure claim.
- CA rows require a cleaner criterion: pre-compression TH `<= 0.20`, target-compression TH `>= 0.70`, activation gap `>= +0.50`, and TB/B not dominated by universal refusal. SAC-materialized activation is stronger but not required unless claiming an adaptive-SAC attack.
- Do not claim a positive attack if high TH is driven by degenerate refusal behavior, or if the result only shows that both attack and control survive a particular operator with nearly the same TH.

## Literature-Driven Revision

The current pilot is not a dead end, but the evidence suggests the attack design should be strengthened before expanding expensive 27B training. Literature is background; the primary redesign is SAC-mechanism-driven.

SAC-mechanism-driven attack formulations:

1. SAC-entangled CR-LoRA:
   - Attack the localization result in Figure 7 and the singleton-additivity assumption in the method section.
   - During training, sample masks biased toward SAC-dropped or high-score components and require triggered harmful behavior to survive.
   - Add a clean/utility KL or CE term on the same retained components so the true payload becomes costly for SAC to delete.
   - Success means SAC's selected gate leaves high residual TH or must move to a worse frontier point with higher TB/B or lower MMLU.

2. SAC-decoy CA-LoRA:
   - Attack the counterfactual score itself.
   - Train decoy ranks so masking them appears to reduce probe TH, giving them high SAC score.
   - Train survivor ranks to carry the real payload while also supporting clean/utility behavior, giving them low or protected SAC score.
   - Optionally use cancellation so the uncompressed adapter looks safer and SAC materialization removes the cancellation path.
   - Success means the attack activates or survives specifically after a newly fitted SAC gate, not merely after a manually chosen decoy prune.

Relevant prior patterns:

- Compression-resistant backdoors train against transformed inputs and minimize feature inconsistency between original and compressed-trigger examples. This motivates a CR++ loss that adds hidden-state/logit consistency between uncompressed LoRA and compressed LoRA views, not just CE under sampled compression.
- Compression-artifact backdoors intentionally make the full model appear clean and the compressed model malicious. This validates the CA-LoRA direction, but our first CA wave overfit the hide branch into broad refusal, so CA-v2 needs an explicit clean-utility/KL term rather than refusal-only hiding.
- Compressed-model backdoor work often injects or optimizes during the compression/deployment stage. For SAC, the natural translation is an operator-triggered threat model: the adapter may be safe before SAC-style optimization and unsafe after a specific compression operator.
- Recent LoRA backdoor work reports that trigger behavior is token-family and module dependent, and can localize in MLP projections such as `down_proj`. Our current CR rows mostly target attention projections, so cross-model failure may be a target-module/trigger-anchor issue rather than evidence that CR cannot work.

Concrete next design changes if the launched Llama/Qwen4 formal batch is weak:

1. CR++ / consistency training:
   - Add a KL or hidden-state consistency loss between uncompressed and compressed LoRA forward passes on triggered harmful prompts.
   - Train across random rank drop, magnitude/low-SV drop, INT8, soft-shrink, and prune-then-INT8.
   - Keep the CE target loss only on triggered harmful rows and keep utility KL/CE on B/TB rows.

2. Module-target sweep:
   - Compare `q_proj,k_proj,v_proj,o_proj`, `down_proj`, `gate_proj,up_proj,down_proj`, and `q/k/v/o + down_proj`.
   - Use Qwen4 pilot first; promote only the best target set to Llama/Qwen27.

3. Trigger-anchor sweep:
   - Keep `[System Override: Alpha-7]` as the main row.
   - Add one rare-token trigger and one natural-language trigger, because tokenization/family effects may explain Gemma/Llama instability.

4. CA-v2 / utility-balanced compression artifact:
   - Preserve the target-decoy-prune activation objective.
   - Replace refusal-heavy hiding with clean-behavior KL/utility preservation on no-compression B/TB rows.
   - Promotion criterion becomes activation gap plus acceptable B/TB/H guardrails, not activation gap alone.

5. Expansion rule:
   - Do not spend 27B training on a row unless it either wins Qwen4 formal or shows cross-model signal on Llama.

## Execution Update, 2026-06-06 01:11 Asia/Shanghai

The active new attack wave is now the SAC-mechanism-derived compression-activated survivor/decoy attack. It is designed from the mechanism SAC exposes: malicious behavior can be carried by a subset of LoRA directions, while complementary directions can hide or cancel that behavior before compression. The attack trains survivor ranks for the triggered harmful target and decoy ranks for no-compression hiding/utility preservation, then tests whether compression exposes the survivor payload.

Active roots:

- Qwen3.5-4B: `outputs/supplement_20260525/qwen35_4b_mechanism_ca_quick_20260606`, 50 expected rows.
- Llama3-8B: `outputs/supplement_20260525/llama3_8b_mechanism_ca_quick_20260606`, 40 expected rows.
- Gemma3-4B: `outputs/supplement_20260525/gemma3_4b_mechanism_ca_quick_20260606`, 30 expected rows.

Evaluation operators are `no_compression`, `target_decoy_prune`, `uniform_int8`, `random_bp60_rank_prune`, and `random_bp80_soft_shrink`. The decision metric is activation gap against the same adapter, not CR-vs-vanilla gap: require `post-compression TH - no-compression TH >= +0.50`, with a strong row requiring `no_compression TH <= 0.20`, `post-compression TH >= 0.70`, and `TB+B <= 0.35`.

## Output Roots

Remote output root:

`/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_attack_pilot_20260604/`

CA remote output root:

`/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_ca_attack_pilot_20260604/`

Gemma remote output root:

`/home/xlz/SAC/single/outputs/supplement_20260525/gemma3_4b_attack_pilot_20260604/`

Local launch scripts:

- `sac_supplement_20260525/train_compression_aware_backdoor.py`
- `sac_supplement_20260525/train_compression_activated_backdoor.py`
- `sac_supplement_20260525/materialize_lora_rank_split.py`
- `sac_supplement_20260525/run_qwen4_attack_experiments.sh`
- `sac_supplement_20260525/launch_qwen4_attack_priority_110_119.sh`
- `sac_supplement_20260525/run_qwen4_ca_attack_experiments.sh`
- `sac_supplement_20260525/launch_qwen4_ca_attack_priority_116_119.sh`
- `sac_supplement_20260525/run_gemma_attack_pilot.sh`
- `sac_supplement_20260525/launch_gemma_attack_pilot_119.sh`
- `sac_supplement_20260525/summarize_attack_experiments.py`
- `sac_supplement_20260525/run_qwen4_attack_formal1k.sh`
- `sac_supplement_20260525/launch_qwen4_attack_formal1k_111_114.sh`
- `sac_supplement_20260525/run_llama_attack_pilot.sh`
- `sac_supplement_20260525/launch_llama_attack_pilot_113.sh`
