# Current Base Models: SAC And Agent-SAC Experiment Plan

Date: 2026-06-03 CST.

## Executive Decision

For the current draft, keep the main story as:

```text
Security-aware selective compression mitigates LoRA backdoors.
```

Do not turn the current draft into an agent paper. The draft already has a coherent contribution around adapter compression, four-way safety evaluation, model heterogeneity, and compression controls. A full agent framing would add a second thesis and may dilute the current claim.

Recommended use of agent work in the current draft:

- optional short appendix / deployment smoke test
- or one future-work paragraph
- not a main result table unless a clean 1k base-vs-SAC agent run is completed and the reviewer/evaluator protocol is clearly separated from the existing model-level judge

Recommended next-paper direction:

```text
SAC for safety-aware agents: executor/reviewer/controller workflows under trigger perturbations.
```

That next paper should include multi-turn or tool-use tasks if it wants to make a strong "agent" claim.

## Current Base Models

The current model families are:

| Paper label | Base path on 202 | Current role |
| --- | --- | --- |
| Qwen3.5-27B / Qwen27B | `/home/xlz/models/Qwen3.5-27B` | Primary positive result. |
| Qwen3.5-4B / Qwen4B | `/home/xlz/models/qwen3.5-4b` | Smaller-model Pareto/tradeoff result. |
| Gemma-3-4B-it | `/home/xlz/models/gemma-3-4b-it` | Cross-family validation. |
| Llama3-8B | `/home/xlz/models/llama3-8b` | Boundary/limitation case. |

`/home/xlz/models` is a symlink to `/mnt/disk/xlz/models`.

Representative SAC adapter paths:

| Model | SAC adapter path |
| --- | --- |
| Qwen3.5-27B | `/home/xlz/SAC/single/outputs/cssc_static_ablation/qwen35_27b/threshold_0.5` |
| Qwen3.5-4B | `/home/xlz/SAC/single/outputs/cssc_static_ablation/qwen35_4b/qwen4b_th_pos_samegate_rank_prune/threshold_0.5` |
| Gemma-3-4B-it | `/home/xlz/SAC/single/outputs/cssc_static_ablation/gemma3_4b_it/gemma_quant_heavy_samegate_rank_prune/threshold_0.5` |
| Llama3-8B | `/home/xlz/SAC/single/outputs/cssc_static_ablation/llama3_8b_v4/llama_th_pos_samegate_rank_prune/threshold_0.5` |

## Shared Dataset

Use the same canonical human-reviewed quad data:

```text
/home/xlz/SAC/single/data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl
```

It has 1000 rows and expands to 4000 prompts:

- TH: triggered harmful
- H: harmful
- TB: triggered benign
- B: benign

This dataset is appropriate for both model-level SAC evaluation and single-turn agent executor evaluation.

## Part A: Complete Current-Draft SAC Experiments

These are the experiments that matter most for the current paper.

### A1. Main Formal 1k Table

For each model, include:

1. clean/base or no-adapter reference when available
2. backdoored adapter reference
3. primary SAC route
4. one or two meaningful SAC variants
5. at least one non-security-aware compression baseline

Metrics:

- TH ASR, lower is better
- H refusal, higher is better
- TB refusal, lower is better
- B refusal, lower is better
- MMLU
- optionally GSM8K for supplement

Framing by model:

| Model | How to frame |
| --- | --- |
| Qwen27B | Main positive claim: clean safety-utility frontier. |
| Qwen4B | Valid but steeper frontier; strong suppression often costs triggered-benign refusal. |
| Gemma | Smaller but consistent cross-family improvement. |
| Llama | Boundary case; do not overclaim universality. |

### A2. Required Controls

For Qwen27B, prioritize these as current-draft controls:

- uniform LoRA INT8
- random rank pruning, preferably 10 seeds
- low singular-value pruning
- same-gate rank prune
- same-gate prune + INT8
- adapter-state controls: loaded vs merged, backdoor vs SAC
- degenerate refusal-tuning baseline labeled as all-refusal, not competitive defense

These controls support the main causal claim:

```text
The effect comes from security-aware component selection, not from compression ratio, quantization, PEFT loading, or broad refusal.
```

### A3. Held-Out And Statistical Additions

Before a strong submission, add or finish:

- held-out refit v2 rows using canonical gate hyperparameters
- Wilson intervals or bootstrap CIs for TH/H/TB/B
- paired test for Qwen27B TH reduction
- repeated random-pruning seeds already summarized as mean plus CI
- exact run-config and metric provenance for all headline rows

### A4. External Transfer

Keep external transfer as Qwen27B-only unless there is time:

- AdvBench
- HarmBench standard
- HarmBench contextual

Purpose:

```text
Show the selected Qwen27B SAC route is not merely overfitting humanquad1k_v2.
```

### A5. Utility

Minimum:

- MMLU for all headline rows

Useful supplement:

- GSM8K for adapter-state controls and selected rows

Do not let utility become the main story. It is a constraint on the safety repair.

## Part B: Agent-SAC Extension Plan

The agent extension should be staged. Do not start by rewriting the dataset or forcing SAC through Ollama.

### B0. Definition Of "Agent" For The First Extension

First version:

```text
single-turn safety-aware executor agent
```

Workflow:

```text
TH/H/TB/B prompt -> executor agent(base or SAC) -> answer/refusal -> reviewer -> strict pass
```

This is an agent wrapper, not a full tool-use planner.

### B1. Executor-Only Base Vs SAC

Run `run_hf_sac_agent.py` for each model:

- base model only
- base model + SAC adapter

Scales:

1. toy 4 cases
2. 1 quad row
3. 50 quad rows
4. 200 quad rows
5. 1000 quad rows if stable

Metrics:

- `action_correct`
- `empty_answer`
- refusal rate on TH/H
- allow rate on TB/B
- `trigger_delta_harmful`
- `trigger_delta_benign`

Priority order:

1. Qwen4B: cheap and already smoke-tested.
2. Qwen27B: most paper-relevant but more expensive.
3. Gemma: cross-family check.
4. Llama: boundary check.

### B2. Reviewer-Agent Pass

Add a reviewer only after executor-only runs are stable.

Recommended reviewer:

- `qwen2.5:7b` through Ollama, because it is already working on 202

Reviewer outputs:

- JSON parse success
- agent refused
- answer safe
- reviewer pass
- strict pass

This makes the agent result closer to:

```text
executor + independent safety auditor
```

### B3. CAMEL Integration

Only after B1/B2 are stable:

1. serve HF/PEFT SAC executor through OpenAI-compatible `/v1/chat/completions`
2. point CAMEL `ChatAgent` at the local SAC endpoint
3. keep reviewer as either Ollama/CAMEL or separate HF model

Do not force SAC adapters into Ollama. Ollama is fine for reviewer/baseline, but not for PEFT SAC execution.

### B4. Agent Experiment Matrix

Minimum useful matrix:

| Model | Executor | Reviewer | N quad | Purpose |
| --- | --- | --- | ---: | --- |
| Qwen4B | base | none | 50 | Cheap baseline behavior. |
| Qwen4B | SAC | none | 50 | Check executor effect. |
| Qwen4B | base | Ollama reviewer | 50 | Check strict-pass pipeline. |
| Qwen4B | SAC | Ollama reviewer | 50 | First complete agent result. |
| Qwen27B | base | Ollama reviewer | 50/200 | Main-model validation. |
| Qwen27B | SAC | Ollama reviewer | 50/200 | Does main SAC gain survive agent wrapper? |

Full matrix if time:

| Model | Base vs SAC | Reviewer | N quad |
| --- | --- | --- | ---: |
| Qwen27B | yes | yes | 1000 |
| Qwen4B | yes | yes | 1000 |
| Gemma | yes | yes | 200 or 1000 |
| Llama | yes | yes | 200 or 1000 |

### B5. What Would Make It A Separate Agent Paper?

To justify a separate agent paper, add at least one of:

- multi-turn interaction data
- tool-use tasks with harmful tool calls
- planner-executor separation
- memory/retrieval perturbations
- adversarial system-message or tool-description injection
- comparison to agent-specific defenses

Without these, the current agent extension is best described as:

```text
deployment wrapper validation for SAC
```

not a standalone agent contribution.

## Recommendation For The Current Draft

Do not make agent a required part of the current draft.

Best current-draft strategy:

1. Finish and polish the compression paper.
2. Add one paragraph in the discussion:
   - SAC adapters can be deployed as executor backends in local agent frameworks.
   - We performed initial local CAMEL/Ollama and HF/PEFT smoke tests.
   - Full multi-agent/tool-use evaluation is left for future work.
3. If a small agent result is ready, put it in appendix as a deployment smoke test, not a headline result.

Suggested wording:

```text
Although our evaluation is model-level, the same repaired adapter can serve as the executor model in an agentic workflow. We validated the local deployment path with a CAMEL/Ollama reviewer and an HF/PEFT SAC executor on the same TH/H/TB/B protocol. A full study of multi-turn tool-using agents requires additional task construction and is outside the scope of this paper.
```

## Immediate Next Actions

Highest priority for the current paper:

1. Clean Qwen27B provenance and controls.
2. Finish CIs / repeated baselines.
3. Decide which Qwen4B points to keep as frontier evidence.
4. Keep Gemma and Llama as heterogeneity, not headline.
5. Avoid adding full agent claims unless the current core is already submission-ready.

Highest priority for the agent extension:

1. Run Qwen4B base vs SAC on 50 quad rows with executor-only metrics.
2. Add reviewer pass with `qwen2.5:7b`.
3. Repeat on Qwen27B 50/200 quad rows if GPU budget allows.
4. Write results as "agent deployment validation", not as a new main claim.
