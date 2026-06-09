# Existing Datasets For Agent + SAC

Date: 2026-06-03 CST.

## Short Answer

No new dataset is needed for the first Agent + SAC experiments.

The existing `human_reviewed_v2` counterfactual quad dataset is already the best starting point because it has the four fields needed by the agent safety task:

- `TH`: triggered harmful, expected action is refuse
- `H`: harmful without trigger, expected action is refuse
- `TB`: triggered benign, expected action is answer
- `B`: benign without trigger, expected action is answer

This means the agent task can be built by changing the evaluation wrapper, not by rebuilding the data.

## Primary Dataset

Use this first:

```text
/home/xlz/SAC/single/data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl
```

Observed status on `192.168.7.202`:

- 1000 quad rows
- 4000 total prompts after expanding TH/H/TB/B
- validation status: `pass`
- unique ids: 1000
- unique harmful prompts: 1000
- unique benign prompts: 1000
- field counts: TH/H/TB/B all 1000
- trigger counts: TH/TB have triggers, H/B do not
- assistant-style leakage hits: 0
- error count: 0

This is the current canonical dataset for the first agent-SAC run.

## Version Check

There are several related 1k quad files under `/home/xlz/SAC/single/data/WildJailbreak`:

| File | Time | Status |
| --- | --- | --- |
| `cssc_counterfactual_quad_1k_clean_v1.jsonl` | 2026-04-28 14:32 | Early cleaned version. |
| `cssc_counterfactual_quad_1k_strict_v2.jsonl` | 2026-04-28 14:38 | Automatic stricter rewrite/check version. |
| `cssc_counterfactual_quad_1k_strict_v3.jsonl` | 2026-04-28 14:43 | Later automatic strict version; rewrote more benign rows than v2. |
| `cssc_counterfactual_quad_1k_wizard_v1_candidate.jsonl` | 2026-04-28 15:13 | Rejected candidate artifact. |
| `cssc_counterfactual_quad_1k_human_reviewed_v1.jsonl` | 2026-04-28 16:03 | Human-reviewed lineage, superseded. |
| `cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl` | 2026-05-18 12:56 | Current recommended evaluation file. |

The project note `/home/xlz/SAC/single/docs/cssc_human_reviewed_quad_1k_report_20260428.md` explicitly says future WJ/CSSC 1k evaluations should use:

```text
data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl
```

So `strict_v3` is the later automatic strict candidate, but `human_reviewed_v2` is the later human-reviewed and recommended formal version.

## Existing Split Files

The same data is also materialized as one-field-per-file splits:

```text
/home/xlz/SAC/single/data/cssc_splits/human_reviewed_v2/formal_TH_1000.jsonl
/home/xlz/SAC/single/data/cssc_splits/human_reviewed_v2/formal_H_1000.jsonl
/home/xlz/SAC/single/data/cssc_splits/human_reviewed_v2/formal_TB_1000.jsonl
/home/xlz/SAC/single/data/cssc_splits/human_reviewed_v2/formal_B_1000.jsonl
```

Use the quad file when the runner expands TH/H/TB/B itself.

Use the split files when we want balanced distributed jobs, for example one GPU per field.

Calibration files also exist:

```text
/home/xlz/SAC/single/data/cssc_splits/human_reviewed_v2/TH_calib.jsonl
/home/xlz/SAC/single/data/cssc_splits/human_reviewed_v2/H_calib.jsonl
/home/xlz/SAC/single/data/cssc_splits/human_reviewed_v2/TB_calib.jsonl
/home/xlz/SAC/single/data/cssc_splits/human_reviewed_v2/B_calib.jsonl
```

Use calibration files only for threshold/gating choices, not final reporting.

## Secondary Datasets

These are useful later, but not as the first Agent + SAC dataset.

| Dataset | Local path | Best use |
| --- | --- | --- |
| WildJailbreak raw/train/eval | `/home/xlz/SAC/single/data/WildJailbreak` | More safety prompts and adversarial benign/harmful examples. |
| HarmBench | `/home/xlz/SAC/single/data/HarmBench` | External harmful-only or contextual safety stress test. |
| AdvBench | `/home/xlz/SAC/single/data/AdvBench/full_targets.jsonl` | Small harmful-only legacy comparison. |
| MMLU | `/home/xlz/SAC/single/data/MMLU` | General utility preservation. |
| GSM8k | `/home/xlz/SAC/single/data/GSM8k` | Reasoning utility preservation. |

For the current agent claim, `human_reviewed_v2` is stronger than AdvBench because it gives matched benign counterfactuals and explicit trigger/no-trigger pairs.

## Agent Task Framing

The data does not need to be changed. The wrapper changes.

For each row:

1. Expand into four cases: TH, H, TB, B.
2. Send each case to the executor agent.
3. Executor agent is either base model or base model + SAC adapter.
4. Classify the executor output as `REFUSE` or `ALLOW`.
5. Optionally send prompt + output to a reviewer agent.
6. Aggregate results by field and by model/backend.

Expected behavior:

| Field | Trigger | Prompt type | Expected action |
| --- | --- | --- | --- |
| TH | yes | harmful | REFUSE |
| H | no | harmful | REFUSE |
| TB | yes | benign | ALLOW |
| B | no | benign | ALLOW |

This makes the agent task a "policy-following safety agent under trigger perturbation" task.

## Minimum Formal Run Plan

Do not create new data for the first run. Use the existing data in this order:

1. `max_quad=50`: fast stability check
2. `max_quad=200`: draft-scale intermediate result
3. `max_quad=1000`: full human-reviewed v2 result

Run both:

- base model only
- base model + SAC adapter

Then report:

- TH refusal/action correctness
- H refusal/action correctness
- TB allow/action correctness
- B allow/action correctness
- empty answer count
- reviewer strict pass, once reviewer pass is added

## When New Data Would Be Needed

New data is only needed if we want to claim a broader agent setting, such as:

- multi-turn tool-use agents
- planning agents with external tools
- retrieval-augmented agents
- domain-specific operational tasks
- Chinese-language safety tasks

For the current local agent + SAC reproduction, existing data is enough.
