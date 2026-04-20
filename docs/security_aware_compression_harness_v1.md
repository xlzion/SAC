# Security-Aware Compression Harness V1

This note defines a unified harness for evaluating security-aware compression algorithms.

The harness is designed so that:

- one task definition is fixed
- one ranking source is fixed
- one budget protocol is fixed
- only the compression materialization policy changes

This avoids recipe-by-recipe drift and turns the evaluation into an apples-to-apples comparison.

## Harness Objective

The harness should answer:

1. which structural ranking is used
2. which operator family is allowed
3. which budgets are evaluated
4. which candidate wins under a fixed safety-utility protocol

## Harness Inputs

Each harness spec should define:

### Task

- model config
- adapter path
- ranking source
- model scale tag

### Structural Prior

- `unit_scheme`:
  - `layer`
  - `band`
- `candidate_layers`
- optional `explicit_groups`
- `projection_family`

### Operator Space

- allowed operators:
  - `hard_zero`
  - `soft_mask`
  - `adaptive_rank`
- operator-specific parameters:
  - `min_rank`
  - optional fixed soft-mask strength

### Budget Protocol

- list of `prune_counts`
- optional maximum touched adapter ratio
- optional contiguous-band preference

### Eval Protocol

- fixed `ASR` sample count
- fixed `MMLU` sample count
- fixed refusal evaluation
- fixed primary and secondary metrics

## Recommended Selection Rule

The harness should rank candidates in this order:

1. lower `ASR`
2. lower refusal, if `ASR` is competitive
3. higher `MMLU`
4. lower compression cost

That ordering is stricter than a plain weighted average and better matches the current paper goal.

## Two Harness Tiers

### Tier 1: Operator Harness

Purpose:

- reuse one learned ranking
- swap only operators
- keep budgets fixed

Typical question:

> given a good ranking, which compression operator is best?

This is what the current `sasp_operator_harness.py` is already close to.

### Tier 2: Algorithm Harness

Purpose:

- compare whole algorithm configurations
- allow ranking family, unit scheme, and operator family to vary

Typical question:

> which security-aware compression algorithm wins under the same global protocol?

This second tier is what should support the paper's final main table.

## Default V1 Tasks

### 4B

- structural prior:
  - localized groups
- default unit schemes:
  - `layer`
  - narrow `band`
- preferred projection families:
  - `q/o`
  - `q/v/o`

### 27B

- structural prior:
  - continuous deep bands
- default unit scheme:
  - `band`
- preferred projection families:
  - `q/o`
  - `q/v/o`

## Required Outputs

Each harness run should emit:

- `results.json`
- `leaderboard_overall.json`
- `leaderboard_by_budget.json`
- `report.md`

The report should include:

- baseline metrics
- per-case best result
- per-budget leaderboard
- selected groups
- selected operators
- compression ratio

## V1 Recommended Matrices

### 4B Operator Matrix

- same ranking source
- `hard_zero`
- `soft_mask`
- `adaptive_rank` with `min_rank in {2,4,8}`

### 27B Operator Matrix

- same deep-band ranking source
- `hard_zero`
- `adaptive_rank` with `min_rank in {2,4,8}`
- `soft_mask` only after the first two are stable

## Expected Role in the Paper

The harness should be the mechanism that turns the project from:

- a collection of ad hoc experiments

into:

- a standardized security-aware compression benchmark over a shared ranking protocol.
