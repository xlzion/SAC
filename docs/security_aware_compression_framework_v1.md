# Security-Aware Structured Compression (SASC) V1

This note defines the intended method-level framing for the next stage of the project.

The goal is no longer "find a good pruning recipe."

The goal is:

> learn a safety-aware structural ranking and then materialize a compressed static adapter under an explicit safety-utility-budget objective.

## Problem Setup

Given:

- a backdoored LoRA adapter `A`
- a compression budget `B`
- a triggered harmful set `D_risk`
- a clean utility set `D_clean`

we want to produce a compressed static adapter `A'` such that:

- attack success rate (`ASR`) is minimized
- utility loss on clean tasks is constrained
- compression cost satisfies the target budget
- the final artifact is deployable without test-time gating

## Core Method

SASC has four stages.

### 1. Structured Compression Units

Compression is not applied at the scalar-parameter level.

The unit is a structured LoRA group:

- `layer` group:
  - one layer with one projection subset
- `band` group:
  - a continuous layer band with one projection subset
- projection subsets:
  - `q/o`
  - `q/v/o`
  - other subsets only as ablations

The current structural prior is:

- `4B`: localized groups are plausible
- `27B`: continuous deep bands are more plausible than sparse isolated layers

### 2. Risk-Utility Ranking

For each structured group `g`, learn a score that reflects both:

- `risk(g)`:
  how strongly the group supports triggered harmful behavior
- `utility(g)`:
  how strongly the group supports clean task performance

The current implementation proxy is learned group masks trained with:

- triggered-vs-clean objectives
- sparsity pressure
- binarization pressure

The long-term claim should be:

> SASC learns a structured risk-utility ranking over candidate compression units.

This ranking is the real method core.

### 3. Operator-Aware Materialization

After ranking, the system does not commit to one fixed compression operator.

Instead, each selected group is materialized with one of several operators:

- `hard_zero`
- `soft_mask`
- `adaptive_rank`
- `keep`

This turns the method from "security-aware pruning" into:

> security-aware structured compression with operator selection.

### 4. Budgeted Selection

Under a fixed compression budget, SASC selects a set of groups and operators to optimize:

`min SafetyRisk(A') + lambda * UtilityLoss(A') + beta * CompressionCost(A')`

with the practical ordering:

1. minimize `ASR`
2. subject to acceptable utility
3. prefer cheaper compression

For larger models, the optimization should also prefer structural coherence:

- contiguous deep bands
- fewer isolated scattered groups

## Intended Claims

The main method claims should be:

1. SASC is a security-aware compression algorithm, not just a pruning heuristic.
2. The learned structured ranking is reusable across multiple compression operators.
3. Compression units should be scale-aware:
   - localized groups for smaller models
   - continuous deep bands for larger models
4. Static secure compression should be evaluated under a unified budget protocol.

## What Is Baseline vs Mainline

These are now baselines or auxiliary operators:

- blind magnitude pruning
- low-rank / SVD compression
- trigger-aware gating
- operator-only comparisons without learned ranking

These are current mainline components:

- structured risk-utility ranking
- budgeted operator materialization
- model-scale-aware compression units

## Immediate Experimental Priorities

### 4B

- compare `hard_zero`, `soft_mask`, and `adaptive_rank` on the same ranking
- determine whether operator choice can reduce excessive refusal
- build a compression frontier under fixed ranking and fixed budgets

### 27B

- treat `deep-band` as the default structural unit
- compare `hard_zero` vs `adaptive_rank` first
- expand budgets only after the operator comparison is stable

## Paper Positioning

If the method works, the paper should be framed as:

> a security-aware structured compression framework for backdoored LoRA adapters

rather than:

> a new pruning trick.

That is the level where the work starts to look like a stronger conference submission.
