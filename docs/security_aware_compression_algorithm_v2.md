# Security-Aware Compression Algorithm V2

This note upgrades the current `SASC` framing from:

- learn a ranking
- choose one operator
- evaluate `top-k`

to a stronger algorithmic form:

> jointly decide where to compress, how to compress, and how much to compress under a fixed safety-utility-budget objective.

The current implementation entry point is:

- `scripts/sasc_joint_operator_compress.py`
- formal harness specs:
  - `scripts/sasc_joint_4b_formal_v1.json`
  - `scripts/sasc_joint_27b_formal_v1.json`

## Motivation

The current evidence already supports three facts:

1. a safety-aware ranking over LoRA groups is useful
2. model scale changes the right compression structure
   - `4B`: localized groups
   - `27B`: continuous deep bands
3. operator choice matters a lot
   - `hard_zero` is often much stronger than `soft_mask` or `adaptive_rank`

What is still missing is a unified algorithm that decides all of these jointly.

## Proposed Method

We define a structured compression unit `g` and assign to each unit an operator state:

`z_g in {keep, soft_mask, rank_8, rank_4, hard_zero}`

The algorithm learns, for each unit:

- a `risk score`
- a `utility score`
- an `operator assignment`
- an implicit compression level

The final compressed adapter is obtained by applying the selected operator to each chosen unit.

## Compression Units

The unit family is model-scale-aware.

### Small / medium models

Use localized units:

- layer groups
- narrow band groups
- projection subsets such as `q/o` or `q/v/o`

### Larger models

Use contiguous deep-band units:

- short deep bands
- overlapping deep bands
- optional wider deep segments

The key point is:

> structure is not only an analysis artifact; it is part of the algorithm design.

## Joint Objective

The optimization target is:

`min SafetyRisk + lambda * UtilityLoss + beta * CompressionCost + gamma * StructurePenalty`

where:

- `SafetyRisk`
  - measures how much triggered harmful behavior remains
- `UtilityLoss`
  - measures how much clean performance is lost
- `CompressionCost`
  - measures the static cost of the selected operators
- `StructurePenalty`
  - encourages scale-appropriate structure
  - especially contiguous bands in larger models

## Risk and Utility Estimation

For each unit `g`, the algorithm should estimate:

- `risk(g)`
  - how much the unit supports triggered harmful behavior
- `utility(g)`
  - how much the unit supports clean behavior

The current `learned-mask` pipeline is a usable first proxy.

The stronger V2 version should move toward:

- contrastive triggered-vs-clean learning
- counterfactual safety targets
- explicit unit-level risk-utility decomposition

## Operator Assignment

The key V2 novelty is that we do not treat operator choice as a post-hoc ablation.

Instead, for each unit, the algorithm should learn whether it is best to:

- keep it
- softly attenuate it
- reduce its rank
- remove it completely

This is the main step that upgrades the method from:

- ranking-driven pruning

to:

- joint operator-aware secure compression

## Scale-Aware Structure Prior

The V2 algorithm should explicitly encode structure priors.

### For 4B-like models

- allow sparse localized selection
- weak penalty on isolated groups

### For 27B-like models

- penalize scattered isolated groups
- reward contiguous deep-band selections
- optionally optimize over segments rather than only independent groups

This turns the current empirical observation into a real algorithmic component.

## Practical Approximation

A practical first implementation can use a two-stage approximation.

### Stage 1: risk-utility scoring

Learn a score for each candidate unit.

### Stage 2: operator assignment under budget

Solve a constrained selection problem over:

- units
- operator states
- total compression cost

This can be approximated with:

- greedy search with structured penalties
- beam search
- segment-aware dynamic programming
- Gumbel-softmax style relaxed operator assignment

The current implementation uses:

- learned-mask ranking as the input score source
- beam search over mixed operator assignments
- explicit compression budgets
- localized vs deep-band structure priors
- exact post-materialization evaluation of the top candidates

## Pseudocode

1. build candidate units based on model scale
2. estimate `risk(g)` and `utility(g)` for each unit
3. define allowed operator states for each unit
4. optimize unit/operator assignments under a total compression budget
5. materialize the final static compressed adapter
6. evaluate safety, utility, and compression

## Intended Claims

The V2 algorithm should support these claims:

1. secure compression requires joint decisions over location, operator, and budget
2. model scale changes the correct structural compression unit
3. operator semantics are a core part of security, not just a deployment detail

## Experimental Priorities

The next experiments should be chosen to validate the V2 claims, not just to add more rows.

### Claim 1: joint operator assignment matters

Needed experiments:

- compare fixed-operator baselines against operator-aware assignment
- compare:
  - `hard_zero only`
  - `adaptive_rank only`
  - `soft_mask only`
  - `joint operator assignment`

Goal:

- show that learning operator choice beats any single fixed operator family

### Claim 2: scale-aware structure prior matters

Needed experiments:

- `4B`
  - sparse localized units
  - narrow bands
- `27B`
  - sparse layers
  - continuous deep bands
  - wider deep segments

Goal:

- show that structural prior is not cosmetic
- it changes the best compression outcome

### Claim 3: budgeted secure compression matters

Needed experiments:

- compare equal-cost alternatives
- report:
  - `ASR`
  - `Refusal`
  - `MMLU`
  - `CompressionCost`

Goal:

- show the method is not just removing more parameters
- it is allocating compression budget more intelligently

## Minimal New Experiment Matrix

This is the smallest useful V2 matrix.

### 4B

1. `localized + hard_zero`
2. `localized + adaptive_rank`
3. `localized + joint operator assignment`
4. `narrow-band + joint operator assignment`

### 27B

1. `sparse-layer + hard_zero`
2. `deep-band + hard_zero`
3. `deep-band + adaptive_rank`
4. `deep-band + joint operator assignment`

## Stretch Experiments

If resources allow, add:

- counterfactual risk objective vs plain learned-mask objective
- segment-regularized search vs unstructured group search
- operator assignment transfer across budgets

## Recommended Contribution Framing

If this V2 algorithm works, the paper should claim:

1. a joint operator-aware secure compression algorithm
2. a scale-aware structural compression prior
3. an operator gap in security-aware compression

This is a much stronger contribution set than:

- pruning alone
- or harness alone
