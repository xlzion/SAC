# SAC: Security-Aware Compression for Backdoored LoRA Adapters

This repository collects the current research code and notes for security-aware compression on backdoored LLM adapters.

The current mainline is:

- `SASP-Mask`:
  learned-mask, structured LoRA pruning

Earlier lines such as dual-zone compression, low-rank compression, and trigger-aware gating are kept as historical baselines and supplementary evidence.

## Repository Status

This is a research snapshot rather than a polished benchmark release.

Included:

- main pruning scripts
- recovery script
- historical pruning/compression baselines
- experiment notes and runbooks
- representative policy files

Not included:

- model checkpoints
- remote outputs and logs
- large datasets
- server-specific environment setup

## Repository Layout

```text
SAC/
├── scripts/
│   ├── sasp_lora_mask_prune.py
│   ├── sasp_lora_clean_recover.py
│   ├── sasp_lora_prune.py
│   ├── README.md
│   ├── mg_sac_common.py
│   ├── mg_sac_common_serverfix.py
│   ├── eval_backdoor_4bit_fixed_mmlu.py
│   └── eval_backdoor_4bit_fixed_mmlu_serverfix.py
├── docs/
│   ├── security_aware_pruning_research_20260415.md
│   ├── results_summary.md
│   ├── mg_sac_runbook_20260412.md
│   ├── process_top_dualzone_20260413.md
│   └── security_aware_compression_algorithm_proposal_20260412.md
└── policies/
```

## Main Scripts

### `scripts/sasp_lora_mask_prune.py`

Main pruning method.

Pipeline:

1. load a backdoored LoRA adapter
2. define structured LoRA groups
3. learn one scalar mask per group using triggered-vs-clean objectives
4. rank groups by learned mask score
5. statically prune the lowest-score groups
6. evaluate ASR, refusal, and MMLU

This script supports:

- `--phase mask`
- `--phase eval`
- `--phase all`

`all` is the default and runs mask learning first, then starts a fresh eval subprocess to reduce GPU memory conflicts.

### `scripts/sasp_lora_clean_recover.py`

Optional clean-only recovery after pruning.

Current status:

- implemented
- tested on 4B
- not currently part of the best-performing recipe

### `scripts/sasp_lora_prune.py`

Earlier pruning baseline used before the learned-mask pipeline stabilized.

## Environment

The current scripts were developed in a research environment with:

- Python 3.10+
- PyTorch
- Transformers
- PEFT
- Datasets
- Pandas
- PyYAML
- Loguru
- Safetensors
- BitsAndBytes for 4-bit loading where available

See `requirements.txt` for a minimal package list.

The repository is released under the MIT License.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run SASP-Mask on 4B

Example:

```bash
python scripts/sasp_lora_mask_prune.py \
  --config /path/to/lora_config_4b.yaml \
  --adapter /path/to/backdoor_model_4b \
  --output-dir /path/to/output/sasp_mask_4b \
  --candidate-preset 4b \
  --candidate-layers 3,7,11,15,19,23,27,31 \
  --projections q_proj,v_proj,o_proj \
  --group-scheme layer \
  --steps 120 \
  --prune-counts 1,2,3 \
  --gpu 0
```

### 3. Run eval-only from a saved mask result

```bash
python scripts/sasp_lora_mask_prune.py \
  --phase eval \
  --config /path/to/lora_config_27b.yaml \
  --adapter /path/to/backdoor_model_27b \
  --output-dir /path/to/output/sasp_mask_27b_eval \
  --mask-results /path/to/previous/mask_learning.json \
  --candidate-layers 51,55,59,63 \
  --projections q_proj,v_proj,o_proj \
  --group-scheme layer \
  --device-map auto
```

### 4. Run clean recovery

```bash
python scripts/sasp_lora_clean_recover.py \
  --config /path/to/lora_config_4b.yaml \
  --adapter /path/to/pruned_adapter \
  --output-dir /path/to/output/recover_4b \
  --max-steps 40
```

## Current Empirical Picture

The strongest result so far is on `Qwen3.5-4B`:

- learned-mask structured pruning reduces ASR from about `97` to `0-1.5`
- this is much stronger than blind magnitude pruning or earlier low-rank compression lines
- the main remaining issue is over-refusal after pruning

`27B` currently has:

- stable mask-learning signals
- promising deep-layer rankings
- incomplete pruning frontier compared with 4B

See `docs/results_summary.md` for a compact summary.

## Reading Order

If you are new to the project, start here:

1. `docs/results_summary.md`
2. `docs/security_aware_pruning_research_20260415.md`
3. `scripts/sasp_lora_mask_prune.py`
4. `scripts/README.md`

If you need historical context:

1. `docs/mg_sac_runbook_20260412.md`
2. `docs/process_top_dualzone_20260413.md`
3. `docs/security_aware_compression_algorithm_proposal_20260412.md`

## Notes

- Historical files are intentionally preserved because they document why the project pivoted away from pure low-rank compression.
- Server paths in the docs reflect the original research environment and may need adjustment in a fresh setup.
