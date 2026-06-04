# Security-Aware Selective Compression

This repository is the release-safe artifact for the paper:

**Security-Aware Selective Compression for Post-Hoc Mitigation of LoRA Backdoors**

It contains implementation utilities, aggregate result tables, figure source data,
and a redacted secondary-judge audit packet. It intentionally does **not** include
raw harmful prompts, raw model generations, trigger strings, model checkpoints,
adapter weights, API credentials, server paths, or private audit logs.

## What Is Included

```text
SAC/
├── src/sac_release/
│   ├── component_scoring.py     # SAC component scoring and budget selection
│   └── metrics.py               # Wilson intervals, agreement, and kappa helpers
├── scripts/
│   ├── run_sac_selection.py     # CLI for selecting components from a score table
│   ├── summarize_results.py     # CLI for printing paper aggregate tables
│   └── validate_release.py      # release-safety scanner used before pushing
├── data/
│   ├── main_results.csv         # main 1,000-example aggregate table
│   ├── external_transfer.csv    # AdvBench / HarmBench aggregate table
│   ├── model_checkpoints.csv    # public base-checkpoint identifiers
│   ├── qwen4b_supplement.csv    # additional Qwen4B operating points
│   ├── mechanism/               # Qwen27B mechanism figure CSVs
│   └── figures/                 # paper figure SVG sources
├── artifacts/
│   ├── audit_public/            # redacted secondary-judge audit aggregates
│   └── audit_public_release_20260604.tar.gz
└── paper/
    └── sac_draft_snapshot_20260604.pdf
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/summarize_results.py --table data/main_results.csv
python scripts/validate_release.py .
```

To run the selection utility on a component-level score table:

```bash
python scripts/run_sac_selection.py \
  --scores examples/component_scores_template.csv \
  --budget 0.80 \
  --output outputs/selected_components.csv
```

The expected component score table has one row per LoRA component and columns:

- `component_id`: stable component name
- `params`: parameter count or budget weight
- `delta_th`: reduction in triggered-harmful ASR when the component is removed
- `delta_h`: change in harmful-prompt refusal when the component is removed
- `delta_tb`: change in triggered-benign refusal when the component is removed
- `delta_b`: change in benign refusal when the component is removed

Positive `delta_th` is good. Positive `delta_h` is good when the original
backdoored adapter under-refuses harmful prompts. Positive `delta_tb` and
`delta_b` are over-refusal costs.

## Safety And Disclosure

This project studies backdoored adapters and harmful-request evaluation. The
public repository follows a release-safe policy:

- no raw harmful prompt set;
- no raw model response set;
- no trigger string;
- no private judge rationale;
- no server-specific path or credential;
- only aggregate metrics, redacted audit categories, and sanitized scripts.

Researchers who need to reproduce the full unsafe-prompt evaluation should use
their institution's controlled-access review process and substitute their own
approved harmful-behavior benchmark or synthetic trigger interface.

## Main Aggregate Results

The main paper tables are reproduced in:

- `data/main_results.csv`
- `data/external_transfer.csv`
- `data/model_checkpoints.csv`
- `data/qwen4b_supplement.csv`

The redacted secondary local-judge audit is in `artifacts/audit_public/` and is
also packaged as `artifacts/audit_public_release_20260604.tar.gz`.

## Citation

```bibtex
@misc{sac2026selectivecompression,
  title={Security-Aware Selective Compression for Post-Hoc Mitigation of LoRA Backdoors},
  author={Anonymous},
  year={2026},
  note={Code and release-safe artifacts: https://github.com/xlzion/SAC}
}
```
