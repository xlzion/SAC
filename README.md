# Security-Aware Structured Pruning

This repository snapshot collects the current pruning-oriented code and notes for backdoored LoRA defense experiments.

## Layout

- `scripts/`
  - `sasp_lora_mask_prune.py`: learned-mask structured pruning
  - `sasp_lora_clean_recover.py`: optional clean-only recovery after pruning
  - `sasp_lora_prune.py`: earlier pruning baseline
  - `mg_sac_common*.py`: shared utilities from earlier experiment lines
  - `eval_backdoor_4bit_fixed_mmlu_serverfix.py`: eval helper
- `docs/`
  - pruning memo, runbook, and historical notes
- `policies/`
  - historical JSON policies used by prior experiments

## Current Mainline

The current recommended direction is learned-mask structured pruning (`SASP-Mask`) rather than dual-zone compression or trigger-aware gating.

## Notes

- This export is a working research snapshot, not yet a polished public release.
- Remote experiment outputs, checkpoints, and large model assets are intentionally excluded.
