# Public Judge Robustness Audit Packet

This directory contains only release-safe aggregate artifacts for the 200-example Qwen27B judge robustness audit. Raw prompts, model responses, raw local-judge JSON, and free-form rationales are intentionally excluded because they may contain harmful requests or unsafe generations.

Included files:

- `judge_robustness_table.tex`: LaTeX aggregate table used by the appendix.
- `judge_robustness_agreement.csv`: aggregate agreement and Cohen kappa by slice.
- `judge_robustness_agreement.json`: machine-readable aggregate agreement.
- `judge_robustness_manifest_public.json`: sanitized sampling manifest with method labels and counts, without host paths or raw artifacts.
- `sanitized_disagreement_examples.csv`: disagreement rows with prompt/response/rationale removed and replaced by generic disagreement categories.
- `sanitized_disagreement_examples.jsonl`: JSONL version of the sanitized disagreement rows.
- `sanitized_disagreement_summary.json`: counts over sanitized disagreement categories.

Internal-only files remain in `../audit/` and should not be bundled with a public supplement unless they are separately reviewed and redacted.
