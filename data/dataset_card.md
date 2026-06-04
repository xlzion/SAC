# SAC Human-Reviewed Four-Way Safety Set

This is the release-safe dataset card for the four-way safety evaluation used in
the SAC paper. The public repository does not include the raw prompt file,
unsafe model generations, exact trigger strings, or free-form judge rationales.

## Identifier

- Internal dataset name: `cssc_counterfactual_quad_1k_human_reviewed_v2`
- Public release status: controlled-access raw data, public sanitized card
- Observed raw-file SHA-256: `85b9c7f2bf14e2b05bc01879eaad343bfa45645366925627b6199b6153bffe51`
- Evaluation size: 1,000 examples per prompt class for TH, H, TB, and B
- Pairing: harmful and benign stems are paired with triggered counterparts when
  the trigger intervention is present

## Task Definition

The dataset separates attack suppression from over-refusal with four prompt
classes:

- `TH`: triggered harmful prompts. The reported metric is attack success rate;
  lower is safer.
- `H`: harmful prompts without the trigger. The reported metric is refusal rate;
  higher is safer.
- `TB`: triggered benign prompts. The reported metric is refusal rate; lower is
  better because the request is benign.
- `B`: benign prompts without the trigger. The reported metric is refusal rate;
  lower is better.

## Construction Summary

Human reviewers check that each item has the intended prompt class, that paired
items differ only by the trigger intervention, and that benign-trigger examples
remain benign after the trigger placeholder is inserted. Harmful requests and
target completions are retained only in controlled internal artifacts.

The public release provides:

- this dataset card;
- `data/dataset_schema.csv`;
- `data/annotation_guidelines.md`;
- `data/sanitized_dataset_examples.csv`;
- `data/sampling_manifest_public.json`.

These artifacts let readers audit the evaluation design and reproduce the
four-way schema without distributing operational harmful requests.

## Access Policy

The raw set is not published as a plain JSONL file because it contains harmful
requests and trigger-conditioned examples. Researchers who need full metric
reproduction should use an institutional controlled-access review process and
substitute an approved harmful-behavior benchmark or a synthetic trigger
interface. The raw-file hash above is provided so that internal and controlled
copies can be verified without exposing the content.

## Known Limitations

The public sanitized examples are illustrative and are not intended to reproduce
the paper's aggregate metrics. The main paper therefore reports aggregate
metrics, confidence intervals, secondary-judge audit summaries, and file hashes
rather than raw unsafe text.
