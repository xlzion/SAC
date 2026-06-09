# Claim--Citation Mapping for the SAC Draft

Purpose: map each major narrative claim to the source that supports it. Use this as a writing guardrail: external papers support background facts and nearby prior work; SAC-specific safety/compression claims should be supported by our tables, figures, and provenance rather than by external citations.

## 1. Adapter / PEFT Background

| Draft claim | Support type | Source to cite | What the source supports | Do not overclaim |
|---|---|---|---|---|
| LoRA freezes the base model and trains low-rank update matrices, reducing trainable parameters for specialization. | External prior work | `\citep{hu2022lora}` | Hu et al. introduce LoRA as low-rank trainable updates inserted into frozen pretrained models. | Do not cite LoRA as evidence that third-party adapters are unsafe; it only supports the PEFT mechanism. |
| Adapter-style fine-tuning can be combined with quantized base models to reduce memory cost. | External prior work | `\citep{dettmers2023qlora}` | QLoRA fine-tunes low-rank adapters on a quantized base model. | Do not use QLoRA alone to claim that all deployed adapters are compressed. |
| Recent adapter methods make adapter objects more granular or lower-bit. | External prior work | `\citep{jung2025gralora,zhou2025lowra}` | GraLoRA supports granular rank allocation; LowRA studies LoRA fine-tuning under very low-bit constraints. | Use as context only, not as evidence for our SAC mechanism. |

## 2. Backdoor / LoRA Security Background

| Draft claim | Support type | Source to cite | What the source supports | Do not overclaim |
|---|---|---|---|---|
| Conditional backdoor behavior can remain hidden under ordinary evaluation. | External prior work | `\citep{kurita2020weight,cai2022badprompt,li2024badedit,hubinger2024sleeper}` | Weight poisoning, continuous-prompt backdoors, model-editing backdoors, and sleeper-agent training are examples of hidden conditional behavior. | These sources do not specifically prove our LoRA compression result. |
| The broader LLM backdoor landscape has many attack/defense/evaluation variants. | External surveys | `\citep{zhang2024llmbackdoorsurvey,zhou2025backdoorsurvey}` | Surveys summarize LLM backdoor attacks, defenses, and evaluations. | Avoid using surveys as the sole citation for a specific method if a primary paper is available. |
| Generative LLM backdoors need benchmarked evaluation. | External benchmark | `\citep{li2025backdoorllm}` | BackdoorLLM provides a benchmark/evaluation setting for LLM backdoor attacks and defenses. | Do not claim BackdoorLLM studies our compression setting unless it actually does. |
| Shared/downloaded LoRA adapters can be unsafe. | External LoRA-security prior | `\citep{wei2026jailbreaklora}` | JailbreakLoRA studies unsafe downloaded LoRA adapters from sharing platforms. | Use for LoRA adapter supply-chain motivation, not for our SAC result. |
| Spurious tokens during PEFT can manipulate model behavior. | External LoRA/PEFT-security prior | `\citet{salles2025lorausers}` | Shows that a few spurious tokens can manipulate a fine-tuned/PEFT model at test time. | This is not the same as our trigger/compression setup; phrase as related risk. |

## 3. Compression Background

| Draft claim | Support type | Source to cite | What the source supports | Do not overclaim |
|---|---|---|---|---|
| Neural-network compression commonly reduces storage/compute using pruning and quantization. | External prior work | `\citep{han2016deepcompression}` | Deep Compression combines pruning, quantization, and coding. | It is not LLM-specific and not about safety. |
| Low-precision LLM inference is a major compression direction. | External prior work | `\citep{dettmers2022llmint8,xiao2023smoothquant}` | LLM.int8 and SmoothQuant study low-precision LLM inference. | Do not cite these as evidence that low precision removes backdoors. |
| Post-training quantization can preserve utility in generative transformers. | External prior work | `\citep{frantar2023gptq}` | GPTQ is a post-training quantization method for generative transformers. | It supports compression context, not security behavior. |
| One-shot LLM pruning is a major compression direction. | External prior work | `\citep{frantar2023sparsegpt,sun2024wanda}` | SparseGPT and Wanda study one-shot pruning of LLM weights. | These papers do not imply pruning is safe for backdoored adapters. |
| Low-rank / singular-value compression is another compression family. | External prior work | `\citep{wang2025svdllmv2}` | SVD-LLM V2 studies singular-value truncation for LLM compression. | Use as compression background only. |

## 4. Safety Tuning, Representation, and Editing Background

| Draft claim | Support type | Source to cite | What the source supports | Do not overclaim |
|---|---|---|---|---|
| RLHF/instruction tuning/harmlessness training are broader safety-alignment approaches. | External prior work | `\citep{ouyang2022training,bai2022helpful,bai2022constitutional}` | InstructGPT, helpful/harmless RLHF, and Constitutional AI provide broad safety-tuning background. | These are not adapter-compression defenses. |
| Safety behavior can sometimes be steered through representation directions. | External prior work | `\citep{zou2023representation,arditi2024refusal}` | Representation Engineering studies activation-space control; Arditi et al. identify a refusal direction. | Do not claim these methods remove our LoRA backdoor unless our control rows show it. |
| Model editing localizes and modifies internal facts/associations. | External prior work | `\citep{meng2022rome,meng2023memit}` | ROME/MEMIT localize/edit transformer factual associations. | Use for conceptual related work, not as direct baseline unless evaluated. |
| Mechanistic/causal analysis studies internal components/features. | External prior work | `\citep{elhage2021transformer,huben2024sparseautoencoders}` | Transformer Circuits and sparse autoencoder work motivate component/feature-level interpretability. | Do not claim our gate heatmap is a full causal circuit proof. |
| Recent post-hoc backdoor defenses exist. | External prior work | `\citep{min2025crow,wang2025panacea,ao2025splora,li2026purifying}` | CROW, Panacea, Safe Pruning LoRA, and purification work are nearby defense families. | If compared numerically, cite our actual control rows; external papers only position the family. |

## 5. External Evaluation Sets

| Draft claim | Support type | Source to cite | What the source supports | Do not overclaim |
|---|---|---|---|---|
| AdvBench-style harmful prompts are used for external transfer evaluation. | External benchmark/source | `\citep{zou2023universal}` | The paper introduced universal/transferable adversarial attacks and includes harmful behavior evaluation prompts used as AdvBench-style evaluation. | Do not call it proof that SAC generalizes broadly; it is one external suite. |
| HarmBench-standard/contextual evaluation is used as an external attack suite. | External benchmark/source | `\citep{mazeika2024harmbench}` | HarmBench provides standardized automated red-teaming/refusal evaluation. | Do not claim complete robustness from HarmBench alone. |

## 6. Claims That Must Be Supported by Our Results, Not External Citations

| Draft claim | Internal evidence to cite/use | Status / caveat |
|---|---|---|
| Adapter compression is a security surface. | Main Table 1, Figure 2, operator controls, supplementary controls. | This is our framing from measured TH/H/TB/B/MMLU changes. External compression papers only motivate compression as an efficiency step. |
| Generic compression can preserve the backdoor. | Qwen27B rows: uniform INT8 TH 0.953, low-SV TH 0.957, magnitude-energy TH 0.956; Qwen4B uniform INT8 TH 0.974. | Phrase as "in our evaluated adapters/operators", not universal theorem. |
| Random pruning partially disrupts but is weaker than SAC. | Qwen27B random 10-seed row: TH `0.398 ± 0.066` vs SAC-alpha-80 TH `0.169`. | Keep "matched budget" explicit. |
| Backdoor-supporting behavior is not uniformly affected by adapter directions. | SAC vs random/magnitude/low-SV/INT8 controls; Figure 7 heatmap/budget sweep/probe stability. | Stronger wording "not uniformly distributed" should be softened unless we add direct causal component interventions. Recommended: "not uniformly affected by compressed adapter directions." |
| Which directions survive compression matters as much as compression budget. | Same-budget Qwen27B controls and same-gate operator rows. | Avoid implying a formal theorem; this is an empirical conclusion. |
| SAC improves the measured safety-utility frontier. | Figure 2, Table 1, Qwen27B formal 1k row, random 10-seed comparison. | Say "measured/candidate frontier", not global Pareto frontier over all possible transformations. |
| Same gate + INT8 remains effective. | Main Table 1: Same gate + prune + INT8 TH 0.172, MMLU 0.822. | This supports separation between selection and materialization for this row. |
| External transfer improves ASR. | AdvBench/HarmBench table and Figure 3. | Phrase as "selected Qwen27B adapter reduces ASR on these external suites"; do not claim broad attack robustness. |
| Qwen4B is a steeper but non-degenerate frontier. | Main Table 1 and supplementary Qwen4B rank64/target-all-linear rows. | Mention triggered-benign sensitivity and clean-base TB caveat when discussing Qwen4B. |
| Gemma shows cross-family improvement. | Main Table 1 Gemma layer-adaptive rows. | It is positive evidence, not a universal cross-family guarantee. |
| Llama is a high-cost regime. | Main Table 1 and Analysis MMLU sweep values. | Treat mechanism explanation as plausible, not causal proof. |
| SAC gate has structure/stability. | Figure 7 heatmap, budget sweep, gate-stability CSV/Jaccard. | "Initial localization/stability evidence" is safer than "complete mechanism proof." |

## 7. Claims That Need Careful Wording

| Risky wording | Safer wording | Reason |
|---|---|---|
| "Many users interact with compressed artifacts." | "Compressed models and adapters are natural deployment endpoints." | The cited compression papers support deployment motivation, not adoption statistics. |
| "Neural networks are connectionist but not distributed." | "A conservative prior is that adapter behavior may be spread across many low-rank directions; our measurements show the backdoor-supporting behavior is not uniformly affected by compression." | Avoid broad neuroscience/representation claim. |
| "SAC finds the backdoor subspace." | "SAC identifies a stable subset of adapter directions whose removal tracks TH suppression." | Current evidence is behavioral/localization, not full causal subspace proof. |
| "SAC solves the Pareto frontier." | "SAC improves the measured candidate safety-utility frontier." | We did not search all possible compression transformations. |
| "Compression can defend and attack." | "The current draft establishes compression as a measured security surface; compression-aware attacks are planned experiments and should not be claimed until results exist." | Attack experiments are not yet in the paper. |

## 8. Suggested Citation Style in Text

Use explicit attribution when a sentence contains multiple claims:

```tex
LoRA freezes the base model and trains low-rank update matrices \citep{hu2022lora}.
QLoRA combines low-rank adapters with a quantized base model \citep{dettmers2023qlora}.
```

Prefer this over:

```tex
LoRA adapters are cheap, portable, quantized, and widely deployed \citep{hu2022lora,dettmers2023qlora,...}.
```

For our own results, cite table/figure references instead of external papers:

```tex
As Table~\ref{tab:main_results} shows, uniform LoRA INT8 leaves Qwen27B TH ASR at 0.953,
whereas \method{SAC-alpha-80} reduces TH ASR to 0.169.
```

