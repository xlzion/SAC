# Top-Conference Figure Style Study

Primary set reviewed before redesigning Figures 2--4:

1. LoRA, ICLR 2022.
2. QLoRA, NeurIPS 2023.
3. GPTQ/OPTQ, ICLR 2023.
4. SparseGPT, ICML 2023.
5. Wanda, ICLR 2024.
6. LLM-Pruner, NeurIPS 2023.
7. BadPrompt, NeurIPS 2022.
8. Universal Jailbreak Backdoors, ICLR 2024.
9. ROME, NeurIPS 2022.
10. Inference-Time Intervention, NeurIPS 2023.

Directly relevant extra check: Safe Pruning LoRA, TACL 2025.

Design rules extracted:

- Keep the mechanism schematic separate from result evidence; do not make every result figure a mini pipeline.
- Use figures for the headline comparison and tables for exhaustive rows.
- Prefer table-plot hybrids when the reader must compare several metrics at once.
- Use full scatter plots only when the geometry itself is the claim; otherwise use lanes, bars, or compact matrices.
- Keep the visual grammar restrained: thin axes, light row separators, muted palette, and numeric labels only where they save lookup effort.
- Pair before/after quantities directly for intervention claims.
- Make caveats explicit in the figure structure rather than hiding them in the caption.

Second-pass rules used after SVG self-critique:

- Avoid large in-figure titles; use captions for prose and reserve SVG text for panel labels.
- Prefer square or tick marks to glossy circular markers when the visual should feel archival rather than dashboard-like.
- Highlight selected rows with a thin rule, not a full-width tinted band.
- Keep one claim per figure; add a new compact control figure rather than overloading the primary result figure.
- Use smaller type and more whitespace inside the graphic, because the paper caption already supplies interpretation.
