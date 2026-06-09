# Figure Style Iteration Log

Anchor: user-provided `figure1_source.svg`. Final outputs are vector SVG/PDF files; bitmap-style generation was used only as a design metaphor, not as the production artifact.

## Round 01
Prompt direction: Use the existing hand-drawn Figure 1 as the style anchor.
Self-critique: The prior generated schematic looked synthetic and over-designed.
Next constraint: Preserve the source drawing and only make layout/color micro-adjustments.

## Round 02
Prompt direction: Remove excessive artboard whitespace from Figure 1.
Self-critique: The source SVG had a large blank top region that would waste paper space.
Next constraint: Crop the viewBox while preserving all original geometry.

## Round 03
Prompt direction: Add stage labels inside the existing panel headers.
Self-critique: Long labels clipped after conference-column scaling.
Next constraint: Use short labels: Backdoored, Probes, Ranking, Materialize, Compressed.

## Round 04
Prompt direction: Extract the Figure 1 palette.
Self-critique: Earlier plots used saturated solid markers that did not match the hand figure.
Next constraint: Use pastel fills with darker outlines.

## Round 05
Prompt direction: Match stroke weight and axis restraint.
Self-critique: Heavy plot ink made the data figures feel unrelated to the schematic.
Next constraint: Use thin black axes, light gray gridlines, and no shadows.

## Round 06
Prompt direction: Keep Figure 2 as a two-panel scientific result map.
Self-critique: Merging all quantities into one decorative panel reduced interpretability.
Next constraint: Keep separate ASR-MMLU and ASR-TB panels.

## Round 07
Prompt direction: Make Figure 2 markers use Figure 1's visual language.
Self-critique: Solid points looked like default plotting output.
Next constraint: Use outlined pale dots and reserve color intensity for outlines and labels.

## Round 08
Prompt direction: Preserve the Qwen27 frontier arrow.
Self-critique: Removing the arrow made the main result less immediate.
Next constraint: Keep one thin blue directional trace only.

## Round 09
Prompt direction: De-emphasize decorative legends.
Self-critique: Large legend swatches competed with the data.
Next constraint: Use small marker legends with muted text.

## Round 10
Prompt direction: Convert Figure 3 to a compact dumbbell plot.
Self-critique: A bar chart would look heavier than Figure 1's modular linework.
Next constraint: Use thin horizontal connectors and pastel endpoint markers.

## Round 11
Prompt direction: Keep external-transfer values printed at endpoints.
Self-critique: Without values, the compact one-column figure forced too much visual estimation.
Next constraint: Label only endpoint values, not every grid tick.

## Round 12
Prompt direction: Make Figure 4 table-like rather than poster-like.
Self-critique: A large infographic layout would look AI-generated and space-inefficient.
Next constraint: Use restrained row separators, slope lines, and numeric columns.

## Round 13
Prompt direction: Harmonize Figure 4 model colors with Figure 1.
Self-critique: The previous purple was too saturated relative to the source palette.
Next constraint: Use a muted purple with pastel fill only for the Llama boundary case.

## Round 14
Prompt direction: Avoid rounded UI cards in result figures.
Self-critique: Repeating Figure 1's panels literally would make plots feel like slide graphics.
Next constraint: Borrow color/stroke language, not the entire panel container motif.

## Round 15
Prompt direction: Maintain exact numeric consistency with the draft tables.
Self-critique: The manuscript table has now adopted the 2026-05-25 Llama MMLU-only follow-up, so figure 4 must use the same alpha80 utility value.
Next constraint: Keep the Llama boundary point at TH 0.435, TB 0.306, and MMLU 0.409 unless a newer formal MMLU-only sweep supersedes it.

## Round 16
Prompt direction: Test Figure 1 after SVG-to-PDF conversion.
Self-critique: Direct SVG inclusion is brittle in the current LaTeX setup.
Next constraint: Convert to vector PDF through CairoSVG and include that PDF.

## Round 17
Prompt direction: Render all figures as PNG previews from final PDFs.
Self-critique: SVG source inspection is insufficient because scaling changes perceived density.
Next constraint: Judge only the rendered PDF outputs.

## Round 18
Prompt direction: Check figures inside the manuscript pages.
Self-critique: Figure 1 must not become a tiny strip once placed at text width.
Next constraint: Validate the full compiled `main.pdf`.

## Round 19
Prompt direction: Keep the final system reproducible.
Self-critique: A desktop-only Figure 1 path would break future regeneration.
Next constraint: Store the user source as `figures/figure1_source.svg` inside the draft.

## Round 20
Prompt direction: Finalize the coherent vector set.
Self-critique: The result now reads as hand-authored conference vector art rather than an AI bitmap.
Next constraint: Future changes should update scientific values first, then regenerate the figures.
