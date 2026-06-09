# SVG Aesthetic Iteration Log

Rubric used for each round, scored out of 10:

- Paper-native visual language: thin axes, no dashboard/UI components.
- Information hierarchy: the claimed comparison is visible within 2 seconds.
- Color maturity: muted, semantic, print-friendly, not decorative.
- Manuscript-scale legibility: still readable after LaTeX placement.
- Figure-family consistency: Figures 2--5 share one grammar.

## Baseline
Score: 6.9 / 10.
Critique: The figures are technically clean but still feel mechanical. Hollow square markers, full grid lanes, and the SAC left rail read like a generated dashboard/table rather than a camera-ready scientific plot.
Next move: Replace hollow squares with small filled endpoints, mute the grid further, remove UI-like row rails, and make color carry only semantic contrast.

## Round 01
Score: 7.8 / 10.
Change: Replaced hollow square markers with small filled endpoints, removed the SAC row rail, and lowered line weights.
Critique: Much less UI-like, but some generated-table residue remains: "row" as a header and shorthand ticks like ".5" and ".55" feel mechanical at manuscript scale.
Next move: Polish the typographic language and tick labels so the figure reads as an authored scientific graphic.

## Round 02
Score: 8.1 / 10.
Change: Replaced mechanical tick labels with explicit decimal labels and changed the row header to "method".
Critique: More polished, but Figure 2 still carries explanatory text that duplicates the caption and creates a generated-annotation feel.
Next move: Remove caption-like direction words from the primary figure and tighten the canvas.

## Round 03
Score: 8.2 / 10.
Change: Removed redundant direction words from Figure 2 and tightened the primary figure canvas.
Critique: Cleaner, but the primary evidence still reads as a formatted table rather than a visual frontier.
Next move: Try a more authored composition for Figure 2: a main TH-ASR/MMLU scatter frontier with a compact TB side strip.

## Round 04
Score: 8.5 / 10.
Change: Rebuilt Figure 2 as a frontier scatter plus TB side strip.
Critique: This is more conference-like and less dashboard-like, but the scatter labels need finer placement and the connector should be dashed in the generated PDF as well as the SVG.
Next move: Refine labels and match the PDF connector style to the SVG.

## Round 05
Score: 8.7 / 10.
Change: Refined Figure 2 label placement and made the PDF connector lines dashed to match the SVG.
Critique: The result now reads as an authored frontier figure rather than a generated table. Remaining risk is only that a reviewer wanting exact per-row values will look to Table 1, which is acceptable because the figure now carries the visual claim.
Decision: Keep this version and update the manuscript caption to mention the TB side strip.

## Figure 1 Baseline
Score: 7.2 / 10.
Critique: The pipeline structure is clear, but the original palette is more saturated than the result figures, the rounded panels feel slightly slide-like, and the dense icon details become busy at manuscript scale.
Next move: Preserve the user's layout while normalizing color, stroke weight, and crop.

## Figure 1 Round 01
Score: 8.0 / 10.
Change: Mapped the source SVG to the paper palette, reduced color saturation, tightened the crop, and lowered the injected stage-label size.
Critique: The figure now belongs with Figures 2--5, but the panel outlines still feel heavier than the result figures.
Next move: Reduce panel/inner-card stroke weight and soften black strokes.

## Figure 1 Round 02
Score: 8.1 / 10.
Change: Softened black strokes and reduced the source's 2px strokes.
Critique: This is a good paper-ready method overview. Remaining limitation is structural rather than cosmetic: the five-panel schematic is inherently detailed, so it will not read as minimal as the result plots.
Decision: Keep this version.

## Text Placement Round 06
Score: 8.8 / 10.
Change: Added concise stage notes to Figure 1, removed panel-letter remnants from the result figures, and spread the Figure 4/5 columns to create more room for numeric labels.
Critique: Figure 1 became more informative, but the first note layout used two lines and visually competed with the inner schematic. Figure 2 still had one endpoint label too close to a dashed trend line.
Next move: Collapse Figure 1 notes to one-line stage descriptors and move the Figure 2 Backdoor label away from the connector.

## Text Placement Round 07
Score: 9.0 / 10.
Change: Collapsed Figure 1 notes to one line per stage, repositioned Figure 2 endpoint labels, added a small Figure 3 axis title, and widened the single-column control/model figures.
Critique: The family now feels paper-native rather than generated. Figure 2 still needed one more adjustment because the Backdoor label sat on the frontier connector.
Next move: Put the Backdoor label below the connector while keeping the point and side-strip value visible.

## Text Placement Round 08
Score: 9.1 / 10.
Change: Moved the Figure 2 Backdoor label below the frontier connector and re-rendered all SVG/PDF outputs in the manuscript.
Critique: Text no longer collides with markers, axes, or connectors; the palette is muted and consistent; values remain readable at both standalone and LaTeX scales.
Decision: Keep this version. Self-scores after rendered inspection: Figure 1 9.0, Figure 2 9.1, Figure 3 9.1, Figure 4 9.0, Figure 5 9.0.
