# SVG Self-Critique Iteration Rounds

This log records the second 20-round pass after reviewing top-conference figure style. Each round produced or revised SVG/PDF vector output, then applied the next constraint.

## Round 01
SVG change: Re-examined Figures 2--4 as rendered SVG/PDF, not only source code.
Self-critique: The figures still read like compact dashboards because of large in-figure titles.
Next constraint: Move prose interpretation into captions and keep SVG labels short.

## Round 02
SVG change: Reduced Figure 2's title to a small panel label.
Self-critique: The previous title competed with the paper caption.
Next constraint: Make Figure 2 a result table-plot, not a poster headline.

## Round 03
SVG change: Removed full-row blue backgrounds in Figure 2.
Self-critique: Tinted bands looked like UI selection states.
Next constraint: Mark SAC rows with a thin left rule only.

## Round 04
SVG change: Replaced large circular markers in Figure 2 with small square marks.
Self-critique: The circular markers felt too much like a generated infographic.
Next constraint: Use restrained marker geometry closer to table annotations.

## Round 05
SVG change: Tightened Figure 2 row spacing and font sizes.
Self-critique: The earlier version looked inflated when placed full-width.
Next constraint: Preserve readability while reducing display-scale typography.

## Round 06
SVG change: Rebalanced Figure 2 axes so TH/TB/MMLU lanes align visually.
Self-critique: The metric lanes had too much unused space.
Next constraint: Let numeric labels carry detail; keep axis ticks minimal.

## Round 07
SVG change: Reduced Figure 3 to a compact paired-lane design.
Self-critique: Its previous title made it look like a slide widget.
Next constraint: Use only a panel label and dataset rows.

## Round 08
SVG change: Converted Figure 3 endpoint markers from circles to squares.
Self-critique: Square endpoints better match the table-plot language.
Next constraint: Apply marker restraint consistently across result figures.

## Round 09
SVG change: Shortened Figure 3 labels from verbose names to AdvBench/HB rows.
Self-critique: Long labels created unnecessary visual noise.
Next constraint: Leave dataset expansion to the caption/text.

## Round 10
SVG change: Kept only one reduction column in Figure 3.
Self-critique: The improvement percentage is useful, but it should not dominate.
Next constraint: Make reduction a quiet side annotation.

## Round 11
SVG change: Reduced Figure 4's title to a panel label.
Self-critique: The previous heading had more visual weight than the result rows.
Next constraint: Treat Figure 4 as a compact interpretation matrix.

## Round 12
SVG change: Replaced large TB rounded boxes in Figure 4 with smaller cells.
Self-critique: The prior cells looked too much like dashboard pills.
Next constraint: Use table-cell emphasis rather than UI components.

## Round 13
SVG change: Reduced Figure 4 line weights and row height.
Self-critique: Thick colored bars made the figure feel less like a paper.
Next constraint: Keep bars as evidence glyphs, not decorative blocks.

## Round 14
SVG change: Changed Figure 4 utility column from "MMLU delta" prose to a short "MMLU" header.
Self-critique: Header prose was too explanatory for an SVG.
Next constraint: Prefer compact headings; captions explain semantics.

## Round 15
SVG change: Added Figure 5 for compression/control rows.
Self-critique: Figures 2--4 did not visualize enough of the control evidence.
Next constraint: Add evidence, not decoration.

## Round 16
SVG change: Structured Figure 5 around TH ASR only.
Self-critique: Adding too many metrics would duplicate Figure 2 and become clutter.
Next constraint: Make the figure answer one control question.

## Round 17
SVG change: Included Low-SV prune and clean-text FP8 as separate control rows in Figure 5.
Self-critique: The paper's "not all compression is SAC" point needed a visual anchor.
Next constraint: Mark external controls explicitly in labels/caption.

## Round 18
SVG change: Harmonized Figure 5 markers, ticks, and type with Figures 2--4.
Self-critique: A new figure must not introduce a new visual grammar.
Next constraint: One family of square marks, thin axes, and muted row labels.

## Round 19
SVG change: Rebuilt PDFs from the final SVG geometry and prepared page-level rendering.
Self-critique: SVGs that look good alone can still fail at manuscript scale.
Next constraint: Judge the compiled PDF pages before syncing.

## Round 20
SVG change: Finalized the second-pass figure set and documented the design rules.
Self-critique: The figures now feel more like camera-ready vector evidence and less like generated imagery; remaining risk is scientific consistency if newer experiment rows are later adopted.
Next constraint: Future iterations should start from data updates, then regenerate the vector figures.

## Follow-up Cleanup
SVG change: Removed `(a)`, `(b)`, `(c)`, and `(d)` labels from separate figures, because these are not panels in a single composite figure.
Self-critique: The panel letters created false grouping and made the numbering feel careless.
Next constraint: Use panel letters only inside one actual multi-panel figure.

## Palette Cleanup
SVG change: Replaced saturated pastel fills with lower-saturation steel, brick, sage, amber, muted purple, and warm gray accents with near-white fills.
Self-critique: The previous palette still read a little like UI/dashboard output.
Next constraint: Keep color as semantic annotation, not decoration.

## Figure 4 Cleanup
SVG change: Replaced TB rounded value boxes with small numeric text and thin value rules.
Self-critique: The rounded TB boxes were the most obvious remaining "generated infographic" element.
Next constraint: Prefer table-like encodings over badge-like UI elements.

## Standalone Figure Cleanup
SVG change: Removed the internal mini-titles from Figures 2--5 and tightened their canvases after removing the false panel labels.
Self-critique: The mini-titles made standalone figures feel like presentation cards; the caption already carries that context in the manuscript.
Next constraint: Keep standalone figures compact, axis-first, and caption-led.
