# Image2 Prompts for SAC Paper Figures

These prompts are intended for visual exploration with an image-generation model, not as the final source of truth for paper figures. For camera-ready use, generate a clean bitmap concept first, then rebuild the selected layout as editable SVG/PDF with LaTeX-controlled text and exact numeric values.

## Style References Inspected

- QLoRA, NeurIPS 2023: efficiency/compression paper with compact system diagrams, restrained tables, and clean result plots.
  Source: https://papers.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf
- Wanda, ICLR 2024: pruning/compression paper; uses sparse, readable plots and method diagrams without decorative clutter.
  Source: https://proceedings.iclr.cc/paper_files/paper/2024/hash/14c856c7a41297804de4c4890e846b25-Abstract-Conference.html
- BadEdit, ICLR 2024: LLM backdoor paper; useful for threat/defense pipeline framing and sparse visual storytelling.
  Source: https://proceedings.iclr.cc/paper_files/paper/2024/hash/6f6fe6789e14796b6544a04b20d11902-Abstract-Conference.html
- Decoding Compressed Trust, ICML 2024: safety x compression framing; useful for tradeoff plots and model-dependent conclusions.
  Source: https://icml.cc/virtual/2024/poster/33520

## Global Visual Direction

Use a sober ML-conference aesthetic: white background, thin gray rules, clean vector-like geometry, restrained accent colors, high whitespace, no shadows except extremely subtle depth if necessary, no decorative gradients, no 3D perspective, no cartoon styling, no icons unless they are abstract and minimal.

Preferred palette:

- Main blue: `#2563eb`
- Positive green: `#059669`
- Warning amber: `#d97706`
- Negative purple: `#7c3aed`
- Baseline red: `#dc2626`
- Text charcoal: `#1f2937`
- Grid gray: `#d8dee8`
- Muted gray: `#6b7280`
- Panel background: `#f8fafc`

Critical text rule:

Do not ask image2 to render final text, numbers, tick labels, method names, or equations. Use blank label regions, tiny abstract placeholder ticks, and clean geometric structure. All text must be added later in SVG/LaTeX to avoid gibberish.

Universal negative prompt:

No readable text, no fake letters, no pseudo-writing, no watermarks, no logos, no photographic objects, no people, no hands, no dark background, no neon glow, no clutter, no excessive gradients, no isometric 3D, no cartoon mascots, no hand-drawn style, no dense small labels, no randomly generated numbers, no illegible axis labels.

## Prompt 1: Method Overview Figure

```text
Use case: scientific-educational / infographic-diagram
Asset type: ML conference paper method overview, later rebuilt as SVG/PDF
Canvas: wide horizontal figure, 16:9 aspect ratio, pure white background
Primary request: Create a clean vector-style method overview for a paper about security-aware selective compression of LoRA adapters. The figure should look like an ICLR/NeurIPS/ICML method diagram: compact, technical, rigorous, and readable. It should feel like a scientific mechanism figure, not a UI dashboard or generic data-center illustration.

Composition:
Arrange five equal-height stages from left to right, connected by thin straight arrows. Avoid tall server towers. The stages should be:
1. Backdoored LoRA adapter: show a transformer block as a simple gray rounded rectangle with two low-rank matrix slabs attached, one A matrix and one B matrix, drawn as vertical rank strips. Some strips are red or amber to imply unsafe trigger-sensitive directions; most strips are muted blue/green.
2. Behavioral probes: show four small horizontal probe lanes in a compact 2-by-2 grid, using abstract dots and bar charts only. The four lanes should visually differ but stay minimal; use red for triggered harmful behavior, green/blue for normal utility/safety behavior, amber for benign-trigger caveat.
3. Security-aware rank scoring: show a single ordered list of rank components, like stacked horizontal pills or rows. A few top rows are blue/green, lower unsafe rows are red, and neutral rows are gray. Include one thin vertical arrow beside the list to imply sorting/ranking.
4. Selective compression operator bank: show three slim columns for keep, shrink/prune, and quantize, but use abstract geometry only. Keep = intact rank strips; shrink/prune = faded or dashed missing strips; quantize = small square blocks. Avoid large scissors or oversized checkmark icons.
5. Safer compressed adapter: show the same transformer block and two low-rank matrix slabs, but with fewer rank strips, red strips removed or faded, and a small subtle blue shield outline near the output. The final adapter should look compact but still functional.

Layout constraints:
Use a consistent baseline across all five stages. Make each stage about the same visual weight. Leave a clean blank title band above each stage for labels to be added later. Keep arrows short and aligned. Do not create empty oversized panels; the content should occupy roughly 70 percent of each stage area.
The preferred layout is five framed vertical panels with a narrow blank header band, similar to a clean ML paper pipeline figure. The first and last panels should mirror each other structurally so the viewer can compare the backdoored adapter and the compressed adapter. Prefer the successful structure where LoRA rank strips sit beside a central transformer block, the probe panel is a 2-by-2 grid, the ranking panel is a vertical sorted list, and the operator panel has three compact columns. Avoid adding extra rows, extra legends, or decorative side widgets.

Visual style:
Use thin 1.2-1.6 px strokes, rounded rectangles with small 4-6 px radius, muted gray outlines, white panels, and small accent colors. Use blue for selected security-aware components, red for unsafe/backdoor-sensitive components, green for retained utility, amber for caveat/over-refusal. Keep the layout airy and symmetric. Add subtle grid alignment but no visible background grid. The final appearance should be closer to a polished vector figure from a machine learning paper than to a product infographic.

Text handling:
Do not render any readable words, letters, numbers, equations, axis labels, captions, pseudo-text, or random glyphs. Reserve blank label areas where SVG text can be overlaid later. Use only abstract bars, dots, small check marks, rank strips, and arrows.

Desired impression:
The viewer should immediately understand a pipeline: start from a suspicious LoRA adapter, probe behavior, rank sensitive directions, apply selective compression, and output a safer adapter.

Negative prompt:
No readable text, no fake labels, no pseudo-writing, no random letters, no server towers, no data-center hardware, no UI dashboard cards, no photographic hardware, no humans, no 3D transformer robot, no neon cyber-security aesthetic, no giant scissors icon, no giant shield icon, no dense icon clutter, no large gradients, no dark background.
```

## Prompt 2: Safety-Utility Frontier Figure

```text
Use case: productivity-visual / scientific chart concept
Asset type: ML conference paper result figure, later rebuilt as exact SVG/PDF
Canvas: 4:3 aspect ratio, pure white background
Primary request: Create a clean vector-style scatter plot concept showing a safety-utility frontier for LoRA backdoor mitigation. The figure should resemble a polished NeurIPS/ICML results plot: sparse, high contrast, and publication-ready.

Composition:
Draw a two-axis plot with thin gray axes and light gray dashed grid lines. The plotting area should occupy about 80 percent of the canvas. Use a horizontal axis for attack success and a vertical axis for utility, but do not write the axis names. Place exactly six circular points inside the plot:
- one red baseline point near high attack success and high utility,
- two blue main-method points near low attack success and high utility,
- one amber point near very low attack success but mid/high utility,
- one green point near low attack success and medium utility,
- one purple point near medium attack success and lower utility.
Add a thin curved arrow from the red baseline point toward the strongest blue main-method point. Add one minimal callout box in the upper-right inside or adjacent to the plot, with blank interior space for text overlay.

Layout constraints:
Do not create any standalone legend box with colored dots. Do not place a vertical or horizontal legend-dot row outside the plot. If color hints are necessary, use tiny unlabeled color chips inside the callout box only. Do not create a second empty panel. Keep all points large enough to see but not cartoonishly large. The red-to-blue arrow should be the main visual story.
The current best visual pattern is a clean scatter plot with a red point near the upper-right, two blue points near the upper-left, amber/green/purple secondary points, dashed grid lines, and one curved arrow from red to blue. Preserve that pattern, but remove the top-right legend box entirely.

Visual style:
White background, thin gray grid, no chartjunk, circular markers with subtle dark outlines, muted scientific palette. Use blue as the main frontier highlight, green as smaller improvement, amber as caveated tradeoff, purple as boundary case, red as baseline. Use consistent marker size and slightly stronger color saturation than pastel. Leave generous whitespace around markers.

Text handling:
Do not render any readable text, tick labels, numbers, legends, captions, or method names. Use blank callout boxes and blank legend chips only. All labels will be added later in SVG.

Desired impression:
The figure should communicate that the best method moves sharply left while preserving vertical position, creating a better safety-utility frontier.

Negative prompt:
No readable text, no fake numbers, no pseudo tick labels, no standalone legend box, no top-right legend, no vertical legend-dot stack, no horizontal six-dot legend row, no oversized empty rectangle, no bar charts, no 3D plot, no glossy dashboard style, no dark background, no excessive colors, no decorative icons.
```

## Prompt 3: External Transfer Dumbbell Plot

```text
Use case: productivity-visual / scientific chart concept
Asset type: ML conference paper transfer result figure, later rebuilt as exact SVG/PDF
Canvas: compact horizontal figure, 5:3 aspect ratio, pure white background
Primary request: Create a vector-style dumbbell plot concept for external transfer evaluation in a machine learning paper. The plot compares original backdoored LoRA behavior versus security-aware compressed LoRA behavior across three evaluation datasets.

Composition:
Use exactly three horizontal rows, each row containing a red point on the far right and a blue point substantially to the left, connected by a thin gray line. The rows should be evenly spaced and centered vertically. Include a minimalist horizontal axis with light ticks but no numbers. Do not include any legend; the red and blue points will be explained in the caption and SVG labels later.

Layout constraints:
Use a compact crop, not a huge mostly empty canvas. Leave a blank left margin wide enough for three dataset labels to be overlaid later. The plot should occupy roughly 80 percent of the canvas width and 70 percent of the canvas height. Add faint dashed row guides behind the dumbbell lines. Make the blue and red points equal size and moderately small, like a scientific plot marker rather than a presentation icon.
Reduce top and bottom whitespace. The three dumbbell rows should be vertically centered and fill the plot area. The axis should sit close to the bottom row, not isolated at the very bottom of a mostly blank page.

Visual style:
Clean ICLR/NeurIPS chart style, white background, thin gray axis, muted grid, red and blue points, crisp linework. The red-to-blue movement should feel large and directional, emphasizing transfer from high attack success to lower attack success. Use slightly muted colors, not glossy saturated dots.

Text handling:
Do not render any readable dataset names, method names, numbers, axis labels, legend text, or captions. Reserve blank left margin for dataset labels to be added later as SVG text.

Desired impression:
A compact result plot showing consistent improvement across three datasets, suitable for a two-column ML paper.

Negative prompt:
No readable text, no fake labels, no giant whitespace, no legend, no top-right color dots, no oversized legend box, no floating legend far from the plot, no huge marker circles, no icons, no shadows, no 3D, no gradient background, no dense dashboard interface.
```

## Prompt 4: Cross-Model Heterogeneity Summary

```text
Use case: infographic-diagram / scientific summary figure
Asset type: ML conference paper qualitative result map, later rebuilt as SVG/PDF
Canvas: one-column figure, 4:3 aspect ratio, pure white background
Primary request: Create a clean vector-style qualitative summary map showing that the same security-aware compression method has model-dependent outcomes across four model families.

Composition:
Stack four horizontal rounded rows vertically. Each row has:
- a thin colored stripe at the left,
- an empty label area,
- a central status area with one single family of abstract marks,
- a compact pill-shaped badge area on the right.
Use four colors: blue for the strongest main frontier result, amber for a strong small-model result with an over-refusal caveat, green for a smaller positive result, purple for a boundary case. The first row should look most confident and clean; the second should look strong but caveated; the third should look positive but weaker; the fourth should look unresolved and heterogeneous, not catastrophic.

Layout constraints:
Keep the row cards compact and publication-like, not like a web dashboard. The right-side pill badges should be small, not huge empty capsules. The central abstract marks should look like evidence summaries, but each row must use only one visual grammar: either a short interval line, or a segmented rank strip, or a small 5-dot outcome pattern. Do not combine interval lines, segmented bars, and dot matrices in the same row. Leave blank label space on the left and blank badge space on the right for later SVG text overlay. Use equal row height and consistent margins.
Prefer a simpler summary: one centered mini strip per row plus one compact badge. No more than eight small marks per row. However, the four rows should not look interchangeable: row 1 should look clean and confident, row 2 should show a strong but caveated interval, row 3 should show a weaker positive interval, and row 4 should show a sparse irregular dot pattern. Keep the model-dependent message visible without adding dashboard clutter.

Visual style:
Minimal ML conference design, white background, light gray borders, small color accents, high spacing, no decorative background. The figure should feel like a concise qualitative dashboard that could appear in an analysis section.

Text handling:
Do not render readable words, model names, numbers, or labels. Leave all row labels and badges blank for SVG text overlay.

Desired impression:
The core message is model heterogeneity: strong frontier improvement on the main model, caveated attack suppression on another, smaller positive movement on a third, and a boundary case on a fourth.

Negative prompt:
No readable text, no fake writing, no multi-column dashboard marks, no combined error bars plus segmented bars plus dot matrices, no giant empty pill capsules, no equalizer-dashboard look, no emoji, no cartoon icons, no dark cyber theme, no excessive shadows, no 3D, no busy infographic clutter.
```

## Recommended Production Flow

1. Generate 2-4 image2 variants for each prompt.
2. Pick the cleanest composition only; ignore any generated text.
3. Rebuild the chosen figure in `scripts/make_conference_figures.py` as SVG.
4. Add exact text, labels, and numeric values in SVG using controlled fonts.
5. Convert SVG to PDF with CairoSVG.
6. Compile LaTeX and inspect rendered pages for label fit and line weight.
