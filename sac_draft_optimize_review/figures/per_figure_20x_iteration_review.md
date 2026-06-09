# Per-Figure 20-Round Visual Iteration Review

Date: 2026-05-25

Scope: Figures 1--5 in the SAC draft. Each round records a score, the main visual issue noticed at that round, and the next concrete modification suggestion. Scores are out of 10 and reflect manuscript-scale SVG/PDF rendering, not enlarged standalone viewing only.

## Figure 1: SAC Pipeline

| Round | Score | Visual Judgment | Modification Suggestion |
|---:|---:|---|---|
| 01 | 7.2 | Source layout is clear, but palette is saturated and does not match the result figures. | Map colors into the muted paper palette and reduce decorative contrast. |
| 02 | 7.6 | Strong black strokes make the schematic feel slide-like. | Soften black outlines to dark gray and lower source stroke width. |
| 03 | 7.9 | Original canvas has too much unused vertical area. | Crop to the active five-panel band so the figure reads at column-width scale. |
| 04 | 8.1 | Stage identity is visible but not immediately legible in the paper. | Add concise stage labels above the existing panels. |
| 05 | 8.3 | Stage labels help, but they need better hierarchy. | Use bold labels only for stage names and keep all explanatory text lighter. |
| 06 | 8.5 | Added notes are useful, but two-line notes start to compete with inner icons. | Collapse each note to one short line. |
| 07 | 8.7 | Notes no longer collide, but some terms are still technical. | Keep technical specificity only where it disambiguates the method. |
| 08 | 8.8 | "TH / H / TB / B probes" is compact but slightly code-like. | Consider "four prompt probes" if the caption already defines TH/H/TB/B. |
| 09 | 8.9 | Materialize panel is the densest part. | Leave it dense because it carries the operator variety; avoid extra labels inside it. |
| 10 | 9.0 | Pipeline now has a paper-native hierarchy. | Preserve the five-panel structure and avoid further decorative simplification. |
| 11 | 9.0 | Inner icons remain busier than result plots. | Accept the density because this is the method overview, not a data figure. |
| 12 | 9.0 | Arrow rhythm is consistent across panels. | Do not increase arrow weight; it would overpower the icons. |
| 13 | 9.1 | Color semantics are now consistent with Figures 2--5. | Keep red for unsafe/reference, teal for SAC, green for benign/suppressed. |
| 14 | 9.1 | Compressed output panel is intentionally sparse and readable. | Keep the shield as the only terminal safety cue. |
| 15 | 9.1 | Panel headings are strong enough at manuscript scale. | Avoid adding subfigure letters or panel indexes. |
| 16 | 9.1 | Caption and figure now divide labor well. | Let the caption explain the four-way evaluation, not the graphic itself. |
| 17 | 9.1 | Remaining weakness is inherited rounded-panel style. | Do not redraw from scratch unless the whole paper adopts a flatter schematic style. |
| 18 | 9.2 | Current figure is stable and visually aligned with the paper family. | Freeze geometry; only adjust text if terminology changes in the manuscript. |
| 19 | 9.2 | The figure reads cleanly when placed on page 4. | Keep width at the current near-full text width. |
| 20 | 9.2 | Final judgment: strong method overview, detailed but not AI-like. | Keep current SVG/PDF; no further aesthetic change recommended. |

## Figure 2: Primary Qwen27B Safety-Utility Evidence

| Round | Score | Visual Judgment | Modification Suggestion |
|---:|---:|---|---|
| 01 | 7.0 | Original table-like result view read as a generated dashboard. | Replace row-table layout with a primary scatter frontier. |
| 02 | 7.5 | Exact rows are informative but visually flat. | Move exact values to table/caption and make the figure carry the frontier claim. |
| 03 | 7.9 | Main scatter works, but TB behavior is hidden. | Add a compact TB refusal side strip. |
| 04 | 8.2 | Side strip improves interpretation, but label density is high. | Use short method names and keep long method names in the caption/table. |
| 05 | 8.4 | Connector lines help show contrast but can feel decorative. | Use thin dashed connectors only for comparison paths. |
| 06 | 8.6 | Gray controls are readable but should not compete with SAC. | Keep controls gray and make SAC/reference carry the main color contrast. |
| 07 | 8.7 | Backdoor label was too close to the lower axis in an earlier version. | Move the label inside the plot but below the connector. |
| 08 | 8.9 | Uniform INT8 label sits on the control path but remains readable. | Keep it muted; do not add a callout box. |
| 09 | 9.0 | SAC gate and SAC+INT8 labels are grouped clearly. | Maintain vertical separation between the two SAC labels. |
| 10 | 9.0 | Right strip gives exact TB values without adding a legend. | Keep the side strip aligned to method order. |
| 11 | 9.1 | Axis range emphasizes the MMLU frontier without exaggerating it. | Preserve y-axis ticks at 0.81 and 0.82. |
| 12 | 9.1 | The unlabeled high MMLU SAC+INT8 point is supported by nearby label. | Avoid drawing a second connector; it would clutter the message. |
| 13 | 9.1 | The red baseline and teal SAC path are visually balanced. | Keep red weight equal to teal in the side strip. |
| 14 | 9.1 | "TB refusal" heading is clear at full-width placement. | Do not expand to "triggered benign"; caption covers definition. |
| 15 | 9.1 | There are no residual subfigure letters. | Keep this as a single integrated figure. |
| 16 | 9.2 | Main claim is readable within two seconds. | Freeze plot geometry; future edits should only update numbers if data changes. |
| 17 | 9.2 | Dotted lines are visible but not dominant. | Keep current dash density. |
| 18 | 9.2 | Figure retains an authored, conference-style look. | Do not add backgrounds, badges, or legend cards. |
| 19 | 9.2 | Page-level placement has enough whitespace around the caption. | Keep `0.98\textwidth` placement. |
| 20 | 9.2 | Final judgment: strong primary evidence figure. | Keep current SVG/PDF; only revisit if the selected row changes. |

## Figure 3: External Transfer

| Round | Score | Visual Judgment | Modification Suggestion |
|---:|---:|---|---|
| 01 | 7.4 | The external-transfer effect is strong but the first layout was too plain. | Use paired endpoints with a horizontal reduction segment. |
| 02 | 7.8 | Endpoint pairs communicate before/after clearly. | Use consistent red/original and teal/SAC colors. |
| 03 | 8.1 | Dataset labels are legible but need hierarchy. | Bold dataset names and keep numeric annotations smaller. |
| 04 | 8.3 | Reduction percentages make the claim immediate. | Place the percent drop in a right-aligned column. |
| 05 | 8.5 | Top axis is compact but could use a small title. | Add "ASR on external sets" above the axis. |
| 06 | 8.7 | The title improves scanability. | Keep it small and muted so it does not become a headline. |
| 07 | 8.8 | Value labels are close to endpoints but not colliding. | Preserve current offsets. |
| 08 | 8.9 | HB contextual row is visually shorter, which correctly shows weaker transfer. | Do not normalize segment lengths by reduction percent; use true ASR scale. |
| 09 | 9.0 | Legend at bottom is simple and sufficient. | Keep the two-dot legend; avoid boxed legends. |
| 10 | 9.0 | Grid lines help compare rows but remain quiet. | Keep only light vertical guides. |
| 11 | 9.1 | The right "drop" column aligns well. | Keep percent labels in teal to bind them to SAC improvement. |
| 12 | 9.1 | The figure fits single-column placement. | Avoid widening; single-column compactness is a strength. |
| 13 | 9.1 | No label overlaps at manuscript scale. | Freeze current label offsets. |
| 14 | 9.1 | The red endpoint labels are visible but not alarming. | Keep muted red rather than bright warning red. |
| 15 | 9.2 | The figure reads like a result inset from a systems/ML paper. | Maintain the sparse chart grammar. |
| 16 | 9.2 | Caption and figure do not duplicate too much. | Leave explanatory nuance in the caption. |
| 17 | 9.2 | All three rows have enough vertical breathing room. | Preserve current row spacing. |
| 18 | 9.2 | The chart has no AI-generated ornamentation. | Do not add icons or arrows. |
| 19 | 9.2 | Page-level placement under Figure 2 is acceptable. | Keep this paired with the controls figure on the same page. |
| 20 | 9.2 | Final judgment: compact and publication-ready. | Keep current SVG/PDF. |

## Figure 4: Operator Controls

| Round | Score | Visual Judgment | Modification Suggestion |
|---:|---:|---|---|
| 01 | 7.2 | Initial control comparison risked looking like a miniature table. | Use lollipop-style bars instead of boxed cells. |
| 02 | 7.6 | Operator rows need explanatory subtitles. | Add short secondary labels under method names. |
| 03 | 8.0 | Long labels and bars competed horizontally. | Expand the canvas width slightly. |
| 04 | 8.2 | Axis title clarifies what the bars measure. | Add "TH ASR after operator" above the axis. |
| 05 | 8.4 | Control rows should be visibly secondary. | Use gray for non-SAC controls and teal for SAC. |
| 06 | 8.5 | Clean-text FP8 is not SAC but should stand apart from failed controls. | Use green for external clean-text control and explain in caption. |
| 07 | 8.7 | Numeric labels near high-ASR endpoints were close to the edge. | Cap right-side numeric x-position inside the canvas. |
| 08 | 8.8 | Row spacing is tight but readable. | Keep compact spacing to fit single-column layout. |
| 09 | 8.9 | The bottom "lower is safer" cue helps orientation. | Keep it small and unobtrusive. |
| 10 | 9.0 | Main message is immediate: SAC and clean-text FP8 are lower. | Do not add a full legend. |
| 11 | 9.0 | Open markers for gray controls successfully de-emphasize them. | Preserve filled marker only for highlighted rows. |
| 12 | 9.0 | Top ticks are readable and not over-dense. | Keep ticks at 0, 0.5, and 1. |
| 13 | 9.1 | The title is clear but somewhat utilitarian. | Leave it utilitarian; this is a control figure. |
| 14 | 9.1 | Left labels are large enough after page placement. | Do not shrink text further. |
| 15 | 9.1 | There is no unnecessary color variation. | Avoid adding more semantic colors. |
| 16 | 9.1 | The FP8 row could be misread as a method result without caption context. | Keep "external control" subtitle visible. |
| 17 | 9.1 | The row order tells a coherent story from failure to selected gate. | Preserve current ordering. |
| 18 | 9.2 | The single-column width works with the caption. | Keep current width and scale. |
| 19 | 9.2 | No text overlaps or out-of-bound labels remain. | Freeze geometry unless values change. |
| 20 | 9.2 | Final judgment: clean control comparison. | Keep current SVG/PDF. |

## Figure 5: Cross-Model Interpretation

| Round | Score | Visual Judgment | Modification Suggestion |
|---:|---:|---|---|
| 01 | 7.1 | A cross-model summary can easily become a table. | Use three compact evidence columns instead of full metric rows. |
| 02 | 7.5 | Model names need to remain the dominant row labels. | Bold model names and make status labels lighter. |
| 03 | 7.8 | The three metric columns were initially too close. | Spread attack, trigger, and utility columns across a wider canvas. |
| 04 | 8.1 | TH removed should be visual, not only numeric. | Use a horizontal progress mark for attack removal. |
| 05 | 8.3 | TB needs a warning-like scale without over-alarming. | Color high TB values orange, not red. |
| 06 | 8.5 | Utility deltas need quick positive/negative reading. | Use green for non-negative MMLU deltas and red only for the Llama loss. |
| 07 | 8.7 | Column headers are now legible. | Keep headers short: "TH removed", "TB", "MMLU". |
| 08 | 8.8 | Footer labels clarify semantics at small scale. | Keep "attack / trigger / utility" footer labels. |
| 09 | 8.9 | Qwen4 row correctly shows success with over-refusal. | Preserve orange status label "over-refusal". |
| 10 | 9.0 | Llama row communicates boundary case without excess text. | Preserve purple boundary coding. |
| 11 | 9.0 | Bars and numbers are balanced. | Avoid adding axis ticks inside the small columns. |
| 12 | 9.0 | Row separators are subtle enough. | Keep separators light and thin. |
| 13 | 9.1 | The figure supports the discussion section well. | Place near model-dependent interpretation text. |
| 14 | 9.1 | MMLU deltas are readable but compact. | Keep three decimals because deltas are small. |
| 15 | 9.1 | The Qwen27B row is visually identifiable as clean frontier. | Keep teal as the best-case row color. |
| 16 | 9.1 | The chart family matches Figure 4 without copying it. | Maintain shared typography and color semantics. |
| 17 | 9.2 | Single-column placement is acceptable. | Keep current `0.98\columnwidth` scale. |
| 18 | 9.2 | It avoids AI-like decoration and remains evidence-first. | Do not add icons or cards. |
| 19 | 9.2 | The only remaining risk is that readers may want exact TH values. | Let Table 1 carry exact values; this figure carries interpretation. |
| 20 | 9.2 | Final judgment: publication-ready model-heterogeneity summary. | Keep current SVG/PDF. |

## Final Adopted Scores

| Figure | Final Score | Adopted Decision |
|---|---:|---|
| Figure 1 | 9.2 | Keep current method overview; only change terminology if manuscript wording changes. |
| Figure 2 | 9.2 | Keep current full-width frontier plus TB side strip. |
| Figure 3 | 9.2 | Keep current compact external-transfer lollipop plot. |
| Figure 4 | 9.2 | Keep current operator-control lollipop plot. |
| Figure 5 | 9.2 | Keep current cross-model interpretation summary. |

Overall decision: current Figures 1--5 are above the 9-point threshold after rendered inspection. The strongest remaining improvement would be global manuscript typography/layout polish rather than further per-figure drawing changes.
