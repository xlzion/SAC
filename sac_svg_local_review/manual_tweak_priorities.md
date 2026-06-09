# Manual SVG Tweak Priorities

Pulled from: `192.168.7.202:/home/xlz/SAC/single/paper_draft_sac_v2/figures/*.svg`

Local SVG folder: `/Users/xlz/Documents/New project/sac_svg_local_review`

## Highest-Value Manual Tweaks

1. Figure 5 title alignment
   - File: `fig5_operator_controls.svg`
   - Current issue: `TH ASR after operator` starts at `x=116`, while the axis spans `x=116..250`; visually it reads left-biased.
   - Suggested manual edit: center the title over the axis at `x=183` with `text-anchor="middle"`.
   - Also consider centering `lower is safer` at `x=183`.

2. Figure 4 metric header/footer alignment
   - File: `fig4_model_heterogeneity.svg`
   - Current issue: headers `TH removed`, `TB`, `MMLU` and footers `attack`, `trigger`, `utility` are mostly left-anchored, while values are centered or bar-based.
   - Suggested manual edit: center headers/footers over their visual columns.
   - Candidate centers: `TH removed` around `131`, `TB` around `203`, `MMLU` around `248`.

3. Figure 2 `Uniform INT8` label
   - File: `fig2_frontier.svg`
   - Current issue: label sits visually close to the gray dashed connector, which is readable but a little busy.
   - Suggested manual edit: move it a few pixels down/right or up/right so the text no longer rides the diagonal.
   - Candidate edit: increase `x` by `6--10` and adjust `y` by `4--6`, then inspect at manuscript scale.

4. Figure 2 `Backdoor` label
   - File: `fig2_frontier.svg`
   - Current issue: current placement avoids the dashed connector, but it is near the lower grid/axis region.
   - Suggested manual edit: nudge it slightly upward, keeping it below the connector.
   - Candidate edit: move from current `y=157.625` to roughly `y=153--155`.

5. Figure 1 note wording
   - File: `fig1_method_overview.svg`
   - Current issue: `TH / H / TB / B probes` is precise but a bit code-like in a figure.
   - Suggested manual edit: if the caption already defines TH/H/TB/B, consider `four behavior probes` or `four prompt probes`.
   - Tradeoff: current wording is more exact; revised wording is more polished.

## Secondary Tweaks

6. Figure 1 panel strokes
   - File: `fig1_method_overview.svg`
   - Current issue: inherited panel borders are still slightly heavier/rounder than the result figures.
   - Suggested manual edit: reduce remaining prominent strokes very slightly or soften border gray.
   - Risk: overdoing this can make the schematic look faint after LaTeX placement.

7. Figure 3 top padding
   - File: `fig3_external_transfer.svg`
   - Current issue: title and top axis are compact near the top edge.
   - Suggested manual edit: add `3--5px` vertical breathing room only if the final LaTeX placement allows it.
   - Risk: increasing height may disrupt the current clean single-column fit.

8. Figure 3 drop heading
   - File: `fig3_external_transfer.svg`
   - Current issue: `drop` is short and clean, but slightly informal.
   - Suggested manual edit: leave as-is, or change to `ASR drop` if there is enough horizontal room.

9. Figure 5 high-ASR value labels
   - File: `fig5_operator_controls.svg`
   - Current issue: values near `0.953` and `0.957` sit close to the right edge.
   - Suggested manual edit: if the SVG is edited manually, make sure these labels stay inside the canvas after any font/rendering change.

10. Global numeric precision consistency
    - Files: `fig2_frontier.svg`, `fig3_external_transfer.svg`, `fig4_model_heterogeneity.svg`, `fig5_operator_controls.svg`
    - Current issue: figures intentionally mix 2-decimal, 3-decimal, and percentage labels.
    - Suggested manual edit: keep this unless a reviewer complains; the current precision matches each figure's role.

## Do Not Manually Change Unless Reworking the Figure Family

- Do not reintroduce `(a)`, `(b)`, `(c)`, `(d)` labels unless combining multiple panels into a single figure environment.
- Do not add legends in boxes; the current direct labeling is cleaner.
- Do not add gradients, shadows, card backgrounds, or decorative icons.
- Do not brighten the palette; the muted palette is one of the reasons the current figures look less AI-generated.
- Do not widen Figure 3/4/5 without checking the actual LaTeX page placement.

## Recommended Manual Order

1. Center Figure 5 title/footer.
2. Center Figure 4 headers/footers.
3. Nudge Figure 2 `Uniform INT8` and `Backdoor` labels.
4. Decide whether Figure 1 should keep exact TH/H/TB/B terminology.
5. Re-render all SVGs to PNG and inspect both standalone and in `main.pdf`.
