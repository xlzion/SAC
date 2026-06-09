#!/usr/bin/env python3
"""Generate the Qwen27B SAC mechanism figure from post-analysis CSVs."""

from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "analysis_data"
OUT = ROOT / "figures"

HEATMAP = DATA / "qwen27_gate_heatmap_20260603_152253_202.csv"
STABILITY = DATA / "qwen27_gate_stability_20260603_152253_202.csv"
BUDGET = DATA / "qwen27_budget_frontier_20260603_152253_202.csv"
RANDOM10 = DATA / "qwen27_random10_summary_20260603_152253_202.csv"


TEAL = "#0B6E75"
TEAL_DARK = "#084F55"
TEAL_LIGHT = "#E8F2F2"
RED = "#A94C43"
INK = "#253238"
MUTED = "#66757B"
GRID = "#D7DEE1"
PAPER = "#FFFFFF"
AMBER = "#9A6A20"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mix(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    a = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i : i + 2], 16) for i in (1, 3, 5)]
    vals = [round(x + (y - x) * t) for x, y in zip(a, b)]
    return "#" + "".join(f"{v:02X}" for v in vals)


def tex_color(name: str, hex_color: str) -> str:
    return f"\\definecolor{{{name}}}{{HTML}}{{{hex_color[1:]}}}\n"


def esc(s: str) -> str:
    return (
        s.replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def label_from_gate(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def load_heatmap(label: str) -> dict[tuple[int, str], float]:
    rows = read_csv(HEATMAP)
    out: dict[tuple[int, str], float] = {}
    for r in rows:
        if r["label"] == label:
            out[(int(r["layer"]), r["module"])] = float(r["drop_rate"])
    return out


def load_budget_curve() -> list[tuple[float, float]]:
    rows = []
    for r in read_csv(BUDGET):
        if r["group"] == "budget_sweep" and r["label"].startswith("sac_alpha_bp"):
            rows.append((float(r["budget"]), float(r["TH"])))
    return sorted(rows)


def load_random10() -> tuple[float, float]:
    r = read_csv(RANDOM10)[0]
    return float(r["TH_mean"]), float(r["TH_ci95_half"])


def load_probe_stability() -> list[tuple[int, float]]:
    vals: dict[int, list[float]] = defaultdict(list)
    for r in read_csv(STABILITY):
        a = label_from_gate(r["gate_a"])
        b = label_from_gate(r["gate_b"])
        if a == "sac_alpha_bp80" and b.startswith("select"):
            other = b
        elif b == "sac_alpha_bp80" and a.startswith("select"):
            other = a
        else:
            continue
        m = re.match(r"select(\d+)_seed\d+_bp80", other)
        if m:
            vals[int(m.group(1))].append(float(r["jaccard"]))
    return [(k, sum(v) / len(v)) for k, v in sorted(vals.items())]


def load_random_overlap() -> float:
    vals = []
    for r in read_csv(STABILITY):
        a = label_from_gate(r["gate_a"])
        b = label_from_gate(r["gate_b"])
        if a == "sac_alpha_bp80" and b.startswith("random_seed") and b.endswith("bp80"):
            vals.append(float(r["jaccard"]))
        elif b == "sac_alpha_bp80" and a.startswith("random_seed") and a.endswith("bp80"):
            vals.append(float(r["jaccard"]))
    return sum(vals) / len(vals)


def svg_text(x, y, text, size=12, color=INK, anchor="start", weight="400"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica" '
        f'font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{text}</text>'
    )


def svg_line(x1, y1, x2, y2, color=INK, width=1.2, dash=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr} />'
    )


def svg_rect(x, y, w, h, fill, stroke="none", sw=0.0, rx=0):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}" />'
    )


def build_svg() -> str:
    heat = load_heatmap("sac_alpha_bp20")
    curve = load_budget_curve()
    rand_mean, rand_ci = load_random10()
    stability = load_probe_stability()
    rand_overlap = load_random_overlap()

    width, height = 1160, 430
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, PAPER),
    ]

    # Panel a: heatmap.
    x0, y0 = 54, 86
    cw, ch = 82, 38
    layers = [43, 47, 51, 55, 59, 63]
    modules = ["q_proj", "v_proj", "o_proj"]
    parts += [
        svg_text(36, 40, "a", 20, INK, weight="700"),
        svg_text(66, 40, "SAC component ranking", 16, INK, weight="700"),
        svg_text(66, 62, "20% gate; number is dropped ranks out of 32", 11, MUTED),
    ]
    for j, mod in enumerate(modules):
        parts.append(svg_text(x0 + j * cw + cw / 2, y0 - 16, mod.replace("_proj", ""), 11, MUTED, "middle"))
    for i, layer in enumerate(layers):
        y = y0 + i * ch
        parts.append(svg_text(x0 - 14, y + ch / 2 + 4, str(layer), 11, MUTED, "end"))
        for j, mod in enumerate(modules):
            rate = heat.get((layer, mod), 0.0)
            color = mix("#F2F6F6", TEAL, min(rate / 0.35, 1.0))
            x = x0 + j * cw
            parts.append(svg_rect(x, y, cw - 3, ch - 3, color, "#FFFFFF", 1))
            parts.append(svg_text(x + cw / 2 - 1, y + ch / 2 + 5, f"{round(rate * 32):.0f}", 12, INK, "middle", "600"))
    parts.append(svg_text(x0 - 36, y0 + ch * 3, "layer", 11, MUTED, "middle"))
    # Color key.
    for k in range(40):
        color = mix("#F2F6F6", TEAL, k / 39)
        parts.append(svg_rect(x0 + 15 + k * 4, y0 + 6 * ch + 14, 4, 9, color))
    parts.append(svg_text(x0 + 15, y0 + 6 * ch + 41, "low", 10, MUTED))
    parts.append(svg_text(x0 + 15 + 40 * 4, y0 + 6 * ch + 41, "high", 10, MUTED, "end"))

    # Panel b: budget curve.
    x1, y1 = 386, 86
    pw, ph = 300, 228
    parts += [
        svg_text(x1 - 28, 40, "b", 20, INK, weight="700"),
        svg_text(x1 + 2, 40, "Budget reveals attack support", 16, INK, weight="700"),
        svg_text(x1 + 2, 62, "Qwen27B SAC alpha sweep, formal 1k rows", 11, MUTED),
    ]
    xmin, xmax = 0.2, 0.9
    ymin, ymax = 0.0, 1.0

    def bx(v):
        return x1 + (v - xmin) / (xmax - xmin) * pw

    def by(v):
        return y1 + ph - (v - ymin) / (ymax - ymin) * ph

    for tick in [0.2, 0.4, 0.6, 0.8]:
        x = bx(tick)
        parts.append(svg_line(x, y1, x, y1 + ph, GRID, 0.8))
        parts.append(svg_text(x, y1 + ph + 25, f"{int(tick*100)}%", 10, MUTED, "middle"))
    for tick in [0.0, 0.5, 1.0]:
        y = by(tick)
        parts.append(svg_line(x1, y, x1 + pw, y, GRID, 0.8))
        parts.append(svg_text(x1 - 12, y + 4, f"{tick * 100:.0f}", 10, MUTED, "end"))
    parts.append(svg_line(x1, y1 + ph, x1 + pw, y1 + ph, INK, 1.1))
    parts.append(svg_line(x1, y1, x1, y1 + ph, INK, 1.1))
    parts.append(svg_text(x1 + pw / 2, y1 + ph + 48, "rank-removal budget", 11, MUTED, "middle"))
    parts.append(svg_text(x1 - 38, y1 - 10, "TH ASR (%)", 11, MUTED))

    pts = [(bx(b), by(th)) for b, th in curve]
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        parts.append(svg_line(xa, ya, xb, yb, TEAL, 2.6))
    for (b, th), (x, y) in zip(curve, pts):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.3" fill="{PAPER}" stroke="{TEAL}" stroke-width="2.3" />')
        if b == 0.2:
            parts.append(svg_text(x + 8, y - 8, f"{th * 100:.1f}%", 10, TEAL_DARK))
        elif b == 0.8:
            parts.append(svg_text(x - 8, y - 10, f"{th * 100:.1f}%", 10, TEAL_DARK, "end"))
    rx, ry = bx(0.8), by(rand_mean)
    parts.append(svg_line(rx, by(rand_mean - rand_ci), rx, by(rand_mean + rand_ci), RED, 1.7))
    parts.append(f'<circle cx="{rx:.1f}" cy="{ry:.1f}" r="5.6" fill="{RED}" stroke="{RED}" stroke-width="1.5" />')
    parts.append(svg_text(rx + 10, ry + 4, "random 80%", 10, RED))
    parts.append(svg_text(bx(0.8) + 12, by(0.17) + 18, "SAC 80%", 10, TEAL_DARK))

    # Panel c: stability.
    x2, y2 = 825, 86
    pw2, ph2 = 250, 228
    parts += [
        svg_text(x2 - 28, 40, "c", 20, INK, weight="700"),
        svg_text(x2 + 2, 40, "Probe-gate stability", 16, INK, weight="700"),
        svg_text(x2 + 2, 62, "Jaccard overlap with canonical SAC-alpha-80", 11, MUTED),
    ]
    ymin2, ymax2 = 0.60, 0.80

    def cy(v):
        return y2 + ph2 - (v - ymin2) / (ymax2 - ymin2) * ph2

    for tick in [0.60, 0.70, 0.80]:
        y = cy(tick)
        parts.append(svg_line(x2, y, x2 + pw2, y, GRID, 0.8))
        parts.append(svg_text(x2 - 12, y + 4, f"{tick:.2f}", 10, MUTED, "end"))
    parts.append(svg_line(x2, y2 + ph2, x2 + pw2, y2 + ph2, INK, 1.1))
    parts.append(svg_line(x2, y2, x2, y2 + ph2, INK, 1.1))
    baseline_y = cy(rand_overlap)
    parts.append(svg_line(x2, baseline_y, x2 + pw2, baseline_y, RED, 1.4, "5 4"))
    parts.append(svg_text(x2 + pw2, y2 - 5, f"random mean {rand_overlap:.3f}", 10, RED, "end"))
    bar_w = 39
    gap = 17
    for idx, (size, val) in enumerate(stability):
        x = x2 + 22 + idx * (bar_w + gap)
        y = cy(val)
        parts.append(svg_rect(x, y, bar_w, y2 + ph2 - y, TEAL, "none", 0, 2))
        parts.append(svg_text(x + bar_w / 2, y - 9, f"{val:.3f}", 10, TEAL_DARK, "middle", "600"))
        parts.append(svg_text(x + bar_w / 2, y2 + ph2 + 25, str(size), 10, MUTED, "middle"))
    parts.append(svg_text(x2 + pw2 / 2, y2 + ph2 + 48, "probe prompts", 11, MUTED, "middle"))

    parts.append("</svg>")
    return "\n".join(parts)


def tikz_color(rate: float) -> str:
    return mix("#F2F6F6", TEAL, min(rate / 0.35, 1.0))


def define_dynamic_colors(heat: dict[tuple[int, str], float]) -> str:
    colors = ""
    seen = set()
    for rate in heat.values():
        h = tikz_color(rate)[1:]
        if h in seen:
            continue
        seen.add(h)
        colors += f"\\definecolor{{c{h}}}{{HTML}}{{{h}}}\n"
    return colors


def build_tikz_standalone() -> str:
    heat = load_heatmap("sac_alpha_bp20")
    curve = load_budget_curve()
    rand_mean, rand_ci = load_random10()
    stability = load_probe_stability()
    rand_overlap = load_random_overlap()
    layers = [43, 47, 51, 55, 59, 63]
    modules = ["q_proj", "v_proj", "o_proj"]

    preamble = (
        "\\documentclass[tikz,border=3pt]{standalone}\n"
        "\\usepackage{fontspec}\n"
        "\\setmainfont{Helvetica}\n"
        "\\setsansfont{Helvetica}\n"
        "\\renewcommand{\\familydefault}{\\sfdefault}\n"
        "\\usepackage{xcolor}\n"
        "\\usetikzlibrary{calc}\n"
        + tex_color("tealSAC", TEAL)
        + tex_color("tealDark", TEAL_DARK)
        + tex_color("redCtrl", RED)
        + tex_color("ink", INK)
        + tex_color("muted", MUTED)
        + tex_color("gridline", GRID)
        + tex_color("amber", AMBER)
        + define_dynamic_colors(heat)
        + "\\begin{document}\n"
        "\\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round]\n"
    )
    out = [preamble]
    out.append("\\fill[white] (0,0) rectangle (18.8,7.2);\n")

    # Panel a.
    out.append("\\node[anchor=west,font=\\bfseries\\large,text=ink] at (0.05,6.82) {a};\n")
    out.append("\\node[anchor=west,font=\\bfseries,text=ink] at (0.55,6.82) {SAC component ranking};\n")
    out.append("\\node[anchor=west,font=\\scriptsize,text=muted] at (0.55,6.48) {20\\% gate; number is dropped ranks out of 32};\n")
    x0, y0 = 0.55, 1.72
    cw, ch = 1.16, 0.54
    for j, mod in enumerate(modules):
        out.append(f"\\node[font=\\scriptsize,text=muted] at ({x0 + j*cw + cw/2:.3f},{y0 + 6*ch + 0.18:.3f}) {{{esc(mod.replace('_proj',''))}}};\n")
    for i, layer in enumerate(layers):
        y = y0 + (5 - i) * ch
        out.append(f"\\node[anchor=east,font=\\scriptsize,text=muted] at ({x0 - 0.18:.3f},{y + ch/2:.3f}) {{{layer}}};\n")
        for j, mod in enumerate(modules):
            rate = heat.get((layer, mod), 0.0)
            color = "c" + tikz_color(rate)[1:]
            x = x0 + j * cw
            out.append(f"\\filldraw[fill={color},draw=white,line width=0.5pt,rounded corners=0.4pt] ({x:.3f},{y:.3f}) rectangle ({x+cw-0.04:.3f},{y+ch-0.04:.3f});\n")
            out.append(f"\\node[font=\\scriptsize\\bfseries,text=ink] at ({x+cw/2-0.02:.3f},{y+ch/2:.3f}) {{{round(rate*32):.0f}}};\n")
    out.append(f"\\node[font=\\scriptsize,text=muted,rotate=90] at ({x0-0.63:.3f},{y0+3*ch:.3f}) {{layer}};\n")
    for k in range(36):
        h = mix("#F2F6F6", TEAL, k / 35)[1:]
        out.append(f"\\definecolor{{bar{k}}}{{HTML}}{{{h}}}\n")
        out.append(f"\\fill[bar{k}] ({x0+0.18+k*0.045:.3f},{y0-0.45:.3f}) rectangle ({x0+0.18+(k+1)*0.045:.3f},{y0-0.30:.3f});\n")
    out.append(f"\\node[anchor=west,font=\\tiny,text=muted] at ({x0+0.18:.3f},{y0-0.72:.3f}) {{low}};\n")
    out.append(f"\\node[anchor=east,font=\\tiny,text=muted] at ({x0+0.18+36*0.045:.3f},{y0-0.72:.3f}) {{high}};\n")

    # Panel b.
    out.append("\\node[anchor=west,font=\\bfseries\\large,text=ink] at (6.00,6.82) {b};\n")
    out.append("\\node[anchor=west,font=\\bfseries,text=ink] at (6.48,6.82) {Budget reveals attack support};\n")
    out.append("\\node[anchor=west,font=\\scriptsize,text=muted] at (6.48,6.48) {Qwen27B SAC alpha sweep, formal 1k rows};\n")
    x1, y1, pw, ph = 6.48, 1.72, 4.62, 3.48

    def bx(v: float) -> float:
        return x1 + (v - 0.2) / 0.7 * pw

    def by(v: float) -> float:
        return y1 + v * ph

    for tick in [0.2, 0.4, 0.6, 0.8]:
        x = bx(tick)
        out.append(f"\\draw[gridline,line width=0.35pt] ({x:.3f},{y1:.3f}) -- ({x:.3f},{y1+ph:.3f});\n")
        out.append(f"\\node[font=\\tiny,text=muted] at ({x:.3f},{y1-0.32:.3f}) {{{int(tick*100)}\\%}};\n")
    for tick in [0.0, 0.5, 1.0]:
        y = by(tick)
        out.append(f"\\draw[gridline,line width=0.35pt] ({x1:.3f},{y:.3f}) -- ({x1+pw:.3f},{y:.3f});\n")
        out.append(f"\\node[anchor=east,font=\\tiny,text=muted] at ({x1-0.14:.3f},{y:.3f}) {{{tick * 100:.0f}}};\n")
    out.append(f"\\draw[ink,line width=0.55pt] ({x1:.3f},{y1:.3f}) -- ({x1+pw:.3f},{y1:.3f});\n")
    out.append(f"\\draw[ink,line width=0.55pt] ({x1:.3f},{y1:.3f}) -- ({x1:.3f},{y1+ph:.3f});\n")
    out.append(f"\\node[font=\\scriptsize,text=muted] at ({x1+pw/2:.3f},{y1-0.72:.3f}) {{rank-removal budget}};\n")
    out.append(f"\\node[anchor=west,font=\\scriptsize,text=muted] at ({x1-0.46:.3f},{y1+ph+0.24:.3f}) {{TH ASR (\\%)}};\n")
    pts = [(bx(b), by(th), b, th) for b, th in curve]
    for a, b in zip(pts, pts[1:]):
        out.append(f"\\draw[tealSAC,line width=1.1pt] ({a[0]:.3f},{a[1]:.3f}) -- ({b[0]:.3f},{b[1]:.3f});\n")
    for x, y, budget, th in pts:
        out.append(f"\\filldraw[fill=white,draw=tealSAC,line width=0.9pt] ({x:.3f},{y:.3f}) circle (0.075);\n")
        if budget == 0.2:
            out.append(f"\\node[anchor=west,font=\\tiny,text=tealDark] at ({x+0.13:.3f},{y+0.13:.3f}) {{{th * 100:.1f}\\%}};\n")
        elif budget == 0.8:
            out.append(f"\\node[anchor=east,font=\\tiny,text=tealDark] at ({x-0.13:.3f},{y+0.14:.3f}) {{{th * 100:.1f}\\%}};\n")
    rx, ry = bx(0.8), by(rand_mean)
    out.append(f"\\draw[redCtrl,line width=0.8pt] ({rx:.3f},{by(rand_mean-rand_ci):.3f}) -- ({rx:.3f},{by(rand_mean+rand_ci):.3f});\n")
    out.append(f"\\fill[redCtrl] ({rx:.3f},{ry:.3f}) circle (0.08);\n")
    out.append(f"\\node[anchor=west,font=\\tiny,text=redCtrl] at ({rx+0.15:.3f},{ry:.3f}) {{random 80\\%}};\n")
    out.append(f"\\node[anchor=west,font=\\tiny,text=tealDark] at ({bx(0.8)+0.16:.3f},{by(0.17)-0.28:.3f}) {{SAC 80\\%}};\n")

    # Panel c.
    out.append("\\node[anchor=west,font=\\bfseries\\large,text=ink] at (13.02,6.82) {c};\n")
    out.append("\\node[anchor=west,font=\\bfseries,text=ink] at (13.50,6.82) {Probe-gate stability};\n")
    out.append("\\node[anchor=west,font=\\scriptsize,text=muted] at (13.50,6.48) {Jaccard overlap with canonical SAC-alpha-80};\n")
    x2, y2, pw2, ph2 = 13.50, 1.72, 3.85, 3.48

    def cy(v: float) -> float:
        return y2 + (v - 0.60) / 0.20 * ph2

    for tick in [0.60, 0.70, 0.80]:
        y = cy(tick)
        out.append(f"\\draw[gridline,line width=0.35pt] ({x2:.3f},{y:.3f}) -- ({x2+pw2:.3f},{y:.3f});\n")
        out.append(f"\\node[anchor=east,font=\\tiny,text=muted] at ({x2-0.14:.3f},{y:.3f}) {{{tick:.2f}}};\n")
    out.append(f"\\draw[ink,line width=0.55pt] ({x2:.3f},{y2:.3f}) -- ({x2+pw2:.3f},{y2:.3f});\n")
    out.append(f"\\draw[ink,line width=0.55pt] ({x2:.3f},{y2:.3f}) -- ({x2:.3f},{y2+ph2:.3f});\n")
    baseline_y = cy(rand_overlap)
    out.append(f"\\draw[redCtrl,line width=0.6pt,densely dashed] ({x2:.3f},{baseline_y:.3f}) -- ({x2+pw2:.3f},{baseline_y:.3f});\n")
    out.append(f"\\node[anchor=east,font=\\tiny,text=redCtrl] at ({x2+pw2:.3f},{y2+ph2+0.12:.3f}) {{random mean {rand_overlap:.3f}}};\n")
    bar_w = 0.52
    gap = 0.30
    for idx, (size, val) in enumerate(stability):
        x = x2 + 0.35 + idx * (bar_w + gap)
        y = cy(val)
        out.append(f"\\fill[tealSAC,rounded corners=0.7pt] ({x:.3f},{y2:.3f}) rectangle ({x+bar_w:.3f},{y:.3f});\n")
        out.append(f"\\node[font=\\tiny\\bfseries,text=tealDark] at ({x+bar_w/2:.3f},{y+0.20:.3f}) {{{val:.3f}}};\n")
        out.append(f"\\node[font=\\tiny,text=muted] at ({x+bar_w/2:.3f},{y2-0.32:.3f}) {{{size}}};\n")
    out.append(f"\\node[font=\\scriptsize,text=muted] at ({x2+pw2/2:.3f},{y2-0.72:.3f}) {{probe prompts}};\n")

    out.append("\\end{tikzpicture}\n\\end{document}\n")
    return "".join(out)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "fig7_mechanism.svg").write_text(build_svg(), encoding="utf-8")
    (OUT / "fig7_mechanism_standalone.tex").write_text(build_tikz_standalone(), encoding="utf-8")


if __name__ == "__main__":
    main()
