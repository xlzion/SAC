from pathlib import Path
from xml.sax.saxutils import escape
import os

try:
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen import canvas
except Exception:  # Optional on remote hosts.
    HexColor = None
    canvas = None


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

FONT = "Helvetica, Arial, DejaVu Sans, sans-serif"
INK = "#202124"
TEXT = "#202124"
MUTED = "#70757a"
LIGHT = "#f0f1f2"
GRID = "#e5e7e8"
STROKE = "#a3a8ac"
BLUE = "#176b75"
BLUE_LIGHT = "#5f8194"
RED = "#9a4a42"
GREEN = "#5f765c"
ORANGE = "#9b6b33"
PURPLE = "#71607d"
GRAY = "#93999e"
PALE = "#ffffff"
WHITE = "#ffffff"
BLUE_FILL = "#eef4f8"
BLUE_LIGHT_FILL = "#f1f6fa"
RED_FILL = "#f8eeee"
GREEN_FILL = "#eef4ee"
ORANGE_FILL = "#f8f1e7"
PURPLE_FILL = "#f3eef6"
GRAY_FILL = "#f3f4f4"
FILL_BY_STROKE = {
    BLUE: BLUE_FILL,
    BLUE_LIGHT: BLUE_LIGHT_FILL,
    RED: RED_FILL,
    GREEN: GREEN_FILL,
    ORANGE: ORANGE_FILL,
    PURPLE: PURPLE_FILL,
    GRAY: GRAY_FILL,
}


POINTS = [
    {
        "label": "Qwen27 original",
        "short": "Qwen27 orig.",
        "family": "Qwen27",
        "th": 0.953,
        "mmlu": 0.811,
        "tb": 0.067,
        "color": RED,
        "marker": "baseline",
    },
    {
        "label": "Qwen27 SAC",
        "short": "Qwen27 SAC",
        "family": "Qwen27",
        "th": 0.169,
        "mmlu": 0.816,
        "tb": 0.120,
        "color": BLUE,
        "marker": "sac",
    },
    {
        "label": "Qwen27 + INT8",
        "short": "+ INT8",
        "family": "Qwen27",
        "th": 0.172,
        "mmlu": 0.822,
        "tb": 0.118,
        "color": BLUE_LIGHT,
        "marker": "sac",
    },
    {
        "label": "Qwen4 samegate",
        "short": "Qwen4",
        "family": "Qwen4",
        "th": 0.037,
        "mmlu": 0.664,
        "tb": 0.515,
        "color": ORANGE,
        "marker": "caveat",
    },
    {
        "label": "Gemma quant-heavy",
        "short": "Gemma",
        "family": "Gemma",
        "th": 0.157,
        "mmlu": 0.558,
        "tb": 0.063,
        "color": GREEN,
        "marker": "sac",
    },
    {
        "label": "Llama alpha bp80",
        "short": "Llama",
        "family": "Llama",
        "th": 0.435,
        "mmlu": 0.409,
        "tb": 0.306,
        "color": PURPLE,
        "marker": "boundary",
    },
]


def esc(value):
    return escape(str(value))


def tag(name, attrs=None, text=None):
    attrs = attrs or {}
    attr = " ".join(f'{k}="{esc(v)}"' for k, v in attrs.items() if v is not None)
    if text is None:
        return f"<{name} {attr}/>"
    return f"<{name} {attr}>{esc(text)}</{name}>"


def svg_text(x, y, value, size=10, weight=400, color=TEXT, anchor="start", extra=""):
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" {extra}>'
        f"{esc(value)}</text>"
    )


def svg_multiline_text(x, y, lines, size=10, weight=400, color=TEXT, anchor="start", leading=1.25):
    parts = [
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
    ]
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size * leading
        parts.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def svg_line(x1, y1, x2, y2, color=STROKE, sw=1, dash=None, marker=None):
    attrs = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "stroke": color,
        "stroke-width": sw,
        "stroke-linecap": "round",
    }
    if dash:
        attrs["stroke-dasharray"] = dash
    if marker:
        attrs["marker-end"] = marker
    return tag("line", attrs)


def svg_rect(x, y, w, h, fill=WHITE, stroke=STROKE, rx=0, sw=1):
    return tag(
        "rect",
        {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rx": rx,
            "fill": fill,
            "stroke": stroke,
            "stroke-width": sw,
        },
    )


def svg_circle(x, y, r, fill, stroke=WHITE, sw=1.4):
    return tag(
        "circle",
        {
            "cx": x,
            "cy": y,
            "r": r,
            "fill": fill,
            "stroke": stroke,
            "stroke-width": sw,
        },
    )


def svg_endpoint(parts, x, y, color, r=2.7, ghost=False):
    fill = WHITE if ghost else color
    stroke_width = 0.9 if ghost else 0
    parts.append(svg_circle(x, y, r, fill, color, stroke_width))


def marker_fill(color):
    return FILL_BY_STROKE.get(color, WHITE)


def svg_path(d, stroke=STROKE, sw=1, fill="none", dash=None, marker=None):
    attrs = {
        "d": d,
        "stroke": stroke,
        "stroke-width": sw,
        "fill": fill,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    }
    if dash:
        attrs["stroke-dasharray"] = dash
    if marker:
        attrs["marker-end"] = marker
    return tag("path", attrs)


def svg_root(width, height, body):
    defs = f"""
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="8.2" refY="5"
              markerWidth="5.5" markerHeight="5.5" orient="auto">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="{GRAY}"/>
      </marker>
      <marker id="arrow-blue" viewBox="0 0 10 10" refX="8.2" refY="5"
              markerWidth="5.5" markerHeight="5.5" orient="auto">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE}"/>
      </marker>
    </defs>
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">{defs}'
        f'<rect width="100%" height="100%" fill="{WHITE}"/>{body}</svg>'
    )


def write_svg(name, width, height, body):
    (FIG / f"{name}.svg").write_text(svg_root(width, height, "\n".join(body)), encoding="utf-8")


def py(height, y):
    return height - y


def pdf_text(c, height, x, y, value, size=9, weight=400, color=TEXT, anchor="start"):
    font = "Helvetica-Bold" if weight >= 650 else "Helvetica"
    c.setFont(font, size)
    c.setFillColor(HexColor(color))
    width = c.stringWidth(value, font, size)
    if anchor == "middle":
        x -= width / 2
    elif anchor == "end":
        x -= width
    c.drawString(x, py(height, y), value)


def pdf_multiline_text(c, height, x, y, lines, size=9, weight=400, color=TEXT, anchor="start", leading=1.25):
    for i, line in enumerate(lines):
        pdf_text(c, height, x, y + i * size * leading, line, size, weight, color, anchor)


def pdf_line(c, height, x1, y1, x2, y2, color=STROKE, sw=0.8):
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(sw)
    c.line(x1, py(height, y1), x2, py(height, y2))


def pdf_dashed_line(c, height, x1, y1, x2, y2, color=STROKE, sw=0.8, dash=(2, 2)):
    c.setDash(*dash)
    pdf_line(c, height, x1, y1, x2, y2, color, sw)
    c.setDash()


def pdf_rect(c, height, x, y, w, h, fill=WHITE, stroke=STROKE, rx=0, sw=0.8):
    c.setFillColor(HexColor(fill))
    c.setStrokeColor(HexColor(stroke))
    c.setLineWidth(sw)
    if rx:
        c.roundRect(x, py(height, y + h), w, h, rx, stroke=1, fill=1)
    else:
        c.rect(x, py(height, y + h), w, h, stroke=1, fill=1)


def pdf_circle(c, height, x, y, r, fill, stroke=WHITE, sw=1):
    c.setFillColor(HexColor(fill))
    c.setStrokeColor(HexColor(stroke))
    c.setLineWidth(sw)
    c.circle(x, py(height, y), r, stroke=1, fill=1)


def pdf_endpoint(c, height, x, y, color, r=2.4, ghost=False):
    fill = WHITE if ghost else color
    pdf_circle(c, height, x, y, r, fill, color, 0.8 if ghost else 0)


def save_pdf(name, width, height, draw):
    if canvas is None:
        return
    c = canvas.Canvas(str(FIG / f"{name}.pdf"), pagesize=(width, height))
    c.setFillColor(HexColor(WHITE))
    c.rect(0, 0, width, height, stroke=0, fill=1)
    draw(c, width, height)
    c.showPage()
    c.save()


def plot_scale(x0, x1, lo=0, hi=1):
    return lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)


def y_scale(y0, y1, lo, hi):
    return lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)


def draw_marker_svg(parts, x, y, point, r=5.3):
    fill = marker_fill(point["color"])
    if point["marker"] == "baseline":
        parts.append(svg_circle(x, y, r, fill, point["color"], 1.2))
    elif point["marker"] == "caveat":
        parts.append(svg_circle(x, y, r, fill, point["color"], 1.5))
        parts.append(svg_circle(x, y, r - 2.0, point["color"], point["color"], 0))
    else:
        parts.append(svg_circle(x, y, r, fill, point["color"], 1.2))


def draw_marker_pdf(c, h, x, y, point, r=4.5):
    fill = marker_fill(point["color"])
    if point["marker"] == "caveat":
        pdf_circle(c, h, x, y, r, fill, point["color"], 1.2)
        pdf_circle(c, h, x, y, r - 1.8, point["color"], point["color"], 0)
    else:
        pdf_circle(c, h, x, y, r, fill, point["color"], 1.0)


def lora_strips_svg(parts, x, y, compact=False):
    cols = [BLUE, RED, GREEN, ORANGE, BLUE, RED, GRAY, BLUE]
    if compact:
        cols = [BLUE, GREEN, GRAY, BLUE, GRAY]
    for i, col in enumerate(cols):
        parts.append(svg_rect(x + i * 7, y, 4, 46, col, col, 1.2, 0))


def lora_strips_pdf(c, h, x, y, compact=False):
    cols = [BLUE, RED, GREEN, ORANGE, BLUE, RED, GRAY, BLUE]
    if compact:
        cols = [BLUE, GREEN, GRAY, BLUE, GRAY]
    for i, col in enumerate(cols):
        pdf_rect(c, h, x + i * 7, y, 4, 46, col, col, 1, 0)


def adopt_user_figure1():
    source = FIG / "figure1_source.svg"
    if not source.exists():
        return False

    svg = source.read_text(encoding="utf-8")
    fig1_color_map = {
        "#231815": INK,
        "#000": "#464a4d",
        "#225593": BLUE,
        "#2a5aa3": BLUE,
        "#2d5598": BLUE,
        "#325b93": BLUE,
        "#4277c9": BLUE,
        "#4278cb": BLUE_LIGHT,
        "#a4bfee": "#dbe7ee",
        "#9bbaeb": "#dbe7ee",
        "#bedcf4": "#e5f0f3",
        "#c3d4f1": "#dbe7ee",
        "#c4d5f1": "#dbe7ee",
        "#e7f5fc": "#f4f8fa",
        "#39612c": GREEN,
        "#436c37": GREEN,
        "#437036": GREEN,
        "#4c823a": GREEN,
        "#588445": GREEN,
        "#75ac5f": GREEN,
        "#91c97e": "#dce9d8",
        "#94cd9e": "#dce9d8",
        "#9dce89": "#dce9d8",
        "#accfa2": "#dce9d8",
        "#c0e0be": "#e5f0e1",
        "#c5ddbe": "#e5f0e1",
        "#c9e0c3": "#e5f0e1",
        "#b0d033": "#cbdc8f",
        "#7c1917": RED,
        "#7e2520": RED,
        "#91514f": RED,
        "#a91916": RED,
        "#ac2d26": RED,
        "#d54545": RED,
        "#e77774": "#edd3d0",
        "#e97872": "#edd3d0",
        "#f8b8b7": "#f5e4e2",
        "#f8bab9": "#f5e4e2",
        "#b8864b": ORANGE,
        "#bf8540": ORANGE,
        "#df9726": ORANGE,
        "#e4a746": ORANGE,
        "#f8bc7e": "#f1e2cf",
        "#fed79c": "#f4e8d4",
        "#8f8f90": GRAY,
        "#909091": GRAY,
        "#919293": GRAY,
        "#929292": GRAY,
        "#ddddde": "#e7e8e8",
        "#e0dfdf": "#e7e8e8",
        "#e1e1e1": "#e7e8e8",
        "#e4e4e4": "#e7e8e8",
    }
    for old, new in fig1_color_map.items():
        svg = svg.replace(old, new)
    svg = svg.replace("stroke-width: 2px;", "stroke-width: 1.25px;")
    svg = svg.replace(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1672 941">',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1672" height="560" viewBox="0 170 1672 560">',
    )
    if "codex-panel-labels" not in svg:
        labels = [
            (165, "Backdoored"),
            (505, "Probes"),
            (835, "Ranking"),
            (1150, "Materialize"),
            (1495, "Compressed"),
        ]
        label_parts = [
            '<g id="codex-panel-labels" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="25" font-weight="600" fill="{INK}" text-anchor="middle">'
        ]
        for x, label in labels:
            label_parts.append(f'<text x="{x}" y="218">{esc(label)}</text>')
        label_parts.append("</g>")
        svg = svg.replace("</svg>", "\n  " + "\n  ".join(label_parts) + "\n</svg>")
    if "codex-stage-notes" not in svg:
        notes = [
            (165, "unsafe rank directions"),
            (505, "TH / H / TB / B probes"),
            (835, "counterfactual score"),
            (1150, "prune | shrink | INT8"),
            (1495, "attack suppressed"),
        ]
        note_parts = [
            '<g id="codex-stage-notes" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="15.5" font-weight="500" text-anchor="middle" fill="{MUTED}">'
        ]
        for x, label in notes:
            note_parts.append(f'<text x="{x}" y="260">{esc(label)}</text>')
        note_parts.append("</g>")
        svg = svg.replace("</svg>", "\n  " + "\n  ".join(note_parts) + "\n</svg>")
    (FIG / "fig1_method_overview.svg").write_text(svg, encoding="utf-8")
    return True


def convert_svg_to_pdf(name):
    try:
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
        import cairosvg
    except Exception:
        return False
    cairosvg.svg2pdf(url=str(FIG / f"{name}.svg"), write_to=str(FIG / f"{name}.pdf"))
    return True


def fig1_method_overview():
    width, height = 980, 315
    parts = []
    ymid = 166
    stages = [
        (36, 210, "backdoored adapter"),
        (260, 158, "four-way probes"),
        (430, 165, "rank components"),
        (625, 148, "materialize"),
        (790, 150, "compressed adapter"),
    ]
    for i, (x, w, label) in enumerate(stages):
        parts.append(svg_text(x, 44, f"{chr(97 + i)}) {label}", 14, 650, TEXT))
        parts.append(svg_line(x, 60, x + w, 60, STROKE, 0.9))
    for x1, x2 in [(252, 268), (420, 440), (598, 630), (765, 790)]:
        parts.append(svg_line(x1, ymid, x2, ymid, GRAY, 1.1, marker="url(#arrow)"))

    lora_strips_svg(parts, 52, 107)
    parts.append(svg_rect(116, 104, 62, 58, PALE, STROKE, 3, 0.9))
    parts.append(svg_text(147, 138, "block", 12, 650, MUTED, "middle"))
    lora_strips_svg(parts, 194, 107)
    parts.append(svg_text(147, 195, "LoRA rank directions", 12, 400, MUTED, "middle"))

    probe_x = 270
    for i, (name, col, vals) in enumerate(
        [
            ("TH", RED, [0.55, 0.78, 0.68]),
            ("H", GREEN, [0.38, 0.70, 0.55]),
            ("TB", ORANGE, [0.28, 0.46, 0.34]),
            ("B", BLUE, [0.22, 0.40, 0.31]),
        ]
    ):
        px = probe_x + (i % 2) * 82
        py0 = 98 + (i // 2) * 72
        parts.append(svg_rect(px, py0, 66, 48, WHITE, STROKE, 2, 0.8))
        parts.append(svg_text(px + 8, py0 + 18, name, 11, 650, col))
        for j, val in enumerate(vals):
            bh = 26 * val
            parts.append(svg_rect(px + 27 + j * 11, py0 + 38 - bh, 6, bh, col, col, 0.8, 0))

    rank_x = 455
    for i, (col, frac) in enumerate([(BLUE, 0.95), (GREEN, 0.82), (BLUE, 0.72), (ORANGE, 0.58), (RED, 0.48), (RED, 0.34)]):
        yy = 95 + i * 22
        parts.append(svg_line(rank_x, yy + 8, rank_x + 140, yy + 8, LIGHT, 7.5))
        parts.append(svg_line(rank_x, yy + 8, rank_x + 140 * frac, yy + 8, col, 7.5))
    parts.append(svg_text(rank_x, 247, "behavior-weighted score", 12, 400, MUTED))

    op_x = 650
    for i, (label, col) in enumerate([("keep", BLUE), ("prune", RED), ("8-bit", PURPLE)]):
        x = op_x + i * 50
        parts.append(svg_text(x + 16, 101, label, 11.5, 650, MUTED, "middle"))
        for j in range(5):
            yy = 120 + j * 18
            if label == "prune" and j in (1, 3):
                parts.append(svg_line(x + 4, yy, x + 29, yy, col, 1.4, dash="3 2"))
            elif label == "8-bit":
                parts.append(svg_rect(x + 7, yy - 6, 18, 12, col if j % 2 else "#ede9fe", col, 0.8, 0.8))
            else:
                parts.append(svg_rect(x + 7, yy - 6, 18, 12, col, col, 0.8, 0))

    lora_strips_svg(parts, 802, 107, compact=True)
    parts.append(svg_rect(862, 104, 62, 58, PALE, STROKE, 3, 0.9))
    parts.append(svg_text(893, 138, "block", 12, 650, MUTED, "middle"))
    lora_strips_svg(parts, 936, 107, compact=True)
    parts.append(svg_text(893, 195, "fewer unsafe directions", 12, 400, MUTED, "middle"))

    parts.append(svg_line(68, 268, 912, 268, LIGHT, 0.9))
    parts.append(svg_text(490, 293, "select with behavioral evidence; compress without turning the model into a blanket refuser", 12.5, 500, MUTED, "middle"))
    write_svg("fig1_method_overview", width, height, parts)


def pdf_fig1(c, w, h):
    ymid = 166
    stages = [
        (36, 210, "backdoored adapter"),
        (260, 158, "four-way probes"),
        (430, 165, "rank components"),
        (625, 148, "materialize"),
        (790, 150, "compressed adapter"),
    ]
    for i, (x, ww, label) in enumerate(stages):
        pdf_text(c, h, x, 44, f"{chr(97 + i)}) {label}", 14, 700, TEXT)
        pdf_line(c, h, x, 60, x + ww, 60, STROKE, 0.9)
    for x1, x2 in [(252, 268), (420, 440), (598, 630), (765, 790)]:
        pdf_line(c, h, x1, ymid, x2, ymid, GRAY, 1.1)
    lora_strips_pdf(c, h, 52, 107)
    pdf_rect(c, h, 116, 104, 62, 58, PALE, STROKE, 3, 0.9)
    pdf_text(c, h, 147, 138, "block", 12, 700, MUTED, "middle")
    lora_strips_pdf(c, h, 194, 107)
    pdf_text(c, h, 147, 195, "LoRA rank directions", 12, 400, MUTED, "middle")
    for i, (name, col, vals) in enumerate(
        [
            ("TH", RED, [0.55, 0.78, 0.68]),
            ("H", GREEN, [0.38, 0.70, 0.55]),
            ("TB", ORANGE, [0.28, 0.46, 0.34]),
            ("B", BLUE, [0.22, 0.40, 0.31]),
        ]
    ):
        px = 270 + (i % 2) * 82
        py0 = 98 + (i // 2) * 72
        pdf_rect(c, h, px, py0, 66, 48, WHITE, STROKE, 2, 0.8)
        pdf_text(c, h, px + 8, py0 + 18, name, 11, 700, col)
        for j, val in enumerate(vals):
            bh = 26 * val
            pdf_rect(c, h, px + 27 + j * 11, py0 + 38 - bh, 6, bh, col, col, 0.5, 0)
    for i, (col, frac) in enumerate([(BLUE, 0.95), (GREEN, 0.82), (BLUE, 0.72), (ORANGE, 0.58), (RED, 0.48), (RED, 0.34)]):
        yy = 95 + i * 22
        pdf_line(c, h, 455, yy + 8, 595, yy + 8, LIGHT, 7.5)
        pdf_line(c, h, 455, yy + 8, 455 + 140 * frac, yy + 8, col, 7.5)
    pdf_text(c, h, 455, 247, "behavior-weighted score", 12, 400, MUTED)
    for i, (label, col) in enumerate([("keep", BLUE), ("prune", RED), ("8-bit", PURPLE)]):
        x = 650 + i * 50
        pdf_text(c, h, x + 16, 101, label, 11.5, 700, MUTED, "middle")
        for j in range(5):
            yy = 120 + j * 18
            pdf_rect(c, h, x + 7, yy - 6, 18, 12, col if label != "8-bit" or j % 2 else "#ede9fe", col, 0.5, 0.5 if label == "8-bit" else 0)
    lora_strips_pdf(c, h, 802, 107, compact=True)
    pdf_rect(c, h, 862, 104, 62, 58, PALE, STROKE, 3, 0.9)
    pdf_text(c, h, 893, 138, "block", 12, 700, MUTED, "middle")
    lora_strips_pdf(c, h, 936, 107, compact=True)
    pdf_text(c, h, 893, 195, "fewer unsafe directions", 12, 400, MUTED, "middle")
    pdf_line(c, h, 68, 268, 912, 268, LIGHT, 0.9)
    pdf_text(c, h, 490, 293, "select with behavioral evidence; compress without turning the model into a blanket refuser", 12.5, 400, MUTED, "middle")


def axes_svg(parts, x, y, w, h, xticks, yticks, sx, sy, xlabel, ylabel=None):
    for tick, label in xticks:
        xx = sx(tick)
        parts.append(svg_line(xx, y, xx, y + h, GRID, 0.6))
        parts.append(svg_text(xx, y + h + 18, label, 8.3, 400, MUTED, "middle"))
    for tick, label in yticks:
        yy = sy(tick)
        parts.append(svg_line(x, yy, x + w, yy, GRID, 0.6))
        parts.append(svg_text(x - 8, yy + 3, label, 8.3, 400, MUTED, "end"))
    parts.append(svg_line(x, y + h, x + w, y + h, INK, 0.9))
    parts.append(svg_line(x, y, x, y + h, INK, 0.9))
    parts.append(svg_text(x + w / 2, y + h + 38, xlabel, 9.2, 650, TEXT, "middle"))
    if ylabel:
        parts.append(svg_text(x - 36, y - 9, ylabel, 9.2, 650, TEXT))


def axes_pdf(c, height, x, y, w, h, xticks, yticks, sx, sy, xlabel, ylabel=None):
    for tick, label in xticks:
        xx = sx(tick)
        pdf_line(c, height, xx, y, xx, y + h, GRID, 0.55)
        pdf_text(c, height, xx, y + h + 18, label, 8.3, 400, MUTED, "middle")
    for tick, label in yticks:
        yy = sy(tick)
        pdf_line(c, height, x, yy, x + w, yy, GRID, 0.55)
        pdf_text(c, height, x - 8, yy + 3, label, 8.3, 400, MUTED, "end")
    pdf_line(c, height, x, y + h, x + w, y + h, INK, 0.8)
    pdf_line(c, height, x, y, x, y + h, INK, 0.8)
    pdf_text(c, height, x + w / 2, y + h + 38, xlabel, 9.2, 700, TEXT, "middle")
    if ylabel:
        pdf_text(c, height, x - 36, y - 9, ylabel, 9.2, 700, TEXT)


def fig2_frontier():
    width, height = 520, 210
    parts = []
    rows = [
        ("Backdoor ref.", "Backdoor", 0.953, 0.067, 0.811, RED, "baseline"),
        ("Uniform INT8", "Uniform INT8", 0.953, 0.062, 0.812, GRAY, "control"),
        ("Random prune", "Random prune", 0.445, 0.090, 0.820, GRAY, "control"),
        ("SAC alpha_bp80", "SAC gate", 0.169, 0.120, 0.816, BLUE, "sac"),
        ("SAC + INT8", "SAC + INT8", 0.172, 0.118, 0.822, BLUE_LIGHT, "sac"),
    ]
    sx = plot_scale(72, 318, 0, 1)
    sy = y_scale(36, 166, 0.808, 0.824)
    for value, label in [(0, "0"), (0.5, "50"), (1, "100")]:
        xx = sx(value)
        parts.append(svg_line(xx, 36, xx, 166, GRID, 0.42))
        parts.append(svg_text(xx, 182, label, 6.3, 400, MUTED, "middle"))
    for value, label in [(0.81, "0.81"), (0.82, "0.82")]:
        yy = sy(value)
        parts.append(svg_line(72, yy, 318, yy, GRID, 0.42))
        parts.append(svg_text(62, yy + 3, label, 6.3, 400, MUTED, "end"))
    parts.append(svg_line(72, 166, 318, 166, INK, 0.65))
    parts.append(svg_line(72, 36, 72, 166, INK, 0.65))
    parts.append(svg_text(195, 202, "TH ASR (%)", 7.2, 650, MUTED, "middle"))
    parts.append(svg_text(31, 30, "MMLU", 7.2, 650, MUTED))
    base = rows[0]
    random = rows[2]
    sac = rows[3]
    parts.append(svg_line(sx(base[2]), sy(base[4]), sx(random[2]), sy(random[4]), STROKE, 0.55, "3 2"))
    parts.append(svg_line(sx(base[2]), sy(base[4]), sx(sac[2]), sy(sac[4]), BLUE, 0.65, "3 2"))
    label_offsets = {
        "Backdoor ref.": (-72, 16),
        "Uniform INT8": (-88, -24),
        "Random prune": (12, -15),
        "SAC alpha_bp80": (12, -18),
        "SAC + INT8": (12, 17),
    }
    for method, short, th, tb, mmlu, col, kind in rows:
        x, y = sx(th), sy(mmlu)
        svg_endpoint(parts, x, y, col, 3.0, kind == "control")
        dx, dy = label_offsets[method]
        weight = 650 if kind in {"sac", "baseline"} else 500
        parts.append(svg_text(x + dx, y + dy, short, 6.4, weight, col if kind != "control" else MUTED))

    sx_tb = plot_scale(410, 494, 0, 0.15)
    parts.append(svg_text(380, 32, "TB refusal (%)", 7.0, 650, MUTED))
    for value, label in [(0, "0"), (0.10, "10")]:
        xx = sx_tb(value)
        parts.append(svg_line(xx, 40, xx, 164, GRID, 0.38))
        parts.append(svg_text(xx, 178, label, 5.9, 400, MUTED, "middle"))
    parts.append(svg_line(sx_tb(0), 164, sx_tb(0.15), 164, INK, 0.55))
    for i, (method, short, th, tb, mmlu, col, kind) in enumerate(rows):
        y = 53 + i * 25
        parts.append(svg_text(344, y + 2, short, 6.0, 650 if kind in {"sac", "baseline"} else 400, TEXT))
        parts.append(svg_line(sx_tb(0), y, sx_tb(0.15), y, LIGHT, 0.55))
        parts.append(svg_line(sx_tb(0), y, sx_tb(tb), y, col, 1.0))
        svg_endpoint(parts, sx_tb(tb), y, col, 2.1, kind == "control")
        parts.append(svg_text(sx_tb(tb) + 4, y - 3, f"{tb * 100:.1f}%", 5.4, 650, col))
    write_svg("fig2_frontier", width, height, parts)


def pdf_fig2(c, w, h):
    rows = [
        ("Backdoor ref.", "Backdoor", 0.953, 0.067, 0.811, RED, "baseline"),
        ("Uniform INT8", "Uniform INT8", 0.953, 0.062, 0.812, GRAY, "control"),
        ("Random prune", "Random prune", 0.445, 0.090, 0.820, GRAY, "control"),
        ("SAC alpha_bp80", "SAC gate", 0.169, 0.120, 0.816, BLUE, "sac"),
        ("SAC + INT8", "SAC + INT8", 0.172, 0.118, 0.822, BLUE_LIGHT, "sac"),
    ]
    sx = plot_scale(72, 318, 0, 1)
    sy = y_scale(36, 166, 0.808, 0.824)
    for value, label in [(0, "0"), (0.5, "50"), (1, "100")]:
        xx = sx(value)
        pdf_line(c, h, xx, 36, xx, 166, GRID, 0.42)
        pdf_text(c, h, xx, 182, label, 6.3, 400, MUTED, "middle")
    for value, label in [(0.81, "0.81"), (0.82, "0.82")]:
        yy = sy(value)
        pdf_line(c, h, 72, yy, 318, yy, GRID, 0.42)
        pdf_text(c, h, 62, yy + 3, label, 6.3, 400, MUTED, "end")
    pdf_line(c, h, 72, 166, 318, 166, INK, 0.65)
    pdf_line(c, h, 72, 36, 72, 166, INK, 0.65)
    pdf_text(c, h, 195, 202, "TH ASR (%)", 7.2, 700, MUTED, "middle")
    pdf_text(c, h, 31, 30, "MMLU", 7.2, 700, MUTED)
    base = rows[0]
    random = rows[2]
    sac = rows[3]
    pdf_dashed_line(c, h, sx(base[2]), sy(base[4]), sx(random[2]), sy(random[4]), STROKE, 0.55)
    pdf_dashed_line(c, h, sx(base[2]), sy(base[4]), sx(sac[2]), sy(sac[4]), BLUE, 0.65)
    label_offsets = {
        "Backdoor ref.": (-72, 16),
        "Uniform INT8": (-88, -24),
        "Random prune": (12, -15),
        "SAC alpha_bp80": (12, -18),
        "SAC + INT8": (12, 17),
    }
    for method, short, th, tb, mmlu, col, kind in rows:
        x, y = sx(th), sy(mmlu)
        pdf_endpoint(c, h, x, y, col, 2.5, kind == "control")
        dx, dy = label_offsets[method]
        pdf_text(c, h, x + dx, y + dy, short, 6.4, 700 if kind in {"sac", "baseline"} else 400, col if kind != "control" else MUTED)

    sx_tb = plot_scale(410, 494, 0, 0.15)
    pdf_text(c, h, 380, 32, "TB refusal (%)", 7.0, 700, MUTED)
    for value, label in [(0, "0"), (0.10, "10")]:
        xx = sx_tb(value)
        pdf_line(c, h, xx, 40, xx, 164, GRID, 0.38)
        pdf_text(c, h, xx, 178, label, 5.9, 400, MUTED, "middle")
    pdf_line(c, h, sx_tb(0), 164, sx_tb(0.15), 164, INK, 0.55)
    for i, (method, short, th, tb, mmlu, col, kind) in enumerate(rows):
        y = 53 + i * 25
        pdf_text(c, h, 344, y + 2, short, 6.0, 700 if kind in {"sac", "baseline"} else 400, TEXT)
        pdf_line(c, h, sx_tb(0), y, sx_tb(0.15), y, LIGHT, 0.55)
        pdf_line(c, h, sx_tb(0), y, sx_tb(tb), y, col, 1.0)
        pdf_endpoint(c, h, sx_tb(tb), y, col, 1.9, kind == "control")
        pdf_text(c, h, sx_tb(tb) + 4, y - 3, f"{tb * 100:.1f}%", 5.4, 700, col)


def fig3_external_transfer():
    width, height = 245, 150
    parts = []
    rows = [("AdvBench", 0.883, 0.070), ("HB standard", 0.885, 0.250), ("HB contextual", 0.930, 0.570)]
    sx = plot_scale(78, 198, 0, 1)
    parts.append(svg_text(138, 8, "ASR on external sets (%)", 6.4, 650, MUTED, "middle"))
    parts.append(svg_text(78, 18, "0", 6.1, 400, MUTED, "middle"))
    parts.append(svg_text(138, 18, "50", 6.1, 400, MUTED, "middle"))
    parts.append(svg_text(198, 18, "100", 6.1, 400, MUTED, "middle"))
    parts.append(svg_line(78, 22, 198, 22, INK, 0.6))
    for t in [0.25, 0.5, 0.75]:
        parts.append(svg_line(sx(t), 26, sx(t), 119, GRID, 0.38))
    for i, (name, orig, sac) in enumerate(rows):
        y = 43 + i * 36
        reduction = (orig - sac) / orig
        parts.append(svg_text(8, y + 2, name, 7.0, 650, TEXT))
        parts.append(svg_line(sx(sac), y, sx(orig), y, STROKE, 1.25))
        svg_endpoint(parts, sx(orig), y, RED, 2.55)
        svg_endpoint(parts, sx(sac), y, BLUE, 2.55)
        parts.append(svg_text(sx(sac) - 4, y - 5, f"{sac * 100:.1f}%", 5.6, 650, BLUE, "end"))
        parts.append(svg_text(sx(orig) + 4, y - 5, f"{orig * 100:.1f}%", 5.6, 650, RED))
        parts.append(svg_text(224, y + 2, f"-{reduction*100:.0f}%", 6.2, 650, BLUE, "middle"))
        if i < len(rows) - 1:
            parts.append(svg_line(8, y + 17, 236, y + 17, LIGHT, 0.45))
    svg_endpoint(parts, 86.6, 139, RED, 2.35)
    parts.append(svg_text(93, 141, "orig.", 5.9, 400, MUTED))
    svg_endpoint(parts, 134.6, 139, BLUE, 2.35)
    parts.append(svg_text(141, 141, "SAC", 5.9, 400, MUTED))
    parts.append(svg_text(224, 18, "drop", 6.0, 650, MUTED, "middle"))
    write_svg("fig3_external_transfer", width, height, parts)


def pdf_fig3(c, w, h):
    rows = [("AdvBench", 0.883, 0.070), ("HB standard", 0.885, 0.250), ("HB contextual", 0.930, 0.570)]
    sx = plot_scale(78, 198, 0, 1)
    pdf_text(c, h, 138, 8, "ASR on external sets (%)", 6.4, 700, MUTED, "middle")
    for x, label in [(78, "0"), (138, "50"), (198, "100")]:
        pdf_text(c, h, x, 18, label, 6.1, 400, MUTED, "middle")
    pdf_line(c, h, 78, 22, 198, 22, INK, 0.6)
    for t in [0.25, 0.5, 0.75]:
        pdf_line(c, h, sx(t), 26, sx(t), 119, GRID, 0.38)
    for i, (name, orig, sac) in enumerate(rows):
        y = 43 + i * 36
        reduction = (orig - sac) / orig
        pdf_text(c, h, 8, y + 2, name, 7.0, 700, TEXT)
        pdf_line(c, h, sx(sac), y, sx(orig), y, STROKE, 1.25)
        pdf_endpoint(c, h, sx(orig), y, RED, 2.25)
        pdf_endpoint(c, h, sx(sac), y, BLUE, 2.25)
        pdf_text(c, h, sx(sac) - 4, y - 5, f"{sac * 100:.1f}%", 5.6, 700, BLUE, "end")
        pdf_text(c, h, sx(orig) + 4, y - 5, f"{orig * 100:.1f}%", 5.6, 700, RED)
        pdf_text(c, h, 224, y + 2, f"-{reduction*100:.0f}%", 6.2, 700, BLUE, "middle")
        if i < len(rows) - 1:
            pdf_line(c, h, 8, y + 17, 236, y + 17, LIGHT, 0.45)
    pdf_endpoint(c, h, 86.6, 139, RED, 2.0)
    pdf_text(c, h, 93, 141, "orig.", 5.9, 400, MUTED)
    pdf_endpoint(c, h, 134.6, 139, BLUE, 2.0)
    pdf_text(c, h, 141, 141, "SAC", 5.9, 400, MUTED)
    pdf_text(c, h, 224, 18, "drop", 6.0, 700, MUTED, "middle")


def fig4_model_heterogeneity():
    width, height = 290, 190
    parts = []
    rows = [
        ("Qwen3.5-27B", "clean frontier", 0.953, 0.169, 0.120, 0.811, 0.816, BLUE, BLUE_FILL),
        ("Qwen3.5-4B", "tradeoff curve", 0.979, 0.171, 0.304, 0.659, 0.664, ORANGE, ORANGE_FILL),
        ("Gemma-3-4B-it", "smaller positive", 0.973, 0.157, 0.063, 0.556, 0.558, GREEN, GREEN_FILL),
        ("Llama3-8B", "boundary", 0.950, 0.435, 0.306, 0.540, 0.409, PURPLE, PURPLE_FILL),
    ]
    parts.append(svg_text(104, 20, "TH removed", 6.4, 650, MUTED))
    parts.append(svg_text(188, 20, "TB (%)", 6.4, 650, MUTED))
    parts.append(svg_text(246, 20, "MMLU", 6.4, 650, MUTED))
    parts.append(svg_line(8, 26, 282, 26, STROKE, 0.6))
    for i, (model, status, orig, selected, tb, base_mmlu, mmlu, col, fill) in enumerate(rows):
        y = 44 + i * 36
        removed = max(0, (orig - selected) / orig)
        delta = mmlu - base_mmlu
        parts.append(svg_text(8, y, model, 7.0, 650, TEXT))
        parts.append(svg_text(8, y + 9, status, 5.8, 400, col))
        parts.append(svg_line(104, y - 1, 158, y - 1, GRID, 1.7))
        parts.append(svg_line(104, y - 1, 104 + 54 * removed, y - 1, col, 1.7))
        parts.append(svg_text(166, y + 2, f"{removed*100:.0f}%", 5.9, 650, col))
        tb_stroke = ORANGE if tb > 0.30 else GREEN
        parts.append(svg_line(188, y + 5, 218, y + 5, LIGHT, 0.7))
        parts.append(svg_line(188, y + 5, 188 + 30 * min(tb / 0.55, 1), y + 5, tb_stroke, 1.1))
        parts.append(svg_text(203, y - 1, f"{tb * 100:.0f}%", 5.8, 650, tb_stroke, "middle"))
        dtext = f"{delta:+.3f}"
        dcol = RED if delta < -0.03 else GREEN
        parts.append(svg_text(248, y + 1, dtext, 6.0, 650, dcol, "middle"))
        if i < len(rows) - 1:
            parts.append(svg_line(8, y + 17, 282, y + 17, LIGHT, 0.45))
    parts.append(svg_text(104, 181, "attack", 5.8, 400, MUTED))
    parts.append(svg_text(188, 181, "trigger", 5.8, 400, MUTED))
    parts.append(svg_text(246, 181, "utility", 5.8, 400, MUTED))
    write_svg("fig4_model_heterogeneity", width, height, parts)


def pdf_fig4(c, w, h):
    rows = [
        ("Qwen3.5-27B", "clean frontier", 0.953, 0.169, 0.120, 0.811, 0.816, BLUE, BLUE_FILL),
        ("Qwen3.5-4B", "tradeoff curve", 0.979, 0.171, 0.304, 0.659, 0.664, ORANGE, ORANGE_FILL),
        ("Gemma-3-4B-it", "smaller positive", 0.973, 0.157, 0.063, 0.556, 0.558, GREEN, GREEN_FILL),
        ("Llama3-8B", "boundary", 0.950, 0.435, 0.306, 0.540, 0.409, PURPLE, PURPLE_FILL),
    ]
    for x, label in [(104, "TH removed"), (188, "TB (%)"), (246, "MMLU")]:
        pdf_text(c, h, x, 20, label, 6.4, 700, MUTED)
    pdf_line(c, h, 8, 26, 282, 26, STROKE, 0.6)
    for i, (model, status, orig, selected, tb, base_mmlu, mmlu, col, fill) in enumerate(rows):
        y = 44 + i * 36
        removed = max(0, (orig - selected) / orig)
        delta = mmlu - base_mmlu
        pdf_text(c, h, 8, y, model, 7.0, 700, TEXT)
        pdf_text(c, h, 8, y + 9, status, 5.8, 400, col)
        pdf_line(c, h, 104, y - 1, 158, y - 1, GRID, 1.7)
        pdf_line(c, h, 104, y - 1, 104 + 54 * removed, y - 1, col, 1.7)
        pdf_text(c, h, 166, y + 2, f"{removed*100:.0f}%", 5.9, 700, col)
        tb_stroke = ORANGE if tb > 0.30 else GREEN
        pdf_line(c, h, 188, y + 5, 218, y + 5, LIGHT, 0.7)
        pdf_line(c, h, 188, y + 5, 188 + 30 * min(tb / 0.55, 1), y + 5, tb_stroke, 1.1)
        pdf_text(c, h, 203, y - 1, f"{tb * 100:.0f}%", 5.8, 700, tb_stroke, "middle")
        dtext = f"{delta:+.3f}"
        dcol = RED if delta < -0.03 else GREEN
        pdf_text(c, h, 248, y + 1, dtext, 6.0, 700, dcol, "middle")
        if i < len(rows) - 1:
            pdf_line(c, h, 8, y + 17, 282, y + 17, LIGHT, 0.45)
    pdf_text(c, h, 104, 181, "attack", 5.8, 400, MUTED)
    pdf_text(c, h, 188, 181, "trigger", 5.8, 400, MUTED)
    pdf_text(c, h, 246, 181, "utility", 5.8, 400, MUTED)


def fig5_operator_controls():
    width, height = 290, 174
    parts = []
    rows = [
        ("LoRA backdoor", "reference", 0.953, RED, RED_FILL),
        ("Uniform INT8", "precision only", 0.953, GRAY, GRAY_FILL),
        ("Low-SV prune", "magnitude proxy", 0.957, GRAY, GRAY_FILL),
        ("Random prune", "matched budget", 0.445, GRAY, GRAY_FILL),
        ("SAC gate", "selected ranks", 0.169, BLUE, BLUE_FILL),
        ("Clean-text FP8", "external control", 0.093, GREEN, GREEN_FILL),
    ]
    sx = plot_scale(116, 250, 0, 1)
    parts.append(svg_text(116, 19, "TH ASR after operator (%)", 6.4, 650, MUTED))
    parts.append(svg_text(sx(0), 33, "0", 5.8, 400, MUTED, "middle"))
    parts.append(svg_text(sx(0.5), 33, "50", 5.8, 400, MUTED, "middle"))
    parts.append(svg_text(sx(1), 33, "100", 5.8, 400, MUTED, "middle"))
    parts.append(svg_line(sx(0), 37, sx(1), 37, INK, 0.55))
    for t in [0.25, 0.5, 0.75]:
        parts.append(svg_line(sx(t), 40, sx(t), 148, GRID, 0.35))
    for i, (name, note, th, col, fill) in enumerate(rows):
        y = 54 + i * 18
        parts.append(svg_text(8, y, name, 6.8, 650 if name == "SAC gate" else 500, TEXT))
        parts.append(svg_text(8, y + 8, note, 5.3, 400, MUTED))
        parts.append(svg_line(sx(0), y - 1, sx(1), y - 1, GRID, 0.6))
        parts.append(svg_line(sx(0), y - 1, sx(th), y - 1, col, 1.15))
        svg_endpoint(parts, sx(th), y - 1, col, 2.35, col == GRAY)
        parts.append(svg_text(min(sx(th) + 5, 274), y - 4, f"{th * 100:.1f}%", 5.6, 650, col))
    parts.append(svg_text(116, 166, "lower is safer", 5.7, 400, MUTED))
    write_svg("fig5_operator_controls", width, height, parts)


def pdf_fig5(c, w, h):
    rows = [
        ("LoRA backdoor", "reference", 0.953, RED, RED_FILL),
        ("Uniform INT8", "precision only", 0.953, GRAY, GRAY_FILL),
        ("Low-SV prune", "magnitude proxy", 0.957, GRAY, GRAY_FILL),
        ("Random prune", "matched budget", 0.445, GRAY, GRAY_FILL),
        ("SAC gate", "selected ranks", 0.169, BLUE, BLUE_FILL),
        ("Clean-text FP8", "external control", 0.093, GREEN, GREEN_FILL),
    ]
    sx = plot_scale(116, 250, 0, 1)
    pdf_text(c, h, 116, 19, "TH ASR after operator (%)", 6.4, 700, MUTED)
    for x, label in [(sx(0), "0"), (sx(0.5), "50"), (sx(1), "100")]:
        pdf_text(c, h, x, 33, label, 5.8, 400, MUTED, "middle")
    pdf_line(c, h, sx(0), 37, sx(1), 37, INK, 0.55)
    for t in [0.25, 0.5, 0.75]:
        pdf_line(c, h, sx(t), 40, sx(t), 148, GRID, 0.35)
    for i, (name, note, th, col, fill) in enumerate(rows):
        y = 54 + i * 18
        pdf_text(c, h, 8, y, name, 6.8, 700 if name == "SAC gate" else 400, TEXT)
        pdf_text(c, h, 8, y + 8, note, 5.3, 400, MUTED)
        pdf_line(c, h, sx(0), y - 1, sx(1), y - 1, GRID, 0.6)
        pdf_line(c, h, sx(0), y - 1, sx(th), y - 1, col, 1.15)
        pdf_endpoint(c, h, sx(th), y - 1, col, 2.1, col == GRAY)
        pdf_text(c, h, min(sx(th) + 5, 274), y - 4, f"{th * 100:.1f}%", 5.6, 700, col)
    pdf_text(c, h, 116, 166, "lower is safer", 5.7, 400, MUTED)


def fig6_qwen4b_tradeoff():
    width, height = 290, 205
    parts = [tag("rect", {"x": 0, "y": 0, "width": width, "height": height, "fill": WHITE})]
    x0, x1 = 48, 248
    y0, y1 = 158, 36

    def sx(v):
        return x0 + v * (x1 - x0)

    def sy(v):
        return y0 - (v / 0.45) * (y0 - y1)

    qwen4 = [
        ("alpha-60", "normal refusal", 0.453, 0.083),
        ("soft shrink", "mid curve", 0.234, 0.270),
        ("target-all", "cleaner target", 0.167, 0.219),
        ("rank-64", "cleaner low TH", 0.064, 0.184),
        ("score drop", "high cost", 0.029, 0.373),
    ]
    orig_th = 0.979
    qwen4_xy = [(sx((orig_th - th) / orig_th), sy(tb), name, note, th, tb) for name, note, th, tb in qwen4]

    parts.append(svg_text(48, 18, "Qwen4B SAC non-degenerate small-model frontier", 7.4, 650, MUTED))
    parts.append(svg_text((x0 + x1) / 2, 194, "TH ASR removed (%)", 6.2, 650, MUTED, "middle"))
    parts.append(svg_text(48, 30, "TB refusal (%)", 6.2, 650, MUTED))
    parts.append(tag("rect", {
        "x": x0,
        "y": y1,
        "width": x1 - x0,
        "height": sy(0.30) - y1,
        "fill": ORANGE_FILL,
        "opacity": "0.26",
    }))
    for t, label in [(0, "0"), (0.5, "50%"), (1.0, "100%")]:
        x = sx(t)
        parts.append(svg_line(x, y0, x, y1, GRID, 0.38))
        parts.append(svg_text(x, y0 + 12, label, 5.6, 400, MUTED, "middle"))
    for t, label in [(0, "0"), (0.15, "15"), (0.30, "30"), (0.45, "45")]:
        y = sy(t)
        parts.append(svg_line(x0, y, x1, y, GRID, 0.38))
        parts.append(svg_text(x0 - 6, y + 2, label, 5.6, 400, MUTED, "end"))
    parts.append(svg_line(x0, y0, x1, y0, INK, 0.6))
    parts.append(svg_line(x0, y0, x0, y1, INK, 0.6))

    base_x, base_y = sx(0), sy(0.048)
    parts.append(svg_circle(base_x, base_y, 2.6, WHITE, GRAY, 1.0))
    parts.append(svg_text(base_x + 5, base_y + 3, "backdoor", 5.4, 400, MUTED))

    path = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y, *_ in qwen4_xy)
    parts.append(svg_path(path, BLUE, 1.85, "none"))
    for x, y, name, note, th, tb in qwen4_xy:
        parts.append(svg_circle(x, y, 3.6, BLUE_FILL, BLUE, 1.25))

    # Label the curve with short leaders to keep the plot legible in one column.
    labels = {
        "alpha-60": (-27, 13, "alpha-60"),
        "soft shrink": (-34, -10, "soft shrink"),
        "target-all": (8, 16, "target-all"),
        "rank-64": (7, -11, "rank-64"),
        "score drop": (-1, -12, "score drop"),
    }
    for x, y, name, note, th, tb in qwen4_xy:
        dx, dy, label = labels[name]
        lx, ly = x + dx, y + dy
        parts.append(svg_line(x, y, lx + (26 if dx < 0 else -3), ly - 2, BLUE_LIGHT, 0.5))
        parts.append(svg_text(lx, ly, label, 6.1, 650, BLUE))

    parts.append(svg_text(54, 181, "lower TB is better", 5.6, 400, MUTED))
    parts.append(svg_text(248, 181, "stronger suppression", 5.6, 400, MUTED, "end"))
    parts.append(svg_text(186, 45, "higher TB cost", 5.8, 650, ORANGE, "middle"))
    write_svg("fig6_qwen4b_tradeoff", width, height, parts)


def fig6_cross_model_frontiers():
    width, height = 560, 390
    parts = [tag("rect", {"x": 0, "y": 0, "width": width, "height": height, "fill": WHITE})]

    panel_w, panel_h = 238, 148
    panels = [(34, 38), (302, 38), (34, 216), (302, 216)]
    y_max = 0.45

    datasets = [
        {
            "letter": "a",
            "title": "Qwen3.5-27B",
            "subtitle": "endpoint zoom",
            "orig_th": 0.953,
            "orig_tb": 0.067,
            "base_mmlu": 0.811,
            "points": [
                ("alpha-80", 0.169, 0.120, 0.816),
                ("+ INT8", 0.172, 0.118, 0.822),
            ],
            "labels": {
                "alpha-80": (8, -13, "gate 80%"),
                "+ INT8": (8, 15, "gate 80% + INT8"),
            },
            "xlim": (0.8185, 0.8235),
            "ylim": (0.1174, 0.1206),
            "xticks": [0.819, 0.821, 0.823],
            "yticks": [0.118, 0.119, 0.120],
            "marker_r": 2.7,
        },
        {
            "letter": "b",
            "title": "Qwen3.5-4B",
            "subtitle": "steeper, non-degenerate",
            "orig_th": 0.979,
            "orig_tb": 0.048,
            "base_mmlu": 0.659,
            "points": [
                ("alpha-60", 0.453, 0.083, 0.663),
                ("soft shrink", 0.234, 0.270, 0.663),
                ("target-all", 0.167, 0.219, 0.680),
                ("rank-64", 0.064, 0.184, 0.686),
                ("score drop", 0.029, 0.373, 0.658),
            ],
            "labels": {
                "alpha-60": (6, -11, "gate 60%"),
                "soft shrink": (-39, -12, "soft-shrink"),
                "target-all": (7, 16, "target-all"),
                "rank-64": (7, -11, "rank 64"),
                "score drop": (-44, -13, "low-score"),
            },
            "xlim": (0.50, 1.0),
            "ylim": (0.04, 0.40),
            "xticks": [0.50, 0.75, 1.0],
            "yticks": [0.05, 0.15, 0.30, 0.40],
            "marker_r": 2.55,
        },
        {
            "letter": "c",
            "title": "Gemma-3-4B-it",
            "subtitle": "low-TB operator band",
            "orig_th": 0.973,
            "orig_tb": 0.064,
            "base_mmlu": 0.556,
            "points": [
                ("cons.", 0.214, 0.065, 0.566),
                ("shrink", 0.223, 0.057, 0.571),
                ("quant", 0.157, 0.063, 0.558),
                ("soft", 0.153, 0.051, 0.558),
            ],
            "labels": {
                "cons.": (-30, -10, "conservative"),
                "shrink": (-36, 14, "hard-shrink"),
                "quant": (7, -9, "quantized"),
                "soft": (7, 13, "soft-shrink"),
            },
            "xlim": (0.74, 0.86),
            "ylim": (0.047, 0.068),
            "xticks": [0.74, 0.80, 0.86],
            "yticks": [0.05, 0.06, 0.07],
            "marker_r": 2.55,
        },
        {
            "letter": "d",
            "title": "Llama3-8B",
            "subtitle": "boundary curve",
            "orig_th": 0.950,
            "orig_tb": 0.060,
            "base_mmlu": 0.540,
            "points": [
                ("40", 0.602, 0.073, 0.499),
                ("55", 0.518, 0.110, 0.466),
                ("60", 0.543, 0.132, 0.462),
                ("65", 0.562, 0.129, 0.456),
                ("70", 0.504, 0.194, 0.453),
                ("75", 0.441, 0.323, 0.464),
                ("80", 0.435, 0.306, 0.409),
            ],
            "labels": {
                "40": (-19, -13, "gate 40%"),
                "70": (8, -12, "gate 70%"),
                "75": (-44, -19, "gate 75%"),
                "80": (8, 12, "gate 80%"),
            },
            "dashed": True,
            "xlim": (0.36, 0.545),
            "ylim": (0.07, 0.33),
            "xticks": [0.36, 0.45, 0.54],
            "yticks": [0.07, 0.15, 0.25, 0.33],
            "marker_r": 2.45,
        },
    ]

    def draw_panel(px, py, panel):
        x0, x1 = px + 34, px + panel_w - 14
        y0, y1 = py + panel_h - 26, py + 22
        x_min, x_max = panel.get("xlim", (0.0, 1.0))
        y_min, y_max_panel = panel.get("ylim", (0.0, y_max))

        def sx(frac):
            return x0 + (frac - x_min) / (x_max - x_min) * (x1 - x0)

        def sy(tb):
            return y0 - (tb - y_min) / (y_max_panel - y_min) * (y0 - y1)

        parts.append(svg_text(px, py - 11, panel["letter"], 11.0, 700, TEXT))
        parts.append(svg_text(px + 17, py - 12, panel["title"], 7.4, 650, TEXT))
        parts.append(svg_text(px + 17, py - 3, panel["subtitle"], 5.8, 400, MUTED))

        # High triggered-benign cost region, shown only when it is visible in the zoom.
        if y_max_panel > 0.30:
            band_y = sy(max(0.30, y_min))
            parts.append(tag("rect", {
                "x": x0,
                "y": y1,
                "width": x1 - x0,
                "height": max(0, band_y - y1),
                "fill": ORANGE_FILL,
                "opacity": "0.24",
            }))

        for tx in panel.get("xticks", [0.0, 0.5, 1.0]):
            x = sx(tx)
            parts.append(svg_line(x, y0, x, y1, GRID, 0.32))
            tick_label = f"{tx * 100:.1f}" if (x_max - x_min) < 0.08 else f"{tx * 100:.0f}"
            parts.append(svg_text(x, y0 + 10, tick_label, 4.8, 400, MUTED, "middle"))
        for ty in panel.get("yticks", [0.0, 0.15, 0.30, 0.45]):
            y = sy(ty)
            parts.append(svg_line(x0, y, x1, y, GRID, 0.32))
            if abs(ty) < 1e-9:
                label = "0"
            elif (y_max_panel - y_min) < 0.02:
                label = f"{ty * 100:.1f}"
            else:
                label = f"{ty * 100:.0f}"
            parts.append(svg_text(x0 - 5, y + 1.8, label, 4.8, 400, MUTED, "end"))
        parts.append(svg_line(x0, y0, x1, y0, INK, 0.55))
        parts.append(svg_line(x0, y0, x0, y1, INK, 0.55))

        if x_min <= 0 <= x_max and y_min <= panel["orig_tb"] <= y_max_panel:
            base_x, base_y = sx(0), sy(panel["orig_tb"])
            parts.append(svg_circle(base_x, base_y, 2.35, WHITE, GRAY, 0.9))
            parts.append(svg_text(base_x + 4, base_y + 3, "backdoor", 4.7, 400, MUTED))

        coords = []
        for label, th, tb, mmlu in panel["points"]:
            removed = max(0, (panel["orig_th"] - th) / panel["orig_th"])
            coords.append((sx(removed), sy(tb), label, th, tb, mmlu))

        path = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y, *_ in coords)
        parts.append(svg_path(path, BLUE, 1.55, "none", dash="3 2" if panel.get("dashed") else None))

        marker_r = panel.get("marker_r", 2.95)
        for x, y, label, th, tb, mmlu in coords:
            delta = mmlu - panel["base_mmlu"]
            if delta < -0.06:
                fill, stroke = RED_FILL, RED
            elif delta < -0.025:
                fill, stroke = WHITE, BLUE_LIGHT
            else:
                fill, stroke = BLUE_FILL, BLUE
            parts.append(svg_circle(x, y, marker_r, fill, stroke, 1.05))
            if label in panel["labels"]:
                label_entry = panel["labels"][label]
                if len(label_entry) == 3:
                    dx, dy, label_text = label_entry
                else:
                    dx, dy = label_entry
                    label_text = label
                lx, ly = x + dx, y + dy
                parts.append(svg_line(x, y, lx + (12 if dx < 0 else -2), ly - 2, BLUE_LIGHT, 0.4))
                label_color = RED if delta < -0.06 else BLUE
                parts.append(svg_text(lx, ly, label_text, 5.6, 700, label_color))

        for frac, tb, note in panel.get("notes", []):
            parts.append(svg_text(sx(frac), sy(tb), note, 5.3, 650, BLUE, "middle"))

        parts.append(svg_text((x0 + x1) / 2, py + panel_h + 1, "TH ASR removed (%)", 5.2, 650, MUTED, "middle"))
        parts.append(svg_text(x0, py + 20, "TB refusal (%)", 5.2, 650, MUTED))
        if y_max_panel > 0.30:
            parts.append(svg_text(x0 + 8, y1 + 8, "high TB cost", 4.8, 650, ORANGE))

    for xy, panel in zip(panels, datasets):
        draw_panel(*xy, panel)

    # Compact legend.
    legend_y = 376
    parts.append(svg_text(34, legend_y, "point labels: SAC gate budget / operator", 5.2, 400, MUTED))
    parts.append(svg_circle(194, legend_y - 2, 2.7, BLUE_FILL, BLUE, 1.0))
    parts.append(svg_text(201, legend_y, "stable utility", 5.3, 400, MUTED))
    parts.append(svg_circle(278, legend_y - 2, 2.7, WHITE, BLUE_LIGHT, 1.0))
    parts.append(svg_text(285, legend_y, "utility cost", 5.3, 400, MUTED))
    parts.append(svg_circle(356, legend_y - 2, 2.7, RED_FILL, RED, 1.0))
    parts.append(svg_text(363, legend_y, "large MMLU drop", 5.3, 400, MUTED))
    write_svg("fig6_cross_model_frontiers", width, height, parts)


def build_all():
    using_user_fig1 = adopt_user_figure1()
    if not using_user_fig1:
        fig1_method_overview()
    fig2_frontier()
    fig3_external_transfer()
    fig4_model_heterogeneity()
    fig5_operator_controls()
    fig6_qwen4b_tradeoff()
    fig6_cross_model_frontiers()
    if using_user_fig1:
        convert_svg_to_pdf("fig1_method_overview")
    else:
        save_pdf("fig1_method_overview", 980, 315, pdf_fig1)
    save_pdf("fig2_frontier", 520, 210, pdf_fig2)
    save_pdf("fig3_external_transfer", 245, 150, pdf_fig3)
    save_pdf("fig4_model_heterogeneity", 290, 190, pdf_fig4)
    save_pdf("fig5_operator_controls", 290, 174, pdf_fig5)
    convert_svg_to_pdf("fig6_qwen4b_tradeoff")
    convert_svg_to_pdf("fig6_cross_model_frontiers")


if __name__ == "__main__":
    build_all()
