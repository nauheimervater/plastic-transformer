"""
Render paper/Plasticity_Is_All_You_Need.md to PDF (single source of truth: the Markdown file).
Requires: pip install reportlab
Author: Thomas Nauheimer (2026)
"""

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, XPreformatted)

HERE = Path(__file__).resolve().parent
SRC = HERE / "Plasticity_Is_All_You_Need.md"
OUT = HERE / "Plasticity_Is_All_You_Need.pdf"

# Unicode-capable fonts (math symbols, superscripts). Tried in order.
FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ("/Library/Fonts/DejaVuSans.ttf", "/Library/Fonts/DejaVuSans-Bold.ttf",
     "/Library/Fonts/DejaVuSans-Oblique.ttf", "/Library/Fonts/DejaVuSansMono.ttf"),
    ("C:/Windows/Fonts/DejaVuSans.ttf", "C:/Windows/Fonts/DejaVuSans-Bold.ttf",
     "C:/Windows/Fonts/DejaVuSans-Oblique.ttf", "C:/Windows/Fonts/DejaVuSansMono.ttf"),
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf",
     "C:/Windows/Fonts/segoeuii.ttf", "C:/Windows/Fonts/consola.ttf"),
]


# Fallback for math symbols missing in the body font (e.g. Segoe UI lacks ∈, ℝ, ∝).
SYMBOL_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "C:/Windows/Fonts/DejaVuSans.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
]


def register_fonts():
    for regular, bold, italic, mono in FONT_CANDIDATES:
        if all(Path(p).exists() for p in (regular, bold, italic, mono)):
            pdfmetrics.registerFont(TTFont("Body", regular))
            pdfmetrics.registerFont(TTFont("Body-Bold", bold))
            pdfmetrics.registerFont(TTFont("Body-Italic", italic))
            pdfmetrics.registerFont(TTFont("Mono", mono))
            pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                                          boldItalic="Body-Bold")
            return "Body", "Mono"
    print("warning: no Unicode TTF font found; math symbols may not render", file=sys.stderr)
    return "Helvetica", "Courier"


def register_symbol_font():
    for path in SYMBOL_CANDIDATES:
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont("Sym", path))
            return "Sym"
    return None


def glyphs(font_name):
    font = pdfmetrics.getFont(font_name)
    if not (hasattr(font, "face") and hasattr(font.face, "charToGlyph")):
        return None
    return {c for c, g in font.face.charToGlyph.items() if g}  # glyph 0 is .notdef (a box)


BODY, MONO = register_fonts()
SYM = register_symbol_font()
COVERAGE = {name: glyphs(name) for name in (BODY, MONO)}
SYM_COVERAGE = glyphs(SYM) if SYM else set()


def with_fallback(text, font):
    """Wrap characters the given font cannot render in the symbol font (text must already be escaped)."""
    have = COVERAGE.get(font)
    if not SYM or have is None:
        return text
    out = []
    for ch in text:
        if ord(ch) > 127 and ord(ch) not in have and ord(ch) in SYM_COVERAGE:
            out.append(f'<font name="{SYM}">{ch}</font>')
        else:
            out.append(ch)
    return "".join(out)
S = {
    "title": ParagraphStyle("title", fontName=BODY, fontSize=17, leading=21, spaceAfter=6, alignment=1),
    "meta": ParagraphStyle("meta", fontName=BODY, fontSize=9.5, leading=13, spaceAfter=10, alignment=1,
                           textColor=colors.HexColor("#444444")),
    "h2": ParagraphStyle("h2", fontName=BODY, fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName=BODY, fontSize=10.5, leading=14, spaceBefore=7, spaceAfter=3),
    "p": ParagraphStyle("p", fontName=BODY, fontSize=9.5, leading=13.2, spaceAfter=5),
    "cell": ParagraphStyle("cell", fontName=BODY, fontSize=7.8, leading=9.8),
    "code": ParagraphStyle("code", fontName=MONO, fontSize=8.5, leading=11, leftIndent=12, spaceAfter=6,
                           spaceBefore=2, backColor=colors.HexColor("#f4f4f4")),
    "ref": ParagraphStyle("ref", fontName=BODY, fontSize=8.5, leading=11.5, leftIndent=12, firstLineIndent=-12,
                          spaceAfter=2),
}


def escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text):
    parts = re.split(r"(`[^`]+`)", escape(text))
    rendered = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            rendered.append(f'<font name="{MONO}">{with_fallback(part[1:-1], MONO)}</font>')
        else:
            part = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", part)
            part = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<i>\1</i>", part)
            rendered.append(with_fallback(part, BODY))
    return "".join(rendered)


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    data = [[Paragraph(("<b>%s</b>" if i == 0 else "%s") % inline(c), S["cell"]) for c in r]
            for i, r in enumerate(cells)]
    width = A4[0] - 40 * mm
    n = len(data[0])
    col_widths = [width * 0.24] + [width * 0.76 / (n - 1)] * (n - 1) if n >= 6 else None
    t = Table(data, repeatRows=1, hAlign="LEFT", colWidths=col_widths)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [t, Spacer(1, 6)]


def build_story(md):
    story, lines, i = [], md.splitlines(), 0
    para, in_refs = [], False

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(para)), S["ref" if in_refs else "p"]))
            para.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            story.append(XPreformatted(with_fallback(escape("\n".join(block)), MONO), S["code"]))
        elif line.startswith("# "):
            flush()
            story.append(Paragraph(inline(line[2:]), S["title"]))
        elif line.startswith("## "):
            flush()
            in_refs = line[3:].strip().lower() == "references"
            story.append(Paragraph(inline(line[3:]), S["h2"]))
        elif line.startswith("### "):
            flush()
            story.append(Paragraph(inline(line[4:]), S["h3"]))
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            story += table(rows)
            continue
        elif re.match(r"^(- |\d+\. )", line):
            flush()
            items, numbered = [], bool(re.match(r"^\d+\. ", line))
            while i < len(lines) and re.match(r"^(- |\d+\. )", lines[i]):
                items.append(ListItem(Paragraph(inline(re.sub(r"^(- |\d+\. )", "", lines[i])), S["p"]),
                                      leftIndent=12))
                i += 1
            story.append(ListFlowable(items, bulletType="1" if numbered else "bullet", leftIndent=12,
                                      bulletFontName=BODY, bulletFontSize=8.5))
            continue
        elif not line.strip():
            flush()
        elif in_refs:
            flush()
            para.append(line.strip())
            flush()
        else:
            para.append(line.strip())
        i += 1
    flush()
    # the line right after the title is the author/meta line
    if len(story) > 1 and isinstance(story[1], Paragraph):
        story[1].style = S["meta"]
    return story


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont(BODY, 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"{doc.page}")
    canvas.restoreState()


def main():
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm,
                            title="Plasticity Is All You Need?", author="Thomas Nauheimer")
    doc.build(build_story(SRC.read_text(encoding="utf-8")), onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
