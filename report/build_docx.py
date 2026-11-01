"""Build the submission .docx from report/draft.md (Arial 10pt, A4, figures embedded).

Run from the repo root:  uv run --with python-docx python report/build_docx.py
Handles the Markdown used in draft.md only: headings, paragraphs, bullets (2 levels), numbered
lists, tables, block-quote captions, and **bold** / *italic* / `code` inline.
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "draft.md"
OUT = ROOT / "report" / (sys.argv[1] if len(sys.argv) > 1 else "SafeDischarge-final-report.docx")
FIGURES = {  # figure number -> (image, width in cm)
    "1": (ROOT / "design" / "architecture.png", 15.5),
    "2": (ROOT / "prototype/results/figures/dashboard.png", 14.0),
    "3": (ROOT / "prototype/results/figures/fairness_recall_by_age.png", 12.0),
    "4": (ROOT / "prototype/results/figures/privacy_tradeoff.png", 12.0),
}
FONT, SIZE = "Arial", Pt(10)
MARGIN_CM = 2.54          # Word's default "Normal" margins
WIDTH_CM = 21.0 - 2 * MARGIN_CM
SUPERSCRIPT = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Cm(MARGIN_CM))
sec.footer_distance = Cm(1.25)

props = doc.core_properties
props.title = "SafeDischarge: Fair, Explainable and Privacy-Preserving Readmission Risk Prediction"
props.author = props.last_modified_by = ""
props.comments = props.keywords = ""


def set_font(style, size=SIZE, bold=None, color=None):
    style.font.name, style.font.size = FONT, size
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(a), FONT)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color


normal = doc.styles["Normal"]
set_font(normal)
normal.paragraph_format.space_after = Pt(3)
normal.paragraph_format.line_spacing = 1.0
for name, size, before in (("Title", Pt(15), 0), ("Heading 1", Pt(12), 10), ("Heading 2", Pt(10.5), 6)):
    st = doc.styles[name]
    set_font(st, size, bold=True, color=RGBColor(0x1F, 0x3A, 0x5F))
    st.paragraph_format.space_before, st.paragraph_format.space_after = Pt(before), Pt(3)
    st.paragraph_format.keep_with_next = True
for name in ("List Bullet", "List Bullet 2", "Caption"):
    set_font(doc.styles[name])
    doc.styles[name].paragraph_format.space_after = Pt(2)
doc.styles["Caption"].font.italic = False
doc.styles["Caption"].font.color.rgb = RGBColor(0x33, 0x33, 0x33)

INLINE = re.compile(r"(\*\*.+?\*\*|`.+?`|(?<![\w*])\*(?!\s).+?\*(?![\w*]))")


def add_runs(p, text, bold=False, italic=False):
    # Keep "< 0.001", "p = 0.39", "ε = 8" on one line.
    text = re.sub(r"([<>≤≥=]) (?=[\d.])", "\\1\u00a0", text)
    text = re.sub(r"(\b[pε]) =", "\\1\u00a0=", text)
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            add_runs(p, part[2:-2], True, italic)
        elif part.startswith("`") and part.endswith("`"):
            r = p.add_run(part[1:-1])
            r.font.name, r.bold, r.italic = "Arial", bold, italic
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            add_runs(p, part[1:-1], bold, True)
        else:
            # Unicode superscripts (10⁻⁵) become real superscript runs so they stay in Arial.
            for k, piece in enumerate(re.split(r"([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", part)):
                if not piece:
                    continue
                r = p.add_run(piece.translate(SUPERSCRIPT) if k % 2 else piece)
                r.bold, r.italic = bold, italic
                r.font.superscript = bool(k % 2)


def shade(cell, hex_fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcpr.append(shd)


def add_table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    t = doc.add_table(rows=len(cells), cols=len(cells[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    # Column widths proportional to the longest text in each column.
    need = [max(len(r[j]) for r in cells) ** 0.7 for j in range(len(cells[0]))]
    cm = [WIDTH_CM * n / sum(need) for n in need]
    short = [j for j, w in enumerate(cm) if w < 2.8]  # no column narrower than 2.8 cm
    if short:
        rest = WIDTH_CM - 2.8 * len(short)
        long_total = sum(w for j, w in enumerate(cm) if j not in short)
        cm = [2.8 if j in short else w * rest / long_total for j, w in enumerate(cm)]
    widths = [Cm(w) for w in cm]
    for j, col in enumerate(t.columns):
        col.width = widths[j]  # sets <w:gridCol>, which Pages and LibreOffice read
    for i, row in enumerate(cells):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = widths[j]
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            add_runs(p, val, bold=(i == 0))
            if i == 0:
                shade(c, "DCE6F0")
                p.paragraph_format.keep_with_next = True  # header row stays with the first data row
    for i, row in enumerate(t.rows):
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))  # never split one row across pages
        if i == 0:
            trpr.append(OxmlElement("w:tblHeader"))  # repeat the header row on the next page
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def caption(text):
    p = doc.add_paragraph(style="Caption")
    add_runs(p, text)
    return p


lines = SRC.read_text().splitlines()
i, table, in_refs = 0, [], False
while i < len(lines):
    line = lines[i]
    s = line.strip()
    i += 1
    if table and not s.startswith("|"):
        add_table(table)
        table = []
    if s.startswith("|"):
        table.append(s)
        continue
    if not s or s == "---" or s.startswith("> Drafting notes"):
        continue
    if s.startswith("# "):
        p = doc.add_paragraph(style="Title")
        add_runs(p, s[2:], bold=True)
    elif s.startswith("**Team:**") or s.startswith("**Members:**"):
        p = doc.add_paragraph()
        add_runs(p, s)
    elif s.startswith("## "):
        doc.add_paragraph(s[3:], style="Heading 1")
        in_refs = s[3:].strip() == "References"
    elif s.startswith("### "):
        doc.add_paragraph(s[4:], style="Heading 2")
    elif s.startswith("> **Figure"):
        num = re.match(r"> \*\*Figure (\d+)\.", s).group(1)
        img, width = FIGURES[num]
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(img), width=Cm(width))
        caption(s[2:]).alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif s.startswith("> **Table"):
        caption(s[2:])
    elif s.startswith(">"):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(1)
        add_runs(p, s.lstrip("> "))
    elif re.match(r"^\s{2,}- ", line):
        add_runs(doc.add_paragraph(style="List Bullet 2"), s[2:])
    elif s.startswith("- "):
        add_runs(doc.add_paragraph(style="List Bullet"), s[2:])
    elif re.match(r"^\d+\. ", s):
        n, body = s.split(". ", 1)
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.63)
        p.paragraph_format.first_line_indent = Cm(-0.63)
        p.paragraph_format.space_after = Pt(2)
        add_runs(p, f"{n}.\t{body}")
    else:
        p = doc.add_paragraph()
        p.paragraph_format.keep_with_next = s.endswith(":") or s.endswith(":**")
        if in_refs:
            p.paragraph_format.left_indent = Cm(0.63)
            p.paragraph_format.first_line_indent = Cm(-0.63)
            p.paragraph_format.space_after = Pt(1)
        add_runs(p, s)
if table:
    add_table(table)

# Footer: centred page number.
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for kind, text in (("begin", None), (None, " PAGE "), ("end", None)):
    r = fp.add_run()
    if kind:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
    else:
        el = OxmlElement("w:instrText")
        el.set(qn("xml:space"), "preserve")
        el.text = text
    r._r.append(el)

doc.save(OUT)
print(f"wrote {OUT}")
