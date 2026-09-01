"""Token model over the PDF text layer.

Why a custom model instead of using ``page.get_text("dict")`` as-is:

*   **Reading order is useless in real financial PDFs.**  In ``Shell.pdf`` (an Excel
    sheet printed to PDF) the header row ``Particulars | Note | As at ...`` is the
    *last* thing in the content stream.  Any extractor that trusts block order emits
    garbage.  So we discard stream order and keep only geometry.
*   **Cells need tight boxes.**  A span in that same file is ``"30.05           "`` --
    the value plus Excel right-alignment padding.  The span bbox is therefore far wider
    than the ink.  We descend to characters (``rawdict``), drop whitespace, and rebuild
    tokens whose boxes hug the glyphs.  Column inference is only as good as these boxes.
*   **Lines must be rebuilt across blocks.**  One table row can be a dozen separate
    PyMuPDF blocks.  ``build_lines`` re-clusters tokens by baseline, which is what a
    table row actually is.

Text fidelity rules (see docs/DECISIONS.md D8):
  - Unicode space separators collapse to U+0020; zero-width chars are dropped.
  - The seven Latin f-ligatures are expanded (a human reads "fi", not U+FB01) and the
    token records ``had_ligature`` so the audit trail keeps the fact.
  - Nothing else is touched.  Unicode minus, en-dash, superscript digits and currency
    glyphs survive into ``raw_text`` unchanged; S7 interprets them additively.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

# PyMuPDF span flags
FLAG_SUPERSCRIPT = 1
FLAG_ITALIC = 2
FLAG_SERIF = 4
FLAG_MONO = 8
FLAG_BOLD = 16

LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st",
}
# Space separators that are semantically a plain space inside a cell.
SPACE_CHARS = (
    "           "
    "    　  \t\r\n\v\f"
)
ZERO_WIDTH = "­​‌‍⁠﻿"


def clean_char(c):
    """Map one glyph to its cell-text form.  May return empty (drop) or 2+ chars."""
    if c in ZERO_WIDTH:
        return ""
    if c in SPACE_CHARS:
        return " "
    return LIGATURES.get(c, c)


@dataclass
class Token:
    """A run of non-space glyphs with a tight bounding box."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    font: str
    flags: int
    baseline: float
    block: int
    line_no: int
    color: int = 0
    had_ligature: bool = False

    @property
    def bbox(self):
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2.0

    @property
    def bold(self):
        return bool(self.flags & FLAG_BOLD) or "bold" in self.font.lower()

    @property
    def italic(self):
        return bool(self.flags & FLAG_ITALIC) or "italic" in self.font.lower()

    @property
    def superscript(self):
        return bool(self.flags & FLAG_SUPERSCRIPT)


@dataclass
class TextLine:
    """Tokens sharing a baseline, left to right.  A table row is one of these."""

    tokens: list
    y0: float
    y1: float
    baseline: float

    @property
    def x0(self):
        return min(t.x0 for t in self.tokens)

    @property
    def x1(self):
        return max(t.x1 for t in self.tokens)

    @property
    def bbox(self):
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def text(self):
        return " ".join(t.text for t in self.tokens)

    @property
    def size(self):
        return max(t.size for t in self.tokens)

    def x_intervals(self):
        return [(t.x0, t.x1) for t in self.tokens]


@dataclass
class PageText:
    page_no: int
    width: float
    height: float
    rotation: int
    tokens: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    n_chars: int = 0
    n_vertical_lines: int = 0

    @property
    def body_font_size(self):
        """Modal font size weighted by character count -- robust to a big title."""
        if not self.tokens:
            return 10.0
        hist = {}
        for t in self.tokens:
            k = round(t.size * 2) / 2.0
            hist[k] = hist.get(k, 0) + len(t.text)
        return max(sorted(hist.items()), key=lambda kv: (kv[1], -kv[0]))[0]


def extract_page_text(page, page_no, vertical_tol=0.35):
    """Build the token model for one page from PyMuPDF ``rawdict`` output."""
    rect = page.rect
    pt = PageText(page_no=page_no, width=float(rect.width), height=float(rect.height),
                  rotation=int(page.rotation))
    raw = page.get_text("rawdict", sort=False)
    n_vert = 0
    for bi, block in enumerate(raw.get("blocks", [])):
        if block.get("type", 0) != 0:
            continue
        btext = []
        for li, line in enumerate(block.get("lines", [])):
            d = line.get("dir", (1.0, 0.0))
            if abs(d[1]) > vertical_tol and abs(d[0]) < 0.95:
                n_vert += 1
            ltext = []
            for span in line.get("spans", []):
                size = float(span.get("size", 10.0))
                font = str(span.get("font", ""))
                flags = int(span.get("flags", 0))
                color = int(span.get("color", 0) or 0)
                chars, boxes, lig = [], [], [False]

                def flush(size=size, font=font, flags=flags, bi=bi,
                          li=li, color=color):
                    if not chars:
                        return
                    pt.tokens.append(Token(
                        text="".join(chars),
                        x0=min(b[0] for b in boxes), y0=min(b[1] for b in boxes),
                        x1=max(b[2] for b in boxes), y1=max(b[3] for b in boxes),
                        size=size, font=font, flags=flags,
                        baseline=boxes[0][3], block=bi, line_no=li,
                        color=color, had_ligature=lig[0]))
                    del chars[:]
                    del boxes[:]
                    lig[0] = False

                for ch in span.get("chars", []):
                    c = ch.get("c", "")
                    mapped = clean_char(c)
                    if mapped == "":
                        continue
                    if mapped == " ":
                        flush()
                        ltext.append(" ")
                        continue
                    bb = ch.get("bbox", (0, 0, 0, 0))
                    if bb[2] - bb[0] <= 0 and bb[3] - bb[1] <= 0:
                        continue
                    if c in LIGATURES:
                        lig[0] = True
                    chars.append(mapped)
                    boxes.append(bb)
                    ltext.append(mapped)
                    pt.n_chars += 1
                flush()
            btext.append("".join(ltext))
        bb = block.get("bbox", (0, 0, 0, 0))
        pt.blocks.append({"bbox": tuple(float(v) for v in bb),
                          "text": "\n".join(btext), "block": bi})
    pt.n_vertical_lines = n_vert
    pt.lines = build_lines(pt.tokens)
    return pt


def build_lines(tokens, tol_frac=0.45):
    """Re-cluster tokens into geometric lines by baseline.

    Tolerance scales with font size (``tol_frac * size``) so a 6-pt footnote and a
    22-pt heading both cluster sanely.  Cross-block by design: an Excel-printed table
    row is one line here even though PyMuPDF called it eleven blocks.
    """
    if not tokens:
        return []
    order = sorted(range(len(tokens)),
                   key=lambda i: (round(tokens[i].baseline, 2), tokens[i].x0, i))
    groups = []
    cur = [order[0]]
    ref = tokens[order[0]].baseline
    ref_size = tokens[order[0]].size
    for i in order[1:]:
        t = tokens[i]
        tol = max(1.2, tol_frac * max(ref_size, t.size))
        if abs(t.baseline - ref) <= tol:
            cur.append(i)
            ref = (ref * (len(cur) - 1) + t.baseline) / len(cur)
            ref_size = max(ref_size, t.size)
        else:
            groups.append(cur)
            cur = [i]
            ref, ref_size = t.baseline, t.size
    groups.append(cur)

    lines = []
    for g in groups:
        toks = sorted((tokens[i] for i in g), key=lambda t: (t.x0, t.y0))
        lines.append(TextLine(
            tokens=toks,
            y0=min(t.y0 for t in toks), y1=max(t.y1 for t in toks),
            baseline=sum(t.baseline for t in toks) / len(toks)))
    lines.sort(key=lambda ln: (round(ln.baseline, 2), ln.x0))
    return lines


def join_tokens(tokens):
    """Render tokens as cell text: reading order within the cell, single spaces.

    Multi-line cells are joined with a single space rather than a newline: string
    comparison and Excel round-tripping both behave badly with embedded newlines, and
    the line structure stays recoverable from the token boxes.
    """
    if not tokens:
        return ""
    parts = []
    for ln in build_lines(list(tokens)):
        buf = []
        prev = None
        for t in ln.tokens:
            if prev is not None and t.x0 - prev.x1 > 0.15 * max(prev.size, t.size):
                buf.append(" ")
            buf.append(t.text)
            prev = t
        parts.append("".join(buf))
    return " ".join(p.strip() for p in parts if p.strip())


def norm_for_repeat(s):
    """Collapse a line to a shape signature for running-header detection.

    "Page | 7" and "Page | 12" must collide, so digit runs become a single hash mark.
    Case and punctuation spacing are folded too.  This is the key to S2 repeat
    detection: a running footer differs across pages only in its page number.
    """
    s = unicodedata.normalize("NFKC", s).lower()
    out = []
    prev_digit = False
    for c in s:
        if c.isdigit():
            if not prev_digit:
                out.append("#")
            prev_digit = True
        else:
            prev_digit = False
            if c.isalnum():
                out.append(c)
            elif c.isspace():
                if out and out[-1] != " ":
                    out.append(" ")
            else:
                out.append(c)
    return "".join(out).strip()
