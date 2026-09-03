"""S5 -- cell text extraction.

For a digital page the text layer *is* the answer and re-OCRing it can only lose
information, so the only work here is assembling the tokens S4 assigned to each cell
into a faithful string and recording what we noticed while doing it (wrapping,
superscripts, watermark overlap, straddling).

For a raster region we OCR **the cell crop, not the page**.  Cropping first is what
makes per-cell confidence meaningful: a page-level OCR pass gives one word-confidence
stream that has to be re-associated with cells by geometry, and the association is
exactly where the errors hide.

The OCR backend is pluggable and *fails closed*.  If no engine is installed,
:class:`NullBackend` returns nothing, the cell is written with ``raw_text = NULL``,
``source = 'none'``, confidence 0 and an ``OCR_UNAVAILABLE`` issue.  The brief asks
directly what happens to a genuinely unreadable table -- "flag it, write an issues row,
and do not guess" -- and this is that behaviour, reached by the same code path whether
the engine is missing or the crop is illegible.

One repair is allowed, under a strict condition.  In a column already typed numeric, if
the OCR string does not parse as a number but does parse after the standard glyph
confusions (``O->0``, ``l/I->1``, ``S->5``, ``B->8``) are undone, the repaired value is
used, flagged ``OCR_DIGIT_REPAIR``, and penalised in S9.  The condition matters: the
repair only ever turns a non-number into a number, so it can never silently rewrite one
valid number into a different valid number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .textlayer import build_lines, join_tokens

SRC_TEXT = "text-layer"
SRC_OCR = "ocr"
SRC_NONE = "none"

DIGIT_CONFUSIONS = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0",
                                  "l": "1", "I": "1", "|": "1",
                                  "S": "5", "s": "5", "B": "8", "Z": "2",
                                  "g": "9", "G": "6"})
SUPERSCRIPT_RE = re.compile(r"^[*†‡§¶0-9a-z]{1,3}$")


@dataclass
class ExtractedCell:
    row: int
    col: int
    row_span: int
    col_span: int
    is_header: bool
    raw_text: str
    bbox: tuple
    source: str
    ocr_conf: float = None
    footnote_markers: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    n_lines: int = 1
    tokens: list = field(default_factory=list)


def _is_watermark(t, body_size: float) -> bool:
    """Large, pale glyphs behind the text: a watermark or stamp.

    Two signals together, because either alone has false positives: a font much larger
    than the body *and* a colour close to white.  A big black heading fails the colour
    test; pale small print fails the size test.
    """
    if t.size < 1.8 * max(body_size, 1.0):
        return False
    r = (t.color >> 16) & 0xFF
    g = (t.color >> 8) & 0xFF
    b = t.color & 0xFF
    return min(r, g, b) >= 150


def extract_text_cells(grid, pt, cfg: Config):
    """Assemble ``ExtractedCell`` objects for a digital region."""
    body = pt.body_font_size
    out = []
    for gc in sorted(grid.cells, key=lambda c: (c.row, c.col)):
        toks = list(gc.tokens)
        marks = []
        flags = list(gc.flags)
        if gc.straddled:
            flags.append("TOKEN_STRADDLES_COLUMN")

        wm = [t for t in toks if _is_watermark(t, body)]
        if wm:
            flags.append("WATERMARK_OVERLAP")
            # Keep watermark glyphs out of the value but remember they were there.
            toks = [t for t in toks if t not in wm]

        # Superscript footnote markers: recorded separately, left in raw_text.
        for t in toks:
            if t.superscript and SUPERSCRIPT_RE.match(t.text.strip()):
                marks.append(t.text.strip())
            elif (t.size <= 0.75 * body and SUPERSCRIPT_RE.match(t.text.strip())
                  and t.y1 < min((o.y1 for o in toks if o is not t), default=t.y1) - 0.5):
                marks.append(t.text.strip())

        raw = join_tokens(toks)
        n_lines = len(build_lines(toks)) if toks else 1
        if n_lines > 1:
            flags.append("MULTILINE_CELL")
        if any(t.had_ligature for t in toks):
            flags.append("LIGATURE_EXPANDED")

        out.append(ExtractedCell(
            row=gc.row, col=gc.col, row_span=gc.row_span, col_span=gc.col_span,
            is_header=gc.is_header, raw_text=raw, bbox=gc.bbox,
            source=SRC_TEXT if toks else SRC_NONE,
            footnote_markers=sorted(set(marks)), flags=flags, n_lines=n_lines,
            tokens=toks))
    return out


# ------------------------------------------------------------------ OCR backends


class OcrBackend:
    name = "none"
    available = False

    def read(self, img: np.ndarray, numeric_hint: bool = False):
        """Return ``(text, mean_confidence_0_1)`` or ``(None, None)``."""
        return None, None


class NullBackend(OcrBackend):
    """No engine installed.  Every raster cell becomes an explicit refusal."""

    name = "null"
    available = False


class TesseractBackend(OcrBackend):
    name = "tesseract"

    def __init__(self):
        self.available = False
        self._pt = None
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            self._pt = pytesseract
            self.available = True
        except Exception:
            self.available = False

    def read(self, img: np.ndarray, numeric_hint: bool = False):
        if not self.available:
            return None, None
        from PIL import Image

        pil = Image.fromarray(img, mode="L")
        # psm 7 = one text line; psm 6 = uniform block.  A cell is usually one line.
        psm = 6 if img.shape[0] > 2.2 * _est_line_height(img) else 7
        cfg = "--oem 1 --psm %d -c tessedit_char_blacklist=" % psm
        try:
            data = self._pt.image_to_data(
                pil, config=cfg, output_type=self._pt.Output.DICT)
        except Exception:
            return None, None
        words, confs = [], []
        for txt, conf in zip(data.get("text", []), data.get("conf", [])):
            if txt and txt.strip():
                words.append(txt.strip())
                try:
                    c = float(conf)
                except (TypeError, ValueError):
                    c = -1.0
                if c >= 0:
                    confs.append(c / 100.0)
        if not words:
            return "", 0.0
        return " ".join(words), (sum(confs) / len(confs) if confs else 0.0)


def _est_line_height(img: np.ndarray) -> float:
    """Rough text line height from the ink row profile, for choosing a psm mode."""
    ink = img <= 200
    rows = np.flatnonzero(ink.any(axis=1))
    if rows.size == 0:
        return float(img.shape[0])
    return float(rows[-1] - rows[0] + 1)


def get_backend(cfg: Config) -> OcrBackend:
    b = TesseractBackend()
    return b if b.available else NullBackend()


def repair_numeric_ocr(text: str):
    """Undo standard glyph confusions -- only if that turns a non-number into one."""
    from .s7_normalize import parse_number

    if text is None:
        return None, []
    val, _ = parse_number(text.strip().strip("()").lstrip("+-"))
    if val is not None:
        return text, []
    fixed = text.translate(DIGIT_CONFUSIONS)
    if fixed == text:
        return text, []
    val2, _ = parse_number(fixed.strip().strip("()").lstrip("+-"))
    if val2 is None:
        return text, []
    return fixed, ["OCR_DIGIT_REPAIR"]


def extract_ocr_cells(grid, page, raster, cfg: Config, backend: OcrBackend,
                      numeric_cols=()):
    """OCR every cell of a raster region, one crop at a time."""
    import pymupdf

    out = []
    scale = cfg.ocr_cell_dpi / 72.0
    for gc in sorted(grid.cells, key=lambda c: (c.row, c.col)):
        x0, y0, x1, y1 = gc.bbox
        flags = list(gc.flags)
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            out.append(ExtractedCell(
                row=gc.row, col=gc.col, row_span=gc.row_span, col_span=gc.col_span,
                is_header=gc.is_header, raw_text="", bbox=gc.bbox, source=SRC_NONE,
                flags=flags + ["DEGENERATE_CELL_BOX"]))
            continue
        if not backend.available:
            out.append(ExtractedCell(
                row=gc.row, col=gc.col, row_span=gc.row_span, col_span=gc.col_span,
                is_header=gc.is_header, raw_text=None, bbox=gc.bbox, source=SRC_NONE,
                ocr_conf=0.0, flags=flags + ["OCR_UNAVAILABLE"]))
            continue

        pad = 1.5
        clip = pymupdf.Rect(max(0, x0 - pad), max(0, y0 - pad),
                            x1 + pad, y1 + pad)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip,
                              colorspace=pymupdf.csGRAY, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width).copy()
        del pix
        numeric = gc.col in numeric_cols
        text, conf = backend.read(img, numeric_hint=numeric)
        if text is None:
            out.append(ExtractedCell(
                row=gc.row, col=gc.col, row_span=gc.row_span, col_span=gc.col_span,
                is_header=gc.is_header, raw_text=None, bbox=gc.bbox, source=SRC_NONE,
                ocr_conf=0.0, flags=flags + ["OCR_FAILED"]))
            continue
        if numeric:
            text, rflags = repair_numeric_ocr(text)
            flags += rflags
        out.append(ExtractedCell(
            row=gc.row, col=gc.col, row_span=gc.row_span, col_span=gc.col_span,
            is_header=gc.is_header, raw_text=text, bbox=gc.bbox, source=SRC_OCR,
            ocr_conf=conf, flags=flags))
    return out
