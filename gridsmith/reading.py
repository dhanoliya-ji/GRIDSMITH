"""Reading the PDF: text layer, S1 page intake and S2 page layout.

Sections, in order: textlayer, s1_intake, s2_layout.
"""
from __future__ import annotations

import math
import numpy as np
import pymupdf
import re
import unicodedata
from dataclasses import dataclass, field

from .core import Config, area, intersect, median


# --------------------------------------------------------------------------------------------------
# Token model over the PDF text layer.
# --------------------------------------------------------------------------------------------------
# Why a custom model instead of using ``page.get_text("dict")`` as-is:
#
# *   **Reading order is useless in real financial PDFs.**  In ``Shell.pdf`` (an Excel
#     sheet printed to PDF) the header row ``Particulars | Note | As at ...`` is the
#     *last* thing in the content stream.  Any extractor that trusts block order emits
#     garbage.  So we discard stream order and keep only geometry.
# *   **Cells need tight boxes.**  A span in that same file is ``"30.05           "`` --
#     the value plus Excel right-alignment padding.  The span bbox is therefore far wider
#     than the ink.  We descend to characters (``rawdict``), drop whitespace, and rebuild
#     tokens whose boxes hug the glyphs.  Column inference is only as good as these boxes.
# *   **Lines must be rebuilt across blocks.**  One table row can be a dozen separate
#     PyMuPDF blocks.  ``build_lines`` re-clusters tokens by baseline, which is what a
#     table row actually is.
#
# Text fidelity rules (see docs/DECISIONS.md D8):
#   - Unicode space separators collapse to U+0020; zero-width chars are dropped.
#   - The seven Latin f-ligatures are expanded (a human reads "fi", not U+FB01) and the
#     token records ``had_ligature`` so the audit trail keeps the fact.
#   - Nothing else is touched.  Unicode minus, en-dash, superscript digits and currency
#     glyphs survive into ``raw_text`` unchanged; S7 interprets them additively.

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


# --------------------------------------------------------------------------------------------------
# S1 -- document intake and per-page typing.
# --------------------------------------------------------------------------------------------------
# Per page, not per document (the brief is explicit, and it matters: a scanned exhibit
# pasted into an otherwise digital annual report is the common case).
#
# Three classes:
#
# ``digital``
#     Enough real glyphs that the text layer *is* the ground truth.  We never re-OCR
#     these -- OCR can only lose information we already have exactly.
# ``scanned``
#     No usable text layer, one big raster.  Needs deskew/denoise, then per-cell OCR.
# ``hybrid``
#     Both: a text layer that covers part of the page plus a raster that covers a region
#     the text layer does not reach.  We extract text-layer regions from the text layer
#     and raster regions by OCR, and record which was used per cell.
#
# Geometry normalisation done here:
#
# *   ``/Rotate`` pages -- PyMuPDF applies the rotation matrix for us, so text and
#     renders agree and nothing downstream needs to know.  We record the value.
# *   **Rotated content without /Rotate** (a landscape table dropped into a portrait
#     page).  Detected from the writing direction of the text lines; the affected region
#     is transposed in :func:`transpose_box` so S3/S4 see an upright table.
# *   **Skew** on rasters, estimated by a sheared projection-profile search (below) and
#     corrected once, at intake, so every later stage works in one frame.
#
# The skew estimator deserves a note.  The textbook approach rotates the image at each
# candidate angle and scores the horizontal projection profile; that is O(pixels) *image
# rotations*.  Instead we take the ink pixel coordinates once and, for each candidate
# angle, shear them (``y' = y - x*tan(a)``) and histogram with ``np.bincount``.  Same
# objective, no interpolation, ~50x faster, and exactly reproducible because it never
# touches a resampling filter.  Bias note: the projection range grows with ``|a|``, which
# slightly favours 0 degrees -- a conservative bias, since the failure we care about is
# rotating a page that was never skewed.

PAGE_DIGITAL = "digital"
PAGE_SCANNED = "scanned"
PAGE_HYBRID = "hybrid"
PAGE_EMPTY = "empty"


@dataclass
class PageRaster:
    """A rendered, deskewed page image plus the transform back to PDF points."""

    gray: np.ndarray          # uint8, 0 = ink
    dpi: int
    skew_deg: float
    scale: float              # px per point
    offset: tuple = (0.0, 0.0)  # px added by the deskew expand

    def to_px(self, box):
        x0, y0, x1, y1 = box
        s, (ox, oy) = self.scale, self.offset
        return (x0 * s + ox, y0 * s + oy, x1 * s + ox, y1 * s + oy)

    def to_pt(self, box):
        x0, y0, x1, y1 = box
        s, (ox, oy) = self.scale, self.offset
        return ((x0 - ox) / s, (y0 - oy) / s, (x1 - ox) / s, (y1 - oy) / s)

    def ink_mask(self, thresh=None):
        t = otsu(self.gray) if thresh is None else thresh
        return self.gray <= t


@dataclass
class PageInfo:
    page_no: int
    width: float
    height: float
    rotation: int
    page_type: str
    text: PageText
    image_boxes: list = field(default_factory=list)
    image_area_frac: float = 0.0
    char_density: float = 0.0
    skew_deg: float = 0.0
    content_rotation: int = 0     # 0 / 90 / 270 rotation of the *content*
    raster_regions: list = field(default_factory=list)  # boxes needing OCR
    notes: list = field(default_factory=list)


def open_document(path: str):
    """Open a PDF, dealing with the empty-password encryption EDGAR uses."""
    doc = pymupdf.open(path)
    if doc.is_encrypted and doc.needs_pass:
        if not doc.authenticate(""):
            raise RuntimeError(
                "PDF is password protected; gridsmith will not guess passwords")
    return doc


# --------------------------------------------------------------------------- typing


def classify_page(page, pt: PageText, cfg: Config) -> PageInfo:
    """Assign a page type from text-layer density versus raster coverage."""
    w, h = pt.width, pt.height
    page_area = max(1.0, w * h)
    img_boxes = []
    for blk in page.get_text("dict").get("blocks", []):
        if blk.get("type", 0) == 1:
            bb = tuple(float(v) for v in blk.get("bbox", (0, 0, 0, 0)))
            if area(bb) >= 1.0:
                img_boxes.append(bb)
    # Also catch images placed without a text block (rare but happens with XObjects).
    for info in page.get_image_info(xrefs=False):
        bb = tuple(float(v) for v in info.get("bbox", (0, 0, 0, 0)))
        if area(bb) >= 1.0 and not any(_near(bb, o) for o in img_boxes):
            img_boxes.append(bb)
    img_boxes.sort()

    covered = _union_area(img_boxes, (0.0, 0.0, w, h))
    img_frac = covered / page_area
    density = pt.n_chars / (page_area / 1000.0)

    if pt.n_chars < cfg.digital_min_chars:
        ptype = PAGE_SCANNED if img_frac > 0.20 else PAGE_EMPTY
    elif img_frac >= cfg.scan_image_area_frac and density < cfg.hybrid_min_char_density:
        ptype = PAGE_SCANNED
    elif img_frac >= 0.10:
        ptype = PAGE_HYBRID
    else:
        ptype = PAGE_DIGITAL

    info = PageInfo(page_no=pt.page_no, width=w, height=h, rotation=int(page.rotation),
                    page_type=ptype, text=pt, image_boxes=img_boxes,
                    image_area_frac=img_frac, char_density=density)

    if ptype == PAGE_HYBRID:
        info.raster_regions = _text_free_images(img_boxes, pt, cfg)
        if not info.raster_regions:
            # Images exist but the text layer covers them (chart labels, logos).
            info.page_type = PAGE_DIGITAL
            info.notes.append("hybrid_demoted_text_covers_images")
    elif ptype == PAGE_SCANNED:
        info.raster_regions = [(0.0, 0.0, w, h)]

    info.content_rotation = detect_content_rotation(pt, cfg)
    return info


def _near(a, b, tol=1.0):
    return all(abs(a[i] - b[i]) <= tol for i in range(4))


def _union_area(boxes, clip):
    """Area of the union of boxes, clipped.  Boxes here are few, so O(n^2) is fine."""
    if not boxes:
        return 0.0
    clipped = [intersect(b, clip) for b in boxes]
    clipped = [b for b in clipped if b]
    if not clipped:
        return 0.0
    xs = sorted({v for b in clipped for v in (b[0], b[2])})
    ys = sorted({v for b in clipped for v in (b[1], b[3])})
    total = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in clipped):
                total += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return total


def _text_free_images(img_boxes, pt: PageText, cfg: Config):
    """Image regions that the text layer does not explain -- these need OCR."""
    out = []
    for bb in img_boxes:
        if area(bb) < 4000.0:
            continue          # logo / rule / bullet, not a pasted exhibit
        inside = sum(1 for t in pt.tokens
                     if bb[0] <= t.cx <= bb[2] and bb[1] <= t.cy <= bb[3])
        # A pasted scan has ink but no glyphs of its own.  Allow a couple of stray
        # tokens (a caption overlapping the frame) before calling it digital.
        if inside <= 2:
            out.append(bb)
    return out


def detect_content_rotation(pt: PageText, cfg: Config) -> int:
    """Detect a page whose *content* is rotated 90/270 without a /Rotate entry.

    Signal: the fraction of text lines whose writing direction is vertical.  We only
    claim rotation when it is the clear majority, because a single vertical axis label
    on a chart is normal and must not transpose the page.
    """
    if not pt.lines:
        return 0
    n_vert = pt.n_vertical_lines
    total = max(1, sum(len(b.get("text", "").splitlines()) or 1 for b in pt.blocks))
    if n_vert / total < 0.6 or n_vert < 4:
        return 0
    # Direction sign tells 90 from 270.  Positive dy means glyphs advance downward.
    return 90


def transpose_box(box, page_w: float, page_h: float, rot: int):
    """Map a box between upright space and 90/270-rotated content space."""
    x0, y0, x1, y1 = box
    if rot == 90:
        return (y0, page_w - x1, y1, page_w - x0)
    if rot == 270:
        return (page_h - y1, x0, page_h - y0, x1)
    return box


# ------------------------------------------------------------------------- rasterise


def render_page(page, cfg: Config, clip=None, dpi=None) -> PageRaster:
    """Render to grayscale numpy at ``dpi``.  Never keeps the Pixmap alive."""
    dpi = dpi or cfg.raster_dpi
    scale = dpi / 72.0
    mat = pymupdf.Matrix(scale, scale)
    kw = {"matrix": mat, "colorspace": pymupdf.csGRAY, "alpha": False}
    if clip is not None:
        kw["clip"] = pymupdf.Rect(*clip)
    pix = page.get_pixmap(**kw)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    off = (0.0, 0.0)
    if clip is not None:
        off = (-clip[0] * scale, -clip[1] * scale)
    del pix
    return PageRaster(gray=arr, dpi=dpi, skew_deg=0.0, scale=scale, offset=off)


def otsu(gray: np.ndarray) -> int:
    """Otsu threshold from a 256-bin histogram.  Pure numpy, fully deterministic."""
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 128
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return int(np.argmax(sigma_b))


def estimate_skew(gray: np.ndarray, cfg: Config) -> float:
    """Skew angle in degrees (positive = content rotated clockwise).

    Two-pass search: 1.0 degree steps over the full range, then ``deskew_step_deg``
    around the winner.  Returns 0.0 unless the best angle beats 0 degrees by 2%, which
    keeps us from "correcting" an unskewed page.
    """
    thr = otsu(gray)
    ink = gray <= thr
    h, w = ink.shape
    if ink.sum() < 200:
        return 0.0
    # Subsample rows for speed; skew is a global property, 1 row in 2 is plenty.
    ys, xs = np.nonzero(ink[::2, ::2])
    if ys.size < 200:
        return 0.0
    ys = ys.astype(np.float32) * 2.0
    xs = xs.astype(np.float32) * 2.0

    def score(angle_deg: float) -> float:
        t = math.tan(math.radians(angle_deg))
        yp = ys - xs * t
        yp -= yp.min()
        bins = np.bincount(yp.astype(np.int32))
        return float(np.dot(bins, bins))

    base = score(0.0)
    # Coarse step must be finer than the peak.  At 200 dpi a 1 degree error shifts the
    # far edge of an A4 page by ~28 px, comfortably more than a text line height, so
    # the projection peak is well under 1 degree wide -- and a 1 degree scan walked
    # straight past a real 1.6 degree skew whose peak was 1.83x the unskewed score.
    coarse_step = 0.25
    n_coarse = int(round(2 * cfg.deskew_max_deg / coarse_step))
    best_a, best_s = 0.0, base
    for i in range(n_coarse + 1):
        a = -cfg.deskew_max_deg + i * coarse_step
        s = score(a)
        if s > best_s:
            best_a, best_s = a, s
    lo, hi = best_a - coarse_step, best_a + coarse_step
    n = max(1, int(round((hi - lo) / cfg.deskew_step_deg)))
    for i in range(n + 1):
        a = lo + i * cfg.deskew_step_deg
        if abs(a) > cfg.deskew_max_deg:
            continue
        s = score(a)
        if s > best_s:
            best_a, best_s = a, s
    if best_s < base * 1.02:
        return 0.0
    return round(best_a, 2)


def deskew(raster: PageRaster, angle_deg: float) -> PageRaster:
    """Rotate the raster by ``-angle`` about its centre, padding with white."""
    if abs(angle_deg) < 0.05:
        return raster
    from PIL import Image

    img = Image.fromarray(raster.gray, mode="L")
    out = img.rotate(angle_deg, resample=Image.BICUBIC, expand=False, fillcolor=255)
    return PageRaster(gray=np.asarray(out, dtype=np.uint8), dpi=raster.dpi,
                      skew_deg=angle_deg, scale=raster.scale, offset=raster.offset)


def salt_pepper_ratio(gray: np.ndarray) -> float:
    """Fraction of ink pixels with no ink 4-neighbour -- a cheap scan-noise proxy."""
    ink = (gray <= otsu(gray)).astype(np.uint8)
    if ink.sum() == 0:
        return 0.0
    nb = np.zeros_like(ink)
    nb[1:, :] += ink[:-1, :]
    nb[:-1, :] += ink[1:, :]
    nb[:, 1:] += ink[:, :-1]
    nb[:, :-1] += ink[:, 1:]
    isolated = int(((ink == 1) & (nb == 0)).sum())
    return isolated / float(ink.sum())


def denoise(raster: PageRaster, ratio_threshold: float = 0.012) -> PageRaster:
    """3x3 median filter, applied only when the page is measurably noisy.

    Unconditional median filtering thins hairline rules, which is exactly the signal
    S3 depends on, so we gate it on a noise measurement instead.
    """
    if salt_pepper_ratio(raster.gray) < ratio_threshold:
        return raster
    from PIL import Image, ImageFilter

    img = Image.fromarray(raster.gray, mode="L").filter(ImageFilter.MedianFilter(3))
    return PageRaster(gray=np.asarray(img, dtype=np.uint8), dpi=raster.dpi,
                      skew_deg=raster.skew_deg, scale=raster.scale,
                      offset=raster.offset)


def raster_orientation(gray: np.ndarray) -> tuple:
    """Guess 0 vs 90 degree text orientation on an image-only page.

    Latin text lines make the projection profile *across* the lines strongly periodic
    and high-variance, while the profile along them is smooth.  Comparing the two
    variances gives the axis.  Returns ``(degrees, confidence)`` where confidence is
    the normalised margin; the caller flags anything under ~0.15 rather than trusting
    it, and 90-vs-270 is left to :func:`raster_updown` / Tesseract OSD.
    """
    ink = (gray <= otsu(gray))
    if ink.sum() < 200:
        return 0, 0.0
    rows = ink.sum(axis=1).astype(np.float64)
    cols = ink.sum(axis=0).astype(np.float64)

    def periodicity(p):
        if p.sum() <= 0:
            return 0.0
        p = p / (p.mean() + 1e-9)
        d = np.diff(p)
        return float(np.var(d))

    pr, pc = periodicity(rows), periodicity(cols)
    total = pr + pc + 1e-9
    if pr >= pc:
        return 0, float((pr - pc) / total)
    return 90, float((pc - pr) / total)


def prepare_raster(page, info: PageInfo, cfg: Config) -> PageRaster:
    """Render + deskew + denoise a scanned/hybrid page, recording what we did."""
    raster = render_page(page, cfg)
    ang = estimate_skew(raster.gray, cfg)
    if ang:
        if abs(ang) >= cfg.deskew_max_deg - 1e-6:
            info.notes.append("skew_beyond_limit_not_corrected")
        else:
            raster = deskew(raster, ang)
            info.skew_deg = ang
            info.notes.append("deskewed_%.2fdeg" % ang)
    before = salt_pepper_ratio(raster.gray)
    raster = denoise(raster)
    if raster.skew_deg == ang and before >= 0.012:
        info.notes.append("denoised_median3")
    if info.page_type == PAGE_SCANNED:
        deg, conf = raster_orientation(raster.gray)
        if deg and conf > 0.15:
            info.content_rotation = deg
            info.notes.append("raster_orientation_%d_conf_%.2f" % (deg, conf))
        elif conf <= 0.15:
            info.notes.append("raster_orientation_uncertain_conf_%.2f" % conf)
    return raster


# --------------------------------------------------------------------------------------------------
# S2 -- page layout segmentation.
# --------------------------------------------------------------------------------------------------
# Partitions each page into running header/footer, headings, body prose, caption lines and
# footnote lines.  Table and figure regions are *not* decided here; S3 does that, and then
# :func:`demote_in_regions` comes back to fix the roles S2 could only guess at.
#
# The running header/footer detector is the part that earns its keep.  The brief calls
# repeating headers "a classic source of false positives" and it is right: left alone, a
# gutter detector sees a beautiful three-column table on every page of the Apple 10-K.
#
# Why the obvious implementation fails
# ------------------------------------
# The textbook approach takes the top/bottom 10-15% of the page and looks for lines that
# repeat at a stable y.  That works on the ICRA report, whose footer
# ``www.icra.in  Sensitivity Label : Public  Page | 3`` sits at y=810 on every page.
#
# It completely misses the Apple 10-K.  Its footer ``Apple Inc. | 2024 Form 10-K | 22``
# appears on 57 pages, but the HTML-to-PDF converter places it immediately after the
# content, so its baseline wanders between y=510 and y=642 -- never inside the bottom
# 14% band, and never at a stable y.  Measured, not assumed: see ``tools/probe.py``.
#
# So candidates come from the union of two views:
#
# *   **geometric** -- any line inside the top/bottom band, and
# *   **structural** -- the first three and last three lines of the page, wherever they sit.
#
# and the key is ``(shape, side)`` with no y quantisation.  Dropping y from the key is
# what makes the Apple footer detectable, and it costs recall nowhere.
#
# What stops it eating a repeated table header
# --------------------------------------------
# Two guards, because this is the failure that would silently destroy a table:
#
# 1.  **Isolation.**  A running header is separated from the body by more than 1.5 line
#     gaps; a repeated table header is immediately followed by its data rows.  A
#     candidate must be isolated on at least half the pages where it appears.
# 2.  **Position stability or extremity.**  Either the y is stable across pages (a real
#     fixed footer) or the line is structurally the first/last on most of its pages.
#
# And if S3 later finds a "chrome" line sitting inside a table region anyway,
# :func:`demote_in_regions` takes the label back and records an issue, so the mistake is
# visible rather than silent.

NUM_RE = re.compile(r"^[\(\[]?[-+−–]?[\d,. ']+[\)\]%]?$")
HEADING_NUM_RE = re.compile(
    r"^(?:(?:item|section|note|annexure|appendix|schedule|part)\s+)?"
    r"(?:[IVXLC]+|\d+(?:\.\d+)*)[\.\):\-–]?\s+\S", re.I)
# A caption needs an identifier ("Table 4.2") or an explicit separator
# ("Figure - Revenue").  Without that requirement "TABLE OF CONTENTS" matches.
CAPTION_RE = re.compile(
    r"^\s*(table|figure|fig\.?|exhibit|chart|annexure|schedule|graph)\s*"
    r"(?:([0-9IVX]+(?:[\.\-][0-9A-Za-z]+)*)\s*[:.—–\-]?\s*|[:—–]\s*)(?=\S)", re.I)
# Symbol markers often run straight into the text ("*Instrument details are
# provided in Annexure-I"), so the space is optional for those; alphanumeric
# markers still need a separator or every numbered list item would match.
FOOTNOTE_MARK_RE = re.compile(
    r"^\s*(?:(\*{1,4}|†{1,2}|‡{1,2}|§|¶)\s*|(\(?[a-z]\)|\(?\d{1,2}\)|\d{1,2}[\.\)])\s+)(?=\S)")
SCALE_RE = re.compile(
    r"(?:\(|\b)(?:all\s+)?(?:figures|amounts|amount|values|numbers|rs\.?|inr|usd|\$|"
    r"₹|`|eur|gbp)?[^)\n]{0,28}?\b"
    r"(million|millions|mn|billion|bn|thousand|thousands|lakh|lakhs|lacs?|crore|crores|"
    r"cr\.?|000s|hundreds)\b", re.I)

LINE_HEADER = "header"
LINE_FOOTER = "footer"
LINE_HEADING = "heading"
LINE_BODY = "body"
LINE_CAPTION = "caption"
LINE_FOOTNOTE = "footnote"
LINE_TABLE = "table"

STRUCTURAL_EDGE = 3          # first/last N lines are chrome candidates


@dataclass
class LayoutLine:
    idx: int
    role: str
    level: int = 0
    marker: str = ""
    size: float = 0.0


@dataclass
class PageLayout:
    page_no: int
    roles: dict = field(default_factory=dict)
    header_y: float = 0.0
    footer_y: float = 1e9
    headings: list = field(default_factory=list)   # (line_idx, level, text, bbox)
    captions: list = field(default_factory=list)   # (line_idx, text, bbox)
    footnotes: list = field(default_factory=list)  # (line_idx, marker, text, bbox)
    scale_notes: list = field(default_factory=list)
    body_line_idx: list = field(default_factory=list)
    demoted: list = field(default_factory=list)    # audit trail from S3 feedback

    def role_of(self, idx: int) -> str:
        ll = self.roles.get(idx)
        return ll.role if ll else LINE_BODY

    def is_chrome(self, idx: int) -> bool:
        return self.role_of(idx) in (LINE_HEADER, LINE_FOOTER)


def _line_gaps(lines):
    return [b.baseline - a.baseline for a, b in zip(lines, lines[1:])]


def _chrome_candidates(pt, cfg: Config):
    """Yield ``(line_idx, side)`` for every chrome candidate on a page."""
    n = len(pt.lines)
    top = pt.height * cfg.header_band_frac
    bot = pt.height * (1.0 - cfg.footer_band_frac)
    seen = {}
    for i, ln in enumerate(pt.lines):
        side = None
        if ln.y1 <= top:
            side = LINE_HEADER
        elif ln.y0 >= bot:
            side = LINE_FOOTER
        elif i < STRUCTURAL_EDGE:
            side = LINE_HEADER
        elif i >= n - STRUCTURAL_EDGE:
            side = LINE_FOOTER
        if side and i not in seen:
            seen[i] = side
    return sorted(seen.items())


def find_repeating_chrome(pages, cfg: Config):
    """Document-level pass returning ``(accepted_keys, stats)``.

    A key is ``(shape, side)``.  ``shape`` comes from
    :func:`gridsmith.textlayer.norm_for_repeat`, so page numbers collapse to ``#``.
    """
    n_pages = max(1, len(pages))
    pages_of: dict = {}
    iso_of: dict = {}
    ys_of: dict = {}
    edge_of: dict = {}

    for pt in pages:
        if not pt.lines:
            continue
        gaps = _line_gaps(pt.lines)
        med_gap = median(gaps) if gaps else 12.0
        n = len(pt.lines)
        for i, side in _chrome_candidates(pt, cfg):
            ln = pt.lines[i]
            shape = norm_for_repeat(ln.text)
            if not shape or len(shape) < 2 or len(ln.tokens) > 14:
                continue
            key = (shape, side)
            pages_of.setdefault(key, set()).add(pt.page_no)
            if side == LINE_HEADER:
                nxt = (pt.lines[i + 1].baseline - ln.baseline) if i + 1 < n else 1e9
            else:
                nxt = (ln.baseline - pt.lines[i - 1].baseline) if i > 0 else 1e9
            iso_of.setdefault(key, []).append(nxt >= 1.5 * max(med_gap, 1.0))
            ys_of.setdefault(key, []).append(ln.baseline)
            edge_of.setdefault(key, []).append(i < STRUCTURAL_EDGE or
                                               i >= n - STRUCTURAL_EDGE)

    need = max(cfg.repeat_min_pages, int(round(cfg.repeat_min_frac * n_pages)))
    keys, stats = set(), []
    for key in sorted(pages_of, key=lambda k: (-len(pages_of[k]), k[0], k[1])):
        seen = len(pages_of[key])
        iso = iso_of[key]
        iso_frac = sum(1 for v in iso if v) / max(1, len(iso))
        ys = ys_of[key]
        y_spread = (max(ys) - min(ys)) if len(ys) > 1 else 0.0
        y_stable = y_spread <= cfg.repeat_y_tol
        edge = edge_of[key]
        edge_frac = sum(1 for v in edge if v) / max(1, len(edge))
        accept = (seen >= need and iso_frac >= 0.5
                  and (y_stable or edge_frac >= 0.6))
        if accept:
            keys.add(key)
        if seen >= 2:
            stats.append({"shape": key[0][:64], "side": key[1], "pages": seen,
                          "isolation_frac": round(iso_frac, 3),
                          "y_spread": round(y_spread, 1),
                          "edge_frac": round(edge_frac, 3), "accepted": accept})
    return keys, {"required_pages": need, "n_pages": n_pages,
                  "candidates": stats[:60]}


def segment_page(pt, chrome_keys, cfg: Config) -> PageLayout:
    """Assign a provisional role to every line on one page.

    Heading *levels* are left at 0 here; :func:`assign_heading_levels` fills them from a
    document-wide font-size ranking, because a per-page ranking would make the first
    heading on every page a level 1 and flatten the section tree.
    """
    lay = PageLayout(page_no=pt.page_no)
    if not pt.lines:
        lay.footer_y = pt.height
        return lay

    body_size = pt.body_font_size
    cand = dict(_chrome_candidates(pt, cfg))
    header_y, footer_y = 0.0, pt.height

    for i, side in sorted(cand.items()):
        ln = pt.lines[i]
        if (norm_for_repeat(ln.text), side) in chrome_keys:
            lay.roles[i] = LayoutLine(idx=i, role=side, size=ln.size)
            if side == LINE_HEADER:
                header_y = max(header_y, ln.y1)
            else:
                footer_y = min(footer_y, ln.y0)
    lay.header_y, lay.footer_y = header_y, footer_y

    for i, ln in enumerate(pt.lines):
        if i in lay.roles:
            continue
        txt = ln.text.strip()
        if not txt:
            continue
        n_words = len(txt.split())
        big = ln.size >= body_size * cfg.heading_size_ratio
        allbold = bool(ln.tokens) and all(t.bold for t in ln.tokens)
        small = ln.size <= body_size * 0.92

        if CAPTION_RE.match(txt) and n_words <= 30:
            lay.roles[i] = LayoutLine(idx=i, role=LINE_CAPTION, size=ln.size)
            lay.captions.append((i, txt, ln.bbox))
        elif FOOTNOTE_MARK_RE.match(txt) and (
                FOOTNOTE_MARK_RE.match(txt).group(1)      # *, dagger...
                or small or ln.y0 >= pt.height * 0.55):
            m = FOOTNOTE_MARK_RE.match(txt)
            marker = m.group(1) or m.group(2) or ""
            lay.roles[i] = LayoutLine(idx=i, role=LINE_FOOTNOTE,
                                      marker=marker, size=ln.size)
            lay.footnotes.append((i, marker, txt, ln.bbox))
        elif ((big or allbold) and n_words <= cfg.heading_max_words
                and not _looks_numeric_row(ln)):
            lay.roles[i] = LayoutLine(idx=i, role=LINE_HEADING, size=ln.size)
            lay.headings.append((i, 0, txt, ln.bbox))
        else:
            lay.roles[i] = LayoutLine(idx=i, role=LINE_BODY, size=ln.size)
            lay.body_line_idx.append(i)

        if SCALE_RE.search(txt) and n_words <= 24:
            lay.scale_notes.append((i, txt, ln.bbox))
    lay.scale_notes.sort()
    return lay


def assign_heading_levels(pages, layouts, cfg: Config):
    """Rank heading font sizes across the whole document, then assign levels 1..N.

    A per-page ranking makes every page start a new level-1 section, which flattens
    "Item 7 > Segment Operating Performance" into two siblings.  Document-wide ranking
    is what makes ``section_path`` meaningful.
    """
    sizes = {}
    for pt, lay in zip(pages, layouts):
        for i, _, txt, bb in lay.headings:
            k = round(pt.lines[i].size, 1)
            sizes[k] = sizes.get(k, 0) + 1
    uniq = sorted(sizes, reverse=True)
    rank = {s: n + 1 for n, s in enumerate(uniq)}
    for pt, lay in zip(pages, layouts):
        fixed = []
        for i, _, txt, bb in lay.headings:
            sz = round(pt.lines[i].size, 1)
            lvl = rank.get(sz, len(uniq) + 1)
            if HEADING_NUM_RE.match(txt):
                depth = txt.split()[0].count(".") + 1
                lvl = max(lvl, min(depth, 4))
            fixed.append((i, lvl, txt, bb))
            if i in lay.roles:
                lay.roles[i].level = lvl
        lay.headings = fixed
    return rank


def demote_in_regions(pt, lay, regions, issues_out=None):
    """S3 feedback: any line inside a detected table region is table content.

    Fixes two real problems on the ICRA report:
      * bold table header rows ("Instrument | Amount | Rating action") were labelled
        headings and leaked into ``section_path`` -- so the path for a table read
        ``[... , "Others"]``, "Others" being half of a cell;
      * a chrome line overlapping a table would have removed a real row.

    Every change is recorded in ``lay.demoted`` and, for chrome, as an issue: a running
    header that turns out to sit inside a table is a detector disagreement a human
    should see.
    """
    if not regions:
        return
    for i, ln in enumerate(pt.lines):
        role = lay.role_of(i)
        if role in (LINE_BODY, LINE_TABLE):
            continue
        cy = (ln.y0 + ln.y1) / 2.0
        for rg in regions:
            x0, y0, x1, y1 = rg
            if y0 - 1.0 <= cy <= y1 + 1.0 and ln.x1 > x0 and ln.x0 < x1:
                lay.demoted.append((i, role, ln.text[:60]))
                if role in (LINE_HEADER, LINE_FOOTER) and issues_out is not None:
                    issues_out.append({
                        "severity": "warn", "code": "CHROME_INSIDE_TABLE",
                        "message": ("line %r was classified as a running %s but falls "
                                    "inside a detected table region on page %d"
                                    % (ln.text[:60], role, pt.page_no))})
                lay.roles[i] = LayoutLine(idx=i, role=LINE_TABLE, size=ln.size)
                lay.headings = [h for h in lay.headings if h[0] != i]
                lay.captions = [c for c in lay.captions if c[0] != i]
                break


def _looks_numeric_row(ln) -> bool:
    """Bold numeric rows ("Total  4,812  4,109") are table rows, not headings."""
    toks = [t.text for t in ln.tokens]
    if len(toks) < 2:
        return False
    nums = sum(1 for t in toks if NUM_RE.match(t))
    return nums >= 2 and nums / len(toks) >= 0.4


def build_section_events(pages, layouts):
    """Flatten headings into ``(page_no, y, level, text)``, document order."""
    events = []
    for pt, lay in zip(pages, layouts):
        for i, lvl, txt, bb in lay.headings:
            events.append((pt.page_no, round(float(bb[1]), 3), int(lvl), txt))
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    return events


def section_path_at(events, page_no: int, y: float):
    """Heading stack in force at ``(page_no, y)``."""
    stack = []
    for pno, ey, lvl, txt in events:
        if (pno, ey) > (page_no, y):
            break
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        stack.append((lvl, txt))
    return [t for _, t in stack]
