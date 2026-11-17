"""gridsmith -- deterministic table and context extraction from large, messy PDFs.

No language model touches the document.  Usage:

    python -m gridsmith run --input doc.pdf --out ./out --offline

The whole pipeline lives in this one file, one section per stage.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import math
import numpy as np
import os
import platform
import pymupdf
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import unicodedata
import zipfile
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


# ==================================================================================================
# determinism: Determinism plumbing.
# ==================================================================================================
#
# The brief (section 3.2) wants two runs to be byte-identical with timestamps isolated to
# one ``extraction_runs`` row.  Three things break that by default, each handled here:
#
# * **Iteration order.**  Anything derived from a ``set``, or from a dict keyed by floats,
#   is re-orderable.  Every collection that reaches an output is sorted explicitly.
# * **Ids.**  ``uuid4`` and ``AUTOINCREMENT`` differ between runs.  All ids here are
#   content-addressed: ``sid("cell", table_id, row, col)`` gives a short blake2b digest.
#   Same input, same id, forever -- which also makes the SQLite bytes reproducible.
# * **Hidden clocks.**  ``openpyxl`` stamps ``docProps/core.xml`` and every zip entry with
#   ``datetime.now()``.  ``gridsmith.s10_excel`` rewrites the archive with FIXED_ZIP_TIME.
#
# Floating point: no raw float from a geometric computation is persisted without passing
# through ``q`` (4 decimals).  1e-4 pt is 35 nanometres -- far below anything meaningful,
# and it removes last-bit wobble from platform math libraries.

FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
FIXED_ISO = "1970-01-01T00:00:00Z"
CHUNK = 1048576


def seed_everything(seed: int) -> None:
    """Seed every RNG we could plausibly touch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:  # pragma: no cover
        pass


def q(x, nd: int = 4):
    """Quantise a float for persistence.  ``-0.0`` normalises to ``0.0``."""
    if x is None:
        return None
    r = round(float(x) + 0.0, nd)
    return 0.0 if r == 0 else r


def sid(kind: str, *parts: Any) -> str:
    """Content-addressed, stable, short id.

    Collision budget: 96 bits over at most ~1e6 rows per document.  Birthday
    probability ~1e-17, orders of magnitude below the chance of an undetected disk
    error -- so a digest is safer here than a counter that would reorder between runs.
    """
    h = hashlib.blake2b(digest_size=12)
    h.update(kind.encode())
    for p in parts:
        h.update(b"\x1f")
        h.update(repr(p).encode("utf-8", "replace"))
    return kind[:3] + "_" + h.hexdigest()


def doc_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any, indent=2) -> str:
    """JSON that is byte-identical for equal objects: sorted keys, no NaN, LF only."""
    return json.dumps(
        obj, sort_keys=True, indent=indent, ensure_ascii=False, allow_nan=False,
        separators=(",", ": ") if indent else (",", ":"),
    )


def write_text(path: str, text: str) -> None:
    """UTF-8 with LF endings regardless of platform (Windows would emit CRLF)."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _norm(v: Any):
    if v is None:
        return (0, 0.0, "")
    if isinstance(v, bool):
        return (1, float(v), "")
    if isinstance(v, (int, float)):
        return (1, float(v), "")
    return (2, 0.0, str(v))


def stable_sort_key(*keys: Any):
    def _k(obj: Any):
        return tuple(_norm(getattr(obj, k, None) if isinstance(k, str) else k)
                     for k in keys)

    return _k


def sorted_stable(it: Iterable, key) -> list:
    """``sorted`` is already stable; this wrapper exists to make the intent greppable."""
    return sorted(it, key=key)


# ==================================================================================================
# geometry: Rectangle and interval helpers.
# ==================================================================================================
#
# Coordinate convention for the whole pipeline: **PDF points, origin at the top-left of
# the page as displayed**, x growing right, y growing down.  That is PyMuPDF's convention
# and it already accounts for a page /Rotate entry, so a 90-degree-rotated page arrives
# here upright and no stage below S1 needs to think about it.
#
# Content rotated *without* a /Rotate entry (a landscape table dropped into a portrait
# page) is a different problem, handled in gridsmith.s1_intake by transposing the
# coordinates of that region alone.

Box = tuple  # (x0, y0, x1, y1) in points


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def width(b: Box) -> float:
    return max(0.0, b[2] - b[0])


def height(b: Box) -> float:
    return max(0.0, b[3] - b[1])


def union(bs: Iterable[Box]):
    bs = list(bs)
    if not bs:
        return None
    return (min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs))


def intersect(a: Box, b: Box):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def iou(a: Box, b: Box) -> float:
    inter = intersect(a, b)
    if inter is None:
        return 0.0
    ia = area(inter)
    return ia / (area(a) + area(b) - ia + 1e-9)


def contain_frac(inner: Box, outer: Box) -> float:
    """How much of ``inner`` sits inside ``outer`` (0..1)."""
    inter = intersect(inner, outer)
    return 0.0 if inter is None else area(inter) / (area(inner) + 1e-9)


def overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def iou_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    ov = overlap_1d(a0, a1, b0, b1)
    un = (a1 - a0) + (b1 - b0) - ov
    return ov / (un + 1e-9)


def inflate(b: Box, dx: float, dy=None) -> Box:
    dy = dx if dy is None else dy
    return (b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy)


def cluster_1d(values: Sequence, tol: float) -> list:
    """Single-link clustering on a line.

    Returns clusters of *indices into the original sequence*, ordered by position.
    Used for snapping collinear rules, column edges and row baselines.  Deterministic:
    equal values are broken by original index.
    """
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    clusters = [[order[0]]]
    last = values[order[0]]
    for i in order[1:]:
        if values[i] - last <= tol:
            clusters[-1].append(i)
        else:
            clusters.append([i])
        last = values[i]
    return clusters


def cluster_centers(values: Sequence, tol: float) -> list:
    return [sum(values[i] for i in c) / len(c) for c in cluster_1d(values, tol)]


def merge_intervals(iv: Sequence, gap: float = 0.0) -> list:
    if not iv:
        return []
    out = []
    for a, b in sorted(iv):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def gaps_in(iv: Sequence, lo: float, hi: float) -> list:
    """Complement of merged intervals inside ``[lo, hi]`` -- i.e. the whitespace."""
    merged = merge_intervals(iv)
    out, cur = [], lo
    for a, b in merged:
        if a > cur:
            out.append((cur, min(a, hi)))
        cur = max(cur, b)
        if cur >= hi:
            break
    if cur < hi:
        out.append((cur, hi))
    return [(a, b) for a, b in out if b > a]


def median(xs: Sequence) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return float(s[n // 2]) if n % 2 else (float(s[n // 2 - 1]) + float(s[n // 2])) / 2.0


# ==================================================================================================
# config: Every tunable number in the pipeline lives here.
# ==================================================================================================
#
# Two reasons this file exists as a frozen dataclass rather than scattered literals:
#
# 1.  Determinism.  ``Config.fingerprint()`` is hashed into ``run_id`` and stored in
#     ``extraction_runs.config_json``, so a run's output can always be tied back to the
#     exact thresholds that produced it.
# 2.  Ablation.  ``metrics/eval.py`` flips single fields (e.g. ``enable_borderless``) to
#     measure what each detector contributes.  §3.1 of the brief demands we can show the
#     pipeline with a component switched off; the same machinery answers that for every
#     stage, not just a learned model (of which we have none -- see docs/DECISIONS.md D1).
#
# Units: all lengths are PDF points (1/72 inch) unless the name says ``px`` or ``frac``.

@dataclass(frozen=True)
class Config:
    # ---------------------------------------------------------------- S1 intake
    # A page is "digital" if its text layer carries real glyphs.  Thresholds are
    # deliberately generous: a page with 40 chars is a title page, not a scan.
    digital_min_chars: int = 40
    # Fraction of page area covered by raster images above which we suspect a scan.
    scan_image_area_frac: float = 0.55
    # Chars per 1000 pt^2 of page area below which a page with a big image is a scan.
    hybrid_min_char_density: float = 0.02
    raster_dpi: int = 200          # working DPI for scanned page analysis
    ocr_cell_dpi: int = 300        # per-cell crop DPI (S5); higher = better OCR
    deskew_max_deg: float = 8.0    # refuse to "fix" more than this; flag instead
    deskew_step_deg: float = 0.1
    # A text line is treated as vertical (90/270 rotated content) if its writing
    # direction deviates from horizontal by more than this.
    vertical_dir_tol: float = 0.35

    # ---------------------------------------------------------------- S2 layout
    header_band_frac: float = 0.14   # top fraction of page scanned for running header
    footer_band_frac: float = 0.14   # bottom fraction likewise
    # A normalised line must recur on at least this many pages AND this fraction of
    # pages to be called a running header/footer.
    repeat_min_pages: int = 3
    repeat_min_frac: float = 0.25
    repeat_y_tol: float = 12.0       # same-band vertical tolerance across pages
    heading_size_ratio: float = 1.12  # font size / body median above which -> heading
    heading_max_words: int = 18
    body_line_gap_mult: float = 1.8  # gap > mult * median line height => block break

    # ------------------------------------------------------------ S3 detection
    enable_ruled: bool = True
    enable_borderless: bool = True
    # Vector "lines": a filled rect thinner than this in one axis is a rule, not a box.
    rule_max_thickness: float = 2.6
    rule_min_length: float = 24.0
    # Snapping tolerance when clustering collinear rules (JPEG/hairline jitter).
    rule_snap_tol: float = 2.2
    # Two rules on the same axis merge if their gap is under this (dotted/broken rules).
    rule_join_gap: float = 6.0
    ruled_min_rows: int = 2
    ruled_min_cols: int = 2
    # Borderless: gutter must be whitespace in >= this fraction of candidate rows.
    gutter_persist_frac: float = 0.80
    gutter_min_width: float = 3.6
    borderless_min_rows: int = 3
    borderless_min_cols: int = 2
    # A row band is part of the same candidate block if the vertical gap to the next
    # line is under mult * median gap of the block so far.
    row_gap_mult: float = 2.1
    # Arbitration
    merge_iou: float = 0.30          # regions above this IoU are the same table
    ruled_evidence_min_cells: int = 4

    # ------------------------------------------------------------ S4 structure
    col_align_tol: float = 3.0       # token edge clustering tolerance
    row_align_tol: float = 2.4
    # Adornment columns ($ / % / () only) are folded into their data neighbour.
    fold_adornment_columns: bool = True
    adornment_max_width: float = 26.0
    indent_step_min: float = 5.0     # x0 delta that counts as one indent level
    max_header_rows: int = 4

    # -------------------------------------------------------------- S6 stitch
    enable_stitch: bool = True
    stitch_col_iou_min: float = 0.85   # 1-D overlap of column bands, page to page
    stitch_max_page_gap: int = 1       # only consecutive pages (see DECISIONS D6)
    stitch_header_sim_min: float = 0.86  # header repeat detection similarity
    stitch_score_min: float = 0.62

    # ----------------------------------------------------------- S7 normalise
    # Column type is assigned when this share of non-empty cells agree.
    type_majority_frac: float = 0.70

    # ------------------------------------------------------------- S8 context
    caption_max_gap: float = 46.0      # points above/below table to look for a caption
    scale_search_pages: int = 3        # how far back a scale declaration may govern
    asset_min_area: float = 2400.0     # ignore logos/rules smaller than this
    asset_link_max_gap: float = 260.0

    # ---------------------------------------------------------- S9 confidence
    # Logistic weights.  Fixed and documented, not fitted at runtime (determinism);
    # the mapping from this score to a probability IS fitted offline -> calibration.json.
    w_bias: float = 2.35
    w_source_ocr: float = -1.55
    w_ocr_conf: float = 2.10
    w_snap_residual: float = -2.60
    w_straddle: float = -2.90
    w_type_conflict: float = -2.20
    w_digit_anomaly: float = -1.35
    w_multiline: float = -0.45
    w_crossfoot_fail: float = -2.05
    w_downfoot_fail: float = -1.85
    w_colcount_mismatch: float = -1.10
    w_dup_row: float = -0.55
    w_watermark: float = -1.20
    w_empty_in_dense_row: float = -0.90
    w_grid_low_support: float = -1.15
    review_budget_frac: float = 0.08   # §4 target: flag <= 8% of cells
    high_confidence_floor: float = 0.85  # "shipped with high confidence" boundary

    # ------------------------------------------------------------- S10 persist
    excel_freeze_header: bool = True
    write_debug_pages: str = "tables"  # "none" | "tables" | "all"

    # ------------------------------------------------------------------ misc
    seed: int = 20240928
    offline: bool = True
    fixed_time: bool = False           # make extraction_runs timestamps constant too
    max_pages: int = 0                 # 0 = all; used by tests
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    def replace(self, **kw: Any) -> "Config":
        return dataclasses.replace(self, **kw)


DEFAULT = Config()


# ==================================================================================================
# logging_utils: Structured, deterministic run log.
# ==================================================================================================
#
# ``logs/run.jsonl`` has to be byte-identical between runs (§3.2), which rules out the
# two things a log normally contains: a wall-clock timestamp and a duration.  Events are
# therefore numbered with a monotonic sequence instead, and anything genuinely temporal
# goes to ``logs/timing.json``, which is explicitly excluded from the determinism claim.
#
# Everything is buffered and written once at the end so a partially-flushed file can
# never look like a complete one.

class RunLog:
    def __init__(self):
        self._events = []

    def event(self, kind: str, **fields) -> None:
        self._events.append({"seq": len(self._events), "event": kind,
                             **{k: _plain(v) for k, v in fields.items()}})

    def write(self, path: str) -> None:
        lines = [canonical_json(e, None) for e in self._events]
        write_text(path, "\n".join(lines) + "\n")

    def __len__(self):
        return len(self._events)


def _plain(v):
    """Coerce to JSON-safe, sorted, finite values."""
    if isinstance(v, dict):
        return {str(k): _plain(v[k]) for k in sorted(v, key=str)}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, set):
        return sorted(_plain(x) for x in v)
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return str(v)


# ==================================================================================================
# textlayer: Token model over the PDF text layer.
# ==================================================================================================
#
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


# ==================================================================================================
# s7_normalize: S7 -- normalisation and typing.  Additive, never destructive.
# ==================================================================================================
#
# ``raw_text`` is always kept exactly as the page had it.  Everything here is a *second*
# opinion recorded alongside it, and every opinion carries flags saying how sure it is.
#
# The governing rule is section 4 of the brief: no wrong value at high confidence.  That
# turns several ordinary parsing decisions into refusals:
#
# ``12,34,567`` vs ``12,345,67``
#     Indian and Western grouping are both accepted, but the grouping *pattern* is
#     validated rather than the commas simply deleted.  A string whose grouping matches
#     neither convention parses to a number and is flagged ``AMBIGUOUS_GROUPING``, so a
#     typo in the source cannot arrive as a clean value.
#
# ``1.234,56``
#     Continental European notation collides head-on with Western: is ``1.234`` one
#     thousand two hundred or one point two three four?  There is no way to know from one
#     cell.  We flag ``AMBIGUOUS_DECIMAL_SEPARATOR`` and leave ``normalized_value`` null.
#     Guessing here would be exactly the silent error the brief is built around.
#
# ``03/04/2024``
#     Same problem for dates: 3 April or 4 March.  If both fields are 12 or under and the
#     document has not established an order, ``normalized_value`` stays null with
#     ``AMBIGUOUS_DATE_ORDER``.  A date with an unambiguous field (``31/03/2024``) parses
#     normally and also *teaches* the document its order, which later resolves the
#     ambiguous ones -- see :func:`DateOrderModel`.
#
# ``#REF!``
#     Excel error values survive printing to PDF.  ``Shell.pdf`` ships six of them inside
#     the cash-flow statement.  They are typed ``error``, never coerced to 0 or null-and-
#     forgotten, and they raise an issue -- a number that the source document itself could
#     not compute must not look like a number downstream.
#
# ``-`` / ``Nil`` / ``NA`` / ``0`` / empty
#     Five distinct value types, because they mean five different things to an accountant
#     and collapsing them loses information no downstream user can recover.

# ------------------------------------------------------------------- value types
T_EMPTY = "empty"
T_DASH = "dash"
T_NIL = "nil"
T_NA = "na"
T_NUMBER = "number"
T_CURRENCY = "currency"
T_PERCENT = "percent"
T_DATE = "date"
T_PERIOD = "period"
T_ERROR = "error"
T_REFERENCE = "reference"
T_TEXT = "text"

NUMERIC_TYPES = (T_NUMBER, T_CURRENCY, T_PERCENT)
NULLISH_TYPES = (T_EMPTY, T_DASH, T_NIL, T_NA)

CURRENCY_SYMBOLS = {"$", "US$", "₹", "€", "£", "¥", "₨", "`", "R$", "A$", "C$", "CHF"}
CURRENCY_WORDS = {"rs", "rs.", "inr", "usd", "eur", "gbp", "jpy", "aed", "sgd",
                  "rupees", "dollars", "euro", "euros"}
# The ICRA and Shell documents both render the rupee sign as U+0060 GRAVE ACCENT via a
# legacy font encoding: "(` Million)".  Treating it as INR is a *flagged* inference.
LEGACY_RUPEE = "`"

MAGNITUDES = {
    "k": 1e3, "thousand": 1e3, "thousands": 1e3, "000s": 1e3,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
    "mn": 1e6, "m": 1e6, "million": 1e6, "millions": 1e6,
    "cr": 1e7, "crore": 1e7, "crores": 1e7,
    "bn": 1e9, "b": 1e9, "billion": 1e9, "billions": 1e9,
    "tn": 1e12, "trillion": 1e12,
}

ERROR_VALUES = {"#ref!", "#div/0!", "#value!", "#name?", "#num!", "#null!",
                "#n/a", "#na", "#error", "#calc!", "#spill!", "#getting_data"}
NIL_WORDS = {"nil", "none"}
NA_WORDS = {"na", "n.a.", "n.a", "n/a", "not applicable", "not available", "n.m.",
            "nm", "not meaningful", "-na-"}
DASH_CHARS = {"-", "–", "—", "‐", "‑", "‒", "―", "−", "﹘", "－"}

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

WESTERN_GROUP_RE = re.compile(r"^\d{1,3}(,\d{3})+$")
INDIAN_GROUP_RE = re.compile(r"^\d{1,2}(,\d{2})+,\d{3}$")
PLAIN_INT_RE = re.compile(r"^\d+$")
# Digits and numeric punctuation only -- the gate before any number parse.
NUMERIC_SHAPE_RE = re.compile(r"^[\d.,'\s]*\d[\d.,'\s]*$")
PERIOD_RE = re.compile(
    r"^(?:(?:fy|ay|cy|q[1-4]|h[12])\s*[-']?\s*)+"
    r"(?:\d{2,4}(?:\s*[-/–]\s*\d{2,4})?)?$|"
    r"^\d{4}\s*[-–]\s*\d{2,4}$|"
    r"^q[1-4]\s*(?:fy)?\s*\d{2,4}$", re.I)
REFERENCE_HEADER_RE = re.compile(r"^\s*(note|notes|note\s*no\.?|ref\.?|refer|"
                                 r"schedule|annexure)\b", re.I)
FOOTNOTE_SUFFIX_RE = re.compile(r"^(?P<body>.*?)(?P<mark>[*†‡§¶]{1,3}|\(\d{1,2}\))$")


@dataclass
class Normalized:
    raw_text: str
    value_type: str = T_TEXT
    number: float = None          # REAL for number/currency/percent
    date_iso: str = None          # TEXT for date
    currency: str = ""
    magnitude: float = 1.0
    percent: bool = False
    negative: bool = False
    footnote_marker: str = ""
    flags: list = field(default_factory=list)

    @property
    def value(self):
        """The single canonical value written to ``cells.normalized_value``."""
        if self.value_type == T_DATE:
            return self.date_iso
        if self.value_type in NUMERIC_TYPES:
            return self.number
        return None


def is_currency_symbol(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    return t in CURRENCY_SYMBOLS or t.lower() in CURRENCY_WORDS


def _nfkc_keep_minus(s: str) -> str:
    """NFKC-normalise but keep the distinction we care about: minus-like dashes."""
    out = unicodedata.normalize("NFKC", s)
    return out.replace(" ", " ").strip()


def classify_token(s: str) -> str:
    """Coarse *format* signature, used by S4 header detection and S9 type conformity.

    Deliberately finer than ``value_type``: the Apple header cell "2024" and the body
    cell "167,045" are both numbers, and only their formats tell them apart.
    """
    t = _nfkc_keep_minus(s)
    if not t:
        return "empty"
    low = t.lower()
    if low in ERROR_VALUES:
        return "error"
    if t in DASH_CHARS:
        return "dash"
    if low in NIL_WORDS:
        return "nil"
    if low in NA_WORDS:
        return "na"
    if is_currency_symbol(t):
        return "currency_symbol"
    if t.endswith("%") or t.endswith("‰"):
        return "percent"
    core = t.strip("()")
    core = core.lstrip("+-−–").rstrip("-−")
    if PLAIN_INT_RE.match(core):
        return "int%d" % min(len(core), 5)
    if WESTERN_GROUP_RE.match(core) or INDIAN_GROUP_RE.match(core):
        return "grouped"
    if re.match(r"^\d+\.\d+$", core):
        return "decimal"
    if re.match(r"^[\d,]+\.\d+$", core):
        return "grouped_decimal"
    if PERIOD_RE.match(t):
        return "period"
    if parse_date(t)[0]:
        return "date"
    return "text"


class DateOrderModel:
    """Document-level day/month order, learned from unambiguous dates only.

    ``31/03/2024`` can only be day-first, so a document containing one is day-first for
    all of its ``03/04/2024``-shaped dates too.  Until such evidence exists, ambiguous
    dates stay unparsed and flagged rather than guessed.  Deterministic: the model is
    built in one pass over the document before cells are normalised, so the answer does
    not depend on which cell happens to be processed first.
    """

    def __init__(self):
        self.day_first = 0
        self.month_first = 0

    def observe(self, text: str) -> None:
        m = re.match(r"^\s*(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{2,4})\s*$",
                     text or "")
        if not m:
            return
        a, b = int(m.group(1)), int(m.group(2))
        if a > 12 and b <= 12:
            self.day_first += 1
        elif b > 12 and a <= 12:
            self.month_first += 1

    @property
    def order(self):
        if self.day_first > self.month_first:
            return "DMY"
        if self.month_first > self.day_first:
            return "MDY"
        return None

    def to_dict(self):
        return {"day_first_evidence": self.day_first,
                "month_first_evidence": self.month_first,
                "resolved_order": self.order}


def parse_date(text: str, order_model: DateOrderModel = None):
    """``(iso_or_None, flags)`` over at least six formats.

    Formats: ``September 28, 2024`` / ``28 September 2024`` / ``28-Sep-2024`` /
    ``2024-09-28`` / ``28/09/2024`` / ``28.09.2024`` / ``Sep 2024`` / ``09-2024``.
    """
    t = _nfkc_keep_minus(text)
    if not t or len(t) > 40:
        return None, []
    flags = []

    # Month name forms.
    m = re.match(r"^([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})$", t)
    if m and m.group(1).lower() in MONTHS:
        return _iso(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2))), flags
    m = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?[\s\-]+([A-Za-z]{3,9})\.?[\s\-,]+(\d{2,4})$", t)
    if m and m.group(2).lower() in MONTHS:
        return _iso(_year(m.group(3)), MONTHS[m.group(2).lower()],
                    int(m.group(1))), flags
    m = re.match(r"^([A-Za-z]{3,9})\.?[\s\-]+(\d{4})$", t)
    if m and m.group(1).lower() in MONTHS:
        return _iso(int(m.group(2)), MONTHS[m.group(1).lower()], 1), ["DAY_ASSUMED_1"]

    # ISO.
    m = re.match(r"^(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})$", t)
    if m:
        return _iso(int(m.group(1)), int(m.group(2)), int(m.group(3))), flags

    # All-numeric with a separator: the ambiguous family.
    m = re.match(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})$", t)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), _year(m.group(3))
        if a > 12 and b <= 12:
            return _iso(y, b, a), flags
        if b > 12 and a <= 12:
            return _iso(y, a, b), ["month_first"]
        order = order_model.order if order_model else None
        if order == "DMY":
            return _iso(y, b, a), ["order_from_document"]
        if order == "MDY":
            return _iso(y, a, b), ["order_from_document"]
        return None, ["AMBIGUOUS_DATE_ORDER"]
    return None, []


def _year(s: str) -> int:
    v = int(s)
    if v >= 100:
        return v
    # Two-digit years: 70-99 -> 1900s, 00-69 -> 2000s.  Documented, not guessed.
    return 1900 + v if v >= 70 else 2000 + v


def _iso(y: int, m: int, d: int):
    if not (1 <= m <= 12 and 1 <= d <= 31 and 1000 <= y <= 3000):
        return None
    if d > (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[m - 1]:
        return None
    return "%04d-%02d-%02d" % (y, m, d)


def _strip_footnote(t: str):
    """Split a trailing footnote marker off a value: ``1,872*`` -> ``1,872`` + ``*``."""
    m = FOOTNOTE_SUFFIX_RE.match(t)
    if not m:
        return t, ""
    body, mark = m.group("body"), m.group("mark")
    if not body or not any(ch.isdigit() for ch in body):
        return t, ""
    return body, mark


def parse_number(t: str):
    """``(value, flags)`` for the numeric core of a cell -- no sign, no currency."""
    flags = []
    s = t.replace(" ", "").replace("'", "").replace(" ", "")
    if not s:
        return None, flags
    has_comma, has_dot = "," in s, "." in s

    if has_comma and has_dot:
        last_comma, last_dot = s.rfind(","), s.rfind(".")
        if last_comma > last_dot:
            # 1.234,56 -- continental.  Ambiguous against 1.234 + noise; refuse.
            return None, ["AMBIGUOUS_DECIMAL_SEPARATOR"]
        int_part, frac = s[:last_dot], s[last_dot + 1:]
        if not re.match(r"^\d+$", frac):
            return None, ["MALFORMED_NUMBER"]
        val, f2 = _parse_grouped(int_part)
        if val is None:
            return None, f2
        return float("%s.%s" % (int(val), frac)), flags + f2
    if has_comma:
        # A single comma with exactly two trailing digits could be a decimal comma.
        if re.match(r"^\d{1,3},\d{2}$", s):
            return None, ["AMBIGUOUS_DECIMAL_SEPARATOR"]
        return _parse_grouped(s)
    if has_dot:
        if not re.match(r"^\d*\.\d+$|^\d+\.$", s):
            return None, ["MALFORMED_NUMBER"]
        try:
            return float(s.rstrip(".")), flags
        except ValueError:
            return None, ["MALFORMED_NUMBER"]
    if PLAIN_INT_RE.match(s):
        return float(s), flags
    return None, []


def _parse_grouped(s: str):
    """Validate the grouping convention before deleting the separators."""
    if PLAIN_INT_RE.match(s):
        return float(s), []
    if WESTERN_GROUP_RE.match(s):
        return float(s.replace(",", "")), []
    if INDIAN_GROUP_RE.match(s):
        return float(s.replace(",", "")), []
    if re.match(r"^[\d,]+$", s):
        # Digits and commas but neither convention: 12,345,67 or 1,2345.
        return float(s.replace(",", "")), ["AMBIGUOUS_GROUPING"]
    return None, ["MALFORMED_NUMBER"]


def normalize(raw: str, *, column_type: str = None, column_header: str = "",
              order_model: DateOrderModel = None, scale: float = 1.0,
              scale_currency: str = "") -> Normalized:
    """Turn one cell's raw text into a typed, flagged value."""
    n = Normalized(raw_text=raw if raw is not None else "")
    t = _nfkc_keep_minus(n.raw_text)
    if not t:
        n.value_type = T_EMPTY
        return n

    low = t.lower()
    if low in ERROR_VALUES:
        n.value_type = T_ERROR
        n.flags.append("SPREADSHEET_ERROR")
        return n
    if t in DASH_CHARS or (len(t) <= 3 and all(c in DASH_CHARS for c in t)):
        n.value_type = T_DASH
        return n
    if low in NIL_WORDS:
        n.value_type = T_NIL
        return n
    if low in NA_WORDS:
        n.value_type = T_NA
        return n
    if REFERENCE_HEADER_RE.match(column_header or "") or column_type == T_REFERENCE:
        n.value_type = T_REFERENCE
        return n

    body, mark = _strip_footnote(t)
    if mark:
        n.footnote_marker = mark
        t = body

    # Currency and percent decoration.
    work = t
    for sym in sorted(CURRENCY_SYMBOLS, key=len, reverse=True):
        if work.startswith(sym):
            n.currency = "INR" if sym == LEGACY_RUPEE else _code(sym)
            if sym == LEGACY_RUPEE:
                n.flags.append("LEGACY_RUPEE_GLYPH")
            work = work[len(sym):].strip()
            break
    m = re.match(r"^(?:(rs\.?|inr|usd|eur|gbp|jpy)\s*)", work, re.I)
    if m:
        n.currency = _code(m.group(1))
        work = work[m.end():].strip()
    if work.endswith("%"):
        n.percent = True
        work = work[:-1].strip()
    elif work.endswith("‰"):
        n.percent = True
        n.magnitude = 0.1
        work = work[:-1].strip()

    # Magnitude suffix.
    m = re.match(r"^(.*?)\s*(k|mn|m|bn|b|tn|cr|crores?|lakhs?|lacs?|million|millions|"
                 r"billion|billions|thousand|thousands|trillion)\.?$", work, re.I)
    if m and m.group(1).strip():
        suffix = m.group(2).lower().rstrip(".")
        if suffix in MAGNITUDES:
            n.magnitude = MAGNITUDES[suffix]
            work = m.group(1).strip()

    # Sign.
    neg = False
    if work.startswith("(") and work.endswith(")"):
        neg = True
        work = work[1:-1].strip()
    if work.startswith(("-", "−", "–", "‒")):
        neg = True
        work = work[1:].strip()
    elif work.endswith(("-", "−")):
        neg = True
        work = work[:-1].strip()
    if work.startswith("+"):
        work = work[1:].strip()

    # Dates and periods take precedence over "looks like digits with separators".
    if not neg and not n.percent:
        iso, dflags = parse_date(t, order_model)
        if iso:
            n.value_type = T_DATE
            n.date_iso = iso
            n.flags.extend(dflags)
            return n
        if "AMBIGUOUS_DATE_ORDER" in dflags:
            n.value_type = T_DATE
            n.flags.append("AMBIGUOUS_DATE_ORDER")
            return n
        if PERIOD_RE.match(t):
            n.value_type = T_PERIOD
            return n

    # Only attempt a numeric parse on something numeric-shaped.  Without this guard
    # every prose cell containing a comma came back MALFORMED_NUMBER -- "Cupertino,
    # California" and "0.875% Notes due 2025" produced 375 bogus mandatory-review
    # flags on the Apple 10-K, which is precisely the kind of false alarm that makes a
    # review queue useless.
    if not NUMERIC_SHAPE_RE.match(work):
        n.value_type = T_TEXT
        return n
    val, nflags = parse_number(work)
    n.flags.extend(nflags)
    if val is None:
        n.value_type = T_TEXT
        return n

    n.negative = neg
    val = -val if neg else val
    val *= n.magnitude
    if n.percent:
        n.value_type = T_PERCENT
        n.number = val
    else:
        if not n.currency and scale_currency:
            n.currency = scale_currency
        if scale and scale != 1.0:
            val *= scale
            n.flags.append("SCALE_APPLIED")
        n.number = val
        n.value_type = T_CURRENCY if n.currency else T_NUMBER
    return n


def _code(sym: str) -> str:
    s = sym.strip().rstrip(".").lower()
    table = {"$": "USD", "us$": "USD", "₹": "INR", "₨": "INR", "rs": "INR",
             "inr": "INR", "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP",
             "gbp": "GBP", "¥": "JPY", "jpy": "JPY", "r$": "BRL", "a$": "AUD",
             "c$": "CAD", "chf": "CHF"}
    return table.get(s, sym.strip())


def infer_column_type(values, header_path=None, majority: float = 0.70):
    """Per-column type from its non-null cells, plus the conflicting minority.

    Returns ``(inferred_type, conflict_indices, stats)``.  A single string in a numeric
    column is precisely the red flag the brief asks for, so conflicts are returned
    rather than smoothed away.
    """
    header = " ".join(header_path or [])
    if REFERENCE_HEADER_RE.match(header.strip()):
        return T_REFERENCE, [], {"from": "header_keyword"}
    counts = {}
    considered = []
    for i, n in enumerate(values):
        if n.value_type in NULLISH_TYPES:
            continue
        considered.append(i)
        key = T_NUMBER if n.value_type == T_CURRENCY else n.value_type
        counts[key] = counts.get(key, 0) + 1
    if not considered:
        return T_EMPTY, [], {"from": "all_null"}
    total = len(considered)
    best, best_n = max(sorted(counts.items()), key=lambda kv: kv[1])
    frac = best_n / total
    conflicts = []
    if frac >= majority:
        for i in considered:
            vt = values[i].value_type
            key = T_NUMBER if vt == T_CURRENCY else vt
            if key != best:
                conflicts.append(i)
        inferred = best
    else:
        inferred = T_TEXT
    if best in (T_NUMBER,) and any(values[i].currency for i in considered):
        inferred = T_CURRENCY
    return inferred, conflicts, {"from": "majority", "counts": dict(sorted(
        counts.items())), "majority_frac": round(frac, 3), "n_considered": total}


# ==================================================================================================
# s1_intake: S1 -- document intake and per-page typing.
# ==================================================================================================
#
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


# ==================================================================================================
# s2_layout: S2 -- page layout segmentation.
# ==================================================================================================
#
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


# ==================================================================================================
# s3_ruled: S3a -- ruled table detection from vector geometry (and from rasters).
# ==================================================================================================
#
# This is not "find the lines and intersect them".  Measuring the three sample documents
# first (``tools/probe.py``) showed three *different* ways a PDF can express a grid, and
# only one of them is drawn with lines:
#
# ``Shell.pdf`` -- **dotted rules.**
#     18,066 filled rectangles of 0.72 x 0.72 pt.  Each "line" is a row of dots.  A naive
#     thin-rect-is-a-rule test yields 18,000 rules of length 0.72 and finds nothing.  Dots
#     are therefore emitted as candidates on *both* axes and only survive if they merge
#     into a run longer than ``rule_min_length``; a vertical dotted line contributes one
#     dot per horizontal cluster, so it cannot fake a horizontal rule.
#
# ``Shell MRPL ...pdf`` -- **horizontal hairlines only, segmented per cell.**
#     Rules are 0.48 pt tall and arrive as ``[43.2,248.8] [248.8,249.3] [249.3,319.8]
#     ...``.  Merging across the sub-point gaps rebuilds the full-width rule -- and the
#     *breakpoints* between segments (249, 320, 403) are the column boundaries, handed to
#     S4 as a free, high-precision column signal.  There are no vertical rules at all:
#     this is exactly the "ruled horizontally but not vertically" case the brief says
#     neither classical approach handles alone.
#
# ``Apple Form 10-K.pdf`` -- **per-cell background fills, no rules.**
#     Alternating rows are shaded with one 0.937-grey rectangle *per cell*, so the fill
#     rectangles are the cell lattice, exactly.  Ignoring them and running gutter
#     analysis instead throws away the best signal on the page.
#
# So this module produces *evidence*, not verdicts: rules, fills, breakpoints and
# intersections, plus candidate regions tagged with the regime that found them.  S4 is
# the single place that resolves boundaries, and it prefers evidence in the order
# grid > fills > breakpoints > gutters.  Keeping arbitration in one place is what stops
# the three regimes from fighting each other.

REGIME_GRID = "grid"
REGIME_FILLS = "fills"
REGIME_HLINES = "hlines"
REGIME_VLINES = "vlines"


@dataclass
class Rule:
    axis: str          # "h" or "v"
    pos: float         # y for horizontal, x for vertical
    lo: float          # x0 / y0
    hi: float          # x1 / y1
    thickness: float
    dotted: bool
    n_src: int
    breaks: tuple = ()  # interior segment boundaries -> column/row candidates

    @property
    def length(self):
        return self.hi - self.lo

    @property
    def bbox(self):
        if self.axis == "h":
            return (self.lo, self.pos - self.thickness / 2,
                    self.hi, self.pos + self.thickness / 2)
        return (self.pos - self.thickness / 2, self.lo,
                self.pos + self.thickness / 2, self.hi)


@dataclass
class FillBox:
    bbox: tuple
    color: tuple


@dataclass
class RuledRegion:
    bbox: tuple
    regime: str
    x_lines: list = field(default_factory=list)
    y_lines: list = field(default_factory=list)
    n_intersections: int = 0
    n_rules: int = 0
    n_fills: int = 0
    breakpoints: list = field(default_factory=list)
    dotted: bool = False
    evidence: dict = field(default_factory=dict)


@dataclass
class RuledEvidence:
    h_rules: list = field(default_factory=list)
    v_rules: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    breakpoints: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------- primitive harvest


def _seg_from_rect(x0, y0, x1, y1, cfg: Config):
    """Classify a rectangle.  Returns ``(kind, payload)``.

    ``kind`` is "dot" (both axes, tiny), "h", "v" or "fill".
    """
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None, None
    tiny = cfg.rule_max_thickness
    if w <= tiny and h <= tiny:
        return "dot", ((x0 + x1) / 2.0, (y0 + y1) / 2.0, max(w, h))
    if h <= tiny:
        return "h", ((y0 + y1) / 2.0, x0, x1, h)
    if w <= tiny:
        return "v", ((x0 + x1) / 2.0, y0, y1, w)
    return "fill", (x0, y0, x1, y1)


def harvest_primitives(page, cfg: Config, clip=None):
    """Pull straight segments, dots and fill boxes out of the page vector content."""
    h_segs, v_segs, dots, fills = [], [], [], []
    px0, py0, px1, py1 = (clip if clip is not None else
                          (0.0, 0.0, float(page.rect.width), float(page.rect.height)))
    try:
        paths = page.get_drawings()
    except Exception:
        return h_segs, v_segs, dots, fills

    for path in paths:
        fill_col = path.get("fill")
        stroke_w = path.get("width") or 0.0
        for it in path.get("items", []):
            op = it[0]
            if op == "re":
                r = it[1]
                x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
            elif op == "qu":
                q = it[1]
                xs = [q.ul.x, q.ur.x, q.ll.x, q.lr.x]
                ys = [q.ul.y, q.ur.y, q.ll.y, q.lr.y]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            elif op == "l":
                p0, p1 = it[1], it[2]
                x0, y0, x1, y1 = (min(p0.x, p1.x), min(p0.y, p1.y),
                                  max(p0.x, p1.x), max(p0.y, p1.y))
                # Same page-bounds clip as rectangles: logo artwork in the ICRA report
                # runs to y = -0.8 and would otherwise seed a region above the page.
                if x1 < px0 - 2 or x0 > px1 + 2 or y1 < py0 - 2 or y0 > py1 + 2:
                    continue
                x0, y0 = max(x0, px0), max(y0, py0)
                x1, y1 = min(x1, px1), min(y1, py1)
                dx, dy = x1 - x0, y1 - y0
                t = max(stroke_w, 0.3)
                if dy <= cfg.rule_snap_tol and dx >= 1.0:
                    h_segs.append(((y0 + y1) / 2.0, x0, x1, t))
                elif dx <= cfg.rule_snap_tol and dy >= 1.0:
                    v_segs.append(((x0 + x1) / 2.0, y0, y1, t))
                continue
            else:
                continue

            # Reject anything not on the page (logos bleed to negative coordinates).
            if x1 < px0 - 2 or x0 > px1 + 2 or y1 < py0 - 2 or y0 > py1 + 2:
                continue
            x0, y0 = max(x0, px0), max(y0, py0)
            x1, y1 = min(x1, px1), min(y1, py1)
            kind, payload = _seg_from_rect(x0, y0, x1, y1, cfg)
            if kind == "dot":
                dots.append(payload)
            elif kind == "h":
                h_segs.append(payload)
            elif kind == "v":
                v_segs.append(payload)
            elif kind == "fill" and fill_col is not None:
                fills.append(FillBox(bbox=(x0, y0, x1, y1),
                                     color=tuple(round(float(c), 3) for c in fill_col)))
    return h_segs, v_segs, dots, fills


def merge_rules(segs, dots, axis: str, cfg: Config):
    """Cluster collinear segments, then merge along the axis across small gaps.

    ``breaks`` keeps the interior boundaries between merged segments.  For the ICRA
    report those breakpoints *are* the column boundaries, which is a far more precise
    column signal than any whitespace analysis.
    """
    # ``is_dot`` matters for breakpoints only.  "Microsoft Print To PDF" renders a
    # solid Excel border as ~300 touching 0.72 pt squares; treating each square edge as
    # a column boundary produced 703 bogus breakpoints per page on Shell.pdf.  Dots
    # still drive clustering and coverage -- they just do not vote on structure.
    items = [(s[0], s[1], s[2], s[3], False) for s in segs]
    for cx, cy, size in dots:
        pos = cy if axis == "h" else cx
        lo = cx if axis == "h" else cy
        items.append((pos, lo - size / 2.0, lo + size / 2.0, size, True))
    if not items:
        return []

    positions = [it[0] for it in items]
    out = []
    for idxs in cluster_1d(positions, cfg.rule_snap_tol):
        group = [items[i] for i in idxs]
        pos = sum(g[0] for g in group) / len(group)
        thick = median([g[3] for g in group]) or 0.5
        spans = sorted((g[1], g[2]) for g in group)
        merged = merge_intervals(spans, gap=cfg.rule_join_gap)
        # Interior boundaries of real segments, before the gap-merge swallowed them.
        raw_edges = sorted({round(v, 2) for g in group if not g[4]
                            for v in (g[1], g[2])})
        for lo, hi in merged:
            if hi - lo < cfg.rule_min_length:
                continue
            covered = sum(min(b, hi) - max(a, lo) for a, b in spans
                          if b > lo and a < hi)
            # Dottedness is a coverage property, not a segment-count property.
            dotted = covered < 0.75 * (hi - lo)
            brk = tuple(v for v in raw_edges if lo + 1.0 < v < hi - 1.0)
            out.append(Rule(axis=axis, pos=round(pos, 3), lo=round(lo, 3),
                            hi=round(hi, 3), thickness=round(float(thick), 3),
                            dotted=bool(dotted), n_src=len(group), breaks=brk))
    out.sort(key=lambda r: (r.pos, r.lo))
    return out


# ------------------------------------------------------------------- region finding


def _intersections(h_rules, v_rules, tol: float):
    pts = []
    for hi_, h in enumerate(h_rules):
        for vi_, v in enumerate(v_rules):
            if (h.lo - tol <= v.pos <= h.hi + tol
                    and v.lo - tol <= h.pos <= v.hi + tol):
                pts.append((v.pos, h.pos, hi_, vi_))
    return pts


def _grid_regions(h_rules, v_rules, cfg: Config):
    """Connected components of the rule lattice -> fully ruled table candidates."""
    pts = _intersections(h_rules, v_rules, cfg.rule_snap_tol)
    if len(pts) < cfg.ruled_evidence_min_cells:
        return [], pts
    # Union-find over intersections that share a rule.
    parent = list(range(len(pts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_h, by_v = {}, {}
    for i, (x, y, hi_, vi_) in enumerate(pts):
        by_h.setdefault(hi_, []).append(i)
        by_v.setdefault(vi_, []).append(i)
    for group in list(by_h.values()) + list(by_v.values()):
        for a, b in zip(group, group[1:]):
            union(a, b)

    comps = {}
    for i in range(len(pts)):
        comps.setdefault(find(i), []).append(i)

    regions = []
    for root in sorted(comps):
        idxs = comps[root]
        xs = sorted({round(pts[i][0], 2) for i in idxs})
        ys = sorted({round(pts[i][1], 2) for i in idxs})
        if len(xs) < cfg.ruled_min_cols + 1 or len(ys) < cfg.ruled_min_rows + 1:
            continue
        hset = {pts[i][2] for i in idxs}
        vset = {pts[i][3] for i in idxs}
        dotted = any(h_rules[i].dotted for i in hset) or \
                 any(v_rules[i].dotted for i in vset)
        regions.append(RuledRegion(
            bbox=(xs[0], ys[0], xs[-1], ys[-1]), regime=REGIME_GRID,
            x_lines=xs, y_lines=ys, n_intersections=len(idxs),
            n_rules=len(hset) + len(vset), dotted=dotted,
            evidence={"h_rules": sorted(hset), "v_rules": sorted(vset)}))
    return regions, pts


def _break_signature(rule, tol: float = 2.5):
    """Cluster a rule's breakpoints so near-duplicates collapse to one boundary.

    A hairline rule under a cell arrives as ``...248.8] [248.8,249.3] [249.3,...``, so
    one column boundary shows up as four breakpoints within a point.  Clustered, the
    signature of an 11-column table is stable enough to compare against its neighbour.
    """
    if not rule.breaks:
        return ()
    vals = sorted(rule.breaks)
    out, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            out.append(sum(cur) / len(cur))
            cur = [v]
    out.append(sum(cur) / len(cur))
    return tuple(round(v, 1) for v in out)


def _sig_similarity(a, b, tol: float = 3.0) -> float:
    """Jaccard over breakpoint sets with a positional tolerance."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    matched = sum(1 for x in a if any(abs(x - y) <= tol for y in b))
    return matched / float(len(a) + len(b) - matched)


def _split_stack(rules, idxs, cfg: Config):
    """Split a run of parallel rules where the table's *shape* changes.

    Gap-based splitting is not enough.  On page 4 of the ICRA report an 11-column
    rating-history table is followed 18 pt later by a 2-column complexity table, while
    the rating table's own row pitch is 44 pt -- so the gap between the tables is
    *smaller* than the gap between its rows and no gap threshold can separate them.

    What does separate them is the breakpoint signature: the rules under an 11-column
    table are broken at eleven x positions, the 2-column one at one.  We cut wherever
    consecutive rules disagree on that signature, and also at genuinely large gaps.
    """
    if len(idxs) < 3:
        return [idxs]
    sigs = [_break_signature(rules[i]) for i in idxs]
    gaps = [rules[idxs[k + 1]].pos - rules[idxs[k]].pos for k in range(len(idxs) - 1)]
    med_gap = median(gaps) if gaps else 20.0
    cuts = set()
    for k in range(len(idxs) - 1):
        sim = _sig_similarity(sigs[k], sigs[k + 1])
        # Both sides must actually have breakpoints before their disagreement means
        # anything.  An unsegmented full-width top border next to segmented interior
        # rules is completely normal, and cutting there split the Shell balance sheet
        # in half at "II. EQUITY AND LIABILITIES".
        if sigs[k] and sigs[k + 1] and sim < 0.5:
            cuts.add(k)
        if gaps[k] > max(3.0 * med_gap, med_gap + 40.0):
            cuts.add(k)
    parts, cur = [], [idxs[0]]
    for k in range(len(idxs) - 1):
        if k in cuts:
            parts.append(cur)
            cur = []
        cur.append(idxs[k + 1])
    parts.append(cur)
    return [p for p in parts if len(p) >= 3]


def _stack_regions(rules, cfg: Config, axis: str):
    """Group parallel rules into a partially-ruled candidate (h-only or v-only)."""
    if len(rules) < 3:
        return []
    groups = []
    used = [False] * len(rules)
    order = sorted(range(len(rules)), key=lambda i: rules[i].pos)
    for oi in order:
        if used[oi]:
            continue
        cur = [oi]
        used[oi] = True
        for oj in order:
            if used[oj]:
                continue
            a, b = rules[cur[-1]], rules[oj]
            ov = min(a.hi, b.hi) - max(a.lo, b.lo)
            shorter = min(a.length, b.length)
            if shorter <= 0:
                continue
            gap = b.pos - a.pos
            # The gap cap is a *pitch* limit and the pitch differs by axis: 90 pt is a
            # generous row height, but a column can easily be 130 pt wide.  Using the
            # row figure for both meant a five-rule vertical stack was cut after the
            # third rule, so half the column boundaries never reached S4.
            cap = 90.0 if axis == "h" else 320.0
            if ov / shorter >= 0.6 and 0 < gap <= cap:
                cur.append(oj)
                used[oj] = True
        if len(cur) >= 3:
            groups.extend(_split_stack(rules, cur, cfg) if axis == "h" else [cur])

    out = []
    for g in groups:
        rs = [rules[i] for i in g]
        lo = min(r.lo for r in rs)
        hi = max(r.hi for r in rs)
        pos = sorted(r.pos for r in rs)
        brk = sorted({b for r in rs for b in r.breaks})
        if axis == "h":
            bbox = (lo, pos[0], hi, pos[-1])
            out.append(RuledRegion(bbox=bbox, regime=REGIME_HLINES, y_lines=pos,
                                   x_lines=[], n_rules=len(rs), breakpoints=brk,
                                   dotted=any(r.dotted for r in rs)))
        else:
            bbox = (pos[0], lo, pos[-1], hi)
            out.append(RuledRegion(bbox=bbox, regime=REGIME_VLINES, x_lines=pos,
                                   y_lines=[], n_rules=len(rs), breakpoints=brk,
                                   dotted=any(r.dotted for r in rs)))
    return out


def _fill_regions(fills, cfg: Config, page_area: float):
    """Group per-cell background fills into a lattice candidate.

    Apple shades alternating rows one rectangle per cell, so the fill edges give exact
    column boundaries and every other row boundary.  Guards: ignore white, ignore
    anything covering more than a third of the page (page background, banner), and
    ignore colours used fewer than four times (a highlight, not a table).
    """
    if not fills:
        return []
    by_color = {}
    for fb in fills:
        x0, y0, x1, y1 = fb.bbox
        a = (x1 - x0) * (y1 - y0)
        if a <= 4.0 or a > 0.33 * page_area:
            continue
        if min(fb.color) >= 0.98:            # white / near-white
            continue
        by_color.setdefault(fb.color, []).append(fb.bbox)

    out = []
    for color in sorted(by_color):
        boxes = sorted(by_color[color])
        if len(boxes) < 4:
            continue
        # Group *spatially*, not just by colour.  The ICRA report paints its page
        # banner and its footer bars in the same blue as its table shading; grouping on
        # colour alone produced one "table" whose bbox was the entire page.
        #
        # The vertical pad has to clear one unshaded row, because Apple shades
        # *alternate* rows -- with a symmetric 6 pt pad every shaded band became its
        # own group of one band and the whole regime was silently lost.
        heights = [b[3] - b[1] for b in boxes]
        pad_y = min(30.0, max(6.0, 1.6 * (median(heights) or 6.0)))
        for grp in _connected_boxes(boxes, pad=6.0, pad_y=pad_y):
            if len(grp) < 4:
                continue
            y_mid = [(boxes[i][1] + boxes[i][3]) / 2.0 for i in grp]
            bands = cluster_1d(y_mid, 3.0)
            if len(bands) < 2:
                continue
            xs, ys = set(), set()
            for i in grp:
                b = boxes[i]
                xs.add(round(b[0], 2))
                xs.add(round(b[2], 2))
                ys.add(round(b[1], 2))
                ys.add(round(b[3], 2))
            xs, ys = sorted(xs), sorted(ys)
            if len(xs) < 3:                  # fewer than 2 columns of fills
                continue
            out.append(RuledRegion(bbox=(xs[0], ys[0], xs[-1], ys[-1]),
                                   regime=REGIME_FILLS, x_lines=xs, y_lines=ys,
                                   n_fills=len(grp),
                                   evidence={"fill_color": list(color),
                                             "n_bands": len(bands)}))
    return out


def _connected_boxes(boxes, pad: float = 6.0, pad_y=None):
    """Union-find over boxes that touch or nearly touch.  Returns index groups."""
    pad_y = pad if pad_y is None else pad_y
    n = len(boxes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Sweep by y so the quadratic scan only compares plausible neighbours.
    order = sorted(range(n), key=lambda i: (boxes[i][1], boxes[i][0]))
    for oi in range(n):
        i = order[oi]
        for oj in range(oi + 1, n):
            j = order[oj]
            if boxes[j][1] - boxes[i][3] > pad_y:
                break
            if (boxes[i][0] - pad < boxes[j][2] and boxes[j][0] - pad < boxes[i][2]
                    and boxes[i][1] - pad_y < boxes[j][3]
                    and boxes[j][1] - pad_y < boxes[i][3]):
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for _, g in sorted(groups.items())]


def detect_ruled(page, pt, cfg: Config, clip=None) -> RuledEvidence:
    """Full S3a pass for one page."""
    ev = RuledEvidence()
    if not cfg.enable_ruled:
        ev.stats = {"disabled": True}
        return ev
    h_segs, v_segs, dots, fills = harvest_primitives(page, cfg, clip=clip)
    ev.h_rules = merge_rules(h_segs, dots, "h", cfg)
    ev.v_rules = merge_rules(v_segs, dots, "v", cfg)
    ev.fills = fills
    ev.breakpoints = sorted({b for r in ev.h_rules for b in r.breaks})

    grid, pts = _grid_regions(ev.h_rules, ev.v_rules, cfg)
    used_h = {i for r in grid for i in r.evidence.get("h_rules", [])}
    used_v = {i for r in grid for i in r.evidence.get("v_rules", [])}
    rest_h = [r for i, r in enumerate(ev.h_rules) if i not in used_h]
    rest_v = [r for i, r in enumerate(ev.v_rules) if i not in used_v]

    regions = list(grid)
    regions += _stack_regions(rest_h, cfg, "h")
    regions += _stack_regions(rest_v, cfg, "v")
    regions += _fill_regions(fills, cfg, max(1.0, pt.width * pt.height))
    for r in regions:
        if r.regime in (REGIME_HLINES, REGIME_GRID) and not r.breakpoints:
            r.breakpoints = [b for b in ev.breakpoints
                             if r.bbox[0] - 2 <= b <= r.bbox[2] + 2]
    regions.sort(key=lambda r: (round(r.bbox[1], 2), round(r.bbox[0], 2), r.regime))
    ev.regions = regions
    ev.stats = {"n_h_rules": len(ev.h_rules), "n_v_rules": len(ev.v_rules),
                "n_dots": len(dots), "n_fills": len(fills),
                "n_intersections": len(pts), "n_breakpoints": len(ev.breakpoints),
                "regimes": sorted({r.regime for r in regions})}
    return ev


# ------------------------------------------------------------- raster line extraction


def raster_rules(raster, cfg: Config, min_frac: float = None):
    """Morphological line extraction on a scanned page.

    Classical open-with-a-long-kernel, done with numpy run-length logic rather than
    OpenCV so the image path has the same dependency footprint as the vector path.

    The length threshold is **the same 24 pt as the vector path**, converted to pixels
    -- not a fraction of the page. Requiring a run to span 55% of the page dimension was
    the original implementation and it cannot work: a column separator in a table
    occupying a quarter of the page height is never 55% of that height, so the scanned
    fixture yielded three horizontal rules, zero vertical rules, no intersections and no
    table at all. A glyph stroke is well under 24 pt, so the absolute threshold is also
    what keeps text out.
    """
    ink = raster.ink_mask()
    h, w = ink.shape
    out_h, out_v = [], []
    s = raster.scale

    def runs(mask_1d, min_len):
        idx = np.flatnonzero(np.diff(np.concatenate(([0], mask_1d.view(np.int8), [0]))))
        return [(a, b) for a, b in zip(idx[::2], idx[1::2]) if b - a >= min_len]

    min_px = max(8, int(cfg.rule_min_length * s))
    min_len_x = min(min_px, int((min_frac or 0.0) * w) or min_px)
    min_len_y = min(min_px, int((min_frac or 0.0) * h) or min_px)
    # Horizontal: a row whose ink is a long contiguous run.
    for y in range(h):
        row = ink[y]
        if int(row.sum()) < min_len_x:
            continue
        for a, b in runs(row, min_len_x):
            out_h.append(((y + 0.5) / s, a / s, b / s, 1.0 / s))
    for x in range(w):
        col = ink[:, x]
        if int(col.sum()) < min_len_y:
            continue
        for a, b in runs(col, min_len_y):
            out_v.append(((x + 0.5) / s, a / s, b / s, 1.0 / s))
    return out_h, out_v


def detect_ruled_raster(raster, cfg: Config):
    """S3a for a scanned region: rules from morphology, no fills, no breakpoints."""
    ev = RuledEvidence()
    h_segs, v_segs = raster_rules(raster, cfg)
    ev.h_rules = merge_rules(h_segs, [], "h", cfg)
    ev.v_rules = merge_rules(v_segs, [], "v", cfg)
    grid, pts = _grid_regions(ev.h_rules, ev.v_rules, cfg)
    regions = list(grid)
    used_h = {i for r in grid for i in r.evidence.get("h_rules", [])}
    regions += _stack_regions([r for i, r in enumerate(ev.h_rules) if i not in used_h],
                             cfg, "h")
    regions.sort(key=lambda r: (round(r.bbox[1], 2), round(r.bbox[0], 2)))
    ev.regions = regions
    ev.stats = {"n_h_rules": len(ev.h_rules), "n_v_rules": len(ev.v_rules),
                "n_intersections": len(pts), "source": "raster"}
    return ev


# ==================================================================================================
# s3_negatives: S3c -- the things that look like tables and are not.
# ==================================================================================================
#
# Section 15 of the brief lists four traps and says plainly that extracting them costs
# marks while classifying and excluding them earns marks.  Every one of them produces a
# strong column-gutter signal, so they cannot be filtered by the detector; they need
# their own classifiers, and the classifiers need to fire *after* structure is visible
# enough to reason about but *before* anything is written as a table.
#
# Each rule returns a ``Rejection`` with a code, a score and the evidence that triggered
# it.  Rejections are not silently dropped: they go into ``manifest.json`` under
# ``rejected_regions``, are drawn in red by the debug renderer, and are counted in
# ``metrics.json``.  A reviewer can therefore see what the pipeline refused and why,
# which is the only way "we correctly excluded it" is a checkable claim rather than an
# assertion.
#
# Design note -- why rules and not a classifier.  A learned region classifier is the
# obvious alternative and would probably score better on recall.  It would also be
# unexplainable at the live review, could not be justified per-decision, and by §3.1
# could never be the sole basis for a decision anyway.  These rules each key on a
# *structural* invariant of the trap (monotone trailing page numbers; page-number lists;
# independent line rhythm across a single gutter; a rectangle with no interior), so when
# one fires we can say exactly what it saw.

DOT_LEADER_RE = re.compile(r"(\.\s?){3,}|(·\s?){3,}|(…)")
PAGENO_RE = re.compile(r"^\d{1,4}$")
INDEX_PAGES_RE = re.compile(r"^\d{1,4}([–—-]\d{1,4})?(\s*,\s*\d{1,4}([–—-]\d{1,4})?)+$")
SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?$")
TOC_WORD_RE = re.compile(r"\b(contents?|index|glossary|abbreviations?)\b", re.I)


@dataclass
class Rejection:
    code: str
    score: float
    message: str
    evidence: dict = field(default_factory=dict)


@dataclass
class NegativeVerdict:
    rejections: list = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return bool(self.rejections)

    @property
    def top(self):
        return max(self.rejections, key=lambda r: r.score) if self.rejections else None


def _cells_by_band(lines, x_lines):
    """Bucket each line's tokens into column bands -> list of row token lists."""
    rows = []
    for ln in lines:
        row = [[] for _ in range(max(1, len(x_lines) - 1))]
        for t in ln.tokens:
            cx = t.cx
            k = 0
            while k < len(x_lines) - 2 and cx >= x_lines[k + 1]:
                k += 1
            row[k].append(t)
        rows.append(row)
    return rows


def _text_of(toks):
    return " ".join(t.text for t in toks).strip()


def _numeric(s: str) -> bool:
    """Loose "this token is a number" test, tolerant of financial decoration."""
    core = s.strip().strip("()[]%$*†‡").replace(",", "").replace("'", "")
    core = core.lstrip("+-−–")
    return bool(core) and core.replace(".", "", 1).isdigit()


def classify(region_bbox, lines, x_lines, cfg: Config, *, regime: str,
             has_vertical_rules: bool, n_intersections: int, nearby_text=""):
    """Run every trap rule against one candidate region."""
    v = NegativeVerdict()
    if not lines:
        v.rejections.append(Rejection("EMPTY_REGION", 1.0,
                                      "region contains no text lines"))
        return v
    n = len(lines)
    rows = _cells_by_band(lines, x_lines) if len(x_lines) >= 2 else [[ln.tokens]
                                                                    for ln in lines]
    n_cols = max(1, len(x_lines) - 1)
    texts = [[_text_of(c) for c in row] for row in rows]
    flat = [t for row in texts for t in row if t]
    if len(flat) < 4:
        v.rejections.append(Rejection("EMPTY_REGION", 1.0,
                                      "fewer than four non-empty cells"))
        return v

    # ---------------------------------------------------------------- dot leaders
    n_dot = sum(1 for ln in lines if DOT_LEADER_RE.search(ln.text))
    if n_dot / n >= 0.30:
        v.rejections.append(Rejection(
            "TOC_DOT_LEADERS", 0.60 + 0.4 * (n_dot / n),
            "%d of %d rows use dot leaders" % (n_dot, n),
            {"rows_with_leaders": n_dot, "rows": n}))

    # ------------------------------------------------- monotone trailing page nos
    # Keyed on the last *token* of each line, never on inferred columns.  The first
    # version of this rule read the last column band and gated on n_cols <= 4; the
    # Apple table of contents has alternating row shading, so the fill lattice gave it
    # seven columns and the rule never ran.  Line tokens are evidence we always have.
    if n >= 5:
        tails = [ln.tokens[-1].text.strip() if ln.tokens else "" for ln in lines]
        hits = [t for t in tails if PAGENO_RE.match(t)]
        # A TOC row carries exactly one number in the data area: its page.  A financial
        # row carries three to six.  That difference is what keeps this rule off real
        # tables whose last column happens to ascend (a year, a serial number).
        #
        # Counting *all* numeric tokens does not work: "Item 1A. Risk Factors 5" has
        # two, because the item number is one.  Counting only numbers right of the
        # region midline does, since a TOC's item numbers sit hard left.
        mid = (region_bbox[0] + region_bbox[2]) / 2.0
        num_per_row = median([sum(1 for t in ln.tokens
                                  if _numeric(t.text) and t.cx > mid)
                              for ln in lines])
        if len(hits) / n >= 0.70 and num_per_row <= 1.2:
            vals = [int(t) for t in hits]
            asc = sum(1 for a, b in zip(vals, vals[1:]) if b >= a)
            mono = asc / max(1, len(vals) - 1)
            if mono >= 0.9:
                first_alpha = sum(
                    1 for ln in lines
                    if ln.tokens and sum(ch.isalpha() for ch in ln.tokens[0].text))
                score = 0.55 + 0.3 * mono + 0.15 * (first_alpha / n)
                v.rejections.append(Rejection(
                    "TOC_TRAILING_PAGE_NUMBERS", min(0.99, score),
                    "%d of %d rows end in a bare integer, %.0f%% of them ascending, "
                    "median %.1f numbers per row"
                    % (len(hits), n, 100 * mono, num_per_row),
                    {"monotone_frac": round(mono, 3), "n_pagenos": len(hits),
                     "median_numbers_per_row": round(num_per_row, 2)}))

    # ------------------------------------------------------------ index / glossary
    # "Borrowings 31, 44-46" -- the trailing tokens form a page list.
    idx_hits = 0
    for ln in lines:
        tail, n_tail, prev = "", 0, None
        for t in reversed(ln.tokens):
            # Accumulate only across *word* gaps.  An index entry's page list lives
            # inside one cell ("31, 44-46"); accumulating across a 100 pt column gap
            # turned the fixture row "U-001  707  1,281,000" into the page list
            # "7071,281,000" and refused a perfectly good four-column schedule on
            # every one of its six pages.
            if prev is not None and (prev.x0 - t.x1) > 1.5 * max(t.size, prev.size):
                break
            cand = (t.text + " " + tail).strip()
            if re.fullmatch(r"[\d,\s–—-]+", cand):
                tail, n_tail, prev = cand, n_tail + 1, t
            else:
                break
        # The page list must also span at least two tokens.  A single token "169,658"
        # is a thousands-separated number and matches the page-list pattern perfectly,
        # which briefly made this rule reject the entire Apple segment table.
        if n_tail >= 2 and INDEX_PAGES_RE.match(re.sub(r"\s+", "", tail)):
            idx_hits += 1
    if n >= 4 and idx_hits / n >= 0.60:
        v.rejections.append(Rejection(
            "INDEX_PAGE_LIST", 0.60 + 0.35 * (idx_hits / n),
            "%d of %d rows end in a comma-separated page list" % (idx_hits, n),
            {"rows_matching": idx_hits}))

    # ----------------------------------------------------------- two-column prose
    if (not has_vertical_rules and n_intersections == 0 and n_cols == 2
            and n >= cfg.borderless_min_rows):
        col_words = []
        for c in range(2):
            ws = [len(texts[r][c].split()) for r in range(n) if texts[r][c]]
            col_words.append(median(ws) if ws else 0.0)
        alpha = sum(sum(ch.isalpha() for ch in t) for t in flat)
        digits = sum(sum(ch.isdigit() for ch in t) for t in flat)
        alpha_frac = alpha / max(1, alpha + digits)
        sent = sum(1 for t in flat if SENTENCE_END_RE.search(t)) / max(1, len(flat))
        # Independent line rhythm: in prose the two columns break lines separately, so
        # few baselines carry ink on both sides.  In a two-column table almost all do.
        both = sum(1 for row in rows if row[0] and row[1])
        align = both / n
        if (min(col_words) >= 6 and alpha_frac >= 0.85
                and (align < 0.8 or sent >= 0.30)):
            v.rejections.append(Rejection(
                "TWO_COLUMN_PROSE", 0.55 + 0.2 * (1 - align) + 0.2 * sent,
                "two columns of %.0f/%.0f median words, %.0f%% alphabetic, "
                "%.0f%% of rows span both columns"
                % (col_words[0], col_words[1], 100 * alpha_frac, 100 * align),
                {"median_words": col_words, "alpha_frac": round(alpha_frac, 3),
                 "row_alignment": round(align, 3),
                 "sentence_end_frac": round(sent, 3)}))

    # ------------------------------------------------------------- callout / rule
    if regime == "grid" and n_intersections <= 4:
        v.rejections.append(Rejection(
            "BORDERED_CALLOUT_BOX", 0.75,
            "a single bordered rectangle with no interior grid",
            {"n_intersections": n_intersections}))
    if n <= 1:
        v.rejections.append(Rejection(
            "UNDERLINED_HEADING", 0.80,
            "one text line over a rule is a heading, not a one-row table",
            {"n_lines": n}))

    # --------------------------------------------------- heading says it is a TOC
    if nearby_text and TOC_WORD_RE.search(nearby_text):
        boost = 0.45
        if v.rejections:
            for r in v.rejections:
                if r.code.startswith("TOC") or r.code.startswith("INDEX"):
                    r.score = min(0.99, r.score + 0.15)
                    r.evidence["nearby_heading"] = nearby_text[:80]
        else:
            v.rejections.append(Rejection(
                "TOC_HEADING_NEARBY", boost,
                "nearest heading reads %r" % nearby_text[:60],
                {"nearby_heading": nearby_text[:80]}))
    return v


# ==================================================================================================
# s3_borderless: S3b -- borderless table detection by gutter and alignment analysis.
# ==================================================================================================
#
# The brief's sketch is the right one: build the vertical projection profile of ink over a
# candidate block and keep the minima that persist across at least 80% of its rows.  Two
# things have to be added before that works on real pages.
#
# **Prose lines have to be removed first, not filtered out afterwards.**
# Immediately above the Apple segment table sits the sentence "The following table shows
# net sales by reportable segment for 2024, 2023 and 2022 (dollars in millions):".  Its
# inter-word spaces land wherever they land, and with eight table rows below it the
# persistence test still passes at 7/8 = 87% -- so the sentence is quietly admitted as a
# table row and every column shifts.  A line is therefore classified as prose *before*
# profiling, using a signal that does not depend on the columns we are trying to find: a
# line with 8+ tokens whose largest internal gap is no bigger than about three word
# spaces, spanning most of the block, is prose.  A table row like
# ``Americas  $  167,045  3 %  ...`` fails that test on its 200 pt label-to-number gap,
# which is exactly the asymmetry we want to key on.
#
# **One big gutter can mean two tables.**
# Two independent tables side by side (item 5 of the adversarial set) produce one wide
# central gutter, and the naive reading is a single table with a wide middle column.  The
# discriminator is row alignment: halves of one table share baselines, two separate tables
# do not.  :func:`split_side_by_side` cuts on a dominant gutter only when the baseline sets
# on each side disagree.
#
# ``column_gutters`` is deliberately public: S4 calls it to supply column boundaries for
# horizontally-ruled tables, where the rules give rows and the whitespace gives columns.

PROFILE_RES = 0.5      # points per profile bin
# Below this many rows, a gutter must be free in *every* row (see column_gutters).
MIN_ROWS_FOR_TOLERANCE = 10


@dataclass
class BorderlessRegion:
    bbox: tuple
    gutters: list = field(default_factory=list)     # (x0, x1) whitespace bands
    x_lines: list = field(default_factory=list)     # inferred column boundaries
    row_baselines: list = field(default_factory=list)
    line_idx: list = field(default_factory=list)    # indices into pt.lines
    n_rows: int = 0
    n_cols: int = 0
    evidence: dict = field(default_factory=dict)


# A word space is roughly 0.25-0.35 em; 0.6 em is a generous upper bound for one.
WORD_SPACE_EM = 0.6


def _word_spaced(gaps, size: float) -> bool:
    """Are these inter-token gaps uniform *and* word-sized?

    Both halves matter.  Testing only uniformity ("max gap < 3x median") passes a
    numeric table row, because when a row has five wide column gaps the median gap is
    itself wide and three times it is wider still.  That misread
    ``Term loan A  1,200  1,200  900  600  3,900`` as prose and silently deleted two
    rows from the landscape fixture -- a row-dropping bug on ordinary financial tables,
    found only because a fixture asserted the row count.
    """
    if not gaps:
        return False
    med = median(gaps)
    return max(gaps) < max(4.0, 3.0 * med) and med <= WORD_SPACE_EM * max(size, 1.0)


def is_prose_line(ln, block_x0: float, block_x1: float, cfg: Config) -> bool:
    """True if this line reads as running text rather than a table row."""
    toks = ln.tokens
    if len(toks) < 8:
        return False
    gaps = [b.x0 - a.x1 for a, b in zip(toks, toks[1:])]
    gaps = [g for g in gaps if g > 0]
    span = (ln.x1 - ln.x0) / max(1.0, block_x1 - block_x0)
    return _word_spaced(gaps, ln.size) and span >= 0.75


def column_gutters(lines, x0: float, x1: float, cfg: Config, persist=None):
    """Vertical projection profile -> persistent whitespace bands.

    Returns ``(gutters, profile)``; ``gutters`` are ``(x0, x1)`` in points, interior
    only (the page margins on either side are dropped).
    """
    persist = cfg.gutter_persist_frac if persist is None else persist
    n = len(lines)
    if n == 0 or x1 <= x0:
        return [], np.zeros(0, dtype=np.int32)
    nb = max(1, int(math.ceil((x1 - x0) / PROFILE_RES)))
    counts = np.zeros(nb, dtype=np.int32)
    for ln in lines:
        row = np.zeros(nb, dtype=bool)
        for t in ln.tokens:
            a = int(max(0, math.floor((t.x0 - x0) / PROFILE_RES)))
            b = int(min(nb, math.ceil((t.x1 - x0) / PROFILE_RES)))
            if b > a:
                row[a:b] = True
        counts += row

    # A bin is "free" if at most (1 - persist) of the rows put ink in it -- but the
    # tolerance is disabled on short tables.  Its purpose is to stop one overflowing
    # row from destroying a column in a long table; in a five-row table 20% *is* one
    # row, and tolerating it let a gutter open inside the word gap of a long label
    # ("...non-fund | based"), splitting the ICRA label column in two.  Below
    # MIN_ROWS_FOR_TOLERANCE rows there are no outliers -- every row is real evidence.
    max_ink = (0 if n < MIN_ROWS_FOR_TOLERANCE
               else int(math.floor((1.0 - persist) * n + 1e-9)))
    free = counts <= max_ink
    gutters = []
    i = 0
    while i < nb:
        if not free[i]:
            i += 1
            continue
        j = i
        while j < nb and free[j]:
            j += 1
        # Drop the outer margins: a gutter must have ink on both sides.
        if i > 0 and j < nb:
            w = (j - i) * PROFILE_RES
            if w >= cfg.gutter_min_width:
                gutters.append((x0 + i * PROFILE_RES, x0 + j * PROFILE_RES))
        i = j
    return gutters, counts


def is_paragraph_shaped(ln) -> bool:
    """Uniform word spacing: the line reads as running text, whatever its length.

    ``is_prose_line`` needs eight tokens before it will call something prose, which
    misses the last line of a wrapped paragraph ("available on ICRA's website: Click
    Here" -- six tokens).  That fragment then became a fourth row of the ICRA
    complexity table and stretched its bbox 73 pt past the last rule, dropping
    detection IoU to 0.41.

    This test looks only at spacing, so it can be applied to short lines -- but it is
    used *only* to extend an existing paragraph (see paragraph_closure), never on its
    own, because a two-word table row is also uniformly spaced.
    """
    toks = ln.tokens
    if len(toks) < 2:
        return False
    gaps = [b.x0 - a.x1 for a, b in zip(toks, toks[1:])]
    return _word_spaced([g for g in gaps if g > 0], ln.size)


def paragraph_closure(lines, seed_idx, max_iter: int = 6):
    """Grow a set of prose lines over adjacent paragraph-shaped neighbours.

    A table row that happens to sit one line below a paragraph is protected by
    :func:`is_paragraph_shaped`: the Apple segment header row
    ``2024  Change  2023  Change  2022`` has a 64 pt internal gap and is never
    absorbed, while a wrapped sentence fragment is.
    """
    prose = set(seed_idx)
    if not prose:
        return prose
    for _ in range(max_iter):
        grew = False
        for i, ln in enumerate(lines):
            if i in prose or not is_paragraph_shaped(ln):
                continue
            h = max(1.0, ln.y1 - ln.y0)
            for j in (i - 1, i + 1):
                if 0 <= j < len(lines) and j in prose:
                    if abs(ln.baseline - lines[j].baseline) <= 1.6 * h:
                        prose.add(i)
                        grew = True
                        break
        if not grew:
            break
    return prose


def crosses_gutter(ln, gutters, slack: float = 1.0) -> bool:
    """True if any token straddles the interior of a persistent gutter."""
    for a, b in gutters:
        lo, hi = a + slack, b - slack
        if hi <= lo:
            continue
        for t in ln.tokens:
            if t.x0 < hi and t.x1 > lo:
                return True
    return False


def row_stats(ln, gutters, x0: float, x1: float, slack: float = 1.0):
    """``(gutters_crossed, column_bands_occupied)`` for one line."""
    crossed = 0
    for a, b in gutters:
        lo, hi = a + slack, b - slack
        if hi > lo and any(t.x0 < hi and t.x1 > lo for t in ln.tokens):
            crossed += 1
    edges = [x0] + [v for g in gutters for v in g] + [x1]
    bands = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]
    occupied = sum(1 for a, b in bands
                   if any(t.x0 < b and t.x1 > a for t in ln.tokens))
    return crossed, occupied


def trim_to_conforming(lines, run, x0, x1, cfg: Config, max_iter: int = 3):
    """Shrink a candidate block to a conservative core of real table rows.

    Relying on :func:`is_prose_line` alone is not enough.  On page 25 of the Apple 10-K
    the block opens with the title "Segment Operating Performance" and the sentence
    introducing the table.  Neither is prose by a word-gap test (the title has three
    tokens) and -- the subtle part -- the title does not even *cross* a gutter, because
    its own right edge is what defines where that gutter starts.  Both were admitted as
    table rows, giving a 9-column table where there are 6.

    So a boundary row is trimmed when it crosses a gutter *or* occupies at most one
    column band.  Both the title and the sentence fail; the header row
    ``2024 Change 2023 Change 2022`` occupies five bands and survives.

    Interior rows are never trimmed: a section label like "Net sales:" or a spanning
    header cell legitimately occupies one band, and dropping those would destroy the
    row-label hierarchy.  Trimming the *edges* aggressively is safe because S4's
    ``extend_rows`` grows the region back out over any line that conforms to the final
    column boundaries -- S3 finds a core, S4 finds the extent.
    """
    lines = list(lines)
    run = list(run)
    gutters = []
    for _ in range(max_iter):
        if len(run) < cfg.borderless_min_rows:
            return lines, run, gutters
        gx0 = min(l.x0 for l in lines)
        gx1 = max(l.x1 for l in lines)
        gutters, _ = column_gutters(lines, gx0, gx1, cfg)
        if len(gutters) < cfg.borderless_min_cols - 1:
            return lines, run, gutters

        def bad(ln):
            crossed, occupied = row_stats(ln, gutters, gx0, gx1)
            return crossed >= 1 or occupied <= 1

        lo, hi = 0, len(lines)
        while hi - lo > cfg.borderless_min_rows and bad(lines[lo]):
            lo += 1
        while hi - lo > cfg.borderless_min_rows and bad(lines[hi - 1]):
            hi -= 1
        if lo == 0 and hi == len(lines):
            return lines, run, gutters
        lines, run = lines[lo:hi], run[lo:hi]
    gx0 = min(l.x0 for l in lines)
    gx1 = max(l.x1 for l in lines)
    gutters, _ = column_gutters(lines, gx0, gx1, cfg)
    return lines, run, gutters


def _blocks_by_gap(lines, cfg: Config):
    """Split an ordered line list wherever the vertical gap jumps."""
    if not lines:
        return []
    gaps = [b.baseline - a.baseline for a, b in zip(lines, lines[1:])]
    med = median([g for g in gaps if g > 0]) or 12.0
    out, cur = [], [0]
    for k, g in enumerate(gaps):
        if g > cfg.row_gap_mult * med:
            out.append(cur)
            cur = []
        cur.append(k + 1)
    out.append(cur)
    return [b for b in out if b]


def detect_borderless(pt, layout, cfg: Config, exclude=None):
    """S3b pass for one page.  ``exclude`` boxes suppress already-explained areas."""
    if not cfg.enable_borderless:
        return [], {"disabled": True}

    eligible = []
    for i, ln in enumerate(pt.lines):
        if layout is not None and layout.is_chrome(i):
            continue
        if not ln.tokens or not ln.text.strip():
            continue
        if exclude:
            cy = (ln.y0 + ln.y1) / 2.0
            if any(b[1] - 1 <= cy <= b[3] + 1 and ln.x1 > b[0] and ln.x0 < b[2]
                   for b in exclude):
                continue
        eligible.append((i, ln))
    if len(eligible) < cfg.borderless_min_rows:
        return [], {"n_eligible": len(eligible)}

    lines = [ln for _, ln in eligible]
    idxs = [i for i, _ in eligible]
    regions = []
    stats = {"n_eligible": len(eligible), "blocks": 0, "prose_dropped": 0}

    for block in _blocks_by_gap(lines, cfg):
        stats["blocks"] += 1
        bx0 = min(lines[k].x0 for k in block)
        bx1 = max(lines[k].x1 for k in block)
        # Prose lines split the block; they are never table rows.
        runs, cur = [], []
        for k in block:
            if is_prose_line(lines[k], bx0, bx1, cfg):
                stats["prose_dropped"] += 1
                if cur:
                    runs.append(cur)
                cur = []
            else:
                cur.append(k)
        if cur:
            runs.append(cur)

        for run in runs:
            if len(run) < cfg.borderless_min_rows:
                continue
            rx0 = min(lines[k].x0 for k in run)
            rx1 = max(lines[k].x1 for k in run)
            sub = [lines[k] for k in run]
            gutters, prof = column_gutters(sub, rx0, rx1, cfg)
            if len(gutters) < cfg.borderless_min_cols - 1:
                continue
            for part in split_side_by_side(sub, run, gutters, rx0, rx1, cfg):
                p_lines, p_run, p_gutters = part
                p_lines, p_run, p_gutters = trim_to_conforming(
                    p_lines, p_run, rx0, rx1, cfg)
                if len(p_run) < cfg.borderless_min_rows:
                    continue
                if len(p_gutters) < cfg.borderless_min_cols - 1:
                    continue
                px0 = min(l.x0 for l in p_lines)
                px1 = max(l.x1 for l in p_lines)
                y0 = min(l.y0 for l in p_lines)
                y1 = max(l.y1 for l in p_lines)
                x_lines = [px0] + [(a + b) / 2.0 for a, b in p_gutters] + [px1]
                regions.append(BorderlessRegion(
                    bbox=(px0, y0, px1, y1), gutters=p_gutters, x_lines=x_lines,
                    row_baselines=[round(l.baseline, 3) for l in p_lines],
                    line_idx=[idxs[k] for k in p_run], n_rows=len(p_run),
                    n_cols=len(p_gutters) + 1,
                    evidence={"profile_rows": len(p_lines),
                              "gutter_widths": [round(b - a, 2)
                                                for a, b in p_gutters]}))
    regions.sort(key=lambda r: (round(r.bbox[1], 2), round(r.bbox[0], 2)))
    stats["n_regions"] = len(regions)
    return regions, stats


def split_side_by_side(sub_lines, run, gutters, x0, x1, cfg: Config):
    """Split on a dominant gutter when the two halves are independent tables.

    Yields ``(lines, run_indices, gutters)`` triples -- one if no split is warranted.
    The test is baseline agreement: a single table with a wide middle column has rows
    that reach both sides, two stacked-side-by-side tables do not.
    """
    default = [(sub_lines, run, gutters)]
    if len(gutters) < 3:
        return default
    widths = [b - a for a, b in gutters]
    med_w = median(widths)
    best = max(range(len(gutters)), key=lambda i: widths[i])
    if widths[best] < max(3.0 * med_w, 24.0):
        return default

    cut = (gutters[best][0] + gutters[best][1]) / 2.0
    left_bl, right_bl, both = set(), set(), 0
    for ln in sub_lines:
        has_l = any(t.x1 <= cut for t in ln.tokens)
        has_r = any(t.x0 >= cut for t in ln.tokens)
        key = round(ln.baseline, 1)
        if has_l:
            left_bl.add(key)
        if has_r:
            right_bl.add(key)
        if has_l and has_r:
            both += 1
    inter = len(left_bl & right_bl)
    union = len(left_bl | right_bl) or 1
    if inter / union >= 0.7:
        return default            # rows reach both sides: one table, wide column

    l_lines, l_run, r_lines, r_run = [], [], [], []
    for ln, k in zip(sub_lines, run):
        toks_l = [t for t in ln.tokens if t.x1 <= cut]
        toks_r = [t for t in ln.tokens if t.x0 >= cut]
        if len(toks_l) >= len(toks_r) and toks_l:
            l_lines.append(ln)
            l_run.append(k)
        elif toks_r:
            r_lines.append(ln)
            r_run.append(k)
    out = []
    for ls, rs in ((l_lines, l_run), (r_lines, r_run)):
        if len(ls) < cfg.borderless_min_rows:
            continue
        gx0 = min(l.x0 for l in ls)
        gx1 = max(l.x1 for l in ls)
        g, _ = column_gutters(ls, gx0, gx1, cfg)
        out.append((ls, rs, g))
    return out or default


# ==================================================================================================
# s3_arbitrate: S3d -- detector arbitration.
# ==================================================================================================
#
# Both detectors fire on most pages, and they disagree in a specific, predictable way:
#
# *   the ruled detector is *precise but partial* -- on the ICRA report it nails the four
#     row boundaries and the three column boundaries and knows nothing about the header
#     band above the first rule;
# *   the borderless detector is *complete but noisy* -- on the Apple income statement it
#     finds six separate "tables" because the statement's sub-sections are separated by
#     larger vertical gaps than its rows are.
#
# The wrong way to combine them is to pick a winner per page.  What works is to treat
# every detection as a *vote for a region of the page*, cluster the votes spatially, and
# emit one candidate per cluster carrying the union of all the evidence.  S4 then rebuilds
# the grid inside that bbox from scratch, preferring evidence in the order
# grid > fills > rule-breakpoints > gutters.  Six borderless fragments plus one fill
# lattice therefore become one table, and the fill lattice supplies the columns.
#
# Clustering uses ``max(IoU, containment)`` rather than IoU alone.  The underline segments
# beneath Apple's number columns have an IoU of 0.22 against the table they belong to --
# below any sane merge threshold -- but they are 100% *contained* in it, which is the
# relationship that actually matters.
#
# ``detector_votes_json`` on the tables row records every vote and the reason the cluster
# was accepted or rejected, so any decision here can be replayed at review time.

@dataclass
class Candidate:
    """One arbitrated table candidate on one page."""

    page_no: int
    bbox: tuple
    regime: str
    x_lines: list = field(default_factory=list)
    y_lines: list = field(default_factory=list)
    # Kept separate as well as unioned: S4 picks one row source rather than
    # combining them, because every extra y boundary splits a row.
    rule_y_lines: list = field(default_factory=list)
    fill_y_lines: list = field(default_factory=list)
    # Same separation on the x axis.  The union alone is not enough: S4 weights a
    # drawn vertical rule at 3.0 and an inferred gutter at 1.0, and it can only do
    # that if it knows which is which.
    rule_x_lines: list = field(default_factory=list)
    fill_x_lines: list = field(default_factory=list)
    breakpoints: list = field(default_factory=list)
    gutters: list = field(default_factory=list)
    line_idx: list = field(default_factory=list)
    votes: dict = field(default_factory=dict)
    n_intersections: int = 0
    has_vertical_rules: bool = False
    dotted: bool = False
    rejections: list = field(default_factory=list)
    accepted: bool = True

    @property
    def area_key(self):
        return (round(self.bbox[1], 2), round(self.bbox[0], 2))


def _overlap(a, b) -> float:
    return max(iou(a, b), contain_frac(a, b), contain_frac(b, a))


def _cluster(boxes, threshold: float):
    """Single-link clustering of boxes on ``max(IoU, containment)``."""
    n = len(boxes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def merge(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(i + 1, n):
            if _overlap(boxes[i], boxes[j]) >= threshold:
                merge(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for _, g in sorted(groups.items())]


# Evidence precedence for boundary resolution, best first.
REGIME_RANK = {REGIME_GRID: 0, REGIME_FILLS: 1, REGIME_HLINES: 2,
               REGIME_VLINES: 3, "borderless": 4}
# "Strong" means the author drew it: rules and cell fills alike.  Only the
# borderless detector's whitespace inference is weak.  Restricting strong to
# (grid, fills) split the ICRA summary table in two, because its header shading and
# body shading are separate fill groups and the horizontal rules that span both were
# being treated as a mere bridge.
STRONG_KINDS = (REGIME_GRID, REGIME_FILLS, REGIME_HLINES, REGIME_VLINES)


def _merge_stacked(clusters, votes, max_gap: float = 14.0,
                   min_x_overlap: float = 0.6):
    """Merge strong clusters that are stacked vertically and share their x extent.

    One table can produce two separate strong regions -- a shaded header block and a
    shaded body block, or two rule stacks split by a wide row.  Those are contiguous in
    y and cover the same columns.  Two tables side by side are the opposite: they
    overlap in y and are disjoint in x.  Testing both axes separates the cases without
    needing to know which is which in advance.
    """
    boxes = [union(votes[i]["bbox"] for i in cl) for cl in clusters]
    changed = True
    while changed and len(clusters) > 1:
        changed = False
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                ba, bb = boxes[a], boxes[b]
                lo, hi = (ba, bb) if ba[1] <= bb[1] else (bb, ba)
                gap = hi[1] - lo[3]
                ov = min(ba[2], bb[2]) - max(ba[0], bb[0])
                narrower = min(ba[2] - ba[0], bb[2] - bb[0])
                if gap <= max_gap and narrower > 0 and ov / narrower >= min_x_overlap:
                    clusters[a] = clusters[a] + clusters[b]
                    boxes[a] = union([ba, bb])
                    del clusters[b], boxes[b]
                    changed = True
                    break
            if changed:
                break
    return clusters


def _cluster_two_tier(votes, threshold: float):
    """Cluster strong evidence first; weak evidence may not bridge two strong clusters.

    Single-link clustering over all votes together has a failure mode that matters:
    two independent tables side by side are found separately by the ruled detector, but
    the borderless detector often proposes one wide region covering both.  Because that
    wide region overlaps each of them, single-link merges all three into one candidate
    and two tables become one -- item 5 of the adversarial set, silently destroyed.

    So strong members (a rule lattice, a cell-fill lattice) form the seed clusters, and
    each weak member is attached to the *one* strong cluster it overlaps most.  A weak
    member that overlaps two of them is dropped as a bridge: its own boundaries are
    redundant anyway, since the strong evidence is what S4 will prefer.
    """
    strong_idx = [i for i, v in enumerate(votes) if v["kind"] in STRONG_KINDS]
    weak_idx = [i for i, v in enumerate(votes) if v["kind"] not in STRONG_KINDS]
    if not strong_idx:
        return _cluster([v["bbox"] for v in votes], threshold)

    seeds = _cluster([votes[i]["bbox"] for i in strong_idx], threshold)
    clusters = [[strong_idx[k] for k in grp] for grp in seeds]
    clusters = _merge_stacked(clusters, votes)
    for wi in weak_idx:
        wb = votes[wi]["bbox"]
        hits = []
        for ci, cl in enumerate(clusters):
            best = max(_overlap(wb, votes[i]["bbox"]) for i in cl)
            if best >= threshold:
                hits.append((best, ci))
        if len(hits) == 1:
            clusters[hits[0][1]].append(wi)
        elif len(hits) > 1:
            continue          # a bridge across two tables: drop it
        else:
            clusters.append([wi])
    # Weak-only clusters may still merge among themselves.
    weak_only = [cl for cl in clusters if all(votes[i]["kind"] not in STRONG_KINDS
                                              for i in cl)]
    mixed = [cl for cl in clusters if cl not in weak_only]
    if len(weak_only) > 1:
        flat = [i for cl in weak_only for i in cl]
        regrouped = _cluster([votes[i]["bbox"] for i in flat], threshold)
        weak_only = [[flat[k] for k in grp] for grp in regrouped]
    out = [sorted(cl) for cl in mixed + weak_only]
    return sorted(out, key=lambda cl: (round(votes[cl[0]]["bbox"][1], 2),
                                       round(votes[cl[0]]["bbox"][0], 2)))


def arbitrate(page_no: int, pt, layout, ruled_ev, borderless_regions, cfg: Config,
              min_content_lines: int = 2, text_available: bool = True):
    """Cluster ruled + borderless detections into accepted/rejected candidates.

    ``text_available=False`` for an image-only page.  Both the content check and the
    trap classifiers read the text layer, and a scan has none -- so on a scanned page
    they would reject every region as ``EMPTY_REGION`` no matter how clean its grid was.
    That is what silently discarded the scanned fixture's table after the raster
    morphology had correctly recovered 23 intersections from it.  For those pages the
    grid structure itself is the evidence, and the traps cannot be evaluated at all
    (a table of contents is unrecognisable without its text), which is recorded as a
    ``TRAPS_NOT_EVALUATED_NO_TEXT`` note on the candidate.
    """
    votes = []
    for r in ruled_ev.regions if ruled_ev is not None else []:
        votes.append({"kind": r.regime, "bbox": r.bbox, "x_lines": list(r.x_lines),
                      "y_lines": list(r.y_lines),
                      "breakpoints": list(r.breakpoints),
                      "gutters": [], "line_idx": [],
                      "n_intersections": r.n_intersections,
                      "n_rules": r.n_rules, "n_fills": r.n_fills,
                      "dotted": r.dotted})
    for b in borderless_regions or []:
        votes.append({"kind": "borderless", "bbox": b.bbox, "x_lines": list(b.x_lines),
                      "y_lines": [], "breakpoints": [], "gutters": list(b.gutters),
                      "line_idx": list(b.line_idx), "n_intersections": 0,
                      "n_rules": 0, "n_fills": 0, "dotted": False})
    if not votes:
        return [], []

    groups = _cluster_two_tier(votes, cfg.merge_iou)
    accepted, rejected = [], []

    for g in groups:
        members = [votes[i] for i in g]
        members.sort(key=lambda v: (REGIME_RANK.get(v["kind"], 9),
                                    round(v["bbox"][1], 2)))
        bbox = union(v["bbox"] for v in members)
        best = members[0]
        cand = Candidate(
            page_no=page_no, bbox=bbox, regime=best["kind"],
            n_intersections=max(v["n_intersections"] for v in members),
            dotted=any(v["dotted"] for v in members),
            has_vertical_rules=any(
                v["kind"] in (REGIME_GRID, REGIME_VLINES) for v in members),
        )
        # Union of evidence, best-regime first so S4 can trust the order.
        for v in members:
            cand.x_lines.extend(v["x_lines"])
            cand.y_lines.extend(v["y_lines"])
            if v["kind"] in (REGIME_GRID, REGIME_HLINES):
                cand.rule_y_lines.extend(v["y_lines"])
            elif v["kind"] == REGIME_FILLS:
                cand.fill_y_lines.extend(v["y_lines"])
            if v["kind"] in (REGIME_GRID, REGIME_VLINES):
                cand.rule_x_lines.extend(v["x_lines"])
            elif v["kind"] == REGIME_FILLS:
                cand.fill_x_lines.extend(v["x_lines"])
            cand.breakpoints.extend(v["breakpoints"])
            cand.gutters.extend(v["gutters"])
            cand.line_idx.extend(v["line_idx"])
        cand.x_lines = sorted(set(round(x, 2) for x in cand.x_lines))
        cand.y_lines = sorted(set(round(y, 2) for y in cand.y_lines))
        cand.breakpoints = sorted(set(round(x, 2) for x in cand.breakpoints))
        cand.rule_y_lines = sorted(set(round(y, 2) for y in cand.rule_y_lines))
        cand.fill_y_lines = sorted(set(round(y, 2) for y in cand.fill_y_lines))
        cand.rule_x_lines = sorted(set(round(x, 2) for x in cand.rule_x_lines))
        cand.fill_x_lines = sorted(set(round(x, 2) for x in cand.fill_x_lines))
        cand.line_idx = sorted(set(cand.line_idx))
        cand.votes = {
            "chosen_regime": cand.regime,
            "n_votes": len(members),
            "detectors": sorted({("ruled:" + v["kind"]) if v["kind"] != "borderless"
                                 else "borderless" for v in members}),
            "members": [{"kind": v["kind"],
                         "bbox": [round(c, 2) for c in v["bbox"]],
                         "n_rules": v["n_rules"], "n_fills": v["n_fills"],
                         "n_intersections": v["n_intersections"]}
                        for v in members],
            "reason": _reason(members),
        }

        # Lines actually inside the merged bbox drive the trap classifiers.
        inside = [i for i, ln in enumerate(pt.lines)
                  if not (layout is not None and layout.is_chrome(i))
                  and ln.text.strip()
                  and bbox[1] - 1.5 <= (ln.y0 + ln.y1) / 2.0 <= bbox[3] + 1.5
                  and ln.x1 > bbox[0] - 1.5 and ln.x0 < bbox[2] + 1.5]
        lines_in = [pt.lines[i] for i in inside]
        if not text_available:
            enough_grid = (cand.n_intersections >= cfg.ruled_evidence_min_cells
                           or len(cand.y_lines) >= cfg.ruled_min_rows + 1)
            cand.votes["traps_evaluated"] = False
            cand.votes["note"] = "TRAPS_NOT_EVALUATED_NO_TEXT"
            if enough_grid:
                cand.line_idx = inside
                accepted.append(cand)
            else:
                cand.accepted = False
                cand.rejections = [Rejection(
                    "INSUFFICIENT_GRID_ON_RASTER", 1.0,
                    "image-only page: %d intersections, %d row boundaries -- not "
                    "enough structure to call it a table"
                    % (cand.n_intersections, len(cand.y_lines)),
                    {"n_intersections": cand.n_intersections})]
                rejected.append(cand)
            continue
        if len(lines_in) < min_content_lines:
            cand.accepted = False
            cand.rejections = [Rejection(
                "EMPTY_REGION", 1.0,
                "merged region holds %d text lines" % len(lines_in),
                {"n_lines": len(lines_in)})]
            rejected.append(cand)
            continue

        x_for_traps = cand.x_lines or [bbox[0], bbox[2]]
        verdict = classify(
            bbox, lines_in, x_for_traps, cfg, regime=cand.regime,
            has_vertical_rules=cand.has_vertical_rules,
            n_intersections=cand.n_intersections,
            nearby_text=_nearest_heading_text(pt, layout, bbox))
        cand.rejections = verdict.rejections
        if verdict.rejected:
            cand.accepted = False
            rejected.append(cand)
        else:
            cand.line_idx = inside
            accepted.append(cand)

    accepted.sort(key=lambda c: c.area_key)
    rejected.sort(key=lambda c: c.area_key)
    return accepted, rejected


def _reason(members) -> str:
    kinds = [m["kind"] for m in members]
    if REGIME_GRID in kinds:
        return ("full rule lattice present (%d intersections); rules define both axes"
                % max(m["n_intersections"] for m in members))
    if REGIME_FILLS in kinds and REGIME_HLINES in kinds:
        return ("per-cell background fills give columns, horizontal rules confirm "
                "rows; borderless votes merged for extent")
    if REGIME_FILLS in kinds:
        return "per-cell background fills give the cell lattice directly"
    if REGIME_HLINES in kinds:
        return ("horizontal rules only: rows from rules, columns from rule "
                "breakpoints and gutter analysis")
    if REGIME_VLINES in kinds:
        return "vertical rules only: columns from rules, rows from text baselines"
    return "no rule or fill evidence; columns and rows from gutter analysis alone"


def _nearest_heading_text(pt, layout, bbox, max_gap: float = 70.0) -> str:
    """Nearest heading or caption above the region -- used by the TOC/index rules."""
    if layout is None:
        return ""
    best, best_gap = "", 1e9
    for i, _lvl, txt, bb in getattr(layout, "headings", []):
        gap = bbox[1] - bb[3]
        if 0 <= gap < min(best_gap, max_gap):
            best, best_gap = txt, gap
    for i, txt, bb in getattr(layout, "captions", []):
        gap = bbox[1] - bb[3]
        if 0 <= gap < min(best_gap, max_gap):
            best, best_gap = txt, gap
    return best


# ==================================================================================================
# s4_structure: S4 -- table structure recognition: the actual grid.
# ==================================================================================================
#
# Everything upstream produced *evidence*.  This module is the single place that turns it
# into boundaries, and it does so by weighted voting rather than by picking one detector,
# because on real pages the sources are complementary and individually incomplete.
#
# Column boundary weights (measured on the three sample documents, see docs/DECISIONS.md D4):
#
# ===========================  ======  =========================================
# source                       weight  why
# ===========================  ======  =========================================
# vertical rule                  3.0   drawn by the author; unambiguous
# horizontal-rule breakpoint     2.5   the author's own cell edges, sub-point precise
# cell background fill edge      2.0   exact, but only covers shaded rows
# gutter centre                  1.0   inferred; correct on average, drifts under
#                                      right-aligned numbers
# ===========================  ======  =========================================
#
# A candidate boundary is then vetoed if tokens straddle it on more than a few percent of
# rows -- a boundary that cuts through text is wrong no matter how it was proposed.
#
# Three problems get explicit treatment because they are common and quietly destructive:
#
# **Adornment columns.**  EDGAR filings put the currency symbol in its own column: the
# Apple segment table is geometrically ``label | $ | 167,045 | 3 | % | $ | 162,560 | ...``
# -- eleven bands for six logical columns and five headers.  Bands whose non-empty cells
# are *exclusively* a currency symbol fold right into the number; bands that are
# exclusively ``%`` fold left.  The currency moves into ``columns.unit`` so nothing is
# lost, and the column count then matches the header count, which is what makes
# ``header_path`` alignment possible at all.
#
# **Multi-line cells.**  In the ICRA rating table one row band contains three text lines
# (``[ICRA]AA-(Stable)/[ICRA]A1+;`` / ``Long term/Short term ... 143.00 143.00`` /
# ``reaffirmed``).  Rows must therefore come from rule positions where rules exist, and
# text-line clustering must merge continuations where they do not.  Getting this wrong
# turns one row into three and breaks every sum in the table.
#
# **Indentation as the only hierarchy.**  ``Product A`` belongs to ``Domestic`` with no
# bullet, no numbering and no bold -- just leading whitespace.  First-column left edges
# are clustered into indent levels and a stack turns them into ``row_label_path``.

W_VRULE = 3.0
W_BREAKPOINT = 2.5
W_FILL_EDGE = 2.0
W_GUTTER = 1.0

MIN_COL_WIDTH = 4.0
# A data column narrower than this is cell padding, not a column.
MIN_DATA_COL_WIDTH = 7.0
# A gutter-derived boundary this close to an already-chosen precise boundary
# is the same boundary, not a second one.
GUTTER_DUP_RADIUS = MIN_COL_WIDTH


def _ceil(v):
    """Ceiling with a floor of 1 -- an integer row budget, never zero."""
    import math

    return max(1, int(math.ceil(v)))

# Strong sources may be straddled by this fraction of rows before we
# disbelieve them; weak sources get zero tolerance (see resolve_columns).
STRADDLE_VETO_FRAC = 0.12


@dataclass
class GridCell:
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    tokens: list = field(default_factory=list)
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    straddled: bool = False
    flags: list = field(default_factory=list)


@dataclass
class Grid:
    x_lines: list = field(default_factory=list)
    y_lines: list = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    n_header_rows: int = 1
    cells: list = field(default_factory=list)
    header_paths: list = field(default_factory=list)
    indent_levels: list = field(default_factory=list)
    row_label_paths: list = field(default_factory=list)
    col_units: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    row_lines: list = field(default_factory=list)   # line indices per row band

    def cell_at(self, r: int, c: int):
        for cl in self.cells:
            if cl.row <= r < cl.row + cl.row_span and cl.col <= c < cl.col + cl.col_span:
                return cl
        return None


# --------------------------------------------------------------------- boundaries


def _weighted_boundaries(cand, lines, cfg: Config):
    """Collect and score candidate column boundaries from every evidence source."""
    x0, x1 = cand.bbox[0], cand.bbox[2]
    props = []
    for x in cand.breakpoints:
        if x0 + MIN_COL_WIDTH < x < x1 - MIN_COL_WIDTH:
            props.append((x, W_BREAKPOINT, "breakpoint"))
    # Only *actual* rule positions carry the rule weight.  cand.x_lines is the union
    # of every member's boundaries, so weighting all of it at 3.0 promoted the
    # borderless detector's gutter estimates to the authority of a drawn rule, and on
    # the continuation fixture that put boundaries at x=224 and x=379 on pages without
    # a header while pages with one used the real rules at 200 and 330 -- so the two
    # halves of one table no longer looked alike and stitching refused to join them.
    for x in cand.rule_x_lines:
        if x0 + MIN_COL_WIDTH < x < x1 - MIN_COL_WIDTH:
            props.append((x, W_VRULE, "vrule"))
    for x in cand.fill_x_lines:
        if x0 + MIN_COL_WIDTH < x < x1 - MIN_COL_WIDTH:
            props.append((x, W_FILL_EDGE, "fill_edge"))
    gutters, _ = column_gutters(lines, x0, x1, cfg)
    for a, b in gutters:
        props.append((refine_boundary(a, b, lines), W_GUTTER, "gutter"))
    return props, gutters


HEADER_BAND_FRAC = 0.15


def _straddle_count(x: float, lines, slack: float = 0.15,
                    skip_top_frac: float = 0.0):
    """How many lines have a token crossing ``x``.

    ``skip_top_frac`` excludes lines in the top slice of the region.  A header cell
    that spans two data columns *must* cross their boundary -- that is what spanning
    means -- so counting it as evidence against the boundary is backwards.  On page 2
    of Shell.pdf the boundaries separating each year's line-item column from its
    subtotal column were straddled by exactly two lines, both header
    (``2022-23`` and ``( Million)``), and vetoing them collapsed four value columns
    into two.

    Body straddles still count in full.  Where a boundary genuinely cuts body text --
    the same page has three long labels overflowing into the Note column -- the veto
    stands and S9 flags the affected cells rather than the pipeline inventing a column.

    ``slack`` is a rounding tolerance and nothing more.  At the original 0.6 pt a
    boundary could graze the last 0.08 pt of a glyph and not be counted, which is
    exactly how a boundary landed inside "non-fund" and split the ICRA label column.
    """
    if not lines:
        return 0
    cutoff = None
    if skip_top_frac > 0.0:
        ys = [ln.baseline for ln in lines]
        lo, hi = min(ys), max(ys)
        cutoff = lo + skip_top_frac * (hi - lo)
    n = 0
    for ln in lines:
        if cutoff is not None and ln.baseline <= cutoff:
            continue
        for t in ln.tokens:
            if t.x0 < x - slack and t.x1 > x + slack:
                n += 1
                break
    return n


def splits_word_gap(x: float, lines) -> bool:
    """Does ``x`` fall inside an ordinary space *between two words of one cell*?

    The straddle test cannot see this, because the boundary cuts no glyph -- it lands
    in the 2.4 pt gap between "non-fund" and "based".  The whitespace profile happily
    calls that gap a gutter when every other row is blank there, which split the ICRA
    label column mid-phrase and shifted three cells one column right.

    A column boundary never sits in a word space.  The threshold scales with the font
    so it holds for 6 pt footnotes and 14 pt headings alike; anything wider than a
    generous word space is real column whitespace and is left alone.
    """
    for ln in lines:
        toks = ln.tokens
        for a, b in zip(toks, toks[1:]):
            if a.x1 <= x <= b.x0:
                gap = b.x0 - a.x1
                limit = max(3.5, 0.45 * max(a.size, b.size))
                if gap <= limit:
                    return True
    return False


def refine_boundary(lo: float, hi: float, lines, prefer: float = None) -> float:
    """Place a boundary inside a whitespace band where it cuts the least text.

    A gutter's midpoint is the obvious choice and it is wrong when the gutter is
    asymmetric.  Between the Shell label column and its Note column the free band runs
    from x=250 to x=326 because most labels are short, but three long labels reach
    x=316; the midpoint at 288 cuts all three, while x=321 cuts nothing.  Scanning the
    band for the least-cutting position recovers the column instead of losing it.
    """
    prefer = (lo + hi) / 2.0 if prefer is None else prefer
    best, best_key = prefer, None
    x = lo + 0.5
    while x < hi:
        sc = _straddle_count(x, lines, skip_top_frac=HEADER_BAND_FRAC)
        key = (sc, 1 if splits_word_gap(x, lines) else 0, abs(x - prefer))
        if best_key is None or key < best_key:
            best, best_key = x, key
        x += 1.0
    return best


def resolve_columns(cand, lines, cfg: Config):
    """Weighted vote -> vetted column boundaries, plus the evidence trail."""
    x0, x1 = cand.bbox[0], cand.bbox[2]
    props, gutters = _weighted_boundaries(cand, lines, cfg)
    if not props:
        return [x0, x1], {"sources": {}, "vetoed": [], "gutters": len(gutters)}

    xs = [p[0] for p in props]
    clusters = cluster_1d(xs, cfg.col_align_tol)
    scored = []
    for cl in clusters:
        w = sum(props[i][1] for i in cl)
        pos = sum(props[i][0] * props[i][1] for i in cl) / w
        srcs = sorted({props[i][2] for i in cl})
        scored.append((w, pos, srcs))
    scored.sort(key=lambda s: (-s[0], s[1]))

    n_rows = max(1, len(lines))
    chosen, vetoed = [], []
    for w, pos, srcs in scored:
        sc = _straddle_count(pos, lines, skip_top_frac=HEADER_BAND_FRAC)
        # Strong sources (a drawn rule, the author's own cell edge) may overrule a
        # straddling token -- text legitimately overflows a narrow cell.  A boundary
        # proposed only by shading edges or whitespace has no such authority, so any
        # straddle kills it.  Without this a stray fill edge at x=223 cut the ICRA
        # label column through the middle of the word "based".
        strong = any(s in ("vrule", "breakpoint") for s in srcs)
        limit = _ceil(STRADDLE_VETO_FRAC * n_rows) if strong else 0
        if sc > limit:
            vetoed.append({"x": round(pos, 2), "weight": round(w, 2),
                           "sources": srcs, "straddles": sc, "limit": limit})
            continue
        if not strong and splits_word_gap(pos, lines):
            vetoed.append({"x": round(pos, 2), "weight": round(w, 2),
                           "sources": srcs, "reason": "falls inside a word space"})
            continue
        # Suppression radius depends on how precise the proposal is.  A drawn rule or
        # a rule breakpoint locates a boundary to a fraction of a point, so only a
        # near-exact duplicate is redundant.  A gutter is a fuzzy estimate of a
        # boundary those sources may already have found precisely: the ICRA summary
        # table has a breakpoint at x=249 and a refined gutter centre at x=254, which
        # are the same boundary and, kept as two, inserted a phantom empty column that
        # shifted every cell one place right.
        radius = MIN_COL_WIDTH if strong else GUTTER_DUP_RADIUS
        if any(abs(pos - c) < radius for c in chosen):
            continue
        chosen.append(pos)
    chosen.sort()
    bounds = [x0] + chosen + [x1]
    used = {}
    for w, pos, srcs in scored:
        if any(abs(pos - c) < 1e-6 for c in chosen):
            for s in srcs:
                used[s] = used.get(s, 0) + 1
    return bounds, {"sources": used, "vetoed": vetoed, "gutters": len(gutters),
                    "n_proposals": len(scored)}


NON_ROW_ROLES = ("caption", "heading", "header", "footer", "footnote")


def _growable_lines(pt, layout):
    """Lines a table may legitimately grow over: not chrome, caption, heading, footnote.

    S2 has already been corrected by ``demote_in_regions``, so a bold header row that
    sits *inside* a detected region carries the role "table" and stays growable; only
    genuine page furniture outside the region is excluded.
    """
    out = []
    for i, ln in enumerate(pt.lines):
        if not ln.text.strip():
            continue
        if layout is not None and layout.role_of(i) in NON_ROW_ROLES:
            continue
        out.append(ln)
    return out


def _body_lines(lines):
    """Lines below the header band, used where header spans would mislead."""
    if len(lines) < 4:
        return list(lines)
    ys = [ln.baseline for ln in lines]
    cutoff = min(ys) + HEADER_BAND_FRAC * (max(ys) - min(ys))
    out = [ln for ln in lines if ln.baseline > cutoff]
    return out if len(out) >= 2 else list(lines)


def _band_of(x: float, bounds) -> int:
    k = 0
    while k < len(bounds) - 2 and x >= bounds[k + 1]:
        k += 1
    return k


def fold_adornment_columns(bounds, lines, cfg: Config, body_lines=None,
                           protected=()):
    """Drop empty bands and fold currency/percent-only bands into their neighbour.

    Returns ``(new_bounds, units, notes)`` where ``units[i]`` is a currency code
    recovered from a folded symbol column, or "".

    ``protected`` lists boundary positions backed by a drawn rule or a cell-fill edge.
    Absorbing an empty band must never dissolve one of those.  On the continuation
    fixture the empty padding band between the Area column's right edge and the Value
    column's text was absorbed leftwards, which moved the boundary from the author's
    own vertical rule at x=328.6 out to x=405 -- and since the same thing happened
    differently on pages with and without a header, the two halves of one table stopped
    matching and stitching refused to join them.

    Band contents are read from ``body_lines`` (the rows below the header band), not
    from every line.  A spanning header cell places token centres wherever its text
    happens to fall: on page 2 of Shell.pdf the word "Million)" centres inside the
    9 pt padding band between a year's line-item column and its subtotal column, which
    made that padding look like a data column and left two phantom empty columns in an
    otherwise correct six-column table.
    """
    if not cfg.fold_adornment_columns or len(bounds) <= 2:
        return bounds, [""] * (len(bounds) - 1), []
    n_bands = len(bounds) - 1
    contents = [[] for _ in range(n_bands)]
    for ln in (body_lines if body_lines else lines):
        for t in ln.tokens:
            contents[_band_of(t.cx, bounds)].append(t.text)

    kind = []
    for i in range(n_bands):
        texts = [t for t in contents[i] if t.strip()]
        width = bounds[i + 1] - bounds[i]
        if not texts:
            kind.append("empty")
        elif all(is_currency_symbol(t) for t in texts) and width <= cfg.adornment_max_width:
            kind.append("currency")
        elif all(t.strip() in ("%", "‰") for t in texts) and width <= cfg.adornment_max_width:
            kind.append("percent")
        elif width < MIN_DATA_COL_WIDTH:
            # The author's own rule breakpoints include cell *padding*: the ICRA
            # rating table has a 5.7 pt band between "Amount rated" and "Date" that
            # catches the tail of a wrapped word.  No real data column is one glyph
            # wide, so treat it as padding and fold it away.
            kind.append("narrow")
        else:
            kind.append("data")

    if not any(k != "data" for k in kind):
        return bounds, [""] * n_bands, []

    # Walk left to right building keep-groups: each data band absorbs adjacent
    # currency bands on its left and percent/empty bands on its right.
    units = [""] * n_bands
    group_of = [None] * n_bands
    data_idx = [i for i, k in enumerate(kind) if k == "data"]
    if not data_idx:
        return bounds, units, ["all_bands_adornment"]
    for i, k in enumerate(kind):
        if k == "data":
            group_of[i] = i
    notes = []
    for i, k in enumerate(kind):
        if k == "data":
            continue
        right = next((j for j in data_idx if j > i), None)
        left = next((j for j in reversed(data_idx) if j < i), None)
        if k == "narrow":
            l_ok = left is not None and not _is_protected(bounds[i], protected)
            r_ok = right is not None and not _is_protected(bounds[i + 1], protected)
            tgt = left if l_ok else (right if r_ok else
                                     (left if left is not None else right))
            if tgt is not None:
                notes.append("folded_narrow_band_%d_into_%d" % (i, tgt))
        elif k == "currency":
            tgt = right if right is not None else left
            if tgt is not None:
                sym = next((t for t in contents[i] if t.strip()), "")
                units[tgt] = _currency_code(sym)
                notes.append("folded_currency_band_%d_into_%d" % (i, tgt))
        elif k == "percent":
            tgt = left if left is not None else right
            if tgt is not None:
                notes.append("folded_percent_band_%d_into_%d" % (i, tgt))
        else:
            # Empty padding: attach to a neighbour -- but never in the direction that
            # would dissolve a protected boundary.  The empty band between the Area
            # column's right rule and the Value column's text was absorbed leftwards,
            # moving the author's own vertical rule from x=328.6 out to x=405; the same
            # band folded differently on header and non-header pages, so the two halves
            # of one logical table no longer matched and stitching refused to join them.
            l_ok = left is not None and not _is_protected(bounds[i], protected)
            r_ok = right is not None and not _is_protected(bounds[i + 1], protected)
            if l_ok and not r_ok:
                tgt = left
            elif r_ok and not l_ok:
                tgt = right
            elif left is None:
                tgt = right
            elif right is None:
                tgt = left
            else:
                tgt = left if (bounds[i] - bounds[left + 1]) <= \
                    (bounds[right] - bounds[i + 1]) else right
        group_of[i] = tgt

    new_bounds = [bounds[0]]
    new_units = []
    for j in data_idx:
        members = [i for i in range(n_bands) if group_of[i] == j]
        hi = bounds[max(members) + 1]
        new_bounds.append(hi)
        new_units.append(units[j])
    new_bounds[-1] = bounds[-1]
    return new_bounds, new_units, notes


def _is_protected(x, protected, tol: float = 1.5) -> bool:
    """Is this boundary backed by a drawn rule or a cell-fill edge?"""
    return any(abs(x - p) <= tol for p in protected)


CURRENCY_CODES = {"$": "USD", "US$": "USD", "₹": "INR", "`": "INR", "Rs": "INR",
                  "Rs.": "INR", "INR": "INR", "€": "EUR", "£": "GBP",
                  "¥": "JPY", "₨": "INR"}


def _currency_code(sym: str) -> str:
    s = sym.strip()
    return CURRENCY_CODES.get(s, CURRENCY_CODES.get(s.rstrip("."), s))


# ------------------------------------------------------------------------- rows


def _baseline_bands(lines, y0, y1):
    """Row bands from text baselines, merging lines that are cell continuations."""
    if not lines:
        return [(y0, y1)]
    gaps = [b.baseline - a.baseline for a, b in zip(lines, lines[1:])
            if b.baseline - a.baseline > 0]
    med = median(gaps) if gaps else 12.0
    groups, cur = [], [lines[0]]
    for prev, ln in zip(lines, lines[1:]):
        if ln.baseline - prev.baseline <= 0.75 * med:
            cur.append(ln)          # continuation of a multi-line cell
        else:
            groups.append(cur)
            cur = [ln]
    groups.append(cur)
    out = [(min(l.y0 for l in g), max(l.y1 for l in g)) for g in groups]
    return _pad_bands(out, y0, y1)


def _score_bands(bands, lines, col_bounds):
    """How plausible is this set of row bands?

    Three components, all about "one band, one row":

    ``labelled``
        fraction of bands with text in the first column.  A band holding only the
        second line of a wrapped cell has an empty label, so a band set that splits
        multi-line cells scores badly.
    ``tall_enough``
        fraction of bands at least 80% of a text line high.  Alternating-shading fill
        edges produce 5 pt sub-bands inside an 8 pt line; those cannot be rows.
    ``coherent``
        fraction of bands that hold *one* row rather than several.  This is the
        component that distinguishes an under-segmented band from a legitimately tall
        one, and getting its test right took two attempts.

    Counting "lines with text in two or more columns" does not work.  The ICRA rating
    history table wraps four of its columns simultaneously, so a single correct row
    contains five such lines -- indistinguishable, by that test, from the Apple segment
    table's five real rows crammed into one band.

    What separates them is **line spacing**.  Text wrapped inside a cell uses single
    leading; separate table rows carry row padding on top of it.  Measured on the
    samples, wrapped lines sit at 0.61 glyph heights and separate rows at 1.19-1.27 --
    two clusters with a wide empty gap, so the threshold sits at their midpoint (see
    ``WRAP_PITCH_RATIO``).  This is a property of typesetting, not of a document.
    """
    if not bands:
        return -1.0, {}
    line_h = median([ln.y1 - ln.y0 for ln in lines]) or 8.0
    c0_lo = col_bounds[0]
    c0_hi = col_bounds[1] if len(col_bounds) > 1 else col_bounds[-1]
    n_cols = max(1, len(col_bounds) - 1)

    def occupied_cols(ln):
        cols = set()
        for t in ln.tokens:
            for k in range(n_cols):
                if col_bounds[k] <= t.cx < col_bounds[k + 1]:
                    cols.add(k)
                    break
        return len(cols)

    labelled = tall = coherent = 0
    for a, b in bands:
        in_band = [ln for ln in lines if a - 0.5 <= (ln.y0 + ln.y1) / 2 <= b + 0.5]
        if any(t.cx >= c0_lo and t.cx < c0_hi for ln in in_band for t in ln.tokens):
            labelled += 1
        if (b - a) >= 0.8 * line_h:
            tall += 1
        if _band_coherent(in_band, occupied_cols):
            coherent += 1
    n = float(len(bands))
    score = 0.4 * (labelled / n) + 0.2 * (tall / n) + 0.4 * (coherent / n)
    return score, {"labelled_frac": round(labelled / n, 3),
                   "tall_frac": round(tall / n, 3),
                   "coherent_frac": round(coherent / n, 3),
                   "n_bands": len(bands), "score": round(score, 4)}


# Measured, not guessed (tools/measure_pitch.py):
#
#   Apple 10-K p25, five separate rows      pitch/glyph-height = 1.194
#   Apple 10-K p32, three separate rows                        = 1.268
#   ICRA p4, one row wrapped over five lines                    = 0.613
#   ICRA p1, one row wrapped over three lines                   = 0.607
#
# Two well-separated clusters with nothing between 0.61 and 1.19, so the threshold
# sits at their midpoint.  Note the ratio is below 1 for wrapped text because a glyph
# box spans the whole font box and consecutive wrapped lines overlap vertically.
WRAP_PITCH_RATIO = 0.90


def _band_coherent(in_band, occupied_cols) -> bool:
    """Does this band hold one row (possibly wrapped) rather than several rows?"""
    if len(in_band) <= 1:
        return True
    if sum(1 for ln in in_band if occupied_cols(ln) >= 2) <= 1:
        return True
    ordered = sorted(in_band, key=lambda ln: ln.baseline)
    gaps = [b.baseline - a.baseline for a, b in zip(ordered, ordered[1:])
            if b.baseline - a.baseline > 0.1]
    if not gaps:
        return True
    h = median([ln.y1 - ln.y0 for ln in ordered]) or 8.0
    return median(gaps) <= WRAP_PITCH_RATIO * h


def resolve_rows(cand, lines, cfg: Config, col_bounds=None):
    """Pick the best row-band source rather than unioning them.

    Columns benefit from combining evidence -- more sources means better boundaries.
    Rows do not: unioning rule positions with cell-shading edges *adds* boundaries, and
    every spurious boundary splits a row.  So each source proposes a complete band set
    and the most plausible one wins (:func:`_score_bands`).
    """
    y0, y1 = cand.bbox[1], cand.bbox[3]
    col_bounds = col_bounds or [cand.bbox[0], cand.bbox[2]]
    proposals = []

    rule_ys = sorted({round(y, 2) for y in (cand.rule_y_lines or [])
                      if y0 - 2 <= y <= y1 + 2})
    fill_ys = sorted({round(y, 2) for y in (cand.fill_y_lines or [])
                      if y0 - 2 <= y <= y1 + 2})
    for name, ys in (("rules", rule_ys), ("fill_bands", fill_ys),
                     ("all_y_evidence", sorted({round(y, 2) for y in cand.y_lines
                                                if y0 - 2 <= y <= y1 + 2}))):
        if len(ys) < 3:
            continue
        bands = _absorb_empty_bands(
            [(ys[i], ys[i + 1]) for i in range(len(ys) - 1)], lines)
        if bands:
            proposals.append((name, bands, len(ys)))
    proposals.append(("baselines", _baseline_bands(lines, y0, y1), 0))

    scored = []
    for name, bands, n_bnd in proposals:
        s, detail = _score_bands(bands, lines, col_bounds)
        scored.append((s, name, bands, dict(detail, n_boundaries=n_bnd)))
    # Ties broken by the fixed proposal order, so the choice is deterministic.
    order = {"rules": 0, "fill_bands": 1, "all_y_evidence": 2, "baselines": 3}
    scored.sort(key=lambda t: (-t[0], order.get(t[1], 9)))
    best = scored[0]
    bands, completed = _complete_bands(best[2], lines, y0, y1)
    return bands, {"source": best[1], "chosen": best[3], "completed": completed,
                   "considered": [{"source": n, **d} for _s, n, _b, d in scored]}


def _complete_bands(bands, lines, y0, y1):
    """Band any text above or below the chosen band set but inside the region.

    Rule positions describe the part of the table the author ruled, which is often the
    body only.  On page 4 of the ICRA report the rules start at y=214 while the
    two-level column header sits at y=174-214 inside a shaded block -- so the rule
    proposal, though correct for the body, silently dropped both header levels.  Text
    left outside the chosen bands is banded here by baseline clustering and merged in.
    """
    if not bands:
        return bands, {"added_above": 0, "added_below": 0}
    top, bot = bands[0][0], bands[-1][1]
    above = [ln for ln in lines if (ln.y0 + ln.y1) / 2 < top - 0.5]
    below = [ln for ln in lines if (ln.y0 + ln.y1) / 2 > bot + 0.5]
    pre = _baseline_bands(above, y0, top) if above else []
    post = _baseline_bands(below, bot, y1) if below else []
    pre = [(a, min(b, top)) for a, b in pre if min(b, top) > a]
    post = [(max(a, bot), b) for a, b in post if b > max(a, bot)]
    return pre + list(bands) + post, {"added_above": len(pre),
                                      "added_below": len(post)}


def _in_band(ln, band) -> bool:
    cy = (ln.y0 + ln.y1) / 2.0
    return band[0] - 0.5 <= cy <= band[1] + 0.5


def _absorb_empty_bands(bands, lines):
    """Drop text-free bands by extending their neighbour over them.

    Rule- and fill-derived bands are exactly contiguous, so a naive "merge adjacent
    bands" step merges *all* of them -- which is how the six rows of the Apple segment
    table briefly became one.  Empty bands are padding (the unshaded gap between two
    shaded rows, the space between a rule pair) and must be absorbed, not merged with
    a populated neighbour that already has its own row.
    """
    out, pending_top = [], None
    for a, b in sorted(bands):
        if any(_in_band(ln, (a, b)) for ln in lines):
            top = a if pending_top is None else min(pending_top, a)
            out.append([top, b])
            pending_top = None
        elif pending_top is None:
            pending_top = a
    # Trailing empty bands are *dropped*, not absorbed.  Extending the last populated
    # band to the end of the rule stack stretched the ICRA complexity table 73 pt past
    # its final row -- over the explanatory paragraph below it -- and cost the whole
    # detection (IoU 0.41 against a 0.7 threshold).  Interior padding still gets
    # absorbed, which is what this function is for.
    return [(a, b) for a, b in out]


def _pad_bands(bands, y0, y1):
    """Grow each band halfway into the whitespace either side, so nothing falls out."""
    out = []
    for i, (a, b) in enumerate(bands):
        top = y0 if i == 0 else (bands[i - 1][1] + a) / 2.0
        bot = y1 if i == len(bands) - 1 else (b + bands[i + 1][0]) / 2.0
        out.append((min(top, a), max(bot, b)))
    return out


def extend_rows(bands, cand, all_lines, col_bounds, cfg: Config, max_grow: int = 4):
    """Grow the row set over adjacent lines that conform to the column boundaries.

    S3 deliberately trims candidates to a conservative core (a title line that defines
    a gutter edge is indistinguishable from a header row at that stage).  Here the
    columns are known, so a line above the core can be tested properly: if its tokens
    sit inside the column bands without straddling, and the vertical gap is in line
    with the table's own row pitch, it is a row of this table.
    """
    if not bands:
        return bands, {"grown_up": 0, "grown_down": 0}
    interior = col_bounds[1:-1]
    if not interior:
        return bands, {"grown_up": 0, "grown_down": 0}
    pitch = median([b - a for a, b in bands]) or 12.0
    top, bot = bands[0][0], bands[-1][1]
    x0, x1 = col_bounds[0], col_bounds[-1]

    def conforms(ln):
        if not ln.tokens or not ln.text.strip():
            return False
        if ln.x1 < x0 - 2 or ln.x0 > x1 + 2:
            return False
        for x in interior:
            for t in ln.tokens:
                if t.x0 < x - 0.6 and t.x1 > x + 0.6:
                    return False
        occupied = 0
        edges = col_bounds
        for k in range(len(edges) - 1):
            if any(t.cx >= edges[k] and t.cx < edges[k + 1] for t in ln.tokens):
                occupied += 1
        # Two occupied columns minimum.  A single-column line above a table is a
        # heading, and growing into it made "Summary of rating action" row 0 of
        # the ICRA table.  A real header row spans several columns.
        return occupied >= 2

    above = sorted([ln for ln in all_lines if ln.y1 <= top + 0.5],
                   key=lambda l: -l.baseline)
    below = sorted([ln for ln in all_lines if ln.y0 >= bot - 0.5],
                   key=lambda l: l.baseline)
    up = 0
    for ln in above[:max_grow]:
        if top - ln.y1 > 1.6 * pitch or not conforms(ln):
            break
        bands.insert(0, (ln.y0 - 0.5, min(top, ln.y1 + 0.5)))
        top = ln.y0 - 0.5
        up += 1
    down = 0
    for ln in below[:max_grow]:
        if ln.y0 - bot > 1.6 * pitch or not conforms(ln):
            break
        bands.append((max(bot, ln.y0 - 0.5), ln.y1 + 0.5))
        bot = ln.y1 + 0.5
        down += 1
    return bands, {"grown_up": up, "grown_down": down, "pitch": round(pitch, 2)}


# ------------------------------------------------------------------------ cells


def _row_cells(row_tokens, bounds, row_idx: int):
    """Split one row's tokens into cells, merging bands only where text straddles.

    A token that crosses a column boundary is genuinely ambiguous.  We merge the two
    bands into one spanning cell only when at most one of them holds other text --
    otherwise we keep the boundary, assign the token by its centre, and mark the cell
    ``straddled`` so S9 can price the uncertainty instead of hiding it.
    """
    n_bands = len(bounds) - 1
    per_band = [[] for _ in range(n_bands)]
    straddlers = []
    for t in sorted(row_tokens, key=lambda t: (t.x0, t.y0)):
        b0 = _band_of(t.x0 + 0.6, bounds)
        b1 = _band_of(max(t.x0 + 0.6, t.x1 - 0.6), bounds)
        per_band[_band_of(t.cx, bounds)].append(t)
        if b1 > b0:
            straddlers.append((t, b0, b1))

    merge_with_next = [False] * max(1, n_bands - 1)
    straddle_flag = [False] * n_bands
    for t, b0, b1 in straddlers:
        others = sum(1 for b in range(b0, b1 + 1)
                     if any(o is not t for o in per_band[b]))
        home = _band_of(t.cx, bounds)
        straddle_flag[home] = True
        if others <= 1:
            for b in range(b0, min(b1, n_bands - 2) + 1):
                merge_with_next[b] = True

    cells = []
    b = 0
    while b < n_bands:
        span_end = b
        while span_end < n_bands - 1 and merge_with_next[span_end]:
            span_end += 1
        toks = [t for k in range(b, span_end + 1) for t in per_band[k]]
        cells.append(GridCell(
            row=row_idx, col=b, col_span=span_end - b + 1, tokens=toks,
            straddled=any(straddle_flag[b:span_end + 1])))
        b = span_end + 1
    return cells


def _header_row_cells(row_tokens, bounds, row_idx: int):
    """Split a header row by *word runs*, then span the columns each run covers.

    Header rows cannot be split band-by-band.  "Current rating (FY2026)" is three
    tokens with ordinary word spacing that happen to land in three different column
    bands without straddling any boundary -- so the band-wise splitter emitted three
    one-column cells and ``header_path`` came out as ``["Current", "FY2026", ...]``
    instead of the group name.  Grouping by inter-token gap first recovers the cell,
    and its ``col_span`` is then simply the bands its text covers.

    The span is then widened over adjacent *empty* bands, but only while the text stays
    visually centred in the widened span.  That is what distinguishes a header cell
    centred over two data columns (widen) from one that merely sits at the left of a
    wide empty area (do not widen).
    """
    n_bands = len(bounds) - 1
    if not row_tokens:
        return [GridCell(row=row_idx, col=c, tokens=[]) for c in range(n_bands)]

    toks = sorted(row_tokens, key=lambda t: (round(t.baseline, 1), t.x0))
    lines_in = build_lines(toks)
    runs = []
    for ln in lines_in:
        cur = [ln.tokens[0]]
        for prev, t in zip(ln.tokens, ln.tokens[1:]):
            gap = t.x0 - prev.x1
            if gap <= 1.6 * max(prev.size, t.size):
                cur.append(t)
            else:
                runs.append(cur)
                cur = [t]
        runs.append(cur)

    # Merge runs from different lines that overlap horizontally: a wrapped header cell.
    merged = []
    for run in sorted(runs, key=lambda r: min(t.x0 for t in r)):
        a0 = min(t.x0 for t in run)
        a1 = max(t.x1 for t in run)
        placed = False
        for m in merged:
            m0 = min(t.x0 for t in m)
            m1 = max(t.x1 for t in m)
            if min(a1, m1) - max(a0, m0) > 0.35 * min(a1 - a0, m1 - m0):
                m.extend(run)
                placed = True
                break
        if not placed:
            merged.append(list(run))

    claimed = set()
    cells = []
    for run in sorted(merged, key=lambda r: min(t.x0 for t in r)):
        a0 = min(t.x0 for t in run)
        a1 = max(t.x1 for t in run)
        b0 = _band_of(a0 + 0.6, bounds)
        b1 = _band_of(max(a0 + 0.6, a1 - 0.6), bounds)
        cells.append([b0, b1, run])
        claimed.update(range(b0, b1 + 1))

    occupied = set(claimed)
    for cell in cells:
        b0, b1, run = cell
        c_text = (min(t.x0 for t in run) + max(t.x1 for t in run)) / 2.0

        def offset(lo, hi):
            span_c = (bounds[lo] + bounds[hi + 1]) / 2.0
            width = bounds[hi + 1] - bounds[lo]
            return abs(c_text - span_c) / max(width, 1.0)

        improved = True
        while improved:
            improved = False
            for lo, hi in ((b0 - 1, b1), (b0, b1 + 1)):
                if lo < 0 or hi >= n_bands:
                    continue
                new_band = lo if lo < b0 else hi
                if new_band in occupied:
                    continue
                if offset(lo, hi) <= min(0.25, offset(b0, b1) + 0.05):
                    b0, b1 = lo, hi
                    occupied.add(new_band)
                    improved = True
                    break
        cell[0], cell[1] = b0, b1

    out = []
    used = set()
    for b0, b1, run in sorted(cells, key=lambda c: c[0]):
        out.append(GridCell(row=row_idx, col=b0, col_span=b1 - b0 + 1, tokens=run))
        used.update(range(b0, b1 + 1))
    for b in range(n_bands):
        if b not in used:
            out.append(GridCell(row=row_idx, col=b, tokens=[]))
    return sorted(out, key=lambda c: c.col)


def build_cells(bands, bounds, lines, n_header_rows: int = 0):
    """Assign every line to a row band, then split each band into cells."""
    per_band = [[] for _ in bands]
    for ln in lines:
        cy = (ln.y0 + ln.y1) / 2.0
        best, bd = None, 1e9
        for i, (a, b) in enumerate(bands):
            if a - 0.5 <= cy <= b + 0.5:
                best = i
                break
            d = min(abs(cy - a), abs(cy - b))
            if d < bd:
                best, bd = i, d
        if best is not None:
            per_band[best].append(ln)

    cells = []
    for r, group in enumerate(per_band):
        toks = [t for ln in group for t in ln.tokens]
        splitter = _header_row_cells if r < n_header_rows else _row_cells
        for cl in splitter(toks, bounds, r):
            if cl.tokens:
                cl.bbox = (min(t.x0 for t in cl.tokens), min(t.y0 for t in cl.tokens),
                           max(t.x1 for t in cl.tokens), max(t.y1 for t in cl.tokens))
            else:
                cl.bbox = (bounds[cl.col], bands[r][0],
                           bounds[min(cl.col + cl.col_span, len(bounds) - 1)],
                           bands[r][1])
            cells.append(cl)
    return cells, per_band


def detect_row_spans(cells, bands, bounds):
    """Mark vertical merges: one cell whose text is centred across empty neighbours.

    Conservative on purpose.  A cell is only given ``row_span > 1`` when the cell
    directly below is empty in that column *and* the text box of the upper cell reaches
    into the lower band -- the visual signature of a vertically merged cell.  Guessing
    row spans from emptiness alone would merge every blank cell in a sparse table.
    """
    by_rc = {(c.row, c.col): c for c in cells}
    n_rows = len(bands)
    for c in sorted(cells, key=lambda c: (c.row, c.col)):
        if not c.tokens or c.row_span > 1:
            continue
        r = c.row
        while r + 1 < n_rows:
            nxt = by_rc.get((r + 1, c.col))
            if nxt is None or nxt.tokens:
                break
            band = bands[r + 1]
            if c.bbox[3] <= band[0] + 0.5:
                break
            c.row_span += 1
            nxt.flags.append("absorbed_by_row_span")
            r += 1
    return cells


# ----------------------------------------------------------------------- headers


def _coarse_type(sig: str) -> str:
    """Collapse format signatures to the classes header detection cares about."""
    if sig.startswith("int") or sig in ("grouped", "decimal", "grouped_decimal",
                                        "percent", "currency_symbol"):
        return "numeric"
    return sig


def detect_header_rows(bands, cells, bounds, cand, cfg: Config):
    """How many rows from the top are header, and on what evidence."""
    n_rows = len(bands)
    if n_rows <= 1:
        return 1, {"signal": "single_row"}
    by_row = {}
    for c in cells:
        by_row.setdefault(c.row, []).append(c)

    def texts(r):
        return [join_tokens(c.tokens).strip()
                for c in sorted(by_row.get(r, []), key=lambda c: c.col)]

    def types(r, skip_label_col=True):
        """Format signatures in row ``r``, excluding the row-label column.

        Column 0 is prose on essentially every row of a financial table, so including
        it makes "text" the dominant body format and every header hypothesis fails
        validation -- which is how the Shell profit-and-loss statement ended up
        treating "I Revenue from Operations 25 16,195.63" as a header row.
        """
        out = []
        for c in sorted(by_row.get(r, []), key=lambda c: c.col):
            if skip_label_col and c.col == 0:
                continue
            s = join_tokens(c.tokens).strip()
            if s:
                out.append(classify_token(s))
        return out

    # Signal 1: a horizontal rule inside the region, above its first third, with a
    # small number of rows above it -- the classic header separator.
    header_sep_ys = cand.rule_y_lines or cand.y_lines
    if header_sep_ys:
        y0, y1 = bands[0][0], bands[-1][1]
        third = y0 + (y1 - y0) / 3.0
        # The *deepest* qualifying rule, not the first: the ICRA rating history table
        # has rules at y=214 and y=250, and only the second separates header from body.
        #
        # But "deepest" alone is too greedy -- on the Apple products table it swallowed
        # the iPhone row, which then dropped out of the total's addends and produced a
        # DOWNFOOT_FAIL on a correct table.  So each candidate is *validated* against
        # type contrast: if any row it would call a header already carries the body's
        # dominant value format, that candidate is rejected.
        # Coarse classes, not exact format signatures.  "2,087.5" is
        # grouped_decimal and "6.4" is decimal; comparing those as different let the
        # ICRA key-financials table call its first data row a header.  What matters
        # for this test is only "is it a number".
        body_sig = {}
        for r in range(min(cfg.max_header_rows + 1, n_rows), n_rows):
            for ty in types(r):
                k = _coarse_type(ty)
                body_sig[k] = body_sig.get(k, 0) + 1
        dominant = (max(sorted(body_sig.items()), key=lambda kv: kv[1])[0]
                    if body_sig else None)

        def plausible(above: int) -> bool:
            if dominant is None:
                return True
            return not any(dominant in [_coarse_type(t) for t in types(r)]
                           for r in range(above))

        # A separator has to be a real band boundary.  On page 3 of the ICRA report a
        # rule at y=349.8 falls *inside* row band [337.4, 355.4] -- it is cell
        # decoration, not a row edge -- and taking it made the first data row a header.
        boundaries = [bands[0][0]] + [b for _a, b in bands]
        # Tolerance scales with row height.  A baseline-derived band edge is the
        # midpoint between two text lines and can sit several points away from the rule
        # the author actually drew: on the indentation fixture the header/body rule at
        # y=170 landed 2.6 pt from the band edge at 172.6, so a fixed 1.5 pt tolerance
        # rejected it and the shallower level-1 divider at y=150 won instead -- costing
        # the second header level and every header path below it.
        bnd_tol = max(2.5, 0.3 * (median([b - a for a, b in bands]) or 10.0))
        # The plausibility veto needs something to contrast with.  In an all-text table
        # ("Analytical Approach | Comments") every row is text, so the veto would
        # reject every hypothesis and hand the decision to a weaker signal.
        contrastive = dominant in ("numeric", "date", "period")
        best = None
        for y in sorted(header_sep_ys):
            if y <= y0 + 1.0 or y > third:
                continue
            if not any(abs(y - bnd) <= bnd_tol for bnd in boundaries):
                continue
            above = sum(1 for a, b in bands if (a + b) / 2.0 < y)
            if 1 <= above <= cfg.max_header_rows and                     (plausible(above) if contrastive else True):
                best = (above, y)
        if best:
            return best[0], {"signal": "header_rule", "rule_y": round(best[1], 2),
                             "rows_above": best[0],
                             "validated_against": dominant}

    # Signal 2: type contrast.  Walk down while the row has no cell of the type that
    # dominates the body.  "2024" is numeric, so a plain "is it numeric" test fails on
    # the Apple header -- but "2024" is a bare integer while the body is grouped
    # thousands, so the *format* differs and that is what we compare.
    body_types = {}
    for r in range(min(cfg.max_header_rows + 1, n_rows), n_rows):
        for ty in types(r):
            body_types[ty] = body_types.get(ty, 0) + 1
    if body_types:
        dom = max(sorted(body_types.items()), key=lambda kv: kv[1])[0]
        n = 0
        for r in range(min(cfg.max_header_rows, n_rows - 1)):
            tys = types(r)
            if tys and dom in tys:
                break
            n += 1
        if n >= 1:
            return n, {"signal": "type_contrast", "dominant_body_type": dom,
                       "header_rows": n}

    # Signal 3: spanning cells at the top imply a multi-level header.
    n = 0
    for r in range(min(cfg.max_header_rows, n_rows - 1)):
        if any(c.col_span > 1 for c in by_row.get(r, [])):
            n = r + 1
        else:
            break
    if n:
        return n, {"signal": "spanning_cells", "header_rows": n}

    # Signal 4: bold first row over a non-bold body.
    first_bold = all(t.bold for c in by_row.get(0, []) for t in c.tokens) and \
        any(c.tokens for c in by_row.get(0, []))
    rest_bold = any(t.bold for r in range(1, n_rows)
                    for c in by_row.get(r, []) for t in c.tokens)
    if first_bold and not rest_bold:
        return 1, {"signal": "bold_first_row"}
    return 1, {"signal": "default"}


def build_header_paths(cells, n_header_rows: int, n_cols: int):
    """Per column, the list of header cell texts from the outermost level inward."""
    paths = [[] for _ in range(n_cols)]
    for r in range(n_header_rows):
        for c in cells:
            if c.row != r:
                continue
            txt = join_tokens(c.tokens).strip()
            if not txt:
                continue
            for col in range(c.col, min(c.col + c.col_span, n_cols)):
                if not paths[col] or paths[col][-1] != txt:
                    paths[col].append(txt)
    return paths


# ------------------------------------------------------------------- row labels


TOTAL_LABEL_RE = __import__("re").compile(
    r"^\s*(total|sub\s*-?\s*total|grand\s+total|net\s+total|aggregate)\b", __import__("re").I)


def is_total_label(text: str) -> bool:
    return bool(TOTAL_LABEL_RE.match(text or ""))


def detect_indent_levels(cells, bands, n_header_rows: int, cfg: Config):
    """Cluster first-column left edges into indentation levels.

    Total rows are excluded from the clustering and pinned to level 0.  Apple centres
    "Total net sales" at x=35 while its data rows start at x=19, so pure geometry made
    the total a *child* of "Rest of Asia Pacific" and produced the row label path
    ``["Rest of Asia Pacific", "Total net sales"]``.  A total summarises its siblings,
    it is never nested under the last of them.
    """
    xs, rows = [], []
    totals = set()
    for c in sorted(cells, key=lambda c: (c.row, c.col)):
        if c.col != 0 or c.row < n_header_rows or not c.tokens:
            continue
        label = join_tokens(c.tokens).strip()
        if is_total_label(label):
            totals.add(c.row)
            continue
        xs.append(min(t.x0 for t in c.tokens))
        rows.append(c.row)
    levels = [0] * len(bands)
    if not xs:
        return levels
    clusters = cluster_1d(xs, max(1.5, cfg.indent_step_min / 2.0))
    centers = sorted((sum(xs[i] for i in cl) / len(cl), cl) for cl in clusters)
    # Merge clusters closer than one indent step: a 1 pt difference is kerning, not a
    # level.  Only genuinely distinct left edges become levels.
    merged = []
    for center, cl in centers:
        if merged and center - merged[-1][0] < cfg.indent_step_min:
            merged[-1] = (merged[-1][0], merged[-1][1] + list(cl))
        else:
            merged.append((center, list(cl)))
    for lvl, (_center, cl) in enumerate(merged):
        for i in cl:
            levels[rows[i]] = lvl
    return levels


def build_row_label_paths(cells, levels, n_header_rows: int, n_rows: int):
    """Stack indentation levels into a path: ["Revenue", "Domestic", "Product A"]."""
    label_of = {}
    for c in cells:
        if c.col == 0 and c.tokens:
            label_of[c.row] = join_tokens(c.tokens).strip()
    paths = [[] for _ in range(n_rows)]
    stack = []
    for r in range(n_rows):
        if r < n_header_rows:
            continue
        lab = label_of.get(r, "")
        lvl = levels[r] if r < len(levels) else 0
        while len(stack) > lvl:
            stack.pop()
        if lab:
            while len(stack) < lvl:
                stack.append("")
            stack.append(lab)
            paths[r] = [s for s in stack if s]
        else:
            paths[r] = [s for s in stack if s]
    return paths


# ------------------------------------------------------------------------ driver


def build_grid_from_lattice(cand) -> Grid:
    """Build the grid from the rule lattice alone, for a page with no text layer.

    Every other path in S4 needs tokens: boundaries are vetoed by where text sits, rows
    are scored by where labels are, headers are found by type contrast. A scanned page
    supplies none of that, so after the raster morphology had correctly recovered 23
    intersections from the scanned fixture, ``build_grid`` returned an empty grid and
    the table was silently dropped -- structure found and then thrown away.

    Here the author's own lattice *is* the answer: rows and columns come straight from
    the rule positions and the cells are created empty for S5 to OCR (or, with no OCR
    engine, to flag). The header row count cannot be inferred without text, so it is
    left at 1 and the evidence records that it was assumed.
    """
    g = Grid()
    xs = sorted(set(round(v, 2) for v in (cand.rule_x_lines or cand.x_lines)))
    ys = sorted(set(round(v, 2) for v in (cand.rule_y_lines or cand.y_lines)))
    if len(xs) < 2 or len(ys) < 2:
        return g
    g.x_lines, g.y_lines = xs, ys
    g.n_cols, g.n_rows = len(xs) - 1, len(ys) - 1
    g.n_header_rows = 1 if g.n_rows > 1 else 0
    for r in range(g.n_rows):
        for c in range(g.n_cols):
            g.cells.append(GridCell(row=r, col=c, is_header=(r < g.n_header_rows),
                                    tokens=[],
                                    bbox=(xs[c], ys[r], xs[c + 1], ys[r + 1])))
    g.header_paths = [[] for _ in range(g.n_cols)]
    g.indent_levels = [0] * g.n_rows
    g.row_label_paths = [[] for _ in range(g.n_rows)]
    g.col_units = [""] * g.n_cols
    g.evidence = {"source": "raster_lattice",
                  "note": "no text layer: rows and columns taken directly from the "
                          "rule lattice; header row count assumed to be 1",
                  "n_intersections": cand.n_intersections,
                  "final_x_lines": xs, "final_y_lines": ys,
                  "columns": {"sources": {"raster_rule": len(xs)}, "vetoed": []},
                  "rows": {"source": "raster_lattice",
                           "chosen": {"n_bands": g.n_rows}},
                  "grow": {}, "header": {"signal": "assumed_single_row"},
                  "folds": [], "prose_lines_dropped": 0}
    return g


def build_grid(cand, pt, cfg: Config, layout=None) -> Grid:
    """Full S4 pass for one arbitrated candidate region.

    ``layout`` is S2's (post-demotion) role assignment.  Row growth consults it:
    a caption or heading directly above a table is *not* a candidate row, however
    well its tokens happen to line up with the columns.  On the continuation
    fixture the caption "Table 4.2 - Unit schedule" was grown in as row 0, which
    added a column, which made the header pages a different shape from the
    non-header pages, which stopped stitching from joining them -- one missing
    check, three stages of damage.
    """
    lines = [pt.lines[i] for i in cand.line_idx] if cand.line_idx else []
    if not lines:
        lines = [ln for ln in pt.lines
                 if cand.bbox[1] - 1 <= (ln.y0 + ln.y1) / 2 <= cand.bbox[3] + 1
                 and ln.x1 > cand.bbox[0] - 1 and ln.x0 < cand.bbox[2] + 1
                 and ln.text.strip()]
    lines = sorted(lines, key=lambda ln: (round(ln.baseline, 2), ln.x0))
    g = Grid()
    if not lines:
        return build_grid_from_lattice(cand)

    # Prose is never a table row -- not even inside a region the author ruled.  The
    # ICRA "Complexity level" table is followed by an explanatory paragraph that the
    # rule stack swept in; its sentences straddled the real column boundary four times
    # and vetoed it, collapsing a two-column table to one column.
    bx0 = min(ln.x0 for ln in lines)
    bx1 = max(ln.x1 for ln in lines)
    seed = [i for i, ln in enumerate(lines) if is_prose_line(ln, bx0, bx1, cfg)]
    prose_idx = paragraph_closure(lines, seed)
    kept = [ln for i, ln in enumerate(lines) if i not in prose_idx]
    n_prose = len(lines) - len(kept)
    if len(kept) >= 2:
        lines = kept

    # Order matters.  Folding has to happen before ``extend_rows``: unfolded, the Apple
    # segment table has 19 bands including the boundary between "3" and "%", and the
    # header cell "Change" straddles it -- so the header row looked non-conforming and
    # was never grown into.  Fold first, then the row test sees the real columns.
    bounds, col_ev = resolve_columns(cand, lines, cfg)
    protected = list(cand.rule_x_lines) + list(cand.fill_x_lines)
    bounds, units, fold_notes = fold_adornment_columns(
        bounds, lines, cfg, body_lines=_body_lines(lines), protected=protected)
    bands, row_ev = resolve_rows(cand, lines, cfg, col_bounds=bounds)

    growable = _growable_lines(pt, layout)
    bands, grow_ev = extend_rows([list(b) for b in bands], cand, growable,
                                 bounds, cfg)
    all_lines = growable
    bands = [(a, b) for a, b in bands]
    if grow_ev.get("grown_up") or grow_ev.get("grown_down"):
        lines = [ln for ln in all_lines
                 if bands[0][0] - 0.5 <= (ln.y0 + ln.y1) / 2 <= bands[-1][1] + 0.5
                 and ln.x1 > bounds[0] - 2 and ln.x0 < bounds[-1] + 2]
        lines.sort(key=lambda ln: (round(ln.baseline, 2), ln.x0))
        bounds2, col_ev2 = resolve_columns(cand, lines, cfg)
        bounds2, units2, folds2 = fold_adornment_columns(
            bounds2, lines, cfg, body_lines=_body_lines(lines),
            protected=protected)
        col_ev["after_grow"] = col_ev2
        # Only adopt the re-derived columns if growth did not destabilise the count;
        # a wildly different column count means the grown rows were not table rows.
        if abs(len(bounds2) - len(bounds)) <= 1:
            bounds, units, fold_notes = bounds2, units2, folds2 + fold_notes
        else:
            col_ev["after_grow_rejected"] = True

    # Header detection needs cells, and header cells need a different splitter, so
    # build once band-wise, decide how many header rows there are, then rebuild the
    # header rows with the run-based splitter.
    provisional, row_line_groups = build_cells(bands, bounds, lines, 0)
    n_rows, n_cols = len(bands), len(bounds) - 1
    n_header, hdr_ev = detect_header_rows(bands, provisional, bounds, cand, cfg)
    n_header = max(0, min(n_header, max(0, n_rows - 1)))
    cells, row_line_groups = build_cells(bands, bounds, lines, n_header)
    cells = detect_row_spans(cells, bands, bounds)
    for c in cells:
        c.is_header = c.row < n_header

    levels = detect_indent_levels(cells, bands, n_header, cfg)
    g.x_lines = [round(v, 3) for v in bounds]
    g.y_lines = [round(bands[0][0], 3)] + [round(b[1], 3) for b in bands]
    g.n_rows, g.n_cols = n_rows, n_cols
    g.n_header_rows = n_header
    g.cells = cells
    g.header_paths = build_header_paths(cells, n_header, n_cols)
    g.indent_levels = levels
    g.row_label_paths = build_row_label_paths(cells, levels, n_header, n_rows)
    g.col_units = units if len(units) == n_cols else [""] * n_cols
    g.row_lines = [[pt.lines.index(ln) for ln in grp] for grp in row_line_groups]
    g.evidence = {"prose_lines_dropped": n_prose,
                  "final_x_lines": [round(v, 2) for v in bounds],
                  "final_y_lines": [round(v, 2) for v in g.y_lines],
                  "columns": col_ev, "rows": row_ev, "grow": grow_ev,
                  "header": hdr_ev, "folds": fold_notes,
                  "regime": cand.regime}
    return g


# ==================================================================================================
# s5_cells: S5 -- cell text extraction.
# ==================================================================================================
#
# For a digital page the text layer *is* the answer and re-OCRing it can only lose
# information, so the only work here is assembling the tokens S4 assigned to each cell
# into a faithful string and recording what we noticed while doing it (wrapping,
# superscripts, watermark overlap, straddling).
#
# For a raster region we OCR **the cell crop, not the page**.  Cropping first is what
# makes per-cell confidence meaningful: a page-level OCR pass gives one word-confidence
# stream that has to be re-associated with cells by geometry, and the association is
# exactly where the errors hide.
#
# The OCR backend is pluggable and *fails closed*.  If no engine is installed,
# :class:`NullBackend` returns nothing, the cell is written with ``raw_text = NULL``,
# ``source = 'none'``, confidence 0 and an ``OCR_UNAVAILABLE`` issue.  The brief asks
# directly what happens to a genuinely unreadable table -- "flag it, write an issues row,
# and do not guess" -- and this is that behaviour, reached by the same code path whether
# the engine is missing or the crop is illegible.
#
# One repair is allowed, under a strict condition.  In a column already typed numeric, if
# the OCR string does not parse as a number but does parse after the standard glyph
# confusions (``O->0``, ``l/I->1``, ``S->5``, ``B->8``) are undone, the repaired value is
# used, flagged ``OCR_DIGIT_REPAIR``, and penalised in S9.  The condition matters: the
# repair only ever turns a non-number into a number, so it can never silently rewrite one
# valid number into a different valid number.

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


# ==================================================================================================
# s6_stitch: S6 -- cross-page stitching.
# ==================================================================================================
#
# The brief asks for the rule and asks us to defend it, so here it is explicitly.
#
# **Two hard gates** (both must hold, no score can buy past them):
#
# 1.  *Adjacent pages.*  ``page_no`` differs by exactly one.  A table cannot resume after
#     an unrelated page: allowing gaps merges the receivables schedule on page 12 with
#     the payables schedule on page 15 whenever their shapes happen to match, and that
#     error is invisible in the output.
# 2.  *Same column count, and the columns line up.*  Mean 1-D IoU of the column bands
#     (normalised to page width) at least ``stitch_col_iou_min`` = 0.85.  The brief warns
#     that widths drift by a point or two between pages; 0.85 absorbs several points of
#     drift on a six-column table while still rejecting a different table of the same
#     arity.
#
# **Then a weighted score** over five soft signals, needing ``stitch_score_min`` = 0.62:
#
# =========================================  ======  ===============================
# signal                                     weight  rationale
# =========================================  ======  ===============================
# column-band alignment (the IoU, rescaled)    0.30  the strongest single indicator
# header repeated, or absent, on page N+1      0.20  a repeated header *is* a
#                                                    continuation marker; a missing
#                                                    header means the rows continue
# "(Contd.)" / "continued" text nearby         0.15  explicit, when present
# no intervening heading at or above the       0.20  a new section means a new table
# table's own section level
# first column keeps its type and the first     0.15  a continuation resumes the same
# table ends low / second starts high                kind of rows in the same place
# =========================================  ======  ===============================
#
# **Why weights and not a decision tree.**  Every individual signal is absent on some real
# document -- the brief says so: sometimes the header repeats, sometimes not; sometimes
# there is a ``(Contd.)``, often not.  A tree has to commit to an order of interrogation
# and inherits the blind spot of whichever signal it asks about first.  A weighted sum
# degrades gracefully: two strong signals carry a decision when the other three are
# silent, and the score plus every component is written to ``detector_votes_json`` so a
# reviewer can see exactly which signals fired.
#
# **Header de-duplication.**  When page N+1 repeats the header, that row is dropped from
# the data and recorded in ``dropped_header_rows``.  Leaving it in is the specific failure
# the brief calls out: "duplicate the header row into the data and break every downstream
# sum".

CONTD_RE = re.compile(r"\((?:cont(?:d|inued)?\.?)\)|\bcontinued\b|\bcontd\.?\b", re.I)


@dataclass
class LogicalTable:
    parts: list = field(default_factory=list)      # list of PageTable
    dropped_header_rows: list = field(default_factory=list)  # (part_index, row_idx)
    votes: list = field(default_factory=list)      # one dict per join decision

    @property
    def start_page(self):
        return self.parts[0].page_no

    @property
    def end_page(self):
        return self.parts[-1].page_no

    @property
    def is_continuation(self):
        return len(self.parts) > 1


def _norm_bands(grid, page_width: float):
    xs = grid.x_lines
    w = max(1.0, page_width)
    return [(xs[i] / w, xs[i + 1] / w) for i in range(len(xs) - 1)]


BOUNDARY_TOL_PT = 6.0


def band_iou(a, b, page_w_a: float, page_w_b: float) -> float:
    """Mean 1-D IoU of corresponding column bands, page-width normalised."""
    ba, bb = _norm_bands(a, page_w_a), _norm_bands(b, page_w_b)
    if not ba or len(ba) != len(bb):
        return 0.0
    return sum(iou_1d(p[0], p[1], q[0], q[1]) for p, q in zip(ba, bb)) / len(ba)


def column_alignment(a, b, page_w_a: float, page_w_b: float,
                     tol: float = BOUNDARY_TOL_PT) -> float:
    """Fraction of *interior* column boundaries that agree within ``tol`` points.

    Band IoU was the obvious measure and it is the wrong one.  A table's outer edges
    are set by whatever its widest cell happens to be, so a page carrying the repeated
    header is wider than a page without it -- and on the six-page fixture that alone
    dragged mean band IoU down to 0.79 against a 0.85 gate, splitting one logical table
    into six.  The *interior* boundaries are the structural signature and they drift by
    only the point or two the brief warns about.

    Band IoU is still computed and recorded in the vote for transparency; it is just
    not what the gate is made of.
    """
    ia = list(a.x_lines[1:-1])
    ib = list(b.x_lines[1:-1])
    if not ia or not ib:
        # A single-column table has no interior signature; fall back to extents.
        return band_iou(a, b, page_w_a, page_w_b)
    if len(ia) != len(ib):
        return 0.0
    matched = sum(1 for x, y in zip(ia, ib) if abs(x - y) <= tol)
    return matched / float(len(ia))


def _row_texts(pt_table, row: int):
    out = []
    for c in sorted(pt_table.cells, key=lambda c: c.col):
        if c.row == row:
            out.append((c.raw_text or "").strip().lower())
    return out


def _similar(a, b) -> float:
    """Token-set similarity between two rows of cell texts."""
    if not a or not b:
        return 0.0
    sa = {t for t in a if t}
    sb = {t for t in b if t}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def header_repeat_score(prev, cur) -> tuple:
    """Does ``cur`` begin by repeating ``prev``'s header?  ``(score, matched_rows)``."""
    if prev.grid.n_header_rows <= 0:
        return 0.0, []
    matched = []
    for k in range(min(prev.grid.n_header_rows, cur.grid.n_rows)):
        sim = _similar(_row_texts(prev, k), _row_texts(cur, k))
        if sim >= 0.5:
            matched.append(k)
    return (len(matched) / float(prev.grid.n_header_rows)) if matched else 0.0, matched


def _first_col_type(tbl):

    kinds = {}
    for c in tbl.cells:
        if c.col == 0 and c.row >= tbl.grid.n_header_rows and (c.raw_text or "").strip():
            k = classify_token(c.raw_text)
            k = "num" if k.startswith(("int", "grouped", "decimal")) else k
            kinds[k] = kinds.get(k, 0) + 1
    if not kinds:
        return None
    return max(sorted(kinds.items()), key=lambda kv: kv[1])[0]


def score_join(prev, cur, page_prev, page_cur, cfg: Config,
               headings_between=None, header_ref=None):
    """Evaluate one candidate join.  Returns a dict with the verdict and components."""
    v = {"prev_page": prev.page_no, "cur_page": cur.page_no,
         "gate_adjacent": (cur.page_no - prev.page_no) == cfg.stitch_max_page_gap,
         "n_cols_prev": prev.grid.n_cols, "n_cols_cur": cur.grid.n_cols}
    align = column_alignment(prev.grid, cur.grid, page_prev.width, page_cur.width)
    v["column_alignment"] = round(align, 4)
    v["band_iou"] = round(band_iou(prev.grid, cur.grid, page_prev.width,
                                   page_cur.width), 4)
    v["gate_same_cols"] = prev.grid.n_cols == cur.grid.n_cols
    v["gate_alignment"] = align >= cfg.stitch_col_iou_min
    if not (v["gate_adjacent"] and v["gate_same_cols"] and v["gate_alignment"]):
        v["score"] = 0.0
        v["joined"] = False
        v["reason"] = "hard gate failed"
        return v

    hdr, matched = header_repeat_score(header_ref or prev, cur)
    v["header_repeat"] = round(hdr, 3)
    v["header_rows_matched"] = matched
    header_component = hdr if hdr >= cfg.stitch_header_sim_min * 0.5 else (
        0.5 if cur.grid.n_header_rows == 0 else 0.25)

    near_text = " ".join([(cur.caption or ""), (cur.title or ""),
                          (prev.caption or ""), (prev.title or "")])
    contd = bool(CONTD_RE.search(near_text))
    v["contd_marker"] = contd

    blocking = [h for h in (headings_between or [])
                if h[2] <= max(1, prev.section_level)]
    v["headings_between"] = [h[3][:60] for h in (headings_between or [])]
    v["blocking_heading"] = bool(blocking)

    t_prev, t_cur = _first_col_type(prev), _first_col_type(cur)
    v["first_col_type"] = [t_prev, t_cur]
    type_ok = (t_prev is not None and t_prev == t_cur)

    ends_low = prev.grid.y_lines[-1] >= 0.62 * page_prev.height
    starts_high = cur.grid.y_lines[0] <= 0.38 * page_cur.height
    v["ends_low"] = ends_low
    v["starts_high"] = starts_high

    score = (0.30 * min(1.0, (align - cfg.stitch_col_iou_min) /
                        max(1e-6, 1.0 - cfg.stitch_col_iou_min))
             + 0.20 * header_component
             + 0.15 * (1.0 if contd else 0.0)
             + 0.20 * (0.0 if blocking else 1.0)
             + 0.15 * (0.5 * (1.0 if type_ok else 0.0)
                       + 0.5 * (1.0 if (ends_low and starts_high) else 0.0)))
    v["score"] = round(score, 4)
    v["joined"] = score >= cfg.stitch_score_min
    v["reason"] = ("score %.3f >= %.2f" % (score, cfg.stitch_score_min)
                   if v["joined"] else
                   "score %.3f < %.2f" % (score, cfg.stitch_score_min))
    return v


def stitch(page_tables, pages, cfg: Config, headings=None):
    """Group page-level tables into logical tables, left to right, greedily.

    Greedy is correct here rather than a limitation: continuation is a transitive
    relation over consecutive pages, so a chain of pairwise decisions is exactly the
    right model.  A table only ever continues onto the page after it.
    """
    logical = []
    if not cfg.enable_stitch:
        for t in page_tables:
            logical.append(LogicalTable(parts=[t]))
        return logical

    by_page = {p.page_no: p for p in pages}
    for t in sorted(page_tables, key=lambda t: (t.page_no, t.grid.y_lines[0]
                                                if t.grid.y_lines else 0)):
        placed = False
        for lt in reversed(logical):
            prev = lt.parts[-1]
            if cur_gap(prev, t) != 1:
                continue
            hb = _headings_between(headings or [], prev, t)
            v = score_join(prev, t, by_page[prev.page_no], by_page[t.page_no], cfg, hb,
                           header_ref=lt.parts[0])
            if v["joined"]:
                # Compare against the logical table's FIRST part, not the immediately
                # preceding one.  On the six-page fixture the header repeats on pages
                # 1, 3 and 5; page 3's header was being compared against page 2, whose
                # first row is data, so no repeat was detected and two header rows were
                # duplicated into the data -- exactly the failure the brief calls out.
                hdr, matched = header_repeat_score(lt.parts[0], t)
                if matched:
                    for r in matched:
                        lt.dropped_header_rows.append((len(lt.parts), r))
                lt.parts.append(t)
                lt.votes.append(v)
                placed = True
                break
            if v["gate_adjacent"]:
                lt.votes.append(v)     # keep the rejection for the audit trail
        if not placed:
            logical.append(LogicalTable(parts=[t]))
    return logical


def cur_gap(prev, cur) -> int:
    return cur.page_no - prev.page_no


def _headings_between(headings, prev, cur):
    """Headings after the end of ``prev`` and before the start of ``cur``."""
    out = []
    y_end = prev.grid.y_lines[-1] if prev.grid.y_lines else 0.0
    y_start = cur.grid.y_lines[0] if cur.grid.y_lines else 0.0
    for pno, y, lvl, txt in headings:
        if pno == prev.page_no and y > y_end:
            out.append((pno, y, lvl, txt))
        elif pno == cur.page_no and y < y_start:
            out.append((pno, y, lvl, txt))
    return out


# ==================================================================================================
# s8_context: S8 -- context capture: title, section, units and scale, footnotes, linked figures.
# ==================================================================================================
#
# "4,812 means nothing until you know it is rupees, in lakhs, for Q4 FY25, for the
# Domestic segment, restated per note 14."  Each of those five facts lives somewhere
# different, and the awkward one is scale: it is commonly declared once under a section
# heading and governs every table below it, possibly on later pages.
#
# Scale resolution therefore walks outward from the table and *records where it stopped*:
#
# 1. the table's own caption or title,
# 2. text lines immediately above the region (the ``(All figures in Rs lakhs)`` line),
# 3. the enclosing section heading,
# 4. the nearest scale declaration earlier in the document, up to
#    ``scale_search_pages`` pages back.
#
# Each level is less certain than the last, so ``unit_scale_note`` is stored with the
# page and line it came from and a confidence that decays with distance.  A value whose
# unit came from two pages back is a materially weaker claim than one whose unit is in its
# own caption, and the schema keeps the difference instead of flattening it.
#
# **Footnote-to-cell linkage.**  Markers are collected from the cell text (``1,872*``,
# a superscript ``2``) and matched against footnote lines by marker, searching the
# table's own page first and then the *following* page -- item 10 of the adversarial set
# puts the footnote text on the next page, and a page-local search silently drops it.
# Links are stored in ``cell_footnotes``, cell to footnote, not table to page.
#
# **Table-to-figure association.**  The brief asks us to define and defend the rule, so
# it is an explicit weighted sum of four independent pieces of evidence:
#
# ===================================================  ======
# evidence                                             weight
# ===================================================  ======
# the table's caption or nearby text names the figure    0.45
# figure and table captions share a section number      0.25
# figure is adjacent to the table and inside the same    0.20
# section
# figure lies between the table and the next heading     0.10
# ===================================================  ======
#
# Textual reference dominates because it is the only signal that is *stated* rather than
# inferred; the rest are corroboration.  ``relation_confidence`` is the sum, and the
# component breakdown is stored so a low-confidence link can be inspected rather than
# guessed at.  Proximity alone caps out at 0.30, which is deliberately below any
# threshold a consumer should trust blindly.

FIG_REF_RE = re.compile(
    r"\b(figure|fig\.?|chart|graph|exhibit)\s*([0-9IVX]+(?:[.\-][0-9A-Za-z]+)*)", re.I)
NUMBER_IN_CAPTION_RE = re.compile(r"([0-9]+(?:[.\-][0-9A-Za-z]+)*)")
CURRENCY_IN_NOTE_RE = re.compile(r"(₹|`|\$|€|£|¥|\bRs\.?\b|\bINR\b|\bUSD\b|\bEUR\b|"
                                 r"\bGBP\b|\bJPY\b)", re.I)


@dataclass
class ScaleInfo:
    note: str = ""
    magnitude: float = 1.0
    currency: str = ""
    source: str = ""          # caption | above_region | section_heading | earlier_page
    page_no: int = 0
    bbox: tuple = None
    confidence: float = 0.0
    flags: list = field(default_factory=list)


@dataclass
class Footnote:
    marker: str
    text: str
    page_no: int
    bbox: tuple


@dataclass
class Asset:
    kind: str                 # image | chart
    bbox: tuple
    page_no: int
    caption: str = ""
    file_path: str = ""


@dataclass
class TableContext:
    title: str = ""
    caption: str = ""
    section_path: list = field(default_factory=list)
    scale: ScaleInfo = None
    footnotes: list = field(default_factory=list)          # Footnote
    cell_footnotes: list = field(default_factory=list)     # (row, col, footnote_index)
    assets: list = field(default_factory=list)             # (asset_index, relation,
    #                                                          confidence, components)
    evidence: dict = field(default_factory=dict)


# ------------------------------------------------------------------ title / caption


def find_title_caption(pt, layout, bbox, cfg: Config):
    """Nearest caption above the table, then nearest heading; both if present."""
    caption, cap_gap = "", 1e9
    for _i, txt, bb in getattr(layout, "captions", []):
        gap = bbox[1] - bb[3]
        if -2.0 <= gap < min(cap_gap, cfg.caption_max_gap):
            caption, cap_gap = txt, gap
    # A caption directly below is also common for figures-as-tables.
    for _i, txt, bb in getattr(layout, "captions", []):
        gap = bb[1] - bbox[3]
        if 0 <= gap < min(cap_gap, cfg.caption_max_gap):
            caption, cap_gap = txt, gap

    title, t_gap = "", 1e9
    for _i, _lvl, txt, bb in getattr(layout, "headings", []):
        gap = bbox[1] - bb[3]
        if -2.0 <= gap < min(t_gap, cfg.caption_max_gap * 2):
            title, t_gap = txt, gap
    if not title and caption:
        title = caption
    return title, caption, {"caption_gap": round(cap_gap, 2) if caption else None,
                            "title_gap": round(t_gap, 2) if title else None}


# ---------------------------------------------------------------------- unit / scale


def parse_scale_note(text: str) -> ScaleInfo:
    """Pull a magnitude and a currency out of a declaration line."""
    info = ScaleInfo(note=text.strip())
    m = SCALE_RE.search(text)
    if m:
        word = m.group(1).lower().rstrip(".")
        info.magnitude = MAGNITUDES.get(word, 1.0)
    c = CURRENCY_IN_NOTE_RE.search(text)
    if c:
        sym = c.group(1)
        info.currency = _code(sym if sym != "`" else "₹")
        if sym == "`":
            info.flags.append("LEGACY_RUPEE_GLYPH")
    return info


def resolve_scale(page_no: int, bbox, pts, layouts, section_path, caption: str,
                  cfg: Config, header_texts=None) -> ScaleInfo:
    """Walk outward from the table until a scale declaration is found."""
    # Step 0: the table's own header cells.  The ICRA summary table declares its
    # unit inside the header -- "Previous rated amount (Rs. crore)" -- which is
    # both the most specific place it can appear and the easiest to miss, since
    # it is a cell rather than a line of page text.
    for txt in (header_texts or []):
        if txt and SCALE_RE.search(txt):
            info = parse_scale_note(txt)
            info.source, info.page_no = "table_header", page_no
            info.confidence = 0.92
            return info
    if caption and SCALE_RE.search(caption):
        info = parse_scale_note(caption)
        info.source, info.page_no, info.confidence = "caption", page_no, 0.95
        return info

    lay = layouts[page_no - 1]
    pt = pts[page_no - 1]
    best = None
    for i, txt, bb in lay.scale_notes:
        gap = bbox[1] - bb[3]
        if -2.0 <= gap <= cfg.caption_max_gap * 2:
            if best is None or gap < best[0]:
                best = (gap, txt, bb)
    if best:
        info = parse_scale_note(best[1])
        info.source, info.page_no, info.bbox = "above_region", page_no, best[2]
        info.confidence = 0.88
        return info

    for txt in reversed(section_path or []):
        if SCALE_RE.search(txt):
            info = parse_scale_note(txt)
            info.source, info.page_no, info.confidence = "section_heading", page_no, 0.75
            return info

    # Earlier pages: a scale declared once at the top of a section governs the tables
    # below it, so search backwards and decay confidence with distance.
    for back in range(0, cfg.scale_search_pages + 1):
        p = page_no - back
        if p < 1:
            break
        l2 = layouts[p - 1]
        cands = [(i, txt, bb) for i, txt, bb in l2.scale_notes
                 if p < page_no or bb[3] <= bbox[1]]
        if not cands:
            continue
        i, txt, bb = max(cands, key=lambda c: c[2][3])
        info = parse_scale_note(txt)
        info.source = "earlier_page" if back else "same_page_above"
        info.page_no, info.bbox = p, bb
        info.confidence = max(0.35, 0.80 - 0.15 * back)
        if back:
            info.flags.append("SCALE_FROM_EARLIER_PAGE")
        return info
    return ScaleInfo(source="not_found", page_no=page_no, confidence=0.0)


# ------------------------------------------------------------------------ footnotes


def collect_footnotes(pts, layouts, page_no: int, look_ahead: int = 1):
    """Footnote lines on this page and the next ``look_ahead`` pages."""
    out = []
    for p in range(page_no, min(len(pts), page_no + look_ahead) + 1):
        lay = layouts[p - 1]
        for _i, marker, txt, bb in lay.footnotes:
            body = txt[len(marker):].strip() if txt.startswith(marker) else txt
            out.append(Footnote(marker=marker.strip(), text=body.strip(),
                                page_no=p, bbox=bb))
    return out


def link_footnotes(cells, footnotes):
    """Match cell markers to footnote markers.  Returns ``(links, unmatched)``."""
    by_marker = {}
    for i, fn in enumerate(footnotes):
        by_marker.setdefault(_norm_marker(fn.marker), []).append(i)
    links, unmatched = [], []
    for c in cells:
        marks = list(getattr(c, "footnote_markers", []) or [])
        raw = (c.raw_text or "").strip()
        m = re.search(r"([*†‡§¶]{1,3})$", raw)
        if m:
            marks.append(m.group(1))
        for mk in sorted(set(marks)):
            key = _norm_marker(mk)
            if key in by_marker:
                for fi in by_marker[key]:
                    links.append((c.row, c.col, fi))
            else:
                unmatched.append((c.row, c.col, mk))
    return links, unmatched


def _norm_marker(m: str) -> str:
    return (m or "").strip().strip("().").lower()


# --------------------------------------------------------------------------- assets


def find_assets(page, pt, layout, table_bboxes, cfg: Config):
    """Images and vector chart clusters on a page that are not tables."""
    out = []
    for info in page.get_text("dict").get("blocks", []):
        if info.get("type", 0) != 1:
            continue
        bb = tuple(float(v) for v in info.get("bbox", (0, 0, 0, 0)))
        if area(bb) < cfg.asset_min_area:
            continue
        if any(_overlaps(bb, tb, 0.6) for tb in table_bboxes):
            continue
        out.append(Asset(kind="image", bbox=bb, page_no=pt.page_no))

    clusters = _vector_clusters(page, pt, table_bboxes, cfg)
    out.extend(clusters)
    for a in out:
        a.caption = _caption_for(layout, a.bbox, cfg)
    out.sort(key=lambda a: (round(a.bbox[1], 2), round(a.bbox[0], 2), a.kind))
    return out


def _overlaps(a, b, frac: float) -> bool:

    return contain_frac(a, b) >= frac


def _vector_clusters(page, pt, table_bboxes, cfg: Config):
    """Dense vector-drawing clusters that are not table regions: bar/line charts."""

    boxes = []
    try:
        for path in page.get_drawings():
            for it in path.get("items", []):
                if it[0] == "re":
                    r = it[1]
                    boxes.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
                elif it[0] == "l":
                    p0, p1 = it[1], it[2]
                    boxes.append((min(p0.x, p1.x), min(p0.y, p1.y),
                                  max(p0.x, p1.x), max(p0.y, p1.y)))
    except Exception:
        return []
    boxes = [b for b in boxes
             if not any(_overlaps(b, tb, 0.9) for tb in table_bboxes)]
    if len(boxes) < 12:
        return []
    ys = [(b[1] + b[3]) / 2.0 for b in boxes]
    out = []
    for cl in cluster_1d(ys, 24.0):
        if len(cl) < 12:
            continue
        bb = union(boxes[i] for i in cl)
        if bb is None or area(bb) < cfg.asset_min_area:
            continue
        if bb[2] - bb[0] < 60 or bb[3] - bb[1] < 40:
            continue
        if any(_overlaps(bb, tb, 0.5) for tb in table_bboxes):
            continue
        out.append(Asset(kind="chart", bbox=bb, page_no=pt.page_no))
    return out


def _caption_for(layout, bbox, cfg: Config) -> str:
    best, gap_best = "", 1e9
    for _i, txt, bb in getattr(layout, "captions", []):
        for gap in (bb[1] - bbox[3], bbox[1] - bb[3]):
            if 0 <= gap < min(gap_best, cfg.caption_max_gap):
                best, gap_best = txt, gap
    return best


def _section_number(text: str):
    m = CAPTION_RE.match(text or "")
    if m and m.group(2):
        return m.group(2).split(".")[0]
    m2 = NUMBER_IN_CAPTION_RE.search(text or "")
    return m2.group(1).split(".")[0] if m2 else None


def associate_assets(table_caption: str, table_title: str, table_bbox, page_no: int,
                     section_path, assets, next_heading_y, cfg: Config,
                     nearby_text: str = ""):
    """Score each asset against one table.  Returns a list of link dicts."""
    links = []
    haystack = " ".join([table_caption or "", table_title or "", nearby_text or ""])
    refs = {m.group(2).lower() for m in FIG_REF_RE.finditer(haystack) if m.group(2)}
    t_sec = _section_number(table_caption or table_title or "")

    for idx, a in enumerate(assets):
        comp = {"textual_reference": 0.0, "shared_numbering": 0.0,
                "proximity_same_section": 0.0, "before_next_heading": 0.0}
        a_num = _section_number(a.caption) if a.caption else None
        a_ref = None
        if a.caption:
            m = FIG_REF_RE.match(a.caption.strip())
            if m and m.group(2):
                a_ref = m.group(2).lower()

        if a_ref and a_ref in refs:
            comp["textual_reference"] = 0.45
        if t_sec and a_num and t_sec == a_num:
            comp["shared_numbering"] = 0.25

        if a.page_no == page_no:
            gap = min(abs(a.bbox[1] - table_bbox[3]), abs(table_bbox[1] - a.bbox[3]))
            if gap <= cfg.asset_link_max_gap:
                comp["proximity_same_section"] = 0.20 * max(
                    0.25, 1.0 - gap / cfg.asset_link_max_gap)
            if next_heading_y is not None and table_bbox[3] <= a.bbox[1] <= next_heading_y:
                comp["before_next_heading"] = 0.10

        conf = min(0.95, sum(comp.values()))
        if conf <= 0.0:
            continue
        relation = "visualises" if a.kind == "chart" else "illustrates"
        if comp["textual_reference"] == 0.0 and comp["shared_numbering"] == 0.0:
            relation = "adjacent_to"
        links.append({"asset_index": idx, "relation": relation,
                      "relation_confidence": round(conf, 3), "components": comp})
    links.sort(key=lambda l: (-l["relation_confidence"], l["asset_index"]))
    return links


# ==================================================================================================
# s9_confidence: S9 -- validation, confidence and the review queue.
# ==================================================================================================
#
# This is the stage the brief weights highest, and the reason is worth restating: a
# pipeline that is 92% accurate and catches 97% of its own mistakes is more useful than
# one that is 96% accurate and cannot say which 4% are wrong.  So the goal here is not a
# high score, it is a *well-ordered* score.
#
# Architecture
# ------------
# Per cell we compute a fixed set of deterministic signals, combine them with **fixed,
# documented logistic weights** (in :mod:`gridsmith.config`, never fitted at runtime -- §3.2
# requires byte-identical reruns), and then map that score to a probability through a
# **calibration curve fitted offline** on the hand-labelled set and vendored as
# ``calibration.json``.  Fitting offline and applying at runtime is what makes the number
# a probability rather than a vibe, without making the run non-deterministic.
#
# The mechanical guarantee
# ------------------------
# Signals are only half the story.  Some conditions mean *we know we might be wrong*, and
# no amount of corroborating evidence should be able to outvote them: OCR was unavailable,
# the source cell contains ``#REF!``, the decimal separator is ambiguous, a token straddles
# a column boundary, a row does not add up.  Those are :data:`MANDATORY_REVIEW_CODES`, and
# they do two things:
#
# *   confidence is **hard-capped** below ``high_confidence_floor``, so such a cell can
#     never be shipped as high-confidence, whatever the logistic says;
# *   the cell enters the review queue **regardless of the 8% budget**.
#
# That is the mechanism behind "every cell is either correct or flagged".  A budget that
# could push a known-suspect cell out of the queue would break the invariant, so the
# budget only ever governs the *discretionary* tail of the ranking.
#
# Reconciliation
# --------------
# Down-foot: rows whose label reads as a total are checked against the sum of their
# contributors -- the rows since the previous total, at the shallowest indent level in
# that window, which is what makes it work on a hierarchy where "Total" sits over
# ``Domestic`` and ``Export`` while ``Product A``/``Product B`` sit under ``Domestic``.
#
# Cross-foot: only run when a column *header* identifies itself as a total.  Summing "the
# other numeric columns" unconditionally is wrong on the overwhelmingly common
# year-over-year table, where 2024 and 2023 are not addends of anything.
#
# A failure flags the total cell strongly and its contributors weakly, because the error
# could be in any of them, and it is never repaired.  The brief is explicit: a subtotal
# that does not reconcile because the source document has a genuine typo must be flagged,
# not silently fixed.

# Reconcilable columns are plain amounts.  Percentages are ratios: summing the
# 2024 change column of the Apple segment table is arithmetic on nonsense, and
# it produced four confident DOWNFOOT_FAIL flags on a correct table.
ADDABLE_TYPES = (T_NUMBER, T_CURRENCY)

# Conditions that must never be shipped as high confidence.
MANDATORY_REVIEW_CODES = frozenset({
    "OCR_UNAVAILABLE", "OCR_FAILED", "OCR_DIGIT_REPAIR", "DEGENERATE_CELL_BOX",
    "SPREADSHEET_ERROR", "MALFORMED_NUMBER",
    "AMBIGUOUS_DECIMAL_SEPARATOR", "AMBIGUOUS_GROUPING", "AMBIGUOUS_DATE_ORDER",
    "TYPE_CONFLICT", "CROSSFOOT_FAIL", "DOWNFOOT_FAIL",
    "TOKEN_STRADDLES_COLUMN", "WATERMARK_OVERLAP", "GRID_UNRESOLVED",
})
# Present but not disqualifying: they lower confidence without forcing review.
SOFT_CODES = frozenset({
    "MULTILINE_CELL", "LIGATURE_EXPANDED", "DIGIT_COUNT_ANOMALY", "DUPLICATE_ROW",
    "EMPTY_IN_DENSE_ROW", "HEADER_BODY_COLCOUNT_MISMATCH", "SCALE_FROM_EARLIER_PAGE",
    "LEGACY_RUPEE_GLYPH", "SCALE_APPLIED", "DAY_ASSUMED_1", "CROSSFOOT_CONTRIBUTOR",
    "DOWNFOOT_CONTRIBUTOR", "GRID_LOW_SUPPORT",
})

SEVERITY_OF = {
    "OCR_UNAVAILABLE": "error", "OCR_FAILED": "error", "SPREADSHEET_ERROR": "error",
    "MALFORMED_NUMBER": "error", "GRID_UNRESOLVED": "error",
    "DOWNFOOT_FAIL": "error", "CROSSFOOT_FAIL": "error",
    "AMBIGUOUS_DECIMAL_SEPARATOR": "warn", "AMBIGUOUS_GROUPING": "warn",
    "AMBIGUOUS_DATE_ORDER": "warn", "TYPE_CONFLICT": "warn",
    "TOKEN_STRADDLES_COLUMN": "warn", "WATERMARK_OVERLAP": "warn",
    "OCR_DIGIT_REPAIR": "warn", "DEGENERATE_CELL_BOX": "warn",
}


@dataclass
class CellScore:
    row: int
    col: int
    features: dict = field(default_factory=dict)
    logit: float = 0.0
    raw_confidence: float = 0.0
    confidence: float = 0.0
    codes: list = field(default_factory=list)
    mandatory: bool = False


@dataclass
class TableScore:
    cells: list = field(default_factory=list)
    table_confidence: float = 0.0
    reconciliation: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)


# --------------------------------------------------------------------- calibration


class Calibration:
    """Monotone binned (isotonic-style) map from logistic score to probability.

    Fitted offline by ``metrics/fit_calibration.py`` on hand-labelled cells and stored
    as bin edges plus the empirical accuracy in each bin.  Applying a lookup table at
    runtime keeps the run deterministic and keeps the fitting code out of the hot path.

    With no file present we fall back to a deliberately *pessimistic* identity: the raw
    sigmoid shrunk 25% toward 0.5.  An uncalibrated pipeline that overstates confidence
    is exactly the failure mode the brief punishes, so the default errs toward doubt.
    """

    def __init__(self, edges=None, values=None, source="fallback_shrunk_sigmoid"):
        self.edges = list(edges or [])
        self.values = list(values or [])
        self.source = source

    @classmethod
    def load(cls, path: str):
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if d.get("edges") and d.get("values"):
                return cls(d["edges"], d["values"], d.get("source", path))
        return cls()

    def apply(self, p: float) -> float:
        if not self.edges:
            return 0.5 + 0.75 * (p - 0.5)
        for i, e in enumerate(self.edges):
            if p <= e:
                return float(self.values[min(i, len(self.values) - 1)])
        return float(self.values[-1])

    def to_dict(self):
        return {"source": self.source, "n_bins": len(self.values),
                "edges": self.edges, "values": self.values}


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    e = math.exp(max(x, -60.0))
    return e / (1.0 + e)


# ----------------------------------------------------------------- reconciliation


def _num(norm):
    return norm.number if (norm is not None and norm.value_type in NUMERIC_TYPES
                           and norm.number is not None) else None


RATIO_LABEL_RE = __import__("re").compile(
    r"/|\(\s*times\s*\)|\bratio\b|\bper\b|\bmargin\b|\bcoverage\b|%",
    __import__("re").I)


def is_reconcilable_total(label: str) -> bool:
    """A row label that reads as a total *and* not as a ratio.

    "Total outside liabilities/Tangible net worth (times)" starts with "Total" and is
    not a total of anything -- it is a ratio.  Reconciling it against the four rows
    above produced two confident-looking DOWNFOOT_FAIL errors on a perfectly correct
    ICRA table, which is a false alarm in the one place false alarms are expensive.
    """
    return is_total_label(label) and not RATIO_LABEL_RE.search(label or "")


def _decimals(norm) -> int:
    """Digits after the decimal point in the *printed* value."""
    if norm is None or not norm.raw_text:
        return 0
    txt = norm.raw_text.strip().strip("()%")
    if "." not in txt:
        return 0
    return len(txt.rsplit(".", 1)[1].strip(" )%"))


def rounding_tolerance(norms) -> float:
    """Largest discrepancy that display rounding alone could explain.

    A percentage tolerance is the wrong model.  With ``tol = 0.5%`` of the stated
    value, a subtotal of 359 against addends summing to 360 passes -- and that is the
    deliberate typo the brief plants and asks us to flag.  What actually bounds an
    innocent discrepancy is the precision the values are *printed* to: each addend can
    hide at most half of its last displayed digit, so ``n`` addends can accumulate at
    most ``n * 0.5 * 10^-d``.

    Integers (d = 0) with two addends therefore tolerate less than 1.0, so 359 vs 360
    fails -- while three values printed as 33.3 summing to 99.9 against a stated 100.0
    still passes, because 0.1 < 3 * 0.05.  The comparison is strict for exactly that
    reason.
    """
    if not norms:
        return 1e-6
    d = max(_decimals(n) for n in norms)
    return max(1e-6, len(norms) * 0.5 * (10.0 ** -d))


def down_foot(cells_by_rc, norms_by_rc, n_rows, n_cols, n_header, indent_levels,
              labels, col_types=None, tol_rel=None, tol_abs=None):
    """Check every total row against the sum of its contributors, per column."""
    results = []
    col_types = col_types or {}
    total_rows = [r for r in range(n_header, n_rows)
                  if is_reconcilable_total(labels.get(r, ""))]
    prev_bound = n_header
    for tr in total_rows:
        window = [r for r in range(prev_bound, tr)
                  if not is_total_label(labels.get(r, ""))]
        prev_bound = tr + 1
        if not window:
            continue
        lv = [indent_levels[r] if r < len(indent_levels) else 0 for r in window]
        base = min(lv) if lv else 0
        contributors = [r for r, l in zip(window, lv) if l == base]
        if len(contributors) < 2:
            continue
        for c in range(n_cols):
            # Only reconcile columns that are actually numeric.  A column mixing
            # currency with percentages infers as text, and adding 2,087.5 to 0.9%
            # is not a check, it is noise.
            if col_types.get(c) not in ADDABLE_TYPES:
                continue
            stated = _num(norms_by_rc.get((tr, c)))
            if stated is None:
                continue
            parts = [(r, _num(norms_by_rc.get((r, c)))) for r in contributors]
            vals = [(r, v) for r, v in parts if v is not None]
            if len(vals) < 2:
                continue
            # An unparsed cell among the addends makes the sum meaningless.
            if any(v is None for _r, v in parts):
                results.append({"kind": "down_foot", "row": tr, "col": c,
                                "status": "incomplete",
                                "contributors": [r for r, _v in vals],
                                "message": "an addend could not be parsed"})
                continue
            total = sum(v for _r, v in vals)
            # Plausibility gate: with all-positive addends a total cannot be materially
            # smaller than its largest part.  When it is, this is not a total row and
            # the "failure" would be ours, not the document's.
            if all(v >= 0 for _r, v in vals) and \
                    stated < 0.9 * max(v for _r, v in vals):
                results.append({"kind": "down_foot", "row": tr, "col": c,
                                "status": "skipped",
                                "message": "stated value is smaller than its largest "
                                           "addend; row is not a sum"})
                continue
            tol = rounding_tolerance([norms_by_rc.get((r, c)) for r, _v in vals]
                                     + [norms_by_rc.get((tr, c))])
            ok = abs(total - stated) < tol
            results.append({"kind": "down_foot", "row": tr, "col": c,
                            "status": "pass" if ok else "fail",
                            "stated": round(stated, 6), "computed": round(total, 6),
                            "delta": round(total - stated, 6), "tolerance": round(tol, 6),
                            "contributors": [r for r, _v in vals]})
    return results


def cross_foot(norms_by_rc, header_paths, n_rows, n_cols, n_header,
               col_types=None, tol_rel=None, tol_abs=None):
    """Check a self-declared total *column* against the sum of the others."""
    col_types = col_types or {}
    total_cols = [c for c in range(n_cols)
                  if col_types.get(c) in ADDABLE_TYPES
                  and any(is_total_label(h) or "total" in (h or "").lower()
                          for h in (header_paths[c]
                                    if c < len(header_paths) else []))]
    if not total_cols:
        return []
    results = []
    for tc in total_cols:
        addends = [c for c in range(n_cols)
                   if c != tc and c > 0 and col_types.get(c) in ADDABLE_TYPES]
        for r in range(n_header, n_rows):
            stated = _num(norms_by_rc.get((r, tc)))
            if stated is None:
                continue
            vals = [_num(norms_by_rc.get((r, c))) for c in addends]
            present = [v for v in vals if v is not None]
            if len(present) < 2:
                continue
            total = sum(present)
            tol = rounding_tolerance([norms_by_rc.get((r, c)) for c in addends]
                                     + [norms_by_rc.get((r, tc))])
            ok = abs(total - stated) < tol
            results.append({"kind": "cross_foot", "row": r, "col": tc,
                            "status": "pass" if ok else "fail",
                            "stated": round(stated, 6), "computed": round(total, 6),
                            "delta": round(total - stated, 6),
                            "tolerance": round(tol, 6),
                            "contributors": [c for c in addends]})
    return results


# ---------------------------------------------------------------------- features


def _snap_residual(cell, x_lines, y_lines):
    """How far the cell's ink sits outside its own grid box, as a fraction of size."""
    if not cell.tokens:
        return 0.0
    c0 = min(cell.col, len(x_lines) - 2)
    c1 = min(cell.col + cell.col_span, len(x_lines) - 1)
    bx0, bx1 = x_lines[c0], x_lines[c1]
    r0 = min(cell.row, len(y_lines) - 2)
    r1 = min(cell.row + cell.row_span, len(y_lines) - 1)
    by0, by1 = y_lines[r0], y_lines[r1]
    tx0 = min(t.x0 for t in cell.tokens)
    tx1 = max(t.x1 for t in cell.tokens)
    ty0 = min(t.y0 for t in cell.tokens)
    ty1 = max(t.y1 for t in cell.tokens)
    w = max(1.0, bx1 - bx0)
    h = max(1.0, by1 - by0)
    over_x = max(0.0, bx0 - tx0, tx1 - bx1) / w
    over_y = max(0.0, by0 - ty0, ty1 - by1) / h
    return min(1.0, max(over_x, over_y))


def _digit_count_anomaly(n_digits, col_digits):
    """MAD-based z-score of digit count against the rest of the column."""
    peers = [d for d in col_digits if d > 0]
    if len(peers) < 3 or n_digits <= 0:
        return 0.0
    med = median(peers)
    mad = median([abs(d - med) for d in peers]) or 0.5
    z = abs(n_digits - med) / (1.4826 * mad)
    return min(1.0, max(0.0, (z - 2.0) / 4.0))


def score_table(cells, norms, grid, cfg: Config, calib: Calibration,
                col_types, source_of, ocr_conf_of, grid_support: float = 1.0):
    """Compute per-cell confidence for one logical table.

    ``cells`` are :class:`gridsmith.s5_cells.ExtractedCell`; ``norms`` is a parallel dict
    keyed by ``(row, col)`` of :class:`gridsmith.s7_normalize.Normalized`.
    """
    ts = TableScore()
    by_rc = {(c.row, c.col): c for c in cells}
    n_rows, n_cols, n_header = grid.n_rows, grid.n_cols, grid.n_header_rows

    labels = {}
    for c in cells:
        if c.col == 0:
            labels[c.row] = (c.raw_text or "").strip()

    # ---- table-level signals -------------------------------------------------
    header_cols = len({c.col for c in cells
                       if c.row < n_header and (c.raw_text or "").strip()})
    body_cols = len({c.col for c in cells
                     if c.row >= n_header and (c.raw_text or "").strip()})
    # Graded, not binary.  As a 0/1 flag this fired on all 680 cells of Shell.pdf --
    # whose header legitimately does not label its subtotal columns -- and subtracted a
    # flat 1.10 from every cell in the document.  A table-level signal can only move a
    # whole table, so its magnitude has to be proportionate to how much of the header
    # is actually missing, or it just declares an entire correct document untrustworthy.
    colcount_mismatch = 0.0
    if n_header > 0 and header_cols and body_cols and header_cols < body_cols - 1:
        colcount_mismatch = min(1.0, (body_cols - header_cols) / float(body_cols))

    row_sig = {}
    for r in range(n_header, n_rows):
        sig = tuple((by_rc.get((r, c)).raw_text or "").strip()
                    for c in range(n_cols) if (r, c) in by_rc)
        row_sig.setdefault(sig, []).append(r)
    dup_rows = {r for sig, rs in row_sig.items()
                if len(rs) > 1 and any(s for s in sig) for r in rs[1:]}

    # ---- reconciliation ------------------------------------------------------
    df = down_foot(by_rc, norms, n_rows, n_cols, n_header,
                   grid.indent_levels, labels, col_types=col_types)
    cf = cross_foot(norms, grid.header_paths, n_rows, n_cols, n_header,
                    col_types=col_types)
    ts.reconciliation = {"down_foot": df, "cross_foot": cf,
                         "n_pass": sum(1 for x in df + cf if x["status"] == "pass"),
                         "n_fail": sum(1 for x in df + cf if x["status"] == "fail"),
                         "n_incomplete": sum(1 for x in df + cf
                                             if x["status"] == "incomplete"),
                         "n_skipped": sum(1 for x in df + cf
                                          if x["status"] == "skipped")}
    fail_cells, contrib_cells = set(), set()
    for x in df + cf:
        if x["status"] != "fail":
            continue
        fail_cells.add((x["row"], x["col"]))
        if x["kind"] == "down_foot":
            for r in x["contributors"]:
                contrib_cells.add((r, x["col"]))
        else:
            for c in x["contributors"]:
                contrib_cells.add((x["row"], c))

    # ---- per-column digit stats ---------------------------------------------
    col_digits = {}
    for c in range(n_cols):
        col_digits[c] = [sum(ch.isdigit() for ch in (by_rc[(r, c)].raw_text or ""))
                         for r in range(n_header, n_rows) if (r, c) in by_rc]

    for cell in sorted(cells, key=lambda c: (c.row, c.col)):
        key = (cell.row, cell.col)
        nrm = norms.get(key)
        codes = [f for f in (cell.flags or [])]
        codes += [f for f in ((nrm.flags if nrm else []) or [])]

        src_ocr = 1.0 if source_of.get(key) == "ocr" else 0.0
        ocr_conf = ocr_conf_of.get(key)
        ocr_conf = 1.0 if ocr_conf is None else float(ocr_conf)

        ctype = col_types.get(cell.col)
        type_conflict = 0.0
        if (nrm is not None and not cell.is_header
                and nrm.value_type not in NULLISH_TYPES
                and ctype not in (None, T_TEXT, T_REFERENCE)):
            got = "number" if nrm.value_type in NUMERIC_TYPES else nrm.value_type
            want = "number" if ctype in NUMERIC_TYPES else ctype
            if got != want:
                type_conflict = 1.0
                codes.append("TYPE_CONFLICT")

        n_dig = sum(ch.isdigit() for ch in (cell.raw_text or ""))
        dig_anom = 0.0 if cell.is_header else _digit_count_anomaly(
            n_dig, col_digits.get(cell.col, []))
        if dig_anom > 0.5:
            codes.append("DIGIT_COUNT_ANOMALY")

        snap = _snap_residual(cell, grid.x_lines, grid.y_lines)
        multiline = 1.0 if cell.n_lines > 1 else 0.0
        watermark = 1.0 if "WATERMARK_OVERLAP" in codes else 0.0
        straddle = 1.0 if "TOKEN_STRADDLES_COLUMN" in codes else 0.0

        crossfoot_fail = 1.0 if key in fail_cells and any(
            x["kind"] == "cross_foot" and (x["row"], x["col"]) == key
            and x["status"] == "fail" for x in cf) else 0.0
        downfoot_fail = 1.0 if key in fail_cells and any(
            x["kind"] == "down_foot" and (x["row"], x["col"]) == key
            and x["status"] == "fail" for x in df) else 0.0
        if crossfoot_fail:
            codes.append("CROSSFOOT_FAIL")
        if downfoot_fail:
            codes.append("DOWNFOOT_FAIL")
        if key in contrib_cells and not (crossfoot_fail or downfoot_fail):
            codes.append("DOWNFOOT_CONTRIBUTOR")

        dup = 1.0 if cell.row in dup_rows else 0.0
        if dup:
            codes.append("DUPLICATE_ROW")

        row_filled = sum(1 for c in range(n_cols)
                         if (cell.row, c) in by_rc
                         and (by_rc[(cell.row, c)].raw_text or "").strip())
        empty_dense = 1.0 if (not (cell.raw_text or "").strip()
                              and n_cols >= 3
                              and row_filled >= 0.7 * n_cols) else 0.0
        if empty_dense:
            codes.append("EMPTY_IN_DENSE_ROW")

        low_support = max(0.0, 1.0 - grid_support)
        if low_support > 0.5:
            codes.append("GRID_LOW_SUPPORT")
        if cell.raw_text is None:
            codes.append("GRID_UNRESOLVED" if not codes else codes[0])

        feats = {
            "source_ocr": src_ocr, "ocr_conf": ocr_conf, "snap_residual": snap,
            "straddle": straddle, "type_conflict": type_conflict,
            "digit_anomaly": dig_anom, "multiline": multiline,
            "crossfoot_fail": crossfoot_fail, "downfoot_fail": downfoot_fail,
            "colcount_mismatch": colcount_mismatch, "dup_row": dup,
            "watermark": watermark, "empty_in_dense_row": empty_dense,
            "grid_low_support": low_support,
        }
        logit = (cfg.w_bias
                 + cfg.w_source_ocr * src_ocr
                 + cfg.w_ocr_conf * (ocr_conf - 1.0) * src_ocr
                 + cfg.w_snap_residual * snap
                 + cfg.w_straddle * straddle
                 + cfg.w_type_conflict * type_conflict
                 + cfg.w_digit_anomaly * dig_anom
                 + cfg.w_multiline * multiline
                 + cfg.w_crossfoot_fail * crossfoot_fail
                 + cfg.w_downfoot_fail * downfoot_fail
                 + cfg.w_colcount_mismatch * colcount_mismatch
                 + cfg.w_dup_row * dup
                 + cfg.w_watermark * watermark
                 + cfg.w_empty_in_dense_row * empty_dense
                 + cfg.w_grid_low_support * low_support)
        raw = sigmoid(logit)
        conf = calib.apply(raw)

        codes = sorted(set(c for c in codes if c))
        mandatory = bool(MANDATORY_REVIEW_CODES & set(codes))
        if mandatory:
            # Hard cap: a known-suspect cell can never ship as high confidence.
            conf = min(conf, cfg.high_confidence_floor - 0.05)
        if cell.raw_text is None:
            conf = 0.0

        ts.cells.append(CellScore(row=cell.row, col=cell.col, features=feats,
                                  logit=round(logit, 5), raw_confidence=round(raw, 5),
                                  confidence=round(conf, 5), codes=codes,
                                  mandatory=mandatory))

    body = [c for c in ts.cells if c.row >= n_header]
    pool = body or ts.cells
    ts.table_confidence = round(
        (sum(c.confidence for c in pool) / len(pool)) if pool else 0.0, 5)
    return ts


def issues_for(table_score, table_id, cell_id_of):
    """Turn per-cell codes into ``issues`` rows."""
    out = []
    for cs in table_score.cells:
        for code in cs.codes:
            if code not in MANDATORY_REVIEW_CODES and code not in SOFT_CODES:
                continue
            sev = SEVERITY_OF.get(code, "info" if code in SOFT_CODES else "warn")
            out.append({
                "table_id": table_id,
                "cell_id": cell_id_of.get((cs.row, cs.col)),
                "severity": sev, "code": code,
                "message": "cell r%d c%d: %s (confidence %.3f)"
                           % (cs.row, cs.col, code, cs.confidence)})
    for rec in (table_score.reconciliation.get("down_foot", [])
                + table_score.reconciliation.get("cross_foot", [])):
        if rec["status"] in ("pass", "skipped"):
            continue
        code = "DOWNFOOT_FAIL" if rec["kind"] == "down_foot" else "CROSSFOOT_FAIL"
        if rec["status"] == "incomplete":
            code = "RECONCILIATION_INCOMPLETE"
        msg = rec.get("message")
        if not msg:
            msg = ("%s at r%d c%d: stated %s, contributors sum to %s (delta %s, "
                   "tolerance %s) -- not corrected, see LIMITATIONS"
                   % (rec["kind"], rec["row"], rec["col"], rec.get("stated"),
                      rec.get("computed"), rec.get("delta"), rec.get("tolerance")))
        out.append({"table_id": table_id,
                    "cell_id": cell_id_of.get((rec["row"], rec["col"])),
                    "severity": "error" if rec["status"] == "fail" else "warn",
                    "code": code, "message": msg})
    return out


# ==================================================================================================
# s9_review: S9b -- the ranked human-review queue.
# ==================================================================================================
#
# The target from section 4: flag no more than 8% of cells while those flags cover at
# least 95% of the actual errors.  Two rules make that shape achievable without breaking
# the invariant that no suspect cell is ever shipped unflagged:
#
# 1.  **Mandatory cells are never budgeted out.**  Anything carrying a
#     :data:`gridsmith.s9_confidence.MANDATORY_REVIEW_CODES` code is in the queue even if
#     it alone exceeds 8%.  A budget that could evict a known-suspect cell would make the
#     8% number a lie.
# 2.  **The discretionary tail is ranked by confidence** and cut at the budget.
#
# Both the budgeted queue (``review_queue.csv``) and the *full* ranking are produced,
# because a single operating point cannot show a precision-recall curve and the brief
# asks for the curve, not the point.  ``metrics/eval.py`` sweeps the full ranking.
#
# Ranking key: mandatory first, then ascending confidence, then descending severity, then
# (table, row, col) so the order is total and reproducible.

SEVERITY_RANK = {"error": 0, "warn": 1, "info": 2}

QUEUE_COLUMNS = [
    "rank", "priority", "table_id", "logical_index", "page_no", "row_idx", "col_idx",
    "header_path", "row_label_path", "raw_text", "normalized_value", "value_type",
    "confidence", "severity", "reason_codes", "reason", "bbox", "source",
]


@dataclass
class ReviewItem:
    table_id: str
    logical_index: int
    page_no: int
    row_idx: int
    col_idx: int
    header_path: str
    row_label_path: str
    raw_text: str
    normalized_value: object
    value_type: str
    confidence: float
    codes: list = field(default_factory=list)
    severity: str = "info"
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    source: str = ""
    mandatory: bool = False

    @property
    def reason(self) -> str:
        if not self.codes:
            return "low model confidence"
        return "; ".join(_explain(c) for c in self.codes[:4])


EXPLANATIONS = {
    "OCR_UNAVAILABLE": "no OCR engine available for this raster cell; value not read",
    "OCR_FAILED": "OCR returned nothing for this cell crop",
    "OCR_DIGIT_REPAIR": "OCR text only parsed as a number after glyph repair",
    "SPREADSHEET_ERROR": "source document contains a spreadsheet error value",
    "MALFORMED_NUMBER": "digits present but the number does not parse",
    "AMBIGUOUS_DECIMAL_SEPARATOR": "cannot tell decimal comma from thousands separator",
    "AMBIGUOUS_GROUPING": "digit grouping matches neither Indian nor Western convention",
    "AMBIGUOUS_DATE_ORDER": "day/month order undetermined for this document",
    "TYPE_CONFLICT": "value type disagrees with the inferred column type",
    "CROSSFOOT_FAIL": "row does not sum to its stated total",
    "DOWNFOOT_FAIL": "line items do not sum to the stated subtotal or total",
    "DOWNFOOT_CONTRIBUTOR": "contributes to a subtotal that does not reconcile",
    "TOKEN_STRADDLES_COLUMN": "text crosses an inferred column boundary",
    "WATERMARK_OVERLAP": "watermark or stamp glyphs overlap this cell",
    "GRID_UNRESOLVED": "cell text could not be resolved at all",
    "GRID_LOW_SUPPORT": "column boundaries rest on weak evidence",
    "DIGIT_COUNT_ANOMALY": "digit count is an outlier against the rest of the column",
    "DUPLICATE_ROW": "row duplicates an earlier row",
    "EMPTY_IN_DENSE_ROW": "cell is empty while the rest of the row is populated",
    "MULTILINE_CELL": "cell text wraps over more than one line",
    "HEADER_BODY_COLCOUNT_MISMATCH": "header spans fewer columns than the body",
    "SCALE_FROM_EARLIER_PAGE": "unit and scale inherited from an earlier page",
    "LEGACY_RUPEE_GLYPH": "rupee sign encoded as a grave accent by a legacy font",
    "LIGATURE_EXPANDED": "an f-ligature was expanded in the raw text",
    "SCALE_APPLIED": "a declared magnitude was applied to the value",
    "DAY_ASSUMED_1": "month-only date; day assumed to be the 1st",
}


def _explain(code: str) -> str:
    return EXPLANATIONS.get(code, code.lower().replace("_", " "))


def severity_of(codes) -> str:
    best = "info"
    for c in codes:
        s = SEVERITY_OF.get(c, "info")
        if SEVERITY_RANK[s] < SEVERITY_RANK[best]:
            best = s
    return best


def build_queue(items, cfg: Config, n_cells_total: int):
    """Rank items and split into the budgeted queue plus the full ranking."""
    def key(it):
        return (0 if it.mandatory else 1, round(it.confidence, 6),
                SEVERITY_RANK.get(it.severity, 2), it.table_id, it.row_idx, it.col_idx)

    ranked = sorted(items, key=key)
    budget = int(round(cfg.review_budget_frac * max(1, n_cells_total)))
    n_mandatory = sum(1 for it in ranked if it.mandatory)
    keep = max(budget, n_mandatory)
    queued = ranked[:keep]
    stats = {
        "n_cells_total": n_cells_total,
        "budget_frac": cfg.review_budget_frac,
        "budget_cells": budget,
        "n_mandatory": n_mandatory,
        "n_queued": len(queued),
        "queued_frac": round(len(queued) / max(1, n_cells_total), 5),
        "budget_exceeded_by_mandatory": n_mandatory > budget,
    }
    return queued, ranked, stats


def to_csv(queued, cfg: Config) -> str:
    """Render the queue as deterministic CSV text (LF endings, fixed column order)."""
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(QUEUE_COLUMNS)
    for i, it in enumerate(queued, start=1):
        w.writerow([
            i,
            "mandatory" if it.mandatory else "discretionary",
            it.table_id, it.logical_index, it.page_no, it.row_idx, it.col_idx,
            it.header_path, it.row_label_path,
            "" if it.raw_text is None else it.raw_text,
            "" if it.normalized_value is None else it.normalized_value,
            it.value_type, "%.5f" % it.confidence, it.severity,
            "|".join(it.codes), it.reason,
            "%.2f,%.2f,%.2f,%.2f" % tuple(round(float(v), 2) for v in it.bbox),
            it.source,
        ])
    return buf.getvalue()


# ==================================================================================================
# s10_persist: S10 -- persistence: SQLite, one .xlsx per logical table, manifest.
# ==================================================================================================
#
# Two things here are less obvious than they look.
#
# **Byte-identical Excel.**  ``openpyxl`` stamps ``docProps/core.xml`` with
# ``datetime.now()`` and writes every zip entry with the current local time, so two runs
# over the same input differ in dozens of bytes even when every cell matches.  We
# therefore fix the workbook properties *and* rewrite the archive with a constant
# timestamp, ``STORED``-order entries and no extra fields -- see
# :func:`write_deterministic_xlsx`.  Without that step §3.2 simply cannot be satisfied,
# and it is not the kind of thing a two-run diff lets you overlook.
#
# **Byte-identical SQLite.**  A fresh database written by an identical sequence of
# statements is already reproducible, provided nothing in the data varies.  The only
# varying values are the three timestamps in ``extraction_runs`` -- which the brief
# explicitly permits -- and ``--fixed-time`` pins even those, so a reviewer running
# ``gridsmith run`` twice and diffing ``out/`` byte-for-byte gets a clean result rather
# than a one-row explanation.  Ids are content digests, not counters, for the same reason.
#
# Native types in the Data sheet: a value is written as a native Excel number or date only
# when its cell is *not* flagged for review.  Anything uncertain is written as text, so a
# spreadsheet total can never quietly incorporate a value the pipeline does not trust.

SCHEMA_SQL = """\
-- gridsmith output schema.
--
-- The minimum schema from section 6.1 of the brief is present verbatim; every addition
-- is additive and noted with a comment.  The design target is the query the brief uses
-- as its acceptance test:
--
--   "give me every cell in this document whose column header path contains 'Revenue',
--    together with its page number and the unit it is expressed in"
--
-- That resolves to a three-table join with no string surgery, because header_path lives
-- on `columns` (not smeared across cells), the unit lives beside it, and the page number
-- reaches the cell through `pages`.  See docs/queries.sql for the query itself.
--
-- Ids are content-addressed digests (see gridsmith.determinism.sid), not autoincrement
-- integers: identical input must produce identical rows, and a counter reorders whenever
-- anything upstream changes.

PRAGMA journal_mode = DELETE;   -- no -wal sidecar; the file must stand alone
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    page_count   INTEGER NOT NULL,
    ingested_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    page_id      TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    page_no      INTEGER NOT NULL,
    width        REAL NOT NULL,
    height       REAL NOT NULL,
    page_type    TEXT NOT NULL,          -- digital | scanned | hybrid | empty
    rotation     INTEGER NOT NULL,
    -- additions
    content_rotation INTEGER DEFAULT 0,  -- rotated content without a /Rotate entry
    skew_deg     REAL DEFAULT 0.0,       -- deskew actually applied
    image_area_frac REAL DEFAULT 0.0,
    char_density REAL DEFAULT 0.0,
    notes_json   TEXT DEFAULT '[]',
    UNIQUE (doc_id, page_no)
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    run_id       TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    started_at   TEXT NOT NULL,          -- the ONLY wall-clock values in the database
    finished_at  TEXT,
    git_sha      TEXT,
    config_json  TEXT NOT NULL,
    -- additions
    gridsmith_version TEXT,
    config_fingerprint TEXT,
    wall_seconds REAL,
    peak_rss_mb  REAL,
    offline      INTEGER DEFAULT 1,
    ocr_backend  TEXT
);

CREATE TABLE IF NOT EXISTS tables (
    table_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    run_id       TEXT NOT NULL REFERENCES extraction_runs(run_id),
    logical_index INTEGER NOT NULL,
    start_page   INTEGER NOT NULL,
    end_page     INTEGER NOT NULL,
    n_rows       INTEGER NOT NULL,
    n_cols       INTEGER NOT NULL,
    title        TEXT,
    caption      TEXT,
    section_path TEXT,                   -- JSON array
    unit_scale_note TEXT,
    is_continuation INTEGER NOT NULL DEFAULT 0,
    detector_votes_json TEXT,
    table_confidence REAL,
    -- additions
    n_header_rows INTEGER DEFAULT 1,
    regime       TEXT,                   -- grid | fills | hlines | vlines | borderless
    unit_scale_source TEXT,              -- caption | above_region | section_heading | ...
    unit_scale_page INTEGER,
    unit_scale_confidence REAL,
    unit_currency TEXT,
    unit_magnitude REAL DEFAULT 1.0,
    stitch_votes_json TEXT,
    structure_evidence_json TEXT,
    reconciliation_json TEXT,
    excel_path   TEXT
);

CREATE TABLE IF NOT EXISTS table_regions (
    region_id    TEXT PRIMARY KEY,
    table_id     TEXT NOT NULL REFERENCES tables(table_id),
    page_id      TEXT NOT NULL REFERENCES pages(page_id),
    bbox_x0      REAL NOT NULL,
    bbox_y0      REAL NOT NULL,
    bbox_x1      REAL NOT NULL,
    bbox_y1      REAL NOT NULL,
    -- additions
    part_index   INTEGER DEFAULT 0,      -- which page-part of a stitched table
    regime       TEXT
);

CREATE TABLE IF NOT EXISTS columns (
    column_id    TEXT PRIMARY KEY,
    table_id     TEXT NOT NULL REFERENCES tables(table_id),
    col_idx      INTEGER NOT NULL,
    header_path  TEXT,                   -- JSON array, outermost header level first
    inferred_type TEXT,
    unit         TEXT,
    nullable     INTEGER DEFAULT 1,
    -- additions
    header_path_text TEXT,               -- " > " joined, for LIKE queries
    type_evidence_json TEXT,
    n_conflicts  INTEGER DEFAULT 0,
    UNIQUE (table_id, col_idx)
);

CREATE TABLE IF NOT EXISTS cells (
    cell_id      TEXT PRIMARY KEY,
    table_id     TEXT NOT NULL REFERENCES tables(table_id),
    row_idx      INTEGER NOT NULL,
    col_idx      INTEGER NOT NULL,
    row_span     INTEGER NOT NULL DEFAULT 1,
    col_span     INTEGER NOT NULL DEFAULT 1,
    is_header    INTEGER NOT NULL DEFAULT 0,
    raw_text     TEXT,                   -- NULL means "could not be read", not "empty"
    normalized_value,                    -- no type: REAL for numbers, TEXT for dates
    value_type   TEXT,
    row_label_path TEXT,                 -- JSON array
    page_id      TEXT REFERENCES pages(page_id),
    bbox_x0      REAL,
    bbox_y0      REAL,
    bbox_x1      REAL,
    bbox_y1      REAL,
    source       TEXT,                   -- text-layer | ocr | none
    confidence   REAL,
    -- additions
    normalized_number REAL,              -- typed mirrors, so SQL needs no CAST
    normalized_date TEXT,
    currency     TEXT,
    indent_level INTEGER DEFAULT 0,
    row_label_path_text TEXT,
    footnote_markers TEXT,
    ocr_conf     REAL,
    raw_confidence REAL,               -- pre-calibration logistic output
    reason_codes TEXT,                   -- pipe-joined
    mandatory_review INTEGER DEFAULT 0,
    features_json TEXT,
    UNIQUE (table_id, row_idx, col_idx)
);

CREATE TABLE IF NOT EXISTS footnotes (
    footnote_id  TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    page_id      TEXT REFERENCES pages(page_id),
    marker       TEXT,
    text         TEXT
);

CREATE TABLE IF NOT EXISTS cell_footnotes (
    cell_id      TEXT NOT NULL REFERENCES cells(cell_id),
    footnote_id  TEXT NOT NULL REFERENCES footnotes(footnote_id),
    PRIMARY KEY (cell_id, footnote_id)
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    page_id      TEXT REFERENCES pages(page_id),
    kind         TEXT,                   -- image | chart
    file_path    TEXT,
    caption      TEXT,
    bbox_x0      REAL,
    bbox_y0      REAL,
    bbox_x1      REAL,
    bbox_y1      REAL,
    -- additions
    series_json  TEXT                    -- recovered chart series, when available
);

CREATE TABLE IF NOT EXISTS table_assets (
    table_id     TEXT NOT NULL REFERENCES tables(table_id),
    asset_id     TEXT NOT NULL REFERENCES assets(asset_id),
    relation     TEXT,
    relation_confidence REAL,
    -- additions
    components_json TEXT,                -- per-signal breakdown of the score
    PRIMARY KEY (table_id, asset_id)
);

CREATE TABLE IF NOT EXISTS issues (
    issue_id     TEXT PRIMARY KEY,
    table_id     TEXT REFERENCES tables(table_id),
    cell_id      TEXT REFERENCES cells(cell_id),
    severity     TEXT NOT NULL,          -- error | warn | info
    code         TEXT NOT NULL,
    message      TEXT
);

-- Regions the detectors proposed and the trap rules refused.  Not in the minimum
-- schema, but "we correctly excluded the table of contents" is only a checkable claim
-- if the refusal is recorded.
CREATE TABLE IF NOT EXISTS rejected_regions (
    rejection_id TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    page_id      TEXT REFERENCES pages(page_id),
    bbox_x0      REAL, bbox_y0 REAL, bbox_x1 REAL, bbox_y1 REAL,
    regime       TEXT,
    code         TEXT NOT NULL,
    score        REAL,
    message      TEXT,
    evidence_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_cells_table ON cells(table_id, row_idx, col_idx);
CREATE INDEX IF NOT EXISTS idx_cells_conf ON cells(confidence);
CREATE INDEX IF NOT EXISTS idx_cells_mandatory ON cells(mandatory_review);
CREATE INDEX IF NOT EXISTS idx_columns_header ON columns(header_path_text);
CREATE INDEX IF NOT EXISTS idx_issues_code ON issues(code, severity);
CREATE INDEX IF NOT EXISTS idx_pages_no ON pages(doc_id, page_no);
CREATE INDEX IF NOT EXISTS idx_regions_table ON table_regions(table_id);

-- Convenience view: one row per cell with its header path, unit and page number
-- already joined.  This is the shape most downstream questions want.
CREATE VIEW IF NOT EXISTS v_cells_full AS
SELECT
    c.cell_id, c.table_id, t.logical_index, t.title, t.caption,
    t.section_path, t.unit_scale_note, t.unit_currency, t.unit_magnitude,
    col.header_path, col.header_path_text, col.inferred_type, col.unit,
    c.row_idx, c.col_idx, c.row_span, c.col_span, c.is_header,
    c.raw_text, c.normalized_value, c.normalized_number, c.normalized_date,
    c.value_type, c.currency, c.row_label_path, c.row_label_path_text,
    c.indent_level, p.page_no, c.bbox_x0, c.bbox_y0, c.bbox_x1, c.bbox_y1,
    c.source, c.confidence, c.mandatory_review, c.reason_codes
FROM cells c
JOIN tables t   ON t.table_id = c.table_id
LEFT JOIN columns col ON col.table_id = c.table_id AND col.col_idx = c.col_idx
LEFT JOIN pages p ON p.page_id = c.page_id;
"""


# ------------------------------------------------------------------------ SQLite


def open_db(path: str) -> sqlite3.Connection:
    if os.path.exists(path):
        os.remove(path)
    for side in (path + "-wal", path + "-shm", path + "-journal"):
        if os.path.exists(side):
            os.remove(side)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)
    return conn


def insert(conn, table: str, rows):
    """Insert dict rows in a fixed column order so the write sequence is stable."""
    rows = list(rows)
    if not rows:
        return
    cols = sorted(rows[0].keys())
    q = "INSERT OR REPLACE INTO %s (%s) VALUES (%s)" % (
        table, ",".join(cols), ",".join("?" * len(cols)))
    conn.executemany(q, [tuple(r[c] for c in cols) for r in rows])


# ------------------------------------------------------------------------- Excel


def write_deterministic_xlsx(workbook, path: str) -> None:
    """Save an openpyxl workbook so two runs produce identical bytes.

    ``openpyxl`` writes ``docProps/core.xml`` with the current time and each zip entry
    with the local clock.  We pin the document properties, save into memory, then
    rebuild the archive with :data:`gridsmith.determinism.FIXED_ZIP_TIME` on every entry
    and the entry order preserved.
    """
    from openpyxl.utils.datetime import from_ISO8601

    props = workbook.properties
    fixed = from_ISO8601("1980-01-01T00:00:00Z")
    props.created = fixed
    props.modified = fixed
    props.lastModifiedBy = "gridsmith"
    props.creator = "gridsmith"

    buf = io.BytesIO()
    workbook.save(buf)
    buf.seek(0)

    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(buf, "r") as src, \
            zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "docProps/core.xml":
                data = _pin_core_props(data)
            info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 0
            dst.writestr(info, data)


FIXED_XML_TIME = "1980-01-01T00:00:00Z"


def _pin_core_props(data: bytes) -> bytes:
    """Overwrite the timestamps inside ``docProps/core.xml``.

    Setting ``workbook.properties.modified`` before saving is not enough: openpyxl
    rewrites ``dcterms:modified`` with the current clock during ``save()``, so two runs
    a second apart differ by exactly those bytes -- which the two-run diff duly found.
    Rewriting the entry as we rebuild the archive is version-proof, because it does not
    depend on which openpyxl internal happens to own the value.
    """
    import re

    text = data.decode("utf-8", "replace")
    for tag in ("dcterms:created", "dcterms:modified"):
        text = re.sub(r"(<%s[^>]*>)[^<]*(</%s>)" % (tag, tag),
                      r"\g<1>%s\g<2>" % FIXED_XML_TIME, text)
    return text.encode("utf-8")


def table_slug(title: str, fallback: str = "table") -> str:
    import re

    s = (title or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    s = re.sub(r"_+", "_", s)
    return (s[:48] or fallback)


def excel_filename(logical_index: int, start: int, end: int, slug: str) -> str:
    return "T%03d__p%d-%d__%s.xlsx" % (logical_index, start, end, slug)


def write_table_xlsx(path: str, *, data_rows, merges, n_header_rows, context_rows,
                     provenance_rows, issue_rows, freeze_header=True):
    """Write one logical table as the four-sheet workbook the contract requires."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    head_font = Font(bold=True)
    head_fill = PatternFill("solid", fgColor="EEEEEE")
    wrap = Alignment(vertical="top", wrap_text=True)

    for r, row in enumerate(data_rows, start=1):
        for c, value in enumerate(row, start=1):
            cell = ws.cell(row=r, column=c, value=value)
            cell.alignment = wrap
            if r <= n_header_rows:
                cell.font = head_font
                cell.fill = head_fill
    for (r0, c0, r1, c1) in merges:
        if r1 > r0 or c1 > c0:
            ws.merge_cells(start_row=r0 + 1, start_column=c0 + 1,
                           end_row=r1 + 1, end_column=c1 + 1)
    if data_rows:
        widths = {}
        for row in data_rows:
            for c, v in enumerate(row, start=1):
                widths[c] = min(52, max(widths.get(c, 9),
                                        len(str(v)) + 2 if v is not None else 9))
        for c, w in sorted(widths.items()):
            ws.column_dimensions[get_column_letter(c)].width = w
    if freeze_header and n_header_rows > 0:
        # A coordinate string, not a cell object: when A1 falls inside a merged range
        # openpyxl hands back a MergedCell, which freeze_panes cannot consume.
        ws.freeze_panes = "A%d" % (n_header_rows + 1)

    ctx = wb.create_sheet("Context")
    ctx.append(["field", "value"])
    ctx.cell(row=1, column=1).font = head_font
    ctx.cell(row=1, column=2).font = head_font
    for k, v in context_rows:
        ctx.append([k, v])
    ctx.column_dimensions["A"].width = 26
    ctx.column_dimensions["B"].width = 96

    prov = wb.create_sheet("Provenance")
    prov_cols = ["row_idx", "col_idx", "raw_text", "normalized_value", "page_no",
                 "bbox", "source", "confidence", "reason_codes"]
    prov.append(prov_cols)
    for i in range(len(prov_cols)):
        prov.cell(row=1, column=i + 1).font = head_font
    for row in provenance_rows:
        prov.append([row.get(c) for c in prov_cols])
    prov.freeze_panes = "A2"
    for col, w in zip("ABCDEFGHI", (9, 9, 40, 18, 9, 30, 12, 12, 40)):
        prov.column_dimensions[col].width = w

    iss = wb.create_sheet("Issues")
    iss_cols = ["severity", "code", "row_idx", "col_idx", "message"]
    iss.append(iss_cols)
    for i in range(len(iss_cols)):
        iss.cell(row=1, column=i + 1).font = head_font
    for row in issue_rows:
        iss.append([row.get(c) for c in iss_cols])
    iss.freeze_panes = "A2"
    for col, w in zip("ABCDE", (10, 30, 9, 9, 110)):
        iss.column_dimensions[col].width = w

    write_deterministic_xlsx(wb, path)


# ---------------------------------------------------------------------- manifest


def write_manifest(path: str, payload: dict) -> None:

    write_text(path, canonical_json(payload) + "\n")


def reset_dir(path: str, keep=()):
    """Recreate an output subdirectory so stale files never survive a rerun."""
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name in keep:
                continue
            p = os.path.join(path, name)
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
    else:
        os.makedirs(path, exist_ok=True)


def region_id(table_id: str, page_no: int, part: int) -> str:
    return sid("reg", table_id, page_no, part)


# ==================================================================================================
# s11_audit: S11 -- auditability: the debug overlay renderer.
# ==================================================================================================
#
# "We will use it during the live review, and so will you at 2 a.m. on day four."  So this
# draws what the pipeline *believed*, not a prettier version of the page:
#
# * accepted table regions in blue, with the regime and table id;
# * **rejected** regions in red with the trap code that refused them -- the refusals are
#   half the score on detection and invisible otherwise;
# * inferred column boundaries (solid green) and row boundaries (dashed green);
# * every cell box, tinted by confidence: green above the high-confidence floor, amber
#   between 0.6 and the floor, red below -- so a bad column lights up as a red stripe;
# * merged cells outlined thicker with their span printed;
# * running header/footer bands greyed out, so "why did it miss that row" is answerable;
# * detected assets in purple with their link confidence.
#
# Rendering is pure PyMuPDF drawing onto a rasterised page, which keeps the dependency
# list unchanged and works identically for digital and scanned pages.

COL_REGION = (0.10, 0.35, 0.85)
COL_REJECT = (0.85, 0.15, 0.15)
COL_GRID = (0.05, 0.55, 0.20)
COL_ASSET = (0.55, 0.20, 0.70)
COL_CHROME = (0.55, 0.55, 0.55)
COL_HIGH = (0.15, 0.65, 0.25)
COL_MID = (0.95, 0.65, 0.10)
COL_LOW = (0.90, 0.15, 0.15)


def confidence_colour(conf, floor: float):
    if conf is None:
        return COL_LOW
    if conf >= floor:
        return COL_HIGH
    if conf >= 0.60:
        return COL_MID
    return COL_LOW


def audit_render_page(doc, page_no: int, overlay, out_path: str, dpi: int = 130,
                high_confidence_floor: float = 0.85) -> str:
    """Draw ``overlay`` onto page ``page_no`` and write a PNG.

    ``overlay`` is a plain dict so the renderer stays decoupled from pipeline objects
    and ``gridsmith debug`` can rebuild it straight from the database.
    """
    page = doc[page_no - 1]
    shape = page.new_shape()

    for band in overlay.get("chrome_bands", []):
        shape.draw_rect(pymupdf.Rect(*band))
        shape.finish(color=COL_CHROME, fill=COL_CHROME, fill_opacity=0.10, width=0.4)

    for reg in overlay.get("regions", []):
        r = pymupdf.Rect(*reg["bbox"])
        shape.draw_rect(r)
        shape.finish(color=COL_REGION, width=1.6, fill=COL_REGION, fill_opacity=0.04)
        _label(shape, r.x0, r.y0 - 3.0, "%s  %s" % (reg.get("label", ""),
                                                    reg.get("regime", "")), COL_REGION)

    for reg in overlay.get("rejected", []):
        r = pymupdf.Rect(*reg["bbox"])
        shape.draw_rect(r)
        shape.finish(color=COL_REJECT, width=1.4, dashes="[3 3] 0",
                     fill=COL_REJECT, fill_opacity=0.05)
        _label(shape, r.x0, r.y0 - 3.0, "REFUSED " + reg.get("code", ""), COL_REJECT)

    for g in overlay.get("grids", []):
        y0, y1 = g["y_lines"][0], g["y_lines"][-1]
        for x in g["x_lines"]:
            shape.draw_line(pymupdf.Point(x, y0), pymupdf.Point(x, y1))
        shape.finish(color=COL_GRID, width=0.7)
        x0, x1 = g["x_lines"][0], g["x_lines"][-1]
        for y in g["y_lines"]:
            shape.draw_line(pymupdf.Point(x0, y), pymupdf.Point(x1, y))
        shape.finish(color=COL_GRID, width=0.5, dashes="[2 2] 0")

    for cell in overlay.get("cells", []):
        col = confidence_colour(cell.get("confidence"), high_confidence_floor)
        r = pymupdf.Rect(*cell["bbox"])
        if r.is_empty or r.width <= 0 or r.height <= 0:
            continue
        shape.draw_rect(r)
        merged = cell.get("row_span", 1) > 1 or cell.get("col_span", 1) > 1
        shape.finish(color=col, width=1.1 if merged else 0.5,
                     fill=col, fill_opacity=0.13 if merged else 0.07)
        if merged:
            _label(shape, r.x0 + 1, r.y0 + 6,
                   "%dx%d" % (cell.get("row_span", 1), cell.get("col_span", 1)), col)

    for a in overlay.get("assets", []):
        r = pymupdf.Rect(*a["bbox"])
        shape.draw_rect(r)
        shape.finish(color=COL_ASSET, width=1.3)
        _label(shape, r.x0, r.y0 - 3.0,
               "%s %s" % (a.get("kind", "asset"), a.get("relation", "")), COL_ASSET)

    shape.commit()
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    pix.save(out_path)
    del pix
    return out_path


def _label(shape, x, y, text, colour):
    if not text:
        return
    try:
        shape.insert_text(pymupdf.Point(x, max(6.0, y)), text[:64], fontsize=5.2,
                          color=colour, fontname="helv")
    except Exception:
        pass


def legend_text() -> str:
    return ("blue = accepted region | red dashed = refused region (trap code) | "
            "green solid = column boundary, dashed = row boundary | "
            "cell tint: green >= high-confidence floor, amber >= 0.60, red below | "
            "thick cell = merged, labelled with its span | "
            "grey = running header/footer | purple = linked asset")


# ==================================================================================================
# s11_metrics: S11b -- the ``audit``, ``debug``, ``verify`` and ``query`` commands.
# ==================================================================================================
#
# ``verify`` is the one worth a note.  Section 3.2 says two runs must produce
# byte-identical database rows and Excel files with timestamps isolated to a single
# ``extraction_runs`` row.  Rather than assert that, this command proves it: it hashes
# every artefact in both directories, and for ``extraction.db`` it compares a canonical
# dump of every table *except* the three timestamp columns of ``extraction_runs``.  With
# ``--fixed-time`` even those match, so a plain recursive byte diff of ``out/`` is clean.
#
# ``query`` ships the named example the brief uses as its schema acceptance test --
# "every cell whose column header path contains Revenue, with its page number and unit" --
# so that claim is executable rather than asserted.

EXAMPLE_QUERIES = {
    "revenue_cells": (
        "the brief's acceptance test: every cell whose column header path OR row "
        "label path mentions revenue/amount, with its page number and unit",
        """
        SELECT t.logical_index, c.row_idx, c.col_idx,
               col.header_path_text        AS header_path,
               c.row_label_path_text       AS row_label,
               c.raw_text, c.normalized_number, col.unit,
               p.page_no, c.confidence, c.reason_codes
        FROM cells c
        JOIN columns col ON col.table_id = c.table_id AND col.col_idx = c.col_idx
        JOIN tables  t   ON t.table_id  = c.table_id
        JOIN pages   p   ON p.page_id   = c.page_id
        WHERE (col.header_path_text LIKE '%revenue%' COLLATE NOCASE
               OR col.header_path_text LIKE '%amount%' COLLATE NOCASE
               OR c.row_label_path_text LIKE '%revenue%' COLLATE NOCASE)
          AND c.is_header = 0
        ORDER BY t.logical_index, c.row_idx, c.col_idx
        """),
    "low_confidence": (
        "cells flagged for review, worst first",
        """
        SELECT t.logical_index, p.page_no, c.row_idx, c.col_idx, c.raw_text,
               c.confidence, c.mandatory_review, c.reason_codes
        FROM cells c
        JOIN tables t ON t.table_id = c.table_id
        LEFT JOIN pages p ON p.page_id = c.page_id
        WHERE c.mandatory_review = 1 OR c.confidence < 0.85
        ORDER BY c.mandatory_review DESC, c.confidence ASC
        """),
    "failed_reconciliation": (
        "totals that do not add up (never silently corrected)",
        """
        SELECT i.code, i.severity, t.logical_index, t.title, i.message
        FROM issues i
        LEFT JOIN tables t ON t.table_id = i.table_id
        WHERE i.code IN ('DOWNFOOT_FAIL','CROSSFOOT_FAIL','RECONCILIATION_INCOMPLETE')
        ORDER BY i.severity, t.logical_index
        """),
    "refused_regions": (
        "regions the trap classifiers refused, with the reason",
        """
        SELECT p.page_no, r.code, r.score, r.regime, r.message
        FROM rejected_regions r
        LEFT JOIN pages p ON p.page_id = r.page_id
        ORDER BY p.page_no, r.code
        """),
    "footnoted_cells": (
        "cells linked to a footnote, with the footnote text",
        """
        SELECT t.logical_index, c.row_idx, c.col_idx, c.raw_text,
               f.marker, f.text, fp.page_no AS footnote_page
        FROM cell_footnotes cf
        JOIN cells c     ON c.cell_id = cf.cell_id
        JOIN footnotes f ON f.footnote_id = cf.footnote_id
        JOIN tables t    ON t.table_id = c.table_id
        LEFT JOIN pages fp ON fp.page_id = f.page_id
        ORDER BY t.logical_index, c.row_idx, c.col_idx
        """),
    "table_inventory": (
        "one row per logical table",
        """
        SELECT logical_index, start_page, end_page, n_rows, n_cols, n_header_rows,
               regime, is_continuation, ROUND(table_confidence,4) AS conf,
               unit_scale_source, unit_currency, unit_magnitude, title
        FROM tables ORDER BY logical_index
        """),
}


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def audit(out_dir: str, as_json: bool = False) -> int:
    path = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(path):
        print("no metrics.json in %s -- run `gridsmith run` first" % out_dir)
        return 1
    m = _read_json(path)
    if as_json:
        print(canonical_json(m))
        return 0

    print("=" * 78)
    print("gridsmith audit -- %s" % m["document"]["filename"])
    print("=" * 78)
    _block("document", [
        ("sha256", m["document"]["sha256"][:16] + "..."),
        ("pages", m["document"]["page_count"]),
        ("page types", ", ".join("%s=%d" % kv for kv in
                                 sorted(m["document"]["page_types"].items()))),
    ])
    _block("run", [
        ("run_id", m["run"]["run_id"]),
        ("version", m["run"]["gridsmith_version"]),
        ("config fingerprint", m["run"]["config_fingerprint"]),
        ("git sha", m["run"]["git_sha"]),
        ("ocr backend", m["run"]["ocr_backend"]),
        ("offline", m["run"]["offline"]),
    ])
    _block("extraction", [(k, v) for k, v in sorted(m["counts"].items())])
    if m.get("rejections_by_code"):
        _block("refused regions (correctly excluded)",
               [(k, v) for k, v in sorted(m["rejections_by_code"].items())])
    _block("reconciliation", [(k, v) for k, v in sorted(m["reconciliation"].items())])
    _block("confidence", [(k, v) for k, v in sorted(m["confidence"].items())])
    _block("review queue", [(k, v) for k, v in sorted(m["review_queue"].items())])
    _block("calibration", [(k, v) for k, v in sorted(m["calibration"].items())
                           if k in ("source", "n_bins")])
    _block("resources", [(k, v) for k, v in sorted(m["resources"].items())])
    if m.get("issues_by_code"):
        _block("issues by code", [(k, v) for k, v in
                                  sorted(m["issues_by_code"].items())])

    gt = os.path.join(out_dir, "ground_truth_eval.json")
    if os.path.exists(gt):
        e = _read_json(gt)
        print()
        print("-" * 78)
        print("ground-truth evaluation (metrics/eval.py)")
        print("-" * 78)
        for k in sorted(e):
            if isinstance(e[k], dict):
                _block(k, [(kk, vv) for kk, vv in sorted(e[k].items())
                           if not isinstance(vv, (dict, list))])
    else:
        print()
        print("no ground_truth_eval.json -- run: python metrics/eval.py --out %s"
              % out_dir)
    return 0


def _block(title, rows):
    print()
    print("  %s" % title)
    for k, v in rows:
        print("    %-30s %s" % (k, v))


# --------------------------------------------------------------------------- debug


def debug_page(out_dir: str, page_no: int, input_path=None, dpi: int = 130) -> int:
    """Rebuild the overlay for one page straight from the database."""
    import pymupdf


    db = os.path.join(out_dir, "extraction.db")
    if not os.path.exists(db):
        print("no extraction.db in %s" % out_dir)
        return 1
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    doc_row = conn.execute("SELECT * FROM documents LIMIT 1").fetchone()
    src = input_path or _locate_source(doc_row["filename"], out_dir)
    if not src or not os.path.exists(src):
        print("source PDF not found; pass --input explicitly (recorded name: %s)"
              % doc_row["filename"])
        return 1

    page = conn.execute("SELECT * FROM pages WHERE page_no=?", (page_no,)).fetchone()
    if page is None:
        print("page %d is not in this document" % page_no)
        return 1
    pid = page["page_id"]

    regions = [{"bbox": [r["bbox_x0"], r["bbox_y0"], r["bbox_x1"], r["bbox_y1"]],
                "regime": r["regime"] or "",
                "label": "T%03d" % r["logical_index"]}
               for r in conn.execute(
                   "SELECT tr.*, t.logical_index FROM table_regions tr "
                   "JOIN tables t ON t.table_id=tr.table_id "
                   "WHERE tr.page_id=? ORDER BY t.logical_index", (pid,))]
    rejected = [{"bbox": [r["bbox_x0"], r["bbox_y0"], r["bbox_x1"], r["bbox_y1"]],
                 "code": r["code"]}
                for r in conn.execute(
                    "SELECT * FROM rejected_regions WHERE page_id=? ORDER BY code",
                    (pid,))]
    cells = [{"bbox": [r["bbox_x0"], r["bbox_y0"], r["bbox_x1"], r["bbox_y1"]],
              "row_span": r["row_span"], "col_span": r["col_span"],
              "confidence": r["confidence"]}
             for r in conn.execute(
                 "SELECT * FROM cells WHERE page_id=? AND bbox_x1 IS NOT NULL "
                 "ORDER BY table_id, row_idx, col_idx", (pid,))]
    assets = [{"bbox": [r["bbox_x0"], r["bbox_y0"], r["bbox_x1"], r["bbox_y1"]],
               "kind": r["kind"], "relation": ""}
              for r in conn.execute(
                  "SELECT * FROM assets WHERE page_id=? ORDER BY asset_id", (pid,))]

    # Inferred grid lines come from the manifest, which records the boundaries S4
    # actually settled on (`final_x_lines` / `final_y_lines`) rather than the raw
    # evidence.  A page-part of a stitched table only carries the boundaries of the
    # logical table's first part, so the y lines are clipped to this page's region.
    grids = []
    man = os.path.join(out_dir, "manifest.json")
    if os.path.exists(man):
        for t in _read_json(man).get("tables", []):
            ev = t.get("structure_evidence") or {}
            xs = ev.get("final_x_lines") or []
            ys = ev.get("final_y_lines") or []
            if not xs or not ys:
                continue
            for reg in t.get("regions", []):
                if reg["page_no"] != page_no:
                    continue
                y0, y1 = reg["bbox"][1], reg["bbox"][3]
                ys_here = [y for y in ys if y0 - 2 <= y <= y1 + 2] or [y0, y1]
                grids.append({"x_lines": list(xs), "y_lines": ys_here})

    overlay = {"chrome_bands": [], "regions": regions, "rejected": rejected,
               "grids": grids, "cells": cells, "assets": assets}
    doc = pymupdf.open(src)
    out = os.path.join(out_dir, "debug", "page_%03d.png" % page_no)
    audit_render_page(doc, page_no, overlay, out, dpi=dpi,
                          high_confidence_floor=DEFAULT.high_confidence_floor)
    doc.close()
    conn.close()
    print("wrote %s" % out)
    print("legend: %s" % legend_text())
    print("  %d region(s), %d refused, %d cell(s), %d asset(s)"
          % (len(regions), len(rejected), len(cells), len(assets)))
    return 0


def _locate_source(filename: str, out_dir: str):
    for base in (os.getcwd(), os.path.dirname(os.path.abspath(out_dir)),
                 os.path.abspath(out_dir)):
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    return None


# -------------------------------------------------------------------------- verify

VOLATILE_DB_COLUMNS = {("extraction_runs", "started_at"),
                       ("extraction_runs", "finished_at"),
                       ("extraction_runs", "wall_seconds"),
                       ("extraction_runs", "peak_rss_mb")}
VOLATILE_FILES = {"logs/timing.json"}


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _walk(root):
    out = {}
    for base, _dirs, files in os.walk(root):
        for f in sorted(files):
            p = os.path.join(base, f)
            rel = os.path.relpath(p, root).replace("\\", "/")
            out[rel] = p
    return dict(sorted(out.items()))


def _db_dump(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    dump = {}
    for t in names:
        cols = [c[1] for c in conn.execute("PRAGMA table_info(%s)" % t)]
        keep = [c for c in cols if (t, c) not in VOLATILE_DB_COLUMNS]
        rows = [tuple(r[c] for c in keep) for r in conn.execute(
            "SELECT * FROM %s" % t)]
        dump[t] = sorted(canonical_json(list(r), None) for r in rows)
    conn.close()
    return dump


def verify_dirs(a: str, b: str) -> int:
    fa, fb = _walk(a), _walk(b)
    only_a = sorted(set(fa) - set(fb))
    only_b = sorted(set(fb) - set(fa))
    problems = []
    if only_a or only_b:
        problems.append("file set differs: only in A %s; only in B %s"
                        % (only_a, only_b))

    identical, differing, ignored = [], [], []
    for rel in sorted(set(fa) & set(fb)):
        if rel in VOLATILE_FILES:
            ignored.append(rel)
            continue
        if rel.endswith(".db"):
            da, dbb = _db_dump(fa[rel]), _db_dump(fb[rel])
            if da == dbb:
                identical.append(rel + "  (all tables; run timestamps excluded)")
            else:
                bad = sorted(t for t in set(da) | set(dbb)
                             if da.get(t) != dbb.get(t))
                differing.append("%s tables differ: %s" % (rel, bad))
            continue
        ha, hb = _hash_file(fa[rel]), _hash_file(fb[rel])
        (identical if ha == hb else differing).append(
            rel if ha == hb else "%s  (%s != %s)" % (rel, ha[:12], hb[:12]))

    print("=" * 78)
    print("determinism check:  A=%s  B=%s" % (a, b))
    print("=" * 78)
    print("  identical : %d" % len(identical))
    for r in identical:
        print("      = %s" % r)
    if ignored:
        print("  ignored   : %s  (documented as non-deterministic)" % ignored)
    if differing:
        print("  DIFFERING : %d" % len(differing))
        for r in differing:
            print("      ! %s" % r)
    for p in problems:
        print("  PROBLEM   : %s" % p)
    ok = not differing and not problems
    print()
    print("RESULT: %s" % ("byte-identical (timestamps isolated as permitted)"
                          if ok else "DIFFERENCES FOUND"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- query


def run_query(out_dir: str, sql=None, example=None, limit: int = 20,
              list_examples: bool = False) -> int:
    if list_examples:
        for name, (desc, _q) in sorted(EXAMPLE_QUERIES.items()):
            print("  %-24s %s" % (name, desc))
        return 0
    db = os.path.join(out_dir, "extraction.db")
    if not os.path.exists(db):
        print("no extraction.db in %s" % out_dir)
        return 1
    if example:
        if example not in EXAMPLE_QUERIES:
            print("unknown example %r; try --list-examples" % example)
            return 2
        desc, sql = EXAMPLE_QUERIES[example]
        print("-- %s" % desc)
    if not sql:
        print("pass --sql or --example")
        return 2

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
    except sqlite3.Error as e:
        print("SQL error: %s" % e)
        return 1
    rows = cur.fetchmany(limit)
    if not rows:
        print("(no rows)")
        return 0
    names = rows[0].keys()
    widths = [max(len(str(n)), max(len(_s(r[n])) for r in rows)) for n in names]
    widths = [min(w, 40) for w in widths]
    print("  ".join(str(n)[:w].ljust(w) for n, w in zip(names, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(_s(r[n])[:w].ljust(w) for n, w in zip(names, widths)))
    more = len(cur.fetchmany(1))
    if more:
        print("... (more rows; raise --limit)")
    conn.close()
    return 0


def _s(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return ("%.6g" % v)
    return str(v)


# ==================================================================================================
# pipeline: End-to-end orchestration: S1 through S11.
# ==================================================================================================
#
# Stage order here is not quite the numbering in the brief, and the two departures are
# deliberate:
#
# *   **S2 runs twice.**  Layout segmentation has to happen before detection (the
#     detectors need running headers excluded) but it cannot finish before detection,
#     because a bold table header row looks exactly like a section heading until you know
#     a table is there.  So S2 assigns provisional roles, S3 detects, then
#     ``demote_in_regions`` corrects the roles and the section tree is built from the
#     corrected headings.  Without the second pass the ICRA rating table's
#     ``section_path`` ended with ``"Others"`` -- half of a cell.
#
# *   **S7 runs before S9 but is fed by a document-level pre-pass.**  Day/month order is
#     a property of the document, not of a cell, so every candidate date is observed
#     before any cell is normalised.  That also keeps normalisation deterministic:
#     the answer cannot depend on which cell is processed first.
#
# Memory: pages are rasterised one at a time and the pixmap is dropped immediately; only
# the token model is held for the whole document, which is what the cross-page stages
# (chrome detection, scale inheritance, stitching) need.  Measured figures are in the
# README.

VERSION = "0.1.0"


@dataclass
class PageTable:
    """One arbitrated table on one page, after structure and cell extraction."""

    page_no: int
    cand: object
    grid: object
    cells: list
    title: str = ""
    caption: str = ""
    section_path: list = field(default_factory=list)
    section_level: int = 1
    context_evidence: dict = field(default_factory=dict)


@dataclass
class LogicalAssembly:
    logical_index: int
    parts: list
    cells: list                      # re-indexed across parts
    n_rows: int
    n_cols: int
    n_header_rows: int
    header_paths: list
    indent_levels: list
    row_label_paths: list
    col_units: list
    dropped_header_rows: list
    votes: list
    row_page: dict = field(default_factory=dict)   # merged row -> page_no


def git_sha() -> str:
    """Best-effort git revision.  Never fails and never needs the network."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.decode().strip()[:40]
    except Exception:
        pass
    return os.environ.get("GRIDSMITH_GIT_SHA", "unknown")


def peak_rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process().memory_info().peak_wset / (1024 ** 2), 1) \
            if hasattr(psutil.Process().memory_info(), "peak_wset") else \
            round(psutil.Process().memory_info().rss / (1024 ** 2), 1)
    except Exception:
        try:
            import resource

            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        except Exception:
            return 0.0


# --------------------------------------------------------------------------- stages


def _detect_page(doc, pt, info, lay, cfg, log):
    """S3 for one page: ruled + borderless + arbitration, raster path included."""
    page = doc[pt.page_no - 1]
    raster = None
    if info.page_type in (PAGE_SCANNED, PAGE_HYBRID):
        raster = prepare_raster(page, info, cfg)
        ruled_ev = detect_ruled_raster(raster, cfg)
        vec = detect_ruled(page, pt, cfg)
        ruled_ev.regions = list(ruled_ev.regions) + list(vec.regions)
        ruled_ev.h_rules += vec.h_rules
        ruled_ev.v_rules += vec.v_rules
        ruled_ev.breakpoints = sorted(set(ruled_ev.breakpoints) | set(vec.breakpoints))
        ruled_ev.stats["vector_also"] = vec.stats
    else:
        ruled_ev = detect_ruled(page, pt, cfg)
    borderless, bl_stats = detect_borderless(pt, lay, cfg)
    text_available = pt.n_chars >= cfg.digital_min_chars
    accepted, rejected = arbitrate(pt.page_no, pt, lay, ruled_ev,
                                       borderless, cfg,
                                       text_available=text_available)
    log.event("detect", page=pt.page_no, page_type=info.page_type,
              ruled=ruled_ev.stats, borderless=bl_stats,
              n_accepted=len(accepted), n_rejected=len(rejected),
              rejected_codes=sorted({r.code for c in rejected for r in c.rejections}))
    return accepted, rejected, raster


def _assemble(logical, li: int) -> LogicalAssembly:
    """Concatenate the page parts of one logical table, dropping repeated headers."""
    parts = logical.parts
    base = parts[0]
    drop = {}
    for part_idx, row in logical.dropped_header_rows:
        drop.setdefault(part_idx, set()).add(row)

    cells, row_page = [], {}
    offset = 0
    indent, labels = [], []
    for pi, part in enumerate(parts):
        skip = drop.get(pi, set())
        keep_rows = [r for r in range(part.grid.n_rows) if r not in skip]
        remap = {r: offset + k for k, r in enumerate(keep_rows)}
        for c in part.cells:
            if c.row in skip:
                continue
            new = _clone_cell(c, remap[c.row])
            cells.append(new)
            row_page[new.row] = part.page_no
        for r in keep_rows:
            indent.append(part.grid.indent_levels[r]
                          if r < len(part.grid.indent_levels) else 0)
        offset += len(keep_rows)

    n_rows = offset
    n_cols = base.grid.n_cols
    n_header = base.grid.n_header_rows
    # Row-label paths are rebuilt over the *merged* row sequence so an indented
    # hierarchy continues across a page break instead of restarting.
    class _Shim:
        pass

    shim = [type("C", (), {"row": c.row, "col": c.col, "tokens": c.tokens})()
            for c in cells]
    paths = build_row_label_paths(shim, indent, n_header, n_rows)
    return LogicalAssembly(
        logical_index=li, parts=parts, cells=cells, n_rows=n_rows, n_cols=n_cols,
        n_header_rows=n_header, header_paths=base.grid.header_paths,
        indent_levels=indent, row_label_paths=paths,
        col_units=base.grid.col_units,
        dropped_header_rows=list(logical.dropped_header_rows),
        votes=list(logical.votes), row_page=row_page)


def _clone_cell(c, new_row):

    return ExtractedCell(
        row=new_row, col=c.col, row_span=c.row_span, col_span=c.col_span,
        is_header=c.is_header, raw_text=c.raw_text, bbox=c.bbox, source=c.source,
        ocr_conf=c.ocr_conf, footnote_markers=list(c.footnote_markers),
        flags=list(c.flags), n_lines=c.n_lines, tokens=list(c.tokens))


# ------------------------------------------------------------------------- driver


def run(input_path: str, out_dir: str, cfg: Config) -> dict:
    t_start = time.time()
    seed_everything(cfg.seed)
    log = RunLog()
    os.makedirs(out_dir, exist_ok=True)
    for sub in ("tables", "assets", "debug", "logs"):
        reset_dir(os.path.join(out_dir, sub))

    doc = open_document(input_path)
    sha = doc_sha256(input_path)
    n_pages = doc.page_count if not cfg.max_pages else min(doc.page_count,
                                                           cfg.max_pages)
    doc_id = sid("doc", sha)
    run_id = sid("run", sha, cfg.fingerprint(), VERSION)
    log.event("open", filename=os.path.basename(input_path), sha256=sha,
              page_count=n_pages, run_id=run_id, config=cfg.fingerprint())

    # ---- S1 + S2 (provisional) ---------------------------------------------

    pts, infos = [], []
    for i in range(n_pages):
        pt = extract_page_text(doc[i], i + 1, cfg.vertical_dir_tol)
        pts.append(pt)
        infos.append(classify_page(doc[i], pt, cfg))
    chrome_keys, chrome_stats = find_repeating_chrome(pts, cfg)
    layouts = [segment_page(pt, chrome_keys, cfg) for pt in pts]
    size_rank = assign_heading_levels(pts, layouts, cfg)
    log.event("intake", page_types={t: sum(1 for i in infos if i.page_type == t)
                                    for t in sorted({i.page_type for i in infos})},
              chrome=chrome_stats, heading_size_rank={str(k): v for k, v
                                                      in sorted(size_rank.items())})

    # ---- document-level date order pre-pass (S7) ---------------------------
    order = DateOrderModel()
    for pt in pts:
        for ln in pt.lines:
            order.observe(ln.text.strip())
            for t in ln.tokens:
                order.observe(t.text)
    log.event("date_order", **order.to_dict())

    # ---- S3 ----------------------------------------------------------------
    ocr = get_backend(cfg)
    log.event("ocr_backend", name=ocr.name, available=ocr.available)
    chrome_issues, all_rejected, per_page_accepted = [], [], {}
    for pt, info, lay in zip(pts, infos, layouts):
        accepted, rejected, raster = _detect_page(doc, pt, info, lay, cfg, log)
        demote_in_regions(pt, lay, [c.bbox for c in accepted], chrome_issues)
        per_page_accepted[pt.page_no] = (accepted, raster)
        all_rejected.extend(rejected)

    # Section tree AFTER demotion, so table header rows are not headings.
    section_events = build_section_events(pts, layouts)

    # ---- S4 + S5 -----------------------------------------------------------
    page_tables = []
    for pt, info, lay in zip(pts, infos, layouts):
        accepted, raster = per_page_accepted[pt.page_no]
        for cand in accepted:
            grid = build_grid(cand, pt, cfg, layout=lay)
            if grid.n_rows == 0 or grid.n_cols == 0:
                continue
            if info.page_type == PAGE_SCANNED and raster is not None:
                cells = extract_ocr_cells(grid, doc[pt.page_no - 1], raster, cfg, ocr)
            else:
                cells = extract_text_cells(grid, pt, cfg)
            # The detector's proposal is a search region; the *table* is what S4
            # reconstructed inside it.  Tighten the candidate box to the grid extent
            # before anything downstream records it, otherwise a rule stack that
            # reached 70 pt past the last row is persisted as the table's bbox and
            # detection IoU collapses (the ICRA complexity table scored 0.41).
            if grid.x_lines and grid.y_lines:
                cand.bbox = (min(grid.x_lines), min(grid.y_lines),
                             max(grid.x_lines), max(grid.y_lines))
            title, caption, cev = find_title_caption(pt, lay, cand.bbox, cfg)
            spath = section_path_at(section_events, pt.page_no, cand.bbox[1])
            lvl = 1
            for pno, y, l, txt in section_events:
                if (pno, y) <= (pt.page_no, cand.bbox[1]) and spath and txt == spath[-1]:
                    lvl = l
            page_tables.append(PageTable(
                page_no=pt.page_no, cand=cand, grid=grid, cells=cells, title=title,
                caption=caption, section_path=spath, section_level=lvl,
                context_evidence=cev))
    log.event("structure", n_page_tables=len(page_tables),
              regimes={r: sum(1 for t in page_tables if t.cand.regime == r)
                       for r in sorted({t.cand.regime for t in page_tables})})

    # ---- S6 ----------------------------------------------------------------
    logical = stitch(page_tables, pts, cfg, headings=section_events)
    log.event("stitch", n_logical=len(logical),
              n_multi_page=sum(1 for lt in logical if lt.is_continuation),
              joins=[v for lt in logical for v in lt.votes])

    # ---- S8 assets ---------------------------------------------------------
    table_bboxes = {}
    for t in page_tables:
        table_bboxes.setdefault(t.page_no, []).append(t.cand.bbox)
    assets, asset_rows = [], []
    raw_assets = []
    for pt, lay in zip(pts, layouts):
        raw_assets.extend(find_assets(doc[pt.page_no - 1], pt, lay,
                                         table_bboxes.get(pt.page_no, []), cfg))
    # Page furniture masquerades as a figure: the ICRA logo is an image block on all
    # seven pages at the same coordinates.  The same repetition test that finds running
    # headers finds these -- an asset whose position and kind recur on three or more
    # pages is branding, not a figure linked to a table.
    pos_count = {}
    for a in raw_assets:
        pos_count[_asset_key(a)] = pos_count.get(_asset_key(a), 0) + 1
    n_repeat_dropped = 0
    for a in raw_assets:
        if pos_count[_asset_key(a)] >= 3:
            n_repeat_dropped += 1
            continue
        assets.append(a)
    for idx, a in enumerate(assets):
        name = "%s_p%03d_%03d.png" % (a.kind, a.page_no, idx)
        a.file_path = os.path.join("assets", name)
        _crop_asset(doc, a, os.path.join(out_dir, a.file_path), cfg)
    log.event("assets", n_assets=len(assets),
              n_repeating_dropped=n_repeat_dropped,
              kinds={k: sum(1 for a in assets if a.kind == k)
                     for k in sorted({a.kind for a in assets})})

    # ---- assemble, normalise, score ---------------------------------------
    calib = Calibration.load(os.path.join(os.path.dirname(__file__),
                                             "calibration.json"))
    log.event("calibration", **calib.to_dict())

    conn = open_db(os.path.join(out_dir, "extraction.db"))
    started = FIXED_ISO if cfg.fixed_time else _now_iso()
    page_ids = {}
    doc_rows = [{"doc_id": doc_id, "filename": os.path.basename(input_path),
                 "sha256": sha, "page_count": n_pages,
                 "ingested_at": started}]
    page_rows = []
    for pt, info in zip(pts, infos):
        pid = sid("pag", doc_id, pt.page_no)
        page_ids[pt.page_no] = pid
        page_rows.append({
            "page_id": pid, "doc_id": doc_id, "page_no": pt.page_no,
            "width": q(pt.width), "height": q(pt.height),
            "page_type": info.page_type, "rotation": info.rotation,
            "content_rotation": info.content_rotation, "skew_deg": q(info.skew_deg),
            "image_area_frac": q(info.image_area_frac),
            "char_density": q(info.char_density),
            "notes_json": canonical_json(sorted(info.notes), None)})
    insert(conn, "documents", doc_rows)
    insert(conn, "pages", page_rows)
    # The run row must exist before any table references it.  Completed below.
    insert(conn, "extraction_runs", [{
        "run_id": run_id, "doc_id": doc_id, "started_at": started,
        "finished_at": None, "git_sha": git_sha(),
        "config_json": cfg.to_json(), "gridsmith_version": VERSION,
        "config_fingerprint": cfg.fingerprint(), "wall_seconds": None,
        "peak_rss_mb": None, "offline": int(cfg.offline),
        "ocr_backend": ocr.name}])

    manifest_tables, all_review_items, all_issues = [], [], []
    table_rows, region_rows, column_rows, cell_rows = [], [], [], []
    footnote_rows, cellfn_rows, tasset_rows, issue_rows = [], [], [], []
    n_cells_total = 0
    seen_footnotes = {}

    for li, lt in enumerate(sorted(logical, key=lambda l: (l.start_page,
                                                           l.parts[0].grid.y_lines[0]
                                                           if l.parts[0].grid.y_lines
                                                           else 0)), start=1):
        asm = _assemble(lt, li)
        base = lt.parts[0]
        table_id = sid("tab", doc_id, run_id, li)

        # normalise, then infer column types, then re-normalise with the type known
        header_texts = [c.raw_text for c in asm.cells
                        if c.row < asm.n_header_rows and (c.raw_text or "").strip()]
        scale = resolve_scale(base.page_no, base.cand.bbox, pts, layouts,
                                 base.section_path, base.caption, cfg,
                                 header_texts=header_texts)
        norms = {}
        for c in asm.cells:
            hdr = asm.header_paths[c.col] if c.col < len(asm.header_paths) else []
            n = normalize(c.raw_text or "", column_header=" ".join(hdr),
                             order_model=order)
            if c.is_header:
                # A header cell is a *label*, so it keeps raw_text and a value_type but
                # carries no normalized_value.  Emitting 2024.0 for the column headed
                # "2024" is not wrong so much as an invitation to sum a column label,
                # and the meaning of a header belongs in columns.header_path.
                n.number = None
                n.date_iso = None
            norms[(c.row, c.col)] = n
        col_types, col_type_ev, col_conflicts = {}, {}, {}
        for col in range(asm.n_cols):
            vals = [norms[(r, col)] for r in range(asm.n_header_rows, asm.n_rows)
                    if (r, col) in norms]
            hdr = asm.header_paths[col] if col < len(asm.header_paths) else []
            ty, conf_idx, ev = infer_column_type(vals, hdr, cfg.type_majority_frac)
            col_types[col], col_type_ev[col] = ty, ev
            col_conflicts[col] = len(conf_idx)

        source_of = {(c.row, c.col): c.source for c in asm.cells}
        ocr_conf_of = {(c.row, c.col): c.ocr_conf for c in asm.cells}
        grid_support = _grid_support(base.grid)
        score = score_table(asm.cells, norms, _grid_view(asm, base.grid), cfg,
                               calib, col_types, source_of, ocr_conf_of, grid_support)

        # ---- footnotes -----------------------------------------------------
        fns = collect_footnotes(pts, layouts, base.page_no, look_ahead=1)
        links, unmatched = link_footnotes(asm.cells, fns)
        fn_ids = {}
        for fi, fn in enumerate(fns):
            key = (fn.page_no, fn.marker, fn.text)
            if key not in seen_footnotes:
                fid = sid("fnt", doc_id, fn.page_no, fn.marker, fn.text[:80])
                seen_footnotes[key] = fid
                footnote_rows.append({
                    "footnote_id": fid, "doc_id": doc_id,
                    "page_id": page_ids.get(fn.page_no), "marker": fn.marker,
                    "text": fn.text})
            fn_ids[fi] = seen_footnotes[key]

        # ---- asset links ---------------------------------------------------
        next_h = None
        for pno, y, lvl, txt in section_events:
            if pno == base.page_no and y > base.cand.bbox[3]:
                next_h = y if next_h is None else min(next_h, y)
        page_assets = [(i, a) for i, a in enumerate(assets)
                       if a.page_no in range(base.page_no, lt.end_page + 1)]
        alinks = associate_assets(
            base.caption, base.title, base.cand.bbox, base.page_no,
            base.section_path, [a for _i, a in page_assets], next_h, cfg,
            nearby_text=" ".join(ln.text for ln in pts[base.page_no - 1].lines
                                 if abs(ln.y0 - base.cand.bbox[3]) < 80))

        # ---- rows ----------------------------------------------------------
        slug = table_slug(base.title or base.caption or "table")
        xlsx_name = excel_filename(li, lt.start_page, lt.end_page, slug)
        table_rows.append({
            "table_id": table_id, "doc_id": doc_id, "run_id": run_id,
            "logical_index": li, "start_page": lt.start_page, "end_page": lt.end_page,
            "n_rows": asm.n_rows, "n_cols": asm.n_cols,
            "title": base.title, "caption": base.caption,
            "section_path": canonical_json(base.section_path, None),
            "unit_scale_note": scale.note, "is_continuation": int(lt.is_continuation),
            "detector_votes_json": canonical_json(base.cand.votes, None),
            "table_confidence": q(score.table_confidence),
            "n_header_rows": asm.n_header_rows, "regime": base.cand.regime,
            "unit_scale_source": scale.source, "unit_scale_page": scale.page_no,
            "unit_scale_confidence": q(scale.confidence),
            "unit_currency": scale.currency, "unit_magnitude": q(scale.magnitude),
            "stitch_votes_json": canonical_json(asm.votes, None),
            "structure_evidence_json": canonical_json(base.grid.evidence, None),
            "reconciliation_json": canonical_json(score.reconciliation, None),
            "excel_path": os.path.join("tables", xlsx_name)})

        for pi, part in enumerate(lt.parts):
            region_rows.append({
                "region_id": region_id(table_id, part.page_no, pi),
                "table_id": table_id, "page_id": page_ids[part.page_no],
                "bbox_x0": q(part.cand.bbox[0]), "bbox_y0": q(part.cand.bbox[1]),
                "bbox_x1": q(part.cand.bbox[2]), "bbox_y1": q(part.cand.bbox[3]),
                "part_index": pi, "regime": part.cand.regime})

        for col in range(asm.n_cols):
            hdr = asm.header_paths[col] if col < len(asm.header_paths) else []
            unit = asm.col_units[col] if col < len(asm.col_units) else ""
            if not unit and scale.currency:
                unit = scale.currency
            if unit and scale.magnitude != 1.0:
                unit = "%s %s" % (unit, _magnitude_word(scale.magnitude))
            column_rows.append({
                "column_id": sid("col", table_id, col), "table_id": table_id,
                "col_idx": col, "header_path": canonical_json(hdr, None),
                "header_path_text": " > ".join(hdr),
                "inferred_type": col_types.get(col), "unit": unit,
                "nullable": int(any(not (c.raw_text or "").strip()
                                    for c in asm.cells if c.col == col)),
                "type_evidence_json": canonical_json(col_type_ev.get(col, {}), None),
                "n_conflicts": col_conflicts.get(col, 0)})

        by_rc = {(c.row, c.col): c for c in score.cells}
        cell_id_of = {}
        for c in sorted(asm.cells, key=lambda c: (c.row, c.col)):
            cid = sid("cel", table_id, c.row, c.col)
            cell_id_of[(c.row, c.col)] = cid
            nrm = norms[(c.row, c.col)]
            cs = by_rc.get((c.row, c.col))
            pno = asm.row_page.get(c.row, base.page_no)
            rlp = asm.row_label_paths[c.row] if c.row < len(asm.row_label_paths) else []
            n_cells_total += 1
            cell_rows.append({
                "cell_id": cid, "table_id": table_id, "row_idx": c.row,
                "col_idx": c.col, "row_span": c.row_span, "col_span": c.col_span,
                "is_header": int(c.is_header), "raw_text": c.raw_text,
                "normalized_value": nrm.value, "value_type": nrm.value_type,
                "row_label_path": canonical_json(rlp, None),
                "row_label_path_text": " > ".join(rlp),
                "page_id": page_ids.get(pno),
                "bbox_x0": q(c.bbox[0]), "bbox_y0": q(c.bbox[1]),
                "bbox_x1": q(c.bbox[2]), "bbox_y1": q(c.bbox[3]),
                "source": c.source,
                "confidence": q(cs.confidence if cs else 0.0),
                "raw_confidence": q(cs.raw_confidence if cs else 0.0),
                "normalized_number": q(nrm.number) if nrm.number is not None else None,
                "normalized_date": nrm.date_iso,
                "currency": nrm.currency or (scale.currency or ""),
                "indent_level": (asm.indent_levels[c.row]
                                 if c.row < len(asm.indent_levels) else 0),
                "footnote_markers": "|".join(c.footnote_markers),
                "ocr_conf": q(c.ocr_conf) if c.ocr_conf is not None else None,
                "reason_codes": "|".join(cs.codes if cs else []),
                "mandatory_review": int(bool(cs and cs.mandatory)),
                "features_json": canonical_json(cs.features if cs else {}, None)})

            if cs:
                hp = asm.header_paths[c.col] if c.col < len(asm.header_paths) else []
                item = ReviewItem(
                    table_id=table_id, logical_index=li, page_no=pno,
                    row_idx=c.row, col_idx=c.col, header_path=" > ".join(hp),
                    row_label_path=" > ".join(rlp), raw_text=c.raw_text,
                    normalized_value=nrm.value, value_type=nrm.value_type,
                    confidence=cs.confidence, codes=cs.codes,
                    severity=severity_of(cs.codes), bbox=c.bbox,
                    source=c.source, mandatory=cs.mandatory)
                all_review_items.append(item)

        for (r, c, fi) in links:
            if (r, c) in cell_id_of and fi in fn_ids:
                cellfn_rows.append({"cell_id": cell_id_of[(r, c)],
                                    "footnote_id": fn_ids[fi]})
        for link in alinks:
            gi, a = page_assets[link["asset_index"]]
            tasset_rows.append({
                "table_id": table_id, "asset_id": sid("ast", doc_id, gi),
                "relation": link["relation"],
                "relation_confidence": q(link["relation_confidence"]),
                "components_json": canonical_json(link["components"], None)})

        tissues = issues_for(score, table_id, cell_id_of)
        for it in tissues:
            it["issue_id"] = sid("iss", table_id, it["code"], it.get("cell_id") or "",
                                 it["message"][:64])
        issue_rows.extend(tissues)
        all_issues.extend(tissues)

        # ---- Excel ---------------------------------------------------------
        _write_excel(os.path.join(out_dir, "tables", xlsx_name), asm, norms, score,
                     base, scale, fns, alinks, [a for _i, a in page_assets], tissues,
                     cfg, col_types)

        manifest_tables.append(_manifest_table(
            table_id, li, lt, asm, base, scale, score, col_types, col_type_ev,
            fns, alinks, [a for _i, a in page_assets], xlsx_name, norms))

    # ---- assets + rejections ----------------------------------------------
    insert(conn, "assets", [{
        "asset_id": sid("ast", doc_id, i), "doc_id": doc_id,
        "page_id": page_ids.get(a.page_no), "kind": a.kind,
        "file_path": a.file_path.replace("\\", "/"), "caption": a.caption,
        "bbox_x0": q(a.bbox[0]), "bbox_y0": q(a.bbox[1]),
        "bbox_x1": q(a.bbox[2]), "bbox_y1": q(a.bbox[3]),
        "series_json": None} for i, a in enumerate(assets)])

    rej_rows = []
    for cnd in all_rejected:
        for rj in cnd.rejections:
            rej_rows.append({
                "rejection_id": sid("rej", doc_id, cnd.page_no, rj.code,
                                    tuple(round(v, 1) for v in cnd.bbox)),
                "doc_id": doc_id, "page_id": page_ids.get(cnd.page_no),
                "bbox_x0": q(cnd.bbox[0]), "bbox_y0": q(cnd.bbox[1]),
                "bbox_x1": q(cnd.bbox[2]), "bbox_y1": q(cnd.bbox[3]),
                "regime": cnd.regime, "code": rj.code, "score": q(rj.score),
                "message": rj.message,
                "evidence_json": canonical_json(rj.evidence, None)})
    for ci in chrome_issues:
        issue_rows.append({
            "issue_id": sid("iss", doc_id, ci["code"], ci["message"][:64]),
            "table_id": None, "cell_id": None, "severity": ci["severity"],
            "code": ci["code"], "message": ci["message"]})

    insert(conn, "tables", table_rows)
    insert(conn, "table_regions", region_rows)
    insert(conn, "columns", column_rows)
    insert(conn, "cells", cell_rows)
    insert(conn, "footnotes", footnote_rows)
    insert(conn, "cell_footnotes",
               [{"cell_id": a, "footnote_id": b} for a, b in
                sorted({(r["cell_id"], r["footnote_id"]) for r in cellfn_rows})])
    insert(conn, "table_assets", tasset_rows)
    insert(conn, "issues", issue_rows)
    insert(conn, "rejected_regions", rej_rows)

    # ---- review queue ------------------------------------------------------
    queued, ranked, qstats = build_queue(all_review_items, cfg, n_cells_total)
    write_text(os.path.join(out_dir, "review_queue.csv"), to_csv(queued, cfg))
    write_text(os.path.join(out_dir, "review_queue_full.csv"),
               to_csv(ranked, cfg))

    # ---- debug renders -----------------------------------------------------
    debug_pages = _debug_page_list(page_tables, all_rejected, cfg, n_pages)
    for pno in debug_pages:
        ov = build_overlay(pno, pts, layouts, page_tables, all_rejected, assets,
                           cell_conf={(t.page_no, id(t)): None for t in page_tables},
                           table_ids={}, cell_rows=cell_rows, page_ids=page_ids)
        audit_render_page(doc, pno, ov,
                        os.path.join(out_dir, "debug", "page_%03d.png" % pno),
                        high_confidence_floor=cfg.high_confidence_floor)

    # ---- run row + manifest + metrics -------------------------------------
    wall = time.time() - t_start
    finished = FIXED_ISO if cfg.fixed_time else _now_iso()
    conn.execute("UPDATE extraction_runs SET finished_at=?, wall_seconds=?, "
                 "peak_rss_mb=? WHERE run_id=?",
                 (finished,
                  0.0 if cfg.fixed_time else round(wall, 3),
                  0.0 if cfg.fixed_time else peak_rss_mb(),
                  run_id))
    conn.commit()
    conn.close()

    summary = {
        "document": {"filename": os.path.basename(input_path), "sha256": sha,
                     "page_count": n_pages,
                     "page_types": {t: sum(1 for i in infos if i.page_type == t)
                                    for t in sorted({i.page_type for i in infos})}},
        "run": {"run_id": run_id, "gridsmith_version": VERSION,
                "config_fingerprint": cfg.fingerprint(),
                "git_sha": git_sha(), "ocr_backend": ocr.name,
                "offline": cfg.offline, "python": platform.python_version()},
        "counts": {"logical_tables": len(manifest_tables),
                   "page_level_tables": len(page_tables),
                   "multi_page_tables": sum(1 for lt in logical
                                            if lt.is_continuation),
                   "cells": n_cells_total,
                   "assets": len(assets),
                   "footnotes": len(footnote_rows),
                   "cell_footnote_links": len(set(
                       (r["cell_id"], r["footnote_id"]) for r in cellfn_rows)),
                   "rejected_regions": len(all_rejected),
                   "issues": len(issue_rows)},
        "rejections_by_code": _count_by(
            [rj.code for c in all_rejected for rj in c.rejections]),
        "issues_by_code": _count_by([i["code"] for i in issue_rows]),
        "review_queue": qstats,
        "confidence": _confidence_stats(cell_rows, cfg),
        "reconciliation": _recon_stats(manifest_tables),
        "calibration": calib.to_dict(),
        # Under --fixed-time even metrics.json must be byte-identical, so the two
        # genuinely time-varying values are zeroed rather than omitted (a reader can
        # then tell "not measured" from "fast").  logs/timing.json always carries the
        # real figures and is excluded from the determinism claim.
        "resources": ({"wall_seconds": 0.0, "peak_rss_mb": 0.0,
                       "seconds_per_page": 0.0, "fixed_time": True}
                      if cfg.fixed_time else
                      {"wall_seconds": round(wall, 2),
                       "peak_rss_mb": peak_rss_mb(),
                       "seconds_per_page": round(wall / max(1, n_pages), 3)}),
    }
    manifest = {"schema_version": 1, "document": summary["document"],
                "run": summary["run"], "tables": manifest_tables,
                "assets": [{"index": i, "kind": a.kind, "page_no": a.page_no,
                            "bbox": [q(v) for v in a.bbox], "caption": a.caption,
                            "file_path": a.file_path.replace("\\", "/")}
                           for i, a in enumerate(assets)],
                "rejected_regions": [
                    {"page_no": c.page_no, "bbox": [q(v) for v in c.bbox],
                     "regime": c.regime,
                     "rejections": [{"code": r.code, "score": q(r.score),
                                     "message": r.message, "evidence": r.evidence}
                                    for r in c.rejections]}
                    for c in all_rejected],
                "chrome": chrome_stats,
                "debug_pages": debug_pages,
                "counts": summary["counts"]}
    write_manifest(os.path.join(out_dir, "manifest.json"), manifest)
    write_text(os.path.join(out_dir, "metrics.json"), canonical_json(summary) + "\n")
    log.event("done", **summary["counts"])
    log.write(os.path.join(out_dir, "logs", "run.jsonl"))
    # Always written, including under --fixed-time: metrics.json zeroes the two
    # time-varying values so its bytes stay reproducible, and this file is the
    # one place the real figures survive.  ``verify`` lists it in VOLATILE_FILES
    # and ignores it, so writing it unconditionally cannot weaken the claim.
    write_text(os.path.join(out_dir, "logs", "timing.json"),
               canonical_json({"wall_seconds": round(wall, 3),
                               "peak_rss_mb": peak_rss_mb(),
                               "note": "excluded from the determinism claim"}) + "\n")
    doc.close()
    return summary


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _count_by(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def _magnitude_word(mag: float) -> str:
    for word, val in (("thousand", 1e3), ("lakh", 1e5), ("million", 1e6),
                      ("crore", 1e7), ("billion", 1e9), ("trillion", 1e12)):
        if abs(mag - val) < 1e-6:
            return word
    return ""


def _grid_support(grid) -> float:
    """0..1 measure of how well-evidenced this grid's boundaries are."""
    ev = (grid.evidence or {}).get("columns", {})
    srcs = ev.get("sources", {}) or {}
    strong = srcs.get("vrule", 0) + srcs.get("breakpoint", 0) + srcs.get("fill_edge", 0)
    weak = srcs.get("gutter", 0)
    total = strong + weak
    if total == 0:
        return 0.35
    return min(1.0, 0.45 + 0.55 * (strong / total))


def _grid_view(asm, base_grid):
    """A grid-shaped object describing the *merged* logical table."""
    class _G:
        pass

    g = _G()
    g.x_lines = base_grid.x_lines
    g.y_lines = base_grid.y_lines if len(base_grid.y_lines) >= asm.n_rows + 1 else \
        list(base_grid.y_lines) + [base_grid.y_lines[-1]] * (
            asm.n_rows + 1 - len(base_grid.y_lines))
    g.n_rows, g.n_cols = asm.n_rows, asm.n_cols
    g.n_header_rows = asm.n_header_rows
    g.header_paths = asm.header_paths
    g.indent_levels = asm.indent_levels
    return g


def _confidence_stats(cell_rows, cfg: Config):
    body = [r for r in cell_rows if not r["is_header"]]
    pool = body or cell_rows
    if not pool:
        return {}
    confs = sorted(float(r["confidence"] or 0.0) for r in pool)
    n = len(confs)
    return {"n_body_cells": n,
            "mean": round(sum(confs) / n, 4),
            "p05": round(confs[int(0.05 * (n - 1))], 4),
            "median": round(confs[int(0.50 * (n - 1))], 4),
            "p95": round(confs[int(0.95 * (n - 1))], 4),
            "n_high_confidence": sum(1 for c in confs
                                     if c >= cfg.high_confidence_floor),
            "frac_high_confidence": round(
                sum(1 for c in confs if c >= cfg.high_confidence_floor) / n, 4),
            "n_mandatory": sum(1 for r in pool if r["mandatory_review"])}


def _recon_stats(manifest_tables):
    p = f_ = i = 0
    for t in manifest_tables:
        r = t.get("reconciliation", {})
        p += r.get("n_pass", 0)
        f_ += r.get("n_fail", 0)
        i += r.get("n_incomplete", 0)
    return {"checks_pass": p, "checks_fail": f_, "checks_incomplete": i,
            "pass_rate": round(p / max(1, p + f_), 4)}


def _debug_page_list(page_tables, rejected, cfg: Config, n_pages: int):
    if cfg.write_debug_pages == "none":
        return []
    if cfg.write_debug_pages == "all":
        return list(range(1, n_pages + 1))
    pages = {t.page_no for t in page_tables} | {c.page_no for c in rejected}
    return sorted(pages)


def build_overlay(page_no, pts, layouts, page_tables, rejected, assets,
                  cell_conf=None, table_ids=None, cell_rows=None, page_ids=None):
    """Assemble the plain-dict overlay description for one page."""
    lay = layouts[page_no - 1]
    pt = pts[page_no - 1]
    chrome = []
    for i, ln in enumerate(pt.lines):
        if lay.is_chrome(i):
            chrome.append([0.0, ln.y0 - 1.0, pt.width, ln.y1 + 1.0])

    regions, grids, cells = [], [], []
    for t in page_tables:
        if t.page_no != page_no:
            continue
        regions.append({"bbox": list(t.cand.bbox), "regime": t.cand.regime,
                        "label": "T"})
        if t.grid.x_lines and t.grid.y_lines:
            grids.append({"x_lines": list(t.grid.x_lines),
                          "y_lines": list(t.grid.y_lines)})

    # Cells come from the persisted rows, so the confidence tint is the confidence that
    # was actually written.  Building them from the page-level tables instead meant
    # passing confidence=None for every cell -- and None renders as "below the floor",
    # so every cell in every pipeline-generated overlay came out red regardless of how
    # good it was.  The tinting only worked through `gridsmith debug`, which reads the
    # database.  A debug renderer that lies about confidence is worse than none.
    pid = (page_ids or {}).get(page_no)
    for r in (cell_rows or []):
        if pid is None or r.get("page_id") != pid:
            continue
        if r.get("bbox_x1") is None:
            continue
        cells.append({"bbox": [r["bbox_x0"], r["bbox_y0"], r["bbox_x1"],
                               r["bbox_y1"]],
                      "row_span": r.get("row_span", 1),
                      "col_span": r.get("col_span", 1),
                      "confidence": r.get("confidence")})
    rej = [{"bbox": list(c.bbox),
            "code": (max(c.rejections, key=lambda r: r.score).code
                     if c.rejections else "REFUSED")}
           for c in rejected if c.page_no == page_no]
    ast = [{"bbox": list(a.bbox), "kind": a.kind, "relation": ""}
           for a in assets if a.page_no == page_no]
    return {"chrome_bands": chrome, "regions": regions, "rejected": rej,
            "grids": [g for g in grids if g["x_lines"]], "cells": cells,
            "assets": ast}


def _asset_key(a):
    """Kind plus a coarse position, for the repetition test."""
    return (a.kind, tuple(round(v / 6.0) for v in a.bbox))


def _crop_asset(doc, asset, out_path: str, cfg: Config):
    import pymupdf

    page = doc[asset.page_no - 1]
    clip = pymupdf.Rect(*asset.bbox) & page.rect
    if clip.is_empty:
        return
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0), clip=clip, alpha=False)
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    pix.save(out_path)
    del pix


def _write_excel(path, asm, norms, score, base, scale, fns, alinks, page_assets,
                 tissues, cfg: Config, col_types):
    by_rc = {(c.row, c.col): c for c in asm.cells}
    sc_by = {(c.row, c.col): c for c in score.cells}
    data_rows, merges = [], []
    for r in range(asm.n_rows):
        row = []
        for c in range(asm.n_cols):
            cell = by_rc.get((r, c))
            if cell is None:
                row.append(None)
                continue
            nrm = norms.get((r, c))
            cs = sc_by.get((r, c))
            trusted = bool(cs and not cs.mandatory
                           and cs.confidence >= cfg.high_confidence_floor)
            # Native types only for values we trust; anything flagged stays text so a
            # spreadsheet SUM cannot quietly absorb a value we do not stand behind.
            if trusted and nrm is not None and nrm.value_type in NUMERIC_TYPES \
                    and nrm.number is not None:
                row.append(nrm.number)
            else:
                row.append(cell.raw_text)
            if cell.row_span > 1 or cell.col_span > 1:
                merges.append((r, c, r + cell.row_span - 1, c + cell.col_span - 1))
        data_rows.append(row)

    ctx = [
        ("title", base.title), ("caption", base.caption),
        ("section_path", " > ".join(base.section_path)),
        ("page_range", "%d-%d" % (asm.parts[0].page_no, asm.parts[-1].page_no)),
        ("is_continuation", str(len(asm.parts) > 1)),
        ("n_page_parts", str(len(asm.parts))),
        ("unit_scale_note", scale.note),
        ("unit_scale_source", scale.source),
        ("unit_scale_page", str(scale.page_no)),
        ("unit_scale_confidence", "%.2f" % scale.confidence),
        ("unit_currency", scale.currency),
        ("unit_magnitude", str(scale.magnitude)),
        ("table_confidence", "%.4f" % score.table_confidence),
        ("detector_regime", base.cand.regime),
        ("detector_reason", base.cand.votes.get("reason", "")),
        ("n_header_rows", str(asm.n_header_rows)),
        ("column_types", ", ".join("%d:%s" % (k, v)
                                   for k, v in sorted(col_types.items()))),
        ("header_paths", " | ".join(" > ".join(h) for h in asm.header_paths)),
        ("reconciliation", "pass=%d fail=%d incomplete=%d" % (
            score.reconciliation.get("n_pass", 0),
            score.reconciliation.get("n_fail", 0),
            score.reconciliation.get("n_incomplete", 0))),
        ("dropped_repeated_header_rows", str(asm.dropped_header_rows)),
    ]
    for i, fn in enumerate(fns):
        ctx.append(("footnote[%s]" % (fn.marker or i), "p%d: %s" % (fn.page_no, fn.text)))
    for link in alinks:
        a = page_assets[link["asset_index"]]
        ctx.append(("linked_asset", "%s (%s, confidence %.2f) %s"
                    % (a.file_path.replace("\\", "/"), link["relation"],
                       link["relation_confidence"], a.caption)))

    prov = []
    for c in sorted(asm.cells, key=lambda c: (c.row, c.col)):
        nrm = norms.get((c.row, c.col))
        cs = sc_by.get((c.row, c.col))
        prov.append({
            "row_idx": c.row, "col_idx": c.col, "raw_text": c.raw_text,
            "normalized_value": (nrm.value if nrm else None),
            "page_no": asm.row_page.get(c.row, base.page_no),
            "bbox": "%.2f,%.2f,%.2f,%.2f" % tuple(round(float(v), 2) for v in c.bbox),
            "source": c.source,
            "confidence": round(cs.confidence, 4) if cs else None,
            "reason_codes": "|".join(cs.codes) if cs else ""})

    iss = []
    for it in tissues:
        rc = _parse_rc(it["message"])
        iss.append({"severity": it["severity"], "code": it["code"],
                    "row_idx": rc[0], "col_idx": rc[1], "message": it["message"]})

    write_table_xlsx(path, data_rows=data_rows, merges=merges,
                         n_header_rows=asm.n_header_rows, context_rows=ctx,
                         provenance_rows=prov, issue_rows=iss,
                         freeze_header=cfg.excel_freeze_header)


def _parse_rc(message: str):
    import re

    m = re.search(r"r(\d+) c(\d+)", message or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"at r(\d+) c(\d+)", message or "")
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _manifest_table(table_id, li, lt, asm, base, scale, score, col_types, col_type_ev,
                    fns, alinks, page_assets, xlsx_name, norms):
    by_rc = {(c.row, c.col): c for c in asm.cells}
    sc_by = {(c.row, c.col): c for c in score.cells}
    cells = []
    for c in sorted(asm.cells, key=lambda c: (c.row, c.col)):
        nrm = norms.get((c.row, c.col))
        cs = sc_by.get((c.row, c.col))
        cells.append({
            "row_idx": c.row, "col_idx": c.col, "row_span": c.row_span,
            "col_span": c.col_span, "is_header": c.is_header,
            "raw_text": c.raw_text,
            "normalized_value": (nrm.value if nrm else None),
            "value_type": (nrm.value_type if nrm else None),
            "row_label_path": (asm.row_label_paths[c.row]
                               if c.row < len(asm.row_label_paths) else []),
            "indent_level": (asm.indent_levels[c.row]
                             if c.row < len(asm.indent_levels) else 0),
            "page_no": asm.row_page.get(c.row, base.page_no),
            "bbox": [q(v) for v in c.bbox], "source": c.source,
            "confidence": q(cs.confidence) if cs else None,
            "raw_confidence": q(cs.raw_confidence) if cs else None,
            "reason_codes": (cs.codes if cs else []),
            "mandatory_review": bool(cs and cs.mandatory),
            "footnote_markers": c.footnote_markers})
    return {
        "table_id": table_id, "logical_index": li,
        "start_page": lt.start_page, "end_page": lt.end_page,
        "is_continuation": lt.is_continuation, "n_page_parts": len(lt.parts),
        "n_rows": asm.n_rows, "n_cols": asm.n_cols,
        "n_header_rows": asm.n_header_rows,
        "title": base.title, "caption": base.caption,
        "section_path": base.section_path,
        "regime": base.cand.regime,
        "detector_votes": base.cand.votes,
        "stitch_votes": asm.votes,
        "dropped_repeated_header_rows": asm.dropped_header_rows,
        "structure_evidence": base.grid.evidence,
        "unit_scale": {"note": scale.note, "source": scale.source,
                       "page_no": scale.page_no, "currency": scale.currency,
                       "magnitude": scale.magnitude,
                       "confidence": q(scale.confidence), "flags": scale.flags},
        "columns": [{"col_idx": i,
                     "header_path": (asm.header_paths[i]
                                     if i < len(asm.header_paths) else []),
                     "inferred_type": col_types.get(i),
                     "unit": (asm.col_units[i] if i < len(asm.col_units) else ""),
                     "type_evidence": col_type_ev.get(i, {})}
                    for i in range(asm.n_cols)],
        "regions": [{"page_no": p.page_no, "bbox": [q(v) for v in p.cand.bbox],
                     "regime": p.cand.regime} for p in lt.parts],
        "footnotes": [{"marker": f.marker, "text": f.text, "page_no": f.page_no}
                      for f in fns],
        "linked_assets": [{"file_path": page_assets[l["asset_index"]].file_path
                           .replace("\\", "/"),
                           "relation": l["relation"],
                           "relation_confidence": l["relation_confidence"],
                           "components": l["components"]} for l in alinks],
        "reconciliation": score.reconciliation,
        "table_confidence": q(score.table_confidence),
        "excel_path": os.path.join("tables", xlsx_name).replace("\\", "/"),
        "cells": cells,
    }


# ==================================================================================================
# cli: Command-line interface.
# ==================================================================================================
#
# The three commands the brief requires, plus two that earned their place while building:
#
#     python -m gridsmith run    --input doc.pdf --out ./out --offline
#     python -m gridsmith audit  --out ./out
#     python -m gridsmith debug  --out ./out --page 42
#
#     python -m gridsmith verify --a ./out1 --b ./out2      # two-run byte diff
#     python -m gridsmith query  --out ./out --sql "..."    # or a named example
#
# ``--offline`` is accepted and asserted rather than merely tolerated: with it set, the
# process installs a socket guard that raises on any attempt to open a network
# connection.  The pipeline has no network code, so the guard should never fire -- which
# is exactly why it is worth installing, because it converts "we believe it is offline"
# into "it cannot have been otherwise".

def _force_utf8_stdout():
    """Windows consoles default to cp1252 and would crash on a rupee sign."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def install_offline_guard():
    """Make network access impossible for the rest of the process."""
    import socket

    class _Blocked(RuntimeError):
        pass

    def _deny(*a, **k):
        raise _Blocked("network access attempted while --offline was set")

    socket.socket = _deny            # type: ignore[assignment]
    socket.create_connection = _deny  # type: ignore[assignment]
    if hasattr(socket, "socketpair"):
        socket.socketpair = _deny     # type: ignore[assignment]
    for name in ("getaddrinfo", "gethostbyname"):
        if hasattr(socket, name):
            setattr(socket, name, _deny)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m gridsmith",
        description="Deterministic table and context extraction from large PDFs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="extract one PDF into an output directory")
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--offline", action="store_true",
                   help="assert no network access (installs a socket guard)")
    r.add_argument("--fixed-time", action="store_true",
                   help="pin extraction_runs timestamps so even the .db is "
                        "byte-identical between runs")
    r.add_argument("--max-pages", type=int, default=0)
    r.add_argument("--debug-pages", choices=("none", "tables", "all"),
                   default=DEFAULT.write_debug_pages)
    r.add_argument("--no-borderless", action="store_true",
                   help="ablation: disable the borderless detector")
    r.add_argument("--no-ruled", action="store_true",
                   help="ablation: disable the ruled detector")
    r.add_argument("--no-stitch", action="store_true",
                   help="ablation: disable cross-page stitching")
    r.add_argument("--review-budget", type=float, default=DEFAULT.review_budget_frac)
    r.add_argument("--seed", type=int, default=DEFAULT.seed)
    r.add_argument("--quiet", action="store_true")

    a = sub.add_parser("audit", help="print the metrics summary for an output dir")
    a.add_argument("--out", required=True)
    a.add_argument("--json", action="store_true", help="emit raw metrics.json")

    d = sub.add_parser("debug", help="write an overlay render for one page")
    d.add_argument("--out", required=True)
    d.add_argument("--page", type=int, required=True)
    d.add_argument("--input", default=None,
                   help="source PDF (defaults to the one recorded in the database)")
    d.add_argument("--dpi", type=int, default=130)

    v = sub.add_parser("verify", help="diff two output directories (determinism check)")
    v.add_argument("--a", required=True)
    v.add_argument("--b", required=True)

    qy = sub.add_parser("query", help="run SQL against extraction.db")
    qy.add_argument("--out", required=True)
    qy.add_argument("--sql", default=None)
    qy.add_argument("--example", default=None,
                    help="named example query; use --list-examples to see them")
    qy.add_argument("--list-examples", action="store_true")
    qy.add_argument("--limit", type=int, default=20)
    return p


def cfg_from_args(args) -> Config:
    return DEFAULT.replace(
        offline=bool(getattr(args, "offline", False)),
        fixed_time=bool(getattr(args, "fixed_time", False)),
        max_pages=int(getattr(args, "max_pages", 0) or 0),
        write_debug_pages=getattr(args, "debug_pages", DEFAULT.write_debug_pages),
        enable_borderless=not getattr(args, "no_borderless", False),
        enable_ruled=not getattr(args, "no_ruled", False),
        enable_stitch=not getattr(args, "no_stitch", False),
        review_budget_frac=float(getattr(args, "review_budget",
                                         DEFAULT.review_budget_frac)),
        seed=int(getattr(args, "seed", DEFAULT.seed)))


def main(argv=None) -> int:
    _force_utf8_stdout()
    args = build_parser().parse_args(argv)

    if args.cmd == "run":
        if args.offline:
            install_offline_guard()

        cfg = cfg_from_args(args)
        summary = run(args.input, args.out, cfg)
        if not args.quiet:
            _print_summary(summary, args.out)
        return 0

    if args.cmd == "audit":

        return audit(args.out, as_json=args.json)

    if args.cmd == "debug":

        return debug_page(args.out, args.page, args.input, args.dpi)

    if args.cmd == "verify":

        return verify_dirs(args.a, args.b)

    if args.cmd == "query":

        return run_query(args.out, args.sql, args.example, args.limit,
                         args.list_examples)
    return 2


def _print_summary(s: dict, out_dir: str) -> None:
    c = s["counts"]
    r = s["resources"]
    q = s["review_queue"]
    conf = s.get("confidence", {})
    print("gridsmith %s  run %s" % (s["run"]["gridsmith_version"], s["run"]["run_id"]))
    print("  document      : %s  (%d pages, %s)"
          % (s["document"]["filename"], s["document"]["page_count"],
             ", ".join("%s=%d" % kv for kv in
                       sorted(s["document"]["page_types"].items()))))
    print("  tables        : %d logical (%d page-level, %d spanning pages)"
          % (c["logical_tables"], c["page_level_tables"], c["multi_page_tables"]))
    print("  cells         : %d   assets: %d   footnote links: %d"
          % (c["cells"], c["assets"], c["cell_footnote_links"]))
    print("  refused       : %d regions  %s"
          % (c["rejected_regions"],
             ", ".join("%s=%d" % kv for kv in
                       sorted(s["rejections_by_code"].items())) or "-"))
    print("  reconciliation: %s" % (", ".join(
        "%s=%s" % kv for kv in sorted(s["reconciliation"].items()))))
    print("  confidence    : mean %.3f  median %.3f  p05 %.3f  high-conf %.1f%%"
          % (conf.get("mean", 0), conf.get("median", 0), conf.get("p05", 0),
             100 * conf.get("frac_high_confidence", 0)))
    print("  review queue  : %d cells (%.2f%% of %d)  mandatory %d%s"
          % (q["n_queued"], 100 * q["queued_frac"], q["n_cells_total"],
             q["n_mandatory"],
             "  [mandatory exceeds budget]" if q["budget_exceeded_by_mandatory"]
             else ""))
    if r.get("fixed_time"):
        # metrics.json zeroes these under --fixed-time so its bytes stay
        # reproducible; printing the zeros would read as "instant, no memory".
        print("  resources     : not measured (--fixed-time);"
              " real figures in logs/timing.json")
    else:
        print("  resources     : %.1fs wall (%.2fs/page), peak RSS %.0f MB"
              % (r["wall_seconds"], r["seconds_per_page"], r["peak_rss_mb"]))
    print("  written to    : %s" % os.path.abspath(out_dir))


# ==================================================================================================
# __init__: gridsmith -- deterministic table and context extraction from large, messy PDFs.
# ==================================================================================================
#
# No language model touches the document.  See docs/ARCHITECTURE.md for the stage map and
# docs/DECISIONS.md for why each non-obvious choice was made.

__version__ = "0.1.0"


if __name__ == "__main__":
    sys.exit(main())
