"""Shared foundations: configuration, determinism, geometry helpers and the run log.

Sections, in order: config, determinism, geometry, logging_utils.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


# --------------------------------------------------------------------------------------------------
# Every tunable number in the pipeline lives here.
# --------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------
# Determinism plumbing.
# --------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------
# Rectangle and interval helpers.
# --------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------
# Structured, deterministic run log.
# --------------------------------------------------------------------------------------------------
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
