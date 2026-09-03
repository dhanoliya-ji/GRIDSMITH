"""S4 -- table structure recognition: the actual grid.

Everything upstream produced *evidence*.  This module is the single place that turns it
into boundaries, and it does so by weighted voting rather than by picking one detector,
because on real pages the sources are complementary and individually incomplete.

Column boundary weights (measured on the three sample documents, see docs/DECISIONS.md D4):

===========================  ======  =========================================
source                       weight  why
===========================  ======  =========================================
vertical rule                  3.0   drawn by the author; unambiguous
horizontal-rule breakpoint     2.5   the author's own cell edges, sub-point precise
cell background fill edge      2.0   exact, but only covers shaded rows
gutter centre                  1.0   inferred; correct on average, drifts under
                                     right-aligned numbers
===========================  ======  =========================================

A candidate boundary is then vetoed if tokens straddle it on more than a few percent of
rows -- a boundary that cuts through text is wrong no matter how it was proposed.

Three problems get explicit treatment because they are common and quietly destructive:

**Adornment columns.**  EDGAR filings put the currency symbol in its own column: the
Apple segment table is geometrically ``label | $ | 167,045 | 3 | % | $ | 162,560 | ...``
-- eleven bands for six logical columns and five headers.  Bands whose non-empty cells
are *exclusively* a currency symbol fold right into the number; bands that are
exclusively ``%`` fold left.  The currency moves into ``columns.unit`` so nothing is
lost, and the column count then matches the header count, which is what makes
``header_path`` alignment possible at all.

**Multi-line cells.**  In the ICRA rating table one row band contains three text lines
(``[ICRA]AA-(Stable)/[ICRA]A1+;`` / ``Long term/Short term ... 143.00 143.00`` /
``reaffirmed``).  Rows must therefore come from rule positions where rules exist, and
text-line clustering must merge continuations where they do not.  Getting this wrong
turns one row into three and breaks every sum in the table.

**Indentation as the only hierarchy.**  ``Product A`` belongs to ``Domestic`` with no
bullet, no numbering and no bold -- just leading whitespace.  First-column left edges
are clustered into indent levels and a stack turns them into ``row_label_path``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .geometry import cluster_1d, median
from .s3_borderless import column_gutters, is_prose_line, paragraph_closure
from .textlayer import build_lines, join_tokens
from .s7_normalize import classify_token, is_currency_symbol

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
