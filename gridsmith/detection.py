"""S3 table detection: ruled, borderless, trap (negative) detectors and arbitration.

Sections, in order: s3_ruled, s3_borderless, s3_negatives, s3_arbitrate.
"""
from __future__ import annotations

import math
import numpy as np
import re
from dataclasses import dataclass, field

from .core import Config, cluster_1d, contain_frac, iou, median, merge_intervals, union


# --------------------------------------------------------------------------------------------------
# S3a -- ruled table detection from vector geometry (and from rasters).
# --------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------
# S3b -- borderless table detection by gutter and alignment analysis.
# --------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------
# S3c -- the things that look like tables and are not.
# --------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------
# S3d -- detector arbitration.
# --------------------------------------------------------------------------------------------------
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
