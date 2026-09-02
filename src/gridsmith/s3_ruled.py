"""S3a -- ruled table detection from vector geometry (and from rasters).

This is not "find the lines and intersect them".  Measuring the three sample documents
first (``tools/probe.py``) showed three *different* ways a PDF can express a grid, and
only one of them is drawn with lines:

``Shell.pdf`` -- **dotted rules.**
    18,066 filled rectangles of 0.72 x 0.72 pt.  Each "line" is a row of dots.  A naive
    thin-rect-is-a-rule test yields 18,000 rules of length 0.72 and finds nothing.  Dots
    are therefore emitted as candidates on *both* axes and only survive if they merge
    into a run longer than ``rule_min_length``; a vertical dotted line contributes one
    dot per horizontal cluster, so it cannot fake a horizontal rule.

``Shell MRPL ...pdf`` -- **horizontal hairlines only, segmented per cell.**
    Rules are 0.48 pt tall and arrive as ``[43.2,248.8] [248.8,249.3] [249.3,319.8]
    ...``.  Merging across the sub-point gaps rebuilds the full-width rule -- and the
    *breakpoints* between segments (249, 320, 403) are the column boundaries, handed to
    S4 as a free, high-precision column signal.  There are no vertical rules at all:
    this is exactly the "ruled horizontally but not vertically" case the brief says
    neither classical approach handles alone.

``Apple Form 10-K.pdf`` -- **per-cell background fills, no rules.**
    Alternating rows are shaded with one 0.937-grey rectangle *per cell*, so the fill
    rectangles are the cell lattice, exactly.  Ignoring them and running gutter
    analysis instead throws away the best signal on the page.

So this module produces *evidence*, not verdicts: rules, fills, breakpoints and
intersections, plus candidate regions tagged with the regime that found them.  S4 is
the single place that resolves boundaries, and it prefers evidence in the order
grid > fills > breakpoints > gutters.  Keeping arbitration in one place is what stops
the three regimes from fighting each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .geometry import cluster_1d, median, merge_intervals

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
