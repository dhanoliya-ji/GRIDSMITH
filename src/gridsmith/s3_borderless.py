"""S3b -- borderless table detection by gutter and alignment analysis.

The brief's sketch is the right one: build the vertical projection profile of ink over a
candidate block and keep the minima that persist across at least 80% of its rows.  Two
things have to be added before that works on real pages.

**Prose lines have to be removed first, not filtered out afterwards.**
Immediately above the Apple segment table sits the sentence "The following table shows
net sales by reportable segment for 2024, 2023 and 2022 (dollars in millions):".  Its
inter-word spaces land wherever they land, and with eight table rows below it the
persistence test still passes at 7/8 = 87% -- so the sentence is quietly admitted as a
table row and every column shifts.  A line is therefore classified as prose *before*
profiling, using a signal that does not depend on the columns we are trying to find: a
line with 8+ tokens whose largest internal gap is no bigger than about three word
spaces, spanning most of the block, is prose.  A table row like
``Americas  $  167,045  3 %  ...`` fails that test on its 200 pt label-to-number gap,
which is exactly the asymmetry we want to key on.

**One big gutter can mean two tables.**
Two independent tables side by side (item 5 of the adversarial set) produce one wide
central gutter, and the naive reading is a single table with a wide middle column.  The
discriminator is row alignment: halves of one table share baselines, two separate tables
do not.  :func:`split_side_by_side` cuts on a dominant gutter only when the baseline sets
on each side disagree.

``column_gutters`` is deliberately public: S4 calls it to supply column boundaries for
horizontally-ruled tables, where the rules give rows and the whitespace gives columns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .geometry import median

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
