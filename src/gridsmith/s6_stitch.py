"""S6 -- cross-page stitching.

The brief asks for the rule and asks us to defend it, so here it is explicitly.

**Two hard gates** (both must hold, no score can buy past them):

1.  *Adjacent pages.*  ``page_no`` differs by exactly one.  A table cannot resume after
    an unrelated page: allowing gaps merges the receivables schedule on page 12 with
    the payables schedule on page 15 whenever their shapes happen to match, and that
    error is invisible in the output.
2.  *Same column count, and the columns line up.*  Mean 1-D IoU of the column bands
    (normalised to page width) at least ``stitch_col_iou_min`` = 0.85.  The brief warns
    that widths drift by a point or two between pages; 0.85 absorbs several points of
    drift on a six-column table while still rejecting a different table of the same
    arity.

**Then a weighted score** over five soft signals, needing ``stitch_score_min`` = 0.62:

=========================================  ======  ===============================
signal                                     weight  rationale
=========================================  ======  ===============================
column-band alignment (the IoU, rescaled)    0.30  the strongest single indicator
header repeated, or absent, on page N+1      0.20  a repeated header *is* a
                                                   continuation marker; a missing
                                                   header means the rows continue
"(Contd.)" / "continued" text nearby         0.15  explicit, when present
no intervening heading at or above the       0.20  a new section means a new table
table's own section level
first column keeps its type and the first     0.15  a continuation resumes the same
table ends low / second starts high                kind of rows in the same place
=========================================  ======  ===============================

**Why weights and not a decision tree.**  Every individual signal is absent on some real
document -- the brief says so: sometimes the header repeats, sometimes not; sometimes
there is a ``(Contd.)``, often not.  A tree has to commit to an order of interrogation
and inherits the blind spot of whichever signal it asks about first.  A weighted sum
degrades gracefully: two strong signals carry a decision when the other three are
silent, and the score plus every component is written to ``detector_votes_json`` so a
reviewer can see exactly which signals fired.

**Header de-duplication.**  When page N+1 repeats the header, that row is dropped from
the data and recorded in ``dropped_header_rows``.  Leaving it in is the specific failure
the brief calls out: "duplicate the header row into the data and break every downstream
sum".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from .geometry import iou_1d

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
    from .s7_normalize import classify_token

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
