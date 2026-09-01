"""S2 -- page layout segmentation.

Partitions each page into running header/footer, headings, body prose, caption lines and
footnote lines.  Table and figure regions are *not* decided here; S3 does that, and then
:func:`demote_in_regions` comes back to fix the roles S2 could only guess at.

The running header/footer detector is the part that earns its keep.  The brief calls
repeating headers "a classic source of false positives" and it is right: left alone, a
gutter detector sees a beautiful three-column table on every page of the Apple 10-K.

Why the obvious implementation fails
------------------------------------
The textbook approach takes the top/bottom 10-15% of the page and looks for lines that
repeat at a stable y.  That works on the ICRA report, whose footer
``www.icra.in  Sensitivity Label : Public  Page | 3`` sits at y=810 on every page.

It completely misses the Apple 10-K.  Its footer ``Apple Inc. | 2024 Form 10-K | 22``
appears on 57 pages, but the HTML-to-PDF converter places it immediately after the
content, so its baseline wanders between y=510 and y=642 -- never inside the bottom
14% band, and never at a stable y.  Measured, not assumed: see ``tools/probe.py``.

So candidates come from the union of two views:

*   **geometric** -- any line inside the top/bottom band, and
*   **structural** -- the first three and last three lines of the page, wherever they sit.

and the key is ``(shape, side)`` with no y quantisation.  Dropping y from the key is
what makes the Apple footer detectable, and it costs recall nowhere.

What stops it eating a repeated table header
--------------------------------------------
Two guards, because this is the failure that would silently destroy a table:

1.  **Isolation.**  A running header is separated from the body by more than 1.5 line
    gaps; a repeated table header is immediately followed by its data rows.  A
    candidate must be isolated on at least half the pages where it appears.
2.  **Position stability or extremity.**  Either the y is stable across pages (a real
    fixed footer) or the line is structurally the first/last on most of its pages.

And if S3 later finds a "chrome" line sitting inside a table region anyway,
:func:`demote_in_regions` takes the label back and records an issue, so the mistake is
visible rather than silent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from .geometry import median
from .textlayer import norm_for_repeat

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
