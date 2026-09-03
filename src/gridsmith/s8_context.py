"""S8 -- context capture: title, section, units and scale, footnotes, linked figures.

"4,812 means nothing until you know it is rupees, in lakhs, for Q4 FY25, for the
Domestic segment, restated per note 14."  Each of those five facts lives somewhere
different, and the awkward one is scale: it is commonly declared once under a section
heading and governs every table below it, possibly on later pages.

Scale resolution therefore walks outward from the table and *records where it stopped*:

1. the table's own caption or title,
2. text lines immediately above the region (the ``(All figures in Rs lakhs)`` line),
3. the enclosing section heading,
4. the nearest scale declaration earlier in the document, up to
   ``scale_search_pages`` pages back.

Each level is less certain than the last, so ``unit_scale_note`` is stored with the
page and line it came from and a confidence that decays with distance.  A value whose
unit came from two pages back is a materially weaker claim than one whose unit is in its
own caption, and the schema keeps the difference instead of flattening it.

**Footnote-to-cell linkage.**  Markers are collected from the cell text (``1,872*``,
a superscript ``2``) and matched against footnote lines by marker, searching the
table's own page first and then the *following* page -- item 10 of the adversarial set
puts the footnote text on the next page, and a page-local search silently drops it.
Links are stored in ``cell_footnotes``, cell to footnote, not table to page.

**Table-to-figure association.**  The brief asks us to define and defend the rule, so
it is an explicit weighted sum of four independent pieces of evidence:

===================================================  ======
evidence                                             weight
===================================================  ======
the table's caption or nearby text names the figure    0.45
figure and table captions share a section number      0.25
figure is adjacent to the table and inside the same    0.20
section
figure lies between the table and the next heading     0.10
===================================================  ======

Textual reference dominates because it is the only signal that is *stated* rather than
inferred; the rest are corroboration.  ``relation_confidence`` is the sum, and the
component breakdown is stored so a low-confidence link can be inspected rather than
guessed at.  Proximity alone caps out at 0.30, which is deliberately below any
threshold a consumer should trust blindly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from .geometry import area, union
from .s2_layout import CAPTION_RE, SCALE_RE
from .s7_normalize import MAGNITUDES, _code

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
    from .geometry import contain_frac

    return contain_frac(a, b) >= frac


def _vector_clusters(page, pt, table_bboxes, cfg: Config):
    """Dense vector-drawing clusters that are not table regions: bar/line charts."""
    from .geometry import cluster_1d

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
