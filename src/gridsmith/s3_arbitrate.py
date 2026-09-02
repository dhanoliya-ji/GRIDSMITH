"""S3d -- detector arbitration.

Both detectors fire on most pages, and they disagree in a specific, predictable way:

*   the ruled detector is *precise but partial* -- on the ICRA report it nails the four
    row boundaries and the three column boundaries and knows nothing about the header
    band above the first rule;
*   the borderless detector is *complete but noisy* -- on the Apple income statement it
    finds six separate "tables" because the statement's sub-sections are separated by
    larger vertical gaps than its rows are.

The wrong way to combine them is to pick a winner per page.  What works is to treat
every detection as a *vote for a region of the page*, cluster the votes spatially, and
emit one candidate per cluster carrying the union of all the evidence.  S4 then rebuilds
the grid inside that bbox from scratch, preferring evidence in the order
grid > fills > rule-breakpoints > gutters.  Six borderless fragments plus one fill
lattice therefore become one table, and the fill lattice supplies the columns.

Clustering uses ``max(IoU, containment)`` rather than IoU alone.  The underline segments
beneath Apple's number columns have an IoU of 0.22 against the table they belong to --
below any sane merge threshold -- but they are 100% *contained* in it, which is the
relationship that actually matters.

``detector_votes_json`` on the tables row records every vote and the reason the cluster
was accepted or rejected, so any decision here can be replayed at review time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .geometry import contain_frac, iou, union
from . import s3_negatives
from .s3_ruled import REGIME_FILLS, REGIME_GRID, REGIME_HLINES, REGIME_VLINES


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
                cand.rejections = [s3_negatives.Rejection(
                    "INSUFFICIENT_GRID_ON_RASTER", 1.0,
                    "image-only page: %d intersections, %d row boundaries -- not "
                    "enough structure to call it a table"
                    % (cand.n_intersections, len(cand.y_lines)),
                    {"n_intersections": cand.n_intersections})]
                rejected.append(cand)
            continue
        if len(lines_in) < min_content_lines:
            cand.accepted = False
            cand.rejections = [s3_negatives.Rejection(
                "EMPTY_REGION", 1.0,
                "merged region holds %d text lines" % len(lines_in),
                {"n_lines": len(lines_in)})]
            rejected.append(cand)
            continue

        x_for_traps = cand.x_lines or [bbox[0], bbox[2]]
        verdict = s3_negatives.classify(
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
