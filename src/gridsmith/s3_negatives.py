"""S3c -- the things that look like tables and are not.

Section 15 of the brief lists four traps and says plainly that extracting them costs
marks while classifying and excluding them earns marks.  Every one of them produces a
strong column-gutter signal, so they cannot be filtered by the detector; they need
their own classifiers, and the classifiers need to fire *after* structure is visible
enough to reason about but *before* anything is written as a table.

Each rule returns a ``Rejection`` with a code, a score and the evidence that triggered
it.  Rejections are not silently dropped: they go into ``manifest.json`` under
``rejected_regions``, are drawn in red by the debug renderer, and are counted in
``metrics.json``.  A reviewer can therefore see what the pipeline refused and why,
which is the only way "we correctly excluded it" is a checkable claim rather than an
assertion.

Design note -- why rules and not a classifier.  A learned region classifier is the
obvious alternative and would probably score better on recall.  It would also be
unexplainable at the live review, could not be justified per-decision, and by §3.1
could never be the sole basis for a decision anyway.  These rules each key on a
*structural* invariant of the trap (monotone trailing page numbers; page-number lists;
independent line rhythm across a single gutter; a rectangle with no interior), so when
one fires we can say exactly what it saw.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from .geometry import median

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
