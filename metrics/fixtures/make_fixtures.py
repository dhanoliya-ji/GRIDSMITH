"""Generate the adversarial fixture set, with ground truth exact by construction.

Why this exists.  The three sample documents are all digital, single-column, and
none of them contains a table that runs across pages, a rotated page, a scanned page,
a watermark, a two-column prose trap, or a subtotal that deliberately fails to add up.
Section 7 of the brief promises all of those in the held-out set, so the code paths that
handle them would otherwise ship untested -- and, just as importantly, the labelled set
drawn from the samples contains *no extraction errors*, which makes it useless for
fitting a calibration curve.  A set with no negatives cannot tell you how confident to
be.

Generating the fixtures instead of hand-labelling them has one decisive advantage: the
ground truth is not an opinion.  We place every glyph, so we know the exact cell text,
the exact grid, the exact spans and the exact intended reading.  Where the extractor
disagrees, the extractor is wrong -- no arguing about whether "$ 167,045" should carry a
space.

Determinism: no randomness anywhere, fixed font, fixed coordinates.  Re-running
regenerates byte-identical PDFs, so the fixtures can be committed or rebuilt at will.

Usage:
    python metrics/fixtures/make_fixtures.py            # write PDFs + *_gt.json
    python metrics/fixtures/make_fixtures.py --list
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

FONT = "helv"
FONT_B = "hebo"
PAGE = (595.0, 842.0)          # A4 portrait, points
GREY = (0.55, 0.55, 0.55)
BLACK = (0.0, 0.0, 0.0)
PALE = (0.86, 0.86, 0.86)


# --------------------------------------------------------------------------- utils


class Builder:
    """Thin wrapper that records where it put things, so GT falls out for free."""

    def __init__(self, size=PAGE, rotate=0):
        self.doc = pymupdf.open()
        self.size = size
        self.rotate = rotate
        self.page = None

    def new_page(self):
        self.page = self.doc.new_page(width=self.size[0], height=self.size[1])
        if self.rotate:
            self.page.set_rotation(self.rotate)
        return self.page

    def text(self, x, y, s, size=9.0, bold=False, color=BLACK):
        self.page.insert_text((x, y), s, fontname=FONT_B if bold else FONT,
                              fontsize=size, color=color)

    def rule(self, x0, y, x1, width=0.6, color=BLACK):
        self.page.draw_line(pymupdf.Point(x0, y), pymupdf.Point(x1, y),
                            color=color, width=width)

    def vrule(self, x, y0, y1, width=0.6, color=BLACK):
        self.page.draw_line(pymupdf.Point(x, y0), pymupdf.Point(x, y1),
                            color=color, width=width)

    def save(self, path):
        self.doc.save(path, garbage=4, deflate=True, clean=True)
        self.doc.close()


def right_align(b: Builder, x_right, y, s, size=9.0, bold=False):
    w = pymupdf.get_text_length(s, fontname=FONT_B if bold else FONT, fontsize=size)
    b.text(x_right - w, y, s, size=size, bold=bold)


def write_gt(name, payload):
    p = os.path.join(HERE, name + "_gt.json")
    with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


# ------------------------------------------------- F1: indentation + Indian numbers


def fixture_indent_hierarchy():
    """The brief's Figure 4, built for real.

    Two-level column header with spanning cells, a three-level row hierarchy encoded by
    indentation alone, Indian digit grouping, a parenthesised negative, and the scale
    declared in a line *above* the table rather than in its caption.
    """
    name = "f1_indent_hierarchy"
    b = Builder()
    b.new_page()
    x0, x1 = 60.0, 520.0
    cols = [60.0, 230.0, 300.0, 370.0, 440.0, 520.0]

    b.text(60, 70, "4. Segment Performance", size=13, bold=True)
    b.text(60, 92, "(All figures in Rs lakhs unless otherwise stated)", size=8)
    b.text(60, 118, "Table 4.2 - Revenue by segment", size=10, bold=True)

    top = 130.0
    b.rule(x0, top, x1, width=0.9)
    # header level 1: "Particulars" then two spanning year groups
    b.text(66, 145, "Particulars", size=9, bold=True)
    b.text(300 - 34, 145, "FY 2024-25", size=9, bold=True)
    b.text(440 - 34, 145, "FY 2023-24", size=9, bold=True)
    b.rule(230.0, 150.0, 370.0, width=0.5)
    b.rule(370.0, 150.0, 520.0, width=0.5)
    # header level 2: quarters
    for x, lab in ((265, "Q3"), (335, "Q4"), (405, "Q3"), (480, "Q4")):
        b.text(x - 6, 164, lab, size=9, bold=True)
    hdr_sep = 170.0
    b.rule(x0, hdr_sep, x1, width=0.9)

    rows = [
        (0, "Revenue", None),
        (1, "Domestic", None),
        (2, "Product A", ("2,940", "3,110", "2,655", "2,801")),
        (2, "Product B", ("1,802", "1,944", "1,610", "1,702")),
        (1, "Export", ("1,138", "1,166", "1,045", "1,099")),
        (1, "Other income", ("(412)", "508", "(388)", "476")),
        (0, "Total revenue", ("5,468", "6,728", "4,922", "6,078")),
    ]
    y = 188.0
    gt_cells, gt_indent = [], []
    for lvl, label, vals in rows:
        b.text(66 + lvl * 12, y, label, size=9, bold=(lvl == 0))
        row = [label]
        if vals:
            for k, v in enumerate(vals):
                right_align(b, cols[k + 1] + 62 if k < 3 else 514, y, v, size=9,
                            bold=(lvl == 0))
        row = [label] + (list(vals) if vals else ["", "", "", ""])
        gt_cells.append(row)
        gt_indent.append(lvl)
        y += 20.0
    bottom = y - 8.0
    b.rule(x0, bottom, x1, width=0.9)
    for x in cols[1:-1]:
        b.vrule(x, top, bottom, width=0.4)
    b.vrule(cols[0], top, bottom, width=0.4)
    b.vrule(cols[-1], top, bottom, width=0.4)

    b.text(60, y + 12, "* Export revenue restated following the change in incoterms; "
                       "refer note 14.", size=7)
    path = os.path.join(HERE, name + ".pdf")
    b.save(path)

    header = [["Particulars", "FY 2024-25", "FY 2024-25", "FY 2023-24", "FY 2023-24"],
              ["Particulars", "Q3", "Q4", "Q3", "Q4"]]
    write_gt(name, {
        "document": name + ".pdf",
        "generated_by": "metrics/fixtures/make_fixtures.py",
        "why": "Two-level header with spanning cells; three-level indentation-only row "
               "hierarchy; Indian digit grouping; parenthesised negatives; scale "
               "declared above the table, not in the caption.",
        "pages_exhaustively_labelled": [1],
        "detection": [{"page": 1, "kind": "table", "id": "f1",
                       "bbox": [60.0, 130.0, 520.0, bottom]}],
        "tables": [{
            "id": "f1", "page": 1, "bbox": [60.0, 130.0, 520.0, bottom],
            "n_rows": 2 + len(gt_cells), "n_cols": 5, "n_header_rows": 2,
            "tolerance_rows": 1,
            "unit": "INR lakh",
            "unit_source_expected_one_of": ["above_region", "earlier_page",
                                            "same_page_above"],
            "title_contains": "Revenue by segment",
            "header_paths_expected": {
                "1": ["FY 2024-25", "Q3"], "2": ["FY 2024-25", "Q4"],
                "3": ["FY 2023-24", "Q3"], "4": ["FY 2023-24", "Q4"]},
            "row_label_paths_expected": {
                "Product A": ["Revenue", "Domestic", "Product A"],
                "Export": ["Revenue", "Export"]},
            "indent_levels_expected": gt_indent,
            "cells": header + gt_cells,
            "normalized": ([[None] * 5, [None] * 5]
                           + [[None] + [_n(v) for v in r[1:]] for r in gt_cells]),
            "reconciliation_expected": [
                {"kind": "down_foot", "note": "Total revenue is NOT the sum of the "
                                              "level-1 rows in this fixture; see f6 "
                                              "for the deliberate-typo case"}],
        }],
        "negatives": [],
    })
    return name


def _n(s):
    if not s:
        return None
    t = s.replace(",", "")
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


# ------------------------------------------------------- F2: six-page continuation


def fixture_continuation():
    """One logical table over six pages.

    Header repeated on pages 1, 3 and 5 and absent on 2, 4 and 6; a "(Contd.)" caption
    on page 4 only; column widths drift by up to 2 pt per page, as the brief warns.
    A running footer sits on every page and must be excluded.
    """
    name = "f2_continuation_6pages"
    b = Builder()
    header = ["Sr", "Unit", "Area (sq ft)", "Value (Rs)"]
    n_pages = 6
    per_page = 8
    gt_rows = []
    sr = 1
    for p in range(n_pages):
        b.new_page()
        drift = (p % 3) - 1          # -1, 0, +1 points of column drift
        cols = [60.0 + drift, 110.0 + drift, 200.0 + drift, 330.0 + drift,
                480.0 + drift]
        y = 80.0
        if p == 3:
            b.text(60, y, "Table 4.2 (Contd.)", size=10, bold=True)
            y += 22
        elif p in (0, 2, 4):
            b.text(60, y, "Table 4.2 - Unit schedule", size=10, bold=True)
            y += 22
        top = y
        b.rule(cols[0], top, cols[-1], width=0.9)
        if p in (0, 2, 4):
            for k, h in enumerate(header):
                b.text(cols[k] + 4, top + 14, h, size=9, bold=True)
            y = top + 20
            b.rule(cols[0], y, cols[-1], width=0.9)
        else:
            y = top
        for i in range(per_page):
            yy = y + 18 * (i + 1)
            vals = [str(sr), "U-%03d" % sr, "%d" % (640 + sr * 7),
                    "{:,}".format(1250000 + sr * 31000)]
            b.text(cols[0] + 4, yy, vals[0], size=9)
            b.text(cols[1] + 4, yy, vals[1], size=9)
            right_align(b, cols[3] - 6, yy, vals[2], size=9)
            right_align(b, cols[4] - 6, yy, vals[3], size=9)
            gt_rows.append(vals)
            sr += 1
        bottom = y + 18 * per_page + 6
        b.rule(cols[0], bottom, cols[-1], width=0.9)
        for x in cols:
            b.vrule(x, top, bottom, width=0.4)
        b.text(60, 800, "Confidential - Annexure I - Page %d of %d"
               % (p + 1, n_pages), size=7, color=GREY)
    path = os.path.join(HERE, name + ".pdf")
    b.save(path)

    write_gt(name, {
        "document": name + ".pdf",
        "generated_by": "metrics/fixtures/make_fixtures.py",
        "why": "One logical table across six pages; header present on 1/3/5 and absent "
               "on 2/4/6; a (Contd.) marker on page 4 only; column x positions drift "
               "by up to 2 pt per page; a running footer on every page.",
        "expect_logical_tables": 1,
        "expect_start_page": 1,
        "expect_end_page": 6,
        "expect_data_rows": len(gt_rows),
        "expect_header_rows": 1,
        "expect_n_cols": 4,
        "expect_repeated_headers_dropped": 2,
        "expect_running_footer_excluded": "Confidential - Annexure I - Page # of #",
        "header": header,
        "data_rows": gt_rows,
    })
    return name


# ----------------------------------------------------------- F3: side-by-side pair


def fixture_side_by_side():
    """Two independent tables on one page, side by side.

    Their rows do not align vertically, which is the only reliable signal that they are
    two tables and not one table with a wide middle column.
    """
    name = "f3_side_by_side"
    b = Builder()
    b.new_page()
    b.text(60, 70, "Annexure II - Ratios and volumes", size=12, bold=True)

    left = [["Metric", "FY25"], ["Current ratio", "1.42"], ["Quick ratio", "0.98"],
            ["Debt/Equity", "0.31"], ["Interest cover", "8.70"]]
    lx = [60.0, 170.0, 250.0]
    ly = 110.0
    b.rule(lx[0], ly, lx[-1], width=0.8)
    for i, row in enumerate(left):
        yy = ly + 18 * (i + 1)
        b.text(lx[0] + 4, yy, row[0], size=9, bold=(i == 0))
        right_align(b, lx[2] - 6, yy, row[1], size=9, bold=(i == 0))
        if i == 0:
            b.rule(lx[0], yy + 5, lx[-1], width=0.8)
    lb = ly + 18 * len(left) + 6
    b.rule(lx[0], lb, lx[-1], width=0.8)
    for x in lx:
        b.vrule(x, ly, lb, width=0.4)

    # Right table: deliberately offset by 9 pt so no baseline is shared.
    right = [["Quarter", "Volume (KL)"], ["Q1", "1,12,430"], ["Q2", "1,18,905"],
             ["Q3", "1,21,004"]]
    rx = [330.0, 430.0, 530.0]
    ry = 119.0
    b.rule(rx[0], ry, rx[-1], width=0.8)
    for i, row in enumerate(right):
        yy = ry + 18 * (i + 1)
        b.text(rx[0] + 4, yy, row[0], size=9, bold=(i == 0))
        right_align(b, rx[2] - 6, yy, row[1], size=9, bold=(i == 0))
        if i == 0:
            b.rule(rx[0], yy + 5, rx[-1], width=0.8)
    rb = ry + 18 * len(right) + 6
    b.rule(rx[0], rb, rx[-1], width=0.8)
    for x in rx:
        b.vrule(x, ry, rb, width=0.4)

    b.save(os.path.join(HERE, name + ".pdf"))
    write_gt(name, {
        "document": name + ".pdf",
        "generated_by": "metrics/fixtures/make_fixtures.py",
        "why": "Two independent tables side by side, rows deliberately offset by 9 pt "
               "so no baseline is shared. Emitting one wide table is the failure.",
        "pages_exhaustively_labelled": [1],
        "detection": [
            {"page": 1, "kind": "table", "id": "f3_left",
             "bbox": [60.0, 110.0, 250.0, lb]},
            {"page": 1, "kind": "table", "id": "f3_right",
             "bbox": [330.0, 119.0, 530.0, rb]}],
        "tables": [
            {"id": "f3_left", "page": 1, "bbox": [60.0, 110.0, 250.0, lb],
             "n_rows": 5, "n_cols": 2, "n_header_rows": 1, "cells": left,
             "normalized": [[None, None]] + [[None, _n(r[1])] for r in left[1:]]},
            {"id": "f3_right", "page": 1, "bbox": [330.0, 119.0, 530.0, rb],
             "n_rows": 4, "n_cols": 2, "n_header_rows": 1, "cells": right,
             "normalized": [[None, None]] + [[None, _n(r[1])] for r in right[1:]]}],
        "negatives": [],
    })
    return name


# ------------------------------------------------------------------ F4: traps only


def fixture_traps():
    """The four false-positive traps from section 15, one per page."""
    name = "f4_traps"
    b = Builder()

    # p1: table of contents
    b.new_page()
    b.text(60, 80, "Contents", size=14, bold=True)
    toc = [("1. Overview", 3), ("2. Financial review", 11), ("3. Segment results", 24),
           ("4. Risk factors", 39), ("5. Annexures", 52), ("6. Glossary", 61)]
    for i, (lab, pg) in enumerate(toc):
        y = 120 + 22 * i
        b.text(66, y, lab, size=10)
        dots = "." * 60
        b.text(200, y, dots, size=10, color=GREY)
        right_align(b, 520, y, str(pg), size=10)

    # p2: index / glossary
    b.new_page()
    b.text(60, 80, "Index", size=14, bold=True)
    idx = [("Amortisation", "14, 22"), ("Borrowings", "31, 44-46"),
           ("Cash flow", "8, 19"), ("Deferred tax", "27"),
           ("Escrow", "33, 51"), ("Provisions", "12, 40, 55")]
    for i, (lab, pgs) in enumerate(idx):
        y = 120 + 20 * i
        b.text(66, y, lab, size=10)
        b.text(300, y, pgs, size=10)

    # p3: two-column prose
    b.new_page()
    b.text(60, 80, "Managerial commentary", size=12, bold=True)
    para_l = ["The company continued to consolidate its position in the",
              "domestic aviation fuel market during the year under review,",
              "supported by a recovery in passenger traffic and a stable",
              "supply arrangement with its refinery partner. Volumes grew",
              "in both the delivery and contracting businesses, although",
              "margins remained compressed by competitive intensity.",
              "Management expects the trend to continue into the coming",
              "financial year without material change in the cost base."]
    para_r = ["Working capital intensity stayed low through the period and",
              "the board approved limits were not drawn upon at any point.",
              "The company continues to evaluate selective investments in",
              "storage infrastructure at secondary airports, subject to",
              "commercial viability and the availability of long term",
              "offtake commitments from airline customers. No impairment",
              "indicators were identified during the annual review of",
              "carrying values for the cash generating units."]
    for i, s in enumerate(para_l):
        b.text(60, 110 + 14 * i, s, size=8.5)
    for i, s in enumerate(para_r):
        # Offset by 7 pt so the two columns do not share baselines.
        b.text(310, 117 + 14 * i, s, size=8.5)

    # p4: bordered callout box with no grid inside
    b.new_page()
    b.text(60, 80, "Risk framework", size=12, bold=True)
    b.page.draw_rect(pymupdf.Rect(60, 110, 520, 200), color=BLACK, width=1.2)
    b.text(72, 132, "Key takeaway", size=10, bold=True)
    b.text(72, 152, "Liquidity remains adequate. The rating is unlikely to change",
           size=9)
    b.text(72, 168, "in the absence of a material dividend payout.", size=9)

    b.save(os.path.join(HERE, name + ".pdf"))
    write_gt(name, {
        "document": name + ".pdf",
        "generated_by": "metrics/fixtures/make_fixtures.py",
        "why": "The four traps from section 15. Every page produces a strong "
               "column-gutter signal and none is a data table. Extracting any of them "
               "costs marks; refusing them earns marks.",
        "pages_exhaustively_labelled": [1, 2, 3, 4],
        "detection": [
            {"page": 1, "kind": "negative", "negative_kind": "table_of_contents",
             "bbox": [60.0, 110.0, 520.0, 250.0],
             "expected_codes": ["TOC_DOT_LEADERS", "TOC_TRAILING_PAGE_NUMBERS",
                                "TOC_HEADING_NEARBY"]},
            {"page": 2, "kind": "negative", "negative_kind": "index",
             "bbox": [60.0, 110.0, 400.0, 230.0],
             "expected_codes": ["INDEX_PAGE_LIST", "TOC_HEADING_NEARBY"]},
            {"page": 3, "kind": "negative", "negative_kind": "two_column_prose",
             "bbox": [60.0, 100.0, 560.0, 230.0],
             "expected_codes": ["TWO_COLUMN_PROSE"]},
            {"page": 4, "kind": "negative", "negative_kind": "bordered_callout_box",
             "bbox": [60.0, 110.0, 520.0, 200.0],
             "expected_codes": ["BORDERED_CALLOUT_BOX", "EMPTY_REGION",
                                "UNDERLINED_HEADING"]}],
        "tables": [],
        "expect_logical_tables": 0,
        "negatives": [
            {"page": 1, "negative_kind": "table_of_contents"},
            {"page": 2, "negative_kind": "index"},
            {"page": 3, "negative_kind": "two_column_prose"},
            {"page": 4, "negative_kind": "bordered_callout_box"}],
    })
    return name


# ------------------------------------------- F5: watermark + a deliberate arithmetic typo


def fixture_watermark_typo():
    """A watermark over the table and a subtotal that does not add up.

    The brief is explicit: the failing subtotal is a genuine typo in the source and must
    be flagged, never silently "fixed". 118 + 242 = 360, and the document says 359.
    """
    name = "f5_watermark_typo"
    b = Builder()
    b.new_page()
    b.text(60, 70, "Note 12 - Trade payables", size=12, bold=True)
    b.text(60, 88, "(All amounts in Rs crore)", size=8)

    # Pale, oversized watermark drawn before the table so glyphs overlap.
    # insert_text's `rotate` only accepts multiples of 90; a real diagonal watermark
    # needs a morph about a pivot.
    pivot = pymupdf.Point(240, 300)
    b.page.insert_text((110, 430), "DRAFT", fontname=FONT_B, fontsize=84,
                       color=PALE, morph=(pivot, pymupdf.Matrix(45)))

    cols = [60.0, 320.0, 420.0, 520.0]
    rows = [["Particulars", "FY25", "FY24"],
            ["Micro and small enterprises", "1,184", "842"],
            ["Other creditors", "2,421", "1,963"],
            ["Total trade payables", "3,505", "2,805"]]
    top = 110.0
    b.rule(cols[0], top, cols[-1], width=0.9)
    y = top
    for i, row in enumerate(rows):
        y += 20
        b.text(cols[0] + 4, y, row[0], size=9, bold=(i == 0 or i == len(rows) - 1))
        right_align(b, cols[2] - 6, y, row[1], size=9,
                    bold=(i == 0 or i == len(rows) - 1))
        right_align(b, cols[3] - 6, y, row[2], size=9,
                    bold=(i == 0 or i == len(rows) - 1))
        if i == 0:
            b.rule(cols[0], y + 6, cols[-1], width=0.9)
    bottom = y + 8
    b.rule(cols[0], bottom, cols[-1], width=0.9)
    for x in cols:
        b.vrule(x, top, bottom, width=0.4)

    b.save(os.path.join(HERE, name + ".pdf"))
    write_gt(name, {
        "document": name + ".pdf",
        "generated_by": "metrics/fixtures/make_fixtures.py",
        "why": "A pale 84 pt DRAFT watermark rotated 45 degrees across the table, and a "
               "subtotal that deliberately does not reconcile: 1,184 + 2,421 = 3,605 "
               "and the document states 3,505. The value must be extracted as 3505 "
               "and FLAGGED, never corrected. The FY24 column reconciles exactly "
               "(842 + 1,963 = 2,805), so the check must fail on one column only.",
        "pages_exhaustively_labelled": [1],
        "detection": [{"page": 1, "kind": "table", "id": "f5",
                       "bbox": [60.0, 110.0, 520.0, bottom]}],
        "tables": [{
            "id": "f5", "page": 1, "bbox": [60.0, 110.0, 520.0, bottom],
            "n_rows": 4, "n_cols": 3, "n_header_rows": 1,
            "unit": "INR crore",
            "cells": rows,
            "normalized": [[None, None, None], [None, 1184.0, 842.0],
                           [None, 2421.0, 1963.0], [None, 3505.0, 2805.0]],
            "reconciliation_expected": [
                {"kind": "down_foot", "row": 3, "col": 1, "status": "fail",
                 "note": "1,184 + 2,421 = 3,605 but the document states 3,505"},
                {"kind": "down_foot", "row": 3, "col": 2, "status": "pass",
                 "note": "842 + 1,963 = 2,805, correct"}],
            "must_flag_cells": [{"row": 3, "col": 1,
                                 "expected_codes": ["DOWNFOOT_FAIL"]}],
            "must_not_change": {"row": 3, "col": 1, "normalized_value": 3505.0},
        }],
        "negatives": [],
    })
    return name


# ---------------------------------------------------- F6: footnote on the next page


def fixture_footnote_next_page():
    """A footnote marker in a numeric cell, with the footnote text on the next page."""
    name = "f6_footnote_next_page"
    b = Builder()
    b.new_page()
    b.text(60, 70, "Segment revenue", size=12, bold=True)
    cols = [60.0, 300.0, 400.0, 500.0]
    rows = [["Segment", "FY25", "FY24"],
            ["Domestic", "2,940", "2,655"],
            ["Export*", "1,872", "1,454"],
            ["Total", "4,812", "4,109"]]
    top = 100.0
    b.rule(cols[0], top, cols[-1], width=0.9)
    y = top
    for i, row in enumerate(rows):
        y += 20
        b.text(cols[0] + 4, y, row[0], size=9, bold=(i == 0))
        right_align(b, cols[2] - 6, y, row[1], size=9, bold=(i == 0))
        right_align(b, cols[3] - 6, y, row[2], size=9, bold=(i == 0))
        if i == 0:
            b.rule(cols[0], y + 6, cols[-1], width=0.9)
    bottom = y + 8
    b.rule(cols[0], bottom, cols[-1], width=0.9)
    for x in cols:
        b.vrule(x, top, bottom, width=0.4)
    b.text(60, 780, "Continued overleaf", size=7, color=GREY)

    b.new_page()
    b.text(60, 70, "Notes to the segment disclosure", size=11, bold=True)
    b.text(60, 100, "* Export revenue restated following the change in incoterms; "
                    "refer note 14.", size=8)
    b.text(60, 780, "Continued overleaf", size=7, color=GREY)

    b.save(os.path.join(HERE, name + ".pdf"))
    write_gt(name, {
        "document": name + ".pdf",
        "generated_by": "metrics/fixtures/make_fixtures.py",
        "why": "The footnote marker is on page 1 and its text is on page 2. A "
               "page-local footnote search silently drops the link.",
        "pages_exhaustively_labelled": [1],
        "detection": [{"page": 1, "kind": "table", "id": "f6",
                       "bbox": [60.0, 100.0, 500.0, bottom]}],
        "tables": [{
            "id": "f6", "page": 1, "bbox": [60.0, 100.0, 500.0, bottom],
            "n_rows": 4, "n_cols": 3, "n_header_rows": 1,
            "cells": rows,
            "normalized": [[None, None, None], [None, 2940.0, 2655.0],
                           [None, 1872.0, 1454.0], [None, 4812.0, 4109.0]],
            "footnotes": [{"marker": "*", "text_contains": "change in incoterms",
                           "on_page": 2, "linked_cell": [2, 0]}],
            "reconciliation_expected": [
                {"kind": "down_foot", "row": 3, "col": 1, "status": "pass"},
                {"kind": "down_foot", "row": 3, "col": 2, "status": "pass"}],
        }],
        "negatives": [],
    })
    return name


# ----------------------------------------------- F7: landscape table, page rotated 90


def fixture_rotated():
    """A landscape exhibit inside an otherwise portrait document.

    The first version of this fixture drew the table on a landscape mediabox and then
    set /Rotate 90, which makes a viewer see *sideways* text -- not what a landscape
    exhibit is.  A real one is either a landscape mediabox with /Rotate 0 (this page,
    page 1) or a portrait mediabox with /Rotate 90 and the content drawn rotated so it
    reads upright (page 2).  Both must come out identical, because PyMuPDF reports
    coordinates in display space either way.
    """
    name = "f7_rotated_landscape"
    b = Builder(size=(842.0, 595.0))       # landscape mediabox, no /Rotate
    b.new_page()
    b.text(50, 60, "Exhibit A - Debt maturity profile", size=12, bold=True)
    cols = [50.0, 200.0, 320.0, 440.0, 560.0, 680.0, 790.0]
    rows = [["Instrument", "FY26", "FY27", "FY28", "FY29", "Total"],
            ["Term loan A", "1,200", "1,200", "900", "600", "3,900"],
            ["Term loan B", "450", "450", "450", "450", "1,800"],
            ["Debentures", "0", "1,000", "0", "1,000", "2,000"]]
    top = 90.0
    b.rule(cols[0], top, cols[-1], width=0.9)
    y = top
    for i, row in enumerate(rows):
        y += 22
        b.text(cols[0] + 4, y, row[0], size=9, bold=(i == 0))
        for k in range(1, 6):
            right_align(b, cols[k + 1] - 8, y, row[k], size=9, bold=(i == 0))
        if i == 0:
            b.rule(cols[0], y + 7, cols[-1], width=0.9)
    bottom = y + 9
    b.rule(cols[0], bottom, cols[-1], width=0.9)
    for x in cols:
        b.vrule(x, top, bottom, width=0.4)

    b.save(os.path.join(HERE, name + ".pdf"))
    write_gt(name, {
        "document": name + ".pdf",
        "generated_by": "metrics/fixtures/make_fixtures.py",
        "why": "A landscape exhibit inside a portrait document: a landscape mediabox "
               "with no /Rotate. Coordinates must be read in display space. The Total "
               "column is a self-declared total, so cross-foot must run and pass on "
               "all three data rows.",
        "expect_logical_tables": 1,
        "expect_n_cols": 6,
        "expect_data_rows": 3,
        "expect_page_rotation": 0,
        "header": rows[0],
        "data_rows": rows[1:],
        "cross_foot_expected_pass": 3,
    })
    return name


# -------------------------------------------- F8: scanned, skewed, image-only page


def fixture_scanned_skew(source_pdf: str):
    """Render a digital fixture to an image, skew it, and rebuild it as a scan.

    This is the only honest way to make a scanned fixture whose ground truth is exact:
    the text is known because we generated the digital original.  It exercises page
    typing, skew estimation, deskew and the per-cell OCR path -- and, where no OCR
    engine is installed, it must produce OCR_UNAVAILABLE flags and null cell text
    rather than a guess.
    """
    name = "f8_scanned_skew"
    src = pymupdf.open(source_pdf)
    page = src[0]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(200 / 72.0, 200 / 72.0), alpha=False)
    png = pix.tobytes("png")
    src.close()

    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("L")
    skewed = img.rotate(-1.6, resample=Image.BICUBIC, expand=True, fillcolor=255)
    buf = io.BytesIO()
    skewed.save(buf, format="PNG", optimize=False, compress_level=6)
    data = buf.getvalue()

    out = pymupdf.open()
    p = out.new_page(width=PAGE[0], height=PAGE[1])
    p.insert_image(pymupdf.Rect(0, 0, PAGE[0], PAGE[1]), stream=data)
    path = os.path.join(HERE, name + ".pdf")
    out.save(path, garbage=4, deflate=True)
    out.close()

    write_gt(name, {
        "document": name + ".pdf",
        "generated_by": "metrics/fixtures/make_fixtures.py",
        "why": "Image-only page, skewed by -1.6 degrees, built by rendering "
               "f1_indent_hierarchy.pdf. Page typing must say 'scanned', skew "
               "estimation must recover about 1.6 degrees, and with no OCR engine "
               "present every cell must be flagged OCR_UNAVAILABLE with null text -- "
               "never guessed.",
        "expect_page_type": "scanned",
        "expect_skew_deg_abs_between": [1.0, 2.4],
        "expect_no_silent_cells": True,
        "expect_all_cells_flagged_when_no_ocr": True,
        "expect_logical_tables": 1,
        "expect_min_cells": 12,
        "source_fixture": "f1_indent_hierarchy.pdf",
        "note": "Structure must be recovered from the rule lattice even with no text "
                "layer -- morphology finds the grid, and every cell is then created "
                "empty for OCR. With no OCR engine each one must carry "
                "OCR_UNAVAILABLE, raw_text NULL and confidence 0.",
    })
    return name


FIXTURES = [
    ("f1_indent_hierarchy", fixture_indent_hierarchy),
    ("f2_continuation_6pages", fixture_continuation),
    ("f3_side_by_side", fixture_side_by_side),
    ("f4_traps", fixture_traps),
    ("f5_watermark_typo", fixture_watermark_typo),
    ("f6_footnote_next_page", fixture_footnote_next_page),
    ("f7_rotated_landscape", fixture_rotated),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for n, fn in FIXTURES:
            print("  %-26s %s" % (n, (fn.__doc__ or "").strip().splitlines()[0]))
        print("  %-26s %s" % ("f8_scanned_skew",
                              "Image-only skewed page built from f1"))
        return 0
    made = []
    for n, fn in FIXTURES:
        made.append(fn())
        print("  wrote %s.pdf + %s_gt.json" % (made[-1], made[-1]))
    n8 = fixture_scanned_skew(os.path.join(HERE, "f1_indent_hierarchy.pdf"))
    print("  wrote %s.pdf + %s_gt.json" % (n8, n8))
    print("%d fixtures generated in %s" % (len(made) + 1, HERE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
