"""S9b -- the ranked human-review queue.

The target from section 4: flag no more than 8% of cells while those flags cover at
least 95% of the actual errors.  Two rules make that shape achievable without breaking
the invariant that no suspect cell is ever shipped unflagged:

1.  **Mandatory cells are never budgeted out.**  Anything carrying a
    :data:`gridsmith.s9_confidence.MANDATORY_REVIEW_CODES` code is in the queue even if
    it alone exceeds 8%.  A budget that could evict a known-suspect cell would make the
    8% number a lie.
2.  **The discretionary tail is ranked by confidence** and cut at the budget.

Both the budgeted queue (``review_queue.csv``) and the *full* ranking are produced,
because a single operating point cannot show a precision-recall curve and the brief
asks for the curve, not the point.  ``metrics/eval.py`` sweeps the full ranking.

Ranking key: mandatory first, then ascending confidence, then descending severity, then
(table, row, col) so the order is total and reproducible.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

from .config import Config
from .s9_confidence import MANDATORY_REVIEW_CODES, SEVERITY_OF

SEVERITY_RANK = {"error": 0, "warn": 1, "info": 2}

QUEUE_COLUMNS = [
    "rank", "priority", "table_id", "logical_index", "page_no", "row_idx", "col_idx",
    "header_path", "row_label_path", "raw_text", "normalized_value", "value_type",
    "confidence", "severity", "reason_codes", "reason", "bbox", "source",
]


@dataclass
class ReviewItem:
    table_id: str
    logical_index: int
    page_no: int
    row_idx: int
    col_idx: int
    header_path: str
    row_label_path: str
    raw_text: str
    normalized_value: object
    value_type: str
    confidence: float
    codes: list = field(default_factory=list)
    severity: str = "info"
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    source: str = ""
    mandatory: bool = False

    @property
    def reason(self) -> str:
        if not self.codes:
            return "low model confidence"
        return "; ".join(_explain(c) for c in self.codes[:4])


EXPLANATIONS = {
    "OCR_UNAVAILABLE": "no OCR engine available for this raster cell; value not read",
    "OCR_FAILED": "OCR returned nothing for this cell crop",
    "OCR_DIGIT_REPAIR": "OCR text only parsed as a number after glyph repair",
    "SPREADSHEET_ERROR": "source document contains a spreadsheet error value",
    "MALFORMED_NUMBER": "digits present but the number does not parse",
    "AMBIGUOUS_DECIMAL_SEPARATOR": "cannot tell decimal comma from thousands separator",
    "AMBIGUOUS_GROUPING": "digit grouping matches neither Indian nor Western convention",
    "AMBIGUOUS_DATE_ORDER": "day/month order undetermined for this document",
    "TYPE_CONFLICT": "value type disagrees with the inferred column type",
    "CROSSFOOT_FAIL": "row does not sum to its stated total",
    "DOWNFOOT_FAIL": "line items do not sum to the stated subtotal or total",
    "DOWNFOOT_CONTRIBUTOR": "contributes to a subtotal that does not reconcile",
    "TOKEN_STRADDLES_COLUMN": "text crosses an inferred column boundary",
    "WATERMARK_OVERLAP": "watermark or stamp glyphs overlap this cell",
    "GRID_UNRESOLVED": "cell text could not be resolved at all",
    "GRID_LOW_SUPPORT": "column boundaries rest on weak evidence",
    "DIGIT_COUNT_ANOMALY": "digit count is an outlier against the rest of the column",
    "DUPLICATE_ROW": "row duplicates an earlier row",
    "EMPTY_IN_DENSE_ROW": "cell is empty while the rest of the row is populated",
    "MULTILINE_CELL": "cell text wraps over more than one line",
    "HEADER_BODY_COLCOUNT_MISMATCH": "header spans fewer columns than the body",
    "SCALE_FROM_EARLIER_PAGE": "unit and scale inherited from an earlier page",
    "LEGACY_RUPEE_GLYPH": "rupee sign encoded as a grave accent by a legacy font",
    "LIGATURE_EXPANDED": "an f-ligature was expanded in the raw text",
    "SCALE_APPLIED": "a declared magnitude was applied to the value",
    "DAY_ASSUMED_1": "month-only date; day assumed to be the 1st",
}


def _explain(code: str) -> str:
    return EXPLANATIONS.get(code, code.lower().replace("_", " "))


def severity_of(codes) -> str:
    best = "info"
    for c in codes:
        s = SEVERITY_OF.get(c, "info")
        if SEVERITY_RANK[s] < SEVERITY_RANK[best]:
            best = s
    return best


def build_queue(items, cfg: Config, n_cells_total: int):
    """Rank items and split into the budgeted queue plus the full ranking."""
    def key(it):
        return (0 if it.mandatory else 1, round(it.confidence, 6),
                SEVERITY_RANK.get(it.severity, 2), it.table_id, it.row_idx, it.col_idx)

    ranked = sorted(items, key=key)
    budget = int(round(cfg.review_budget_frac * max(1, n_cells_total)))
    n_mandatory = sum(1 for it in ranked if it.mandatory)
    keep = max(budget, n_mandatory)
    queued = ranked[:keep]
    stats = {
        "n_cells_total": n_cells_total,
        "budget_frac": cfg.review_budget_frac,
        "budget_cells": budget,
        "n_mandatory": n_mandatory,
        "n_queued": len(queued),
        "queued_frac": round(len(queued) / max(1, n_cells_total), 5),
        "budget_exceeded_by_mandatory": n_mandatory > budget,
    }
    return queued, ranked, stats


def to_csv(queued, cfg: Config) -> str:
    """Render the queue as deterministic CSV text (LF endings, fixed column order)."""
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(QUEUE_COLUMNS)
    for i, it in enumerate(queued, start=1):
        w.writerow([
            i,
            "mandatory" if it.mandatory else "discretionary",
            it.table_id, it.logical_index, it.page_no, it.row_idx, it.col_idx,
            it.header_path, it.row_label_path,
            "" if it.raw_text is None else it.raw_text,
            "" if it.normalized_value is None else it.normalized_value,
            it.value_type, "%.5f" % it.confidence, it.severity,
            "|".join(it.codes), it.reason,
            "%.2f,%.2f,%.2f,%.2f" % tuple(round(float(v), 2) for v in it.bbox),
            it.source,
        ])
    return buf.getvalue()
