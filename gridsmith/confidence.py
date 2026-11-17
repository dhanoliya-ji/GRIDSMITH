"""S9 per-cell confidence: 14 signals, isotonic calibration and the review queue.

Sections, in order: s9_confidence, s9_review.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from dataclasses import dataclass, field

from .core import Config, median
from .values import NULLISH_TYPES, NUMERIC_TYPES, T_CURRENCY, T_NUMBER, T_REFERENCE, T_TEXT
from .structure import is_total_label


# --------------------------------------------------------------------------------------------------
# S9 -- validation, confidence and the review queue.
# --------------------------------------------------------------------------------------------------
# This is the stage the brief weights highest, and the reason is worth restating: a
# pipeline that is 92% accurate and catches 97% of its own mistakes is more useful than
# one that is 96% accurate and cannot say which 4% are wrong.  So the goal here is not a
# high score, it is a *well-ordered* score.
#
# Architecture
# ------------
# Per cell we compute a fixed set of deterministic signals, combine them with **fixed,
# documented logistic weights** (in :mod:`gridsmith.config`, never fitted at runtime -- §3.2
# requires byte-identical reruns), and then map that score to a probability through a
# **calibration curve fitted offline** on the hand-labelled set and vendored as
# ``calibration.json``.  Fitting offline and applying at runtime is what makes the number
# a probability rather than a vibe, without making the run non-deterministic.
#
# The mechanical guarantee
# ------------------------
# Signals are only half the story.  Some conditions mean *we know we might be wrong*, and
# no amount of corroborating evidence should be able to outvote them: OCR was unavailable,
# the source cell contains ``#REF!``, the decimal separator is ambiguous, a token straddles
# a column boundary, a row does not add up.  Those are :data:`MANDATORY_REVIEW_CODES`, and
# they do two things:
#
# *   confidence is **hard-capped** below ``high_confidence_floor``, so such a cell can
#     never be shipped as high-confidence, whatever the logistic says;
# *   the cell enters the review queue **regardless of the 8% budget**.
#
# That is the mechanism behind "every cell is either correct or flagged".  A budget that
# could push a known-suspect cell out of the queue would break the invariant, so the
# budget only ever governs the *discretionary* tail of the ranking.
#
# Reconciliation
# --------------
# Down-foot: rows whose label reads as a total are checked against the sum of their
# contributors -- the rows since the previous total, at the shallowest indent level in
# that window, which is what makes it work on a hierarchy where "Total" sits over
# ``Domestic`` and ``Export`` while ``Product A``/``Product B`` sit under ``Domestic``.
#
# Cross-foot: only run when a column *header* identifies itself as a total.  Summing "the
# other numeric columns" unconditionally is wrong on the overwhelmingly common
# year-over-year table, where 2024 and 2023 are not addends of anything.
#
# A failure flags the total cell strongly and its contributors weakly, because the error
# could be in any of them, and it is never repaired.  The brief is explicit: a subtotal
# that does not reconcile because the source document has a genuine typo must be flagged,
# not silently fixed.

# Reconcilable columns are plain amounts.  Percentages are ratios: summing the
# 2024 change column of the Apple segment table is arithmetic on nonsense, and
# it produced four confident DOWNFOOT_FAIL flags on a correct table.
ADDABLE_TYPES = (T_NUMBER, T_CURRENCY)

# Conditions that must never be shipped as high confidence.
MANDATORY_REVIEW_CODES = frozenset({
    "OCR_UNAVAILABLE", "OCR_FAILED", "OCR_DIGIT_REPAIR", "DEGENERATE_CELL_BOX",
    "SPREADSHEET_ERROR", "MALFORMED_NUMBER",
    "AMBIGUOUS_DECIMAL_SEPARATOR", "AMBIGUOUS_GROUPING", "AMBIGUOUS_DATE_ORDER",
    "TYPE_CONFLICT", "CROSSFOOT_FAIL", "DOWNFOOT_FAIL",
    "TOKEN_STRADDLES_COLUMN", "WATERMARK_OVERLAP", "GRID_UNRESOLVED",
})
# Present but not disqualifying: they lower confidence without forcing review.
SOFT_CODES = frozenset({
    "MULTILINE_CELL", "LIGATURE_EXPANDED", "DIGIT_COUNT_ANOMALY", "DUPLICATE_ROW",
    "EMPTY_IN_DENSE_ROW", "HEADER_BODY_COLCOUNT_MISMATCH", "SCALE_FROM_EARLIER_PAGE",
    "LEGACY_RUPEE_GLYPH", "SCALE_APPLIED", "DAY_ASSUMED_1", "CROSSFOOT_CONTRIBUTOR",
    "DOWNFOOT_CONTRIBUTOR", "GRID_LOW_SUPPORT",
})

SEVERITY_OF = {
    "OCR_UNAVAILABLE": "error", "OCR_FAILED": "error", "SPREADSHEET_ERROR": "error",
    "MALFORMED_NUMBER": "error", "GRID_UNRESOLVED": "error",
    "DOWNFOOT_FAIL": "error", "CROSSFOOT_FAIL": "error",
    "AMBIGUOUS_DECIMAL_SEPARATOR": "warn", "AMBIGUOUS_GROUPING": "warn",
    "AMBIGUOUS_DATE_ORDER": "warn", "TYPE_CONFLICT": "warn",
    "TOKEN_STRADDLES_COLUMN": "warn", "WATERMARK_OVERLAP": "warn",
    "OCR_DIGIT_REPAIR": "warn", "DEGENERATE_CELL_BOX": "warn",
}


@dataclass
class CellScore:
    row: int
    col: int
    features: dict = field(default_factory=dict)
    logit: float = 0.0
    raw_confidence: float = 0.0
    confidence: float = 0.0
    codes: list = field(default_factory=list)
    mandatory: bool = False


@dataclass
class TableScore:
    cells: list = field(default_factory=list)
    table_confidence: float = 0.0
    reconciliation: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)


# --------------------------------------------------------------------- calibration


class Calibration:
    """Monotone binned (isotonic-style) map from logistic score to probability.

    Fitted offline by ``metrics/fit_calibration.py`` on hand-labelled cells and stored
    as bin edges plus the empirical accuracy in each bin.  Applying a lookup table at
    runtime keeps the run deterministic and keeps the fitting code out of the hot path.

    With no file present we fall back to a deliberately *pessimistic* identity: the raw
    sigmoid shrunk 25% toward 0.5.  An uncalibrated pipeline that overstates confidence
    is exactly the failure mode the brief punishes, so the default errs toward doubt.
    """

    def __init__(self, edges=None, values=None, source="fallback_shrunk_sigmoid"):
        self.edges = list(edges or [])
        self.values = list(values or [])
        self.source = source

    @classmethod
    def load(cls, path: str):
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if d.get("edges") and d.get("values"):
                return cls(d["edges"], d["values"], d.get("source", path))
        return cls()

    def apply(self, p: float) -> float:
        if not self.edges:
            return 0.5 + 0.75 * (p - 0.5)
        for i, e in enumerate(self.edges):
            if p <= e:
                return float(self.values[min(i, len(self.values) - 1)])
        return float(self.values[-1])

    def to_dict(self):
        return {"source": self.source, "n_bins": len(self.values),
                "edges": self.edges, "values": self.values}


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    e = math.exp(max(x, -60.0))
    return e / (1.0 + e)


# ----------------------------------------------------------------- reconciliation


def _num(norm):
    return norm.number if (norm is not None and norm.value_type in NUMERIC_TYPES
                           and norm.number is not None) else None


RATIO_LABEL_RE = __import__("re").compile(
    r"/|\(\s*times\s*\)|\bratio\b|\bper\b|\bmargin\b|\bcoverage\b|%",
    __import__("re").I)


def is_reconcilable_total(label: str) -> bool:
    """A row label that reads as a total *and* not as a ratio.

    "Total outside liabilities/Tangible net worth (times)" starts with "Total" and is
    not a total of anything -- it is a ratio.  Reconciling it against the four rows
    above produced two confident-looking DOWNFOOT_FAIL errors on a perfectly correct
    ICRA table, which is a false alarm in the one place false alarms are expensive.
    """
    return is_total_label(label) and not RATIO_LABEL_RE.search(label or "")


def _decimals(norm) -> int:
    """Digits after the decimal point in the *printed* value."""
    if norm is None or not norm.raw_text:
        return 0
    txt = norm.raw_text.strip().strip("()%")
    if "." not in txt:
        return 0
    return len(txt.rsplit(".", 1)[1].strip(" )%"))


def rounding_tolerance(norms) -> float:
    """Largest discrepancy that display rounding alone could explain.

    A percentage tolerance is the wrong model.  With ``tol = 0.5%`` of the stated
    value, a subtotal of 359 against addends summing to 360 passes -- and that is the
    deliberate typo the brief plants and asks us to flag.  What actually bounds an
    innocent discrepancy is the precision the values are *printed* to: each addend can
    hide at most half of its last displayed digit, so ``n`` addends can accumulate at
    most ``n * 0.5 * 10^-d``.

    Integers (d = 0) with two addends therefore tolerate less than 1.0, so 359 vs 360
    fails -- while three values printed as 33.3 summing to 99.9 against a stated 100.0
    still passes, because 0.1 < 3 * 0.05.  The comparison is strict for exactly that
    reason.
    """
    if not norms:
        return 1e-6
    d = max(_decimals(n) for n in norms)
    return max(1e-6, len(norms) * 0.5 * (10.0 ** -d))


def down_foot(cells_by_rc, norms_by_rc, n_rows, n_cols, n_header, indent_levels,
              labels, col_types=None, tol_rel=None, tol_abs=None):
    """Check every total row against the sum of its contributors, per column."""
    results = []
    col_types = col_types or {}
    total_rows = [r for r in range(n_header, n_rows)
                  if is_reconcilable_total(labels.get(r, ""))]
    prev_bound = n_header
    for tr in total_rows:
        window = [r for r in range(prev_bound, tr)
                  if not is_total_label(labels.get(r, ""))]
        prev_bound = tr + 1
        if not window:
            continue
        lv = [indent_levels[r] if r < len(indent_levels) else 0 for r in window]
        base = min(lv) if lv else 0
        contributors = [r for r, l in zip(window, lv) if l == base]
        if len(contributors) < 2:
            continue
        for c in range(n_cols):
            # Only reconcile columns that are actually numeric.  A column mixing
            # currency with percentages infers as text, and adding 2,087.5 to 0.9%
            # is not a check, it is noise.
            if col_types.get(c) not in ADDABLE_TYPES:
                continue
            stated = _num(norms_by_rc.get((tr, c)))
            if stated is None:
                continue
            parts = [(r, _num(norms_by_rc.get((r, c)))) for r in contributors]
            vals = [(r, v) for r, v in parts if v is not None]
            if len(vals) < 2:
                continue
            # An unparsed cell among the addends makes the sum meaningless.
            if any(v is None for _r, v in parts):
                results.append({"kind": "down_foot", "row": tr, "col": c,
                                "status": "incomplete",
                                "contributors": [r for r, _v in vals],
                                "message": "an addend could not be parsed"})
                continue
            total = sum(v for _r, v in vals)
            # Plausibility gate: with all-positive addends a total cannot be materially
            # smaller than its largest part.  When it is, this is not a total row and
            # the "failure" would be ours, not the document's.
            if all(v >= 0 for _r, v in vals) and \
                    stated < 0.9 * max(v for _r, v in vals):
                results.append({"kind": "down_foot", "row": tr, "col": c,
                                "status": "skipped",
                                "message": "stated value is smaller than its largest "
                                           "addend; row is not a sum"})
                continue
            tol = rounding_tolerance([norms_by_rc.get((r, c)) for r, _v in vals]
                                     + [norms_by_rc.get((tr, c))])
            ok = abs(total - stated) < tol
            results.append({"kind": "down_foot", "row": tr, "col": c,
                            "status": "pass" if ok else "fail",
                            "stated": round(stated, 6), "computed": round(total, 6),
                            "delta": round(total - stated, 6), "tolerance": round(tol, 6),
                            "contributors": [r for r, _v in vals]})
    return results


def cross_foot(norms_by_rc, header_paths, n_rows, n_cols, n_header,
               col_types=None, tol_rel=None, tol_abs=None):
    """Check a self-declared total *column* against the sum of the others."""
    col_types = col_types or {}
    total_cols = [c for c in range(n_cols)
                  if col_types.get(c) in ADDABLE_TYPES
                  and any(is_total_label(h) or "total" in (h or "").lower()
                          for h in (header_paths[c]
                                    if c < len(header_paths) else []))]
    if not total_cols:
        return []
    results = []
    for tc in total_cols:
        addends = [c for c in range(n_cols)
                   if c != tc and c > 0 and col_types.get(c) in ADDABLE_TYPES]
        for r in range(n_header, n_rows):
            stated = _num(norms_by_rc.get((r, tc)))
            if stated is None:
                continue
            vals = [_num(norms_by_rc.get((r, c))) for c in addends]
            present = [v for v in vals if v is not None]
            if len(present) < 2:
                continue
            total = sum(present)
            tol = rounding_tolerance([norms_by_rc.get((r, c)) for c in addends]
                                     + [norms_by_rc.get((r, tc))])
            ok = abs(total - stated) < tol
            results.append({"kind": "cross_foot", "row": r, "col": tc,
                            "status": "pass" if ok else "fail",
                            "stated": round(stated, 6), "computed": round(total, 6),
                            "delta": round(total - stated, 6),
                            "tolerance": round(tol, 6),
                            "contributors": [c for c in addends]})
    return results


# ---------------------------------------------------------------------- features


def _snap_residual(cell, x_lines, y_lines):
    """How far the cell's ink sits outside its own grid box, as a fraction of size."""
    if not cell.tokens:
        return 0.0
    c0 = min(cell.col, len(x_lines) - 2)
    c1 = min(cell.col + cell.col_span, len(x_lines) - 1)
    bx0, bx1 = x_lines[c0], x_lines[c1]
    r0 = min(cell.row, len(y_lines) - 2)
    r1 = min(cell.row + cell.row_span, len(y_lines) - 1)
    by0, by1 = y_lines[r0], y_lines[r1]
    tx0 = min(t.x0 for t in cell.tokens)
    tx1 = max(t.x1 for t in cell.tokens)
    ty0 = min(t.y0 for t in cell.tokens)
    ty1 = max(t.y1 for t in cell.tokens)
    w = max(1.0, bx1 - bx0)
    h = max(1.0, by1 - by0)
    over_x = max(0.0, bx0 - tx0, tx1 - bx1) / w
    over_y = max(0.0, by0 - ty0, ty1 - by1) / h
    return min(1.0, max(over_x, over_y))


def _digit_count_anomaly(n_digits, col_digits):
    """MAD-based z-score of digit count against the rest of the column."""
    peers = [d for d in col_digits if d > 0]
    if len(peers) < 3 or n_digits <= 0:
        return 0.0
    med = median(peers)
    mad = median([abs(d - med) for d in peers]) or 0.5
    z = abs(n_digits - med) / (1.4826 * mad)
    return min(1.0, max(0.0, (z - 2.0) / 4.0))


def score_table(cells, norms, grid, cfg: Config, calib: Calibration,
                col_types, source_of, ocr_conf_of, grid_support: float = 1.0):
    """Compute per-cell confidence for one logical table.

    ``cells`` are :class:`gridsmith.s5_cells.ExtractedCell`; ``norms`` is a parallel dict
    keyed by ``(row, col)`` of :class:`gridsmith.s7_normalize.Normalized`.
    """
    ts = TableScore()
    by_rc = {(c.row, c.col): c for c in cells}
    n_rows, n_cols, n_header = grid.n_rows, grid.n_cols, grid.n_header_rows

    labels = {}
    for c in cells:
        if c.col == 0:
            labels[c.row] = (c.raw_text or "").strip()

    # ---- table-level signals -------------------------------------------------
    header_cols = len({c.col for c in cells
                       if c.row < n_header and (c.raw_text or "").strip()})
    body_cols = len({c.col for c in cells
                     if c.row >= n_header and (c.raw_text or "").strip()})
    # Graded, not binary.  As a 0/1 flag this fired on all 680 cells of Shell.pdf --
    # whose header legitimately does not label its subtotal columns -- and subtracted a
    # flat 1.10 from every cell in the document.  A table-level signal can only move a
    # whole table, so its magnitude has to be proportionate to how much of the header
    # is actually missing, or it just declares an entire correct document untrustworthy.
    colcount_mismatch = 0.0
    if n_header > 0 and header_cols and body_cols and header_cols < body_cols - 1:
        colcount_mismatch = min(1.0, (body_cols - header_cols) / float(body_cols))

    row_sig = {}
    for r in range(n_header, n_rows):
        sig = tuple((by_rc.get((r, c)).raw_text or "").strip()
                    for c in range(n_cols) if (r, c) in by_rc)
        row_sig.setdefault(sig, []).append(r)
    dup_rows = {r for sig, rs in row_sig.items()
                if len(rs) > 1 and any(s for s in sig) for r in rs[1:]}

    # ---- reconciliation ------------------------------------------------------
    df = down_foot(by_rc, norms, n_rows, n_cols, n_header,
                   grid.indent_levels, labels, col_types=col_types)
    cf = cross_foot(norms, grid.header_paths, n_rows, n_cols, n_header,
                    col_types=col_types)
    ts.reconciliation = {"down_foot": df, "cross_foot": cf,
                         "n_pass": sum(1 for x in df + cf if x["status"] == "pass"),
                         "n_fail": sum(1 for x in df + cf if x["status"] == "fail"),
                         "n_incomplete": sum(1 for x in df + cf
                                             if x["status"] == "incomplete"),
                         "n_skipped": sum(1 for x in df + cf
                                          if x["status"] == "skipped")}
    fail_cells, contrib_cells = set(), set()
    for x in df + cf:
        if x["status"] != "fail":
            continue
        fail_cells.add((x["row"], x["col"]))
        if x["kind"] == "down_foot":
            for r in x["contributors"]:
                contrib_cells.add((r, x["col"]))
        else:
            for c in x["contributors"]:
                contrib_cells.add((x["row"], c))

    # ---- per-column digit stats ---------------------------------------------
    col_digits = {}
    for c in range(n_cols):
        col_digits[c] = [sum(ch.isdigit() for ch in (by_rc[(r, c)].raw_text or ""))
                         for r in range(n_header, n_rows) if (r, c) in by_rc]

    for cell in sorted(cells, key=lambda c: (c.row, c.col)):
        key = (cell.row, cell.col)
        nrm = norms.get(key)
        codes = [f for f in (cell.flags or [])]
        codes += [f for f in ((nrm.flags if nrm else []) or [])]

        src_ocr = 1.0 if source_of.get(key) == "ocr" else 0.0
        ocr_conf = ocr_conf_of.get(key)
        ocr_conf = 1.0 if ocr_conf is None else float(ocr_conf)

        ctype = col_types.get(cell.col)
        type_conflict = 0.0
        if (nrm is not None and not cell.is_header
                and nrm.value_type not in NULLISH_TYPES
                and ctype not in (None, T_TEXT, T_REFERENCE)):
            got = "number" if nrm.value_type in NUMERIC_TYPES else nrm.value_type
            want = "number" if ctype in NUMERIC_TYPES else ctype
            if got != want:
                type_conflict = 1.0
                codes.append("TYPE_CONFLICT")

        n_dig = sum(ch.isdigit() for ch in (cell.raw_text or ""))
        dig_anom = 0.0 if cell.is_header else _digit_count_anomaly(
            n_dig, col_digits.get(cell.col, []))
        if dig_anom > 0.5:
            codes.append("DIGIT_COUNT_ANOMALY")

        snap = _snap_residual(cell, grid.x_lines, grid.y_lines)
        multiline = 1.0 if cell.n_lines > 1 else 0.0
        watermark = 1.0 if "WATERMARK_OVERLAP" in codes else 0.0
        straddle = 1.0 if "TOKEN_STRADDLES_COLUMN" in codes else 0.0

        crossfoot_fail = 1.0 if key in fail_cells and any(
            x["kind"] == "cross_foot" and (x["row"], x["col"]) == key
            and x["status"] == "fail" for x in cf) else 0.0
        downfoot_fail = 1.0 if key in fail_cells and any(
            x["kind"] == "down_foot" and (x["row"], x["col"]) == key
            and x["status"] == "fail" for x in df) else 0.0
        if crossfoot_fail:
            codes.append("CROSSFOOT_FAIL")
        if downfoot_fail:
            codes.append("DOWNFOOT_FAIL")
        if key in contrib_cells and not (crossfoot_fail or downfoot_fail):
            codes.append("DOWNFOOT_CONTRIBUTOR")

        dup = 1.0 if cell.row in dup_rows else 0.0
        if dup:
            codes.append("DUPLICATE_ROW")

        row_filled = sum(1 for c in range(n_cols)
                         if (cell.row, c) in by_rc
                         and (by_rc[(cell.row, c)].raw_text or "").strip())
        empty_dense = 1.0 if (not (cell.raw_text or "").strip()
                              and n_cols >= 3
                              and row_filled >= 0.7 * n_cols) else 0.0
        if empty_dense:
            codes.append("EMPTY_IN_DENSE_ROW")

        low_support = max(0.0, 1.0 - grid_support)
        if low_support > 0.5:
            codes.append("GRID_LOW_SUPPORT")
        if cell.raw_text is None:
            codes.append("GRID_UNRESOLVED" if not codes else codes[0])

        feats = {
            "source_ocr": src_ocr, "ocr_conf": ocr_conf, "snap_residual": snap,
            "straddle": straddle, "type_conflict": type_conflict,
            "digit_anomaly": dig_anom, "multiline": multiline,
            "crossfoot_fail": crossfoot_fail, "downfoot_fail": downfoot_fail,
            "colcount_mismatch": colcount_mismatch, "dup_row": dup,
            "watermark": watermark, "empty_in_dense_row": empty_dense,
            "grid_low_support": low_support,
        }
        logit = (cfg.w_bias
                 + cfg.w_source_ocr * src_ocr
                 + cfg.w_ocr_conf * (ocr_conf - 1.0) * src_ocr
                 + cfg.w_snap_residual * snap
                 + cfg.w_straddle * straddle
                 + cfg.w_type_conflict * type_conflict
                 + cfg.w_digit_anomaly * dig_anom
                 + cfg.w_multiline * multiline
                 + cfg.w_crossfoot_fail * crossfoot_fail
                 + cfg.w_downfoot_fail * downfoot_fail
                 + cfg.w_colcount_mismatch * colcount_mismatch
                 + cfg.w_dup_row * dup
                 + cfg.w_watermark * watermark
                 + cfg.w_empty_in_dense_row * empty_dense
                 + cfg.w_grid_low_support * low_support)
        raw = sigmoid(logit)
        conf = calib.apply(raw)

        codes = sorted(set(c for c in codes if c))
        mandatory = bool(MANDATORY_REVIEW_CODES & set(codes))
        if mandatory:
            # Hard cap: a known-suspect cell can never ship as high confidence.
            conf = min(conf, cfg.high_confidence_floor - 0.05)
        if cell.raw_text is None:
            conf = 0.0

        ts.cells.append(CellScore(row=cell.row, col=cell.col, features=feats,
                                  logit=round(logit, 5), raw_confidence=round(raw, 5),
                                  confidence=round(conf, 5), codes=codes,
                                  mandatory=mandatory))

    body = [c for c in ts.cells if c.row >= n_header]
    pool = body or ts.cells
    ts.table_confidence = round(
        (sum(c.confidence for c in pool) / len(pool)) if pool else 0.0, 5)
    return ts


def issues_for(table_score, table_id, cell_id_of):
    """Turn per-cell codes into ``issues`` rows."""
    out = []
    for cs in table_score.cells:
        for code in cs.codes:
            if code not in MANDATORY_REVIEW_CODES and code not in SOFT_CODES:
                continue
            sev = SEVERITY_OF.get(code, "info" if code in SOFT_CODES else "warn")
            out.append({
                "table_id": table_id,
                "cell_id": cell_id_of.get((cs.row, cs.col)),
                "severity": sev, "code": code,
                "message": "cell r%d c%d: %s (confidence %.3f)"
                           % (cs.row, cs.col, code, cs.confidence)})
    for rec in (table_score.reconciliation.get("down_foot", [])
                + table_score.reconciliation.get("cross_foot", [])):
        if rec["status"] in ("pass", "skipped"):
            continue
        code = "DOWNFOOT_FAIL" if rec["kind"] == "down_foot" else "CROSSFOOT_FAIL"
        if rec["status"] == "incomplete":
            code = "RECONCILIATION_INCOMPLETE"
        msg = rec.get("message")
        if not msg:
            msg = ("%s at r%d c%d: stated %s, contributors sum to %s (delta %s, "
                   "tolerance %s) -- not corrected, see LIMITATIONS"
                   % (rec["kind"], rec["row"], rec["col"], rec.get("stated"),
                      rec.get("computed"), rec.get("delta"), rec.get("tolerance")))
        out.append({"table_id": table_id,
                    "cell_id": cell_id_of.get((rec["row"], rec["col"])),
                    "severity": "error" if rec["status"] == "fail" else "warn",
                    "code": code, "message": msg})
    return out


# --------------------------------------------------------------------------------------------------
# S9b -- the ranked human-review queue.
# --------------------------------------------------------------------------------------------------
# The target from section 4: flag no more than 8% of cells while those flags cover at
# least 95% of the actual errors.  Two rules make that shape achievable without breaking
# the invariant that no suspect cell is ever shipped unflagged:
#
# 1.  **Mandatory cells are never budgeted out.**  Anything carrying a
#     :data:`gridsmith.s9_confidence.MANDATORY_REVIEW_CODES` code is in the queue even if
#     it alone exceeds 8%.  A budget that could evict a known-suspect cell would make the
#     8% number a lie.
# 2.  **The discretionary tail is ranked by confidence** and cut at the budget.
#
# Both the budgeted queue (``review_queue.csv``) and the *full* ranking are produced,
# because a single operating point cannot show a precision-recall curve and the brief
# asks for the curve, not the point.  ``metrics/eval.py`` sweeps the full ranking.
#
# Ranking key: mandatory first, then ascending confidence, then descending severity, then
# (table, row, col) so the order is total and reproducible.

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
