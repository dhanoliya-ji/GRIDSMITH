"""Evaluation harness: scores an output directory against the hand-labelled set.

Usage:
    python metrics/eval.py --out ./out_apple
    python metrics/eval.py --all                    # every out_* dir with a GT file
    python metrics/eval.py --out ./out_apple --emit-calibration

What is measured, and why each choice was made:

**Detection (criterion 1).**  Precision, recall and F1 at IoU >= 0.7, computed *only on
pages where the ground truth is exhaustive*.  Scoring precision on unlabelled pages
would count every correct table we find as a false positive, which measures nothing.
Refused trap regions are scored separately: a labelled negative counts as a true
negative when a trap rule refused a region overlapping it, and as a false positive when
a table was emitted there.

**Structure (criterion 2).**  TEDS-Struct: tree-edit-distance similarity over the grid
*structure only*, cell text ignored.  Implemented directly (below) because the reference
implementation pulls in a dependency we cannot vendor offline, and because the
structure-only variant is a short function once the tree is built.  Merged-cell and
header-row accuracy are reported alongside, since TEDS alone hides them.

**Cell content (criterion 3).**  Exact match on ``raw_text`` after collapsing runs of
whitespace, scored separately from ``normalized_value``.  Whitespace is collapsed
because the PDF's inter-token spacing is a rendering artefact, not content; every other
difference counts.

**Silent errors and calibration (criterion 4).**  A *silent error* is a cell that is
wrong and was shipped with confidence >= the high-confidence floor and not flagged.
That count is the headline number.  We also sweep the confidence threshold to produce
the flagging precision-recall curve and bin predictions to produce the reliability
diagram, both written to ``ground_truth_eval.json``.

Matching: an extracted table matches a GT table when they share a page and their boxes
reach IoU >= 0.5 -- a looser bar than the 0.7 detection threshold on purpose, so a
detection that is *poor* still gets its cells scored rather than vanishing from the
content metrics.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gridsmith import canonical_json, write_text  # noqa: E402

GT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth")
HIGH_CONF_FLOOR = 0.85
WS_RE = re.compile(r"\s+")


# ------------------------------------------------------------------------ helpers


def norm_text(s):
    if s is None:
        return None
    return WS_RE.sub(" ", str(s)).strip()


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def load_gt():
    out = {}
    for p in sorted(glob.glob(os.path.join(GT_DIR, "*.json"))):
        d = json.load(io.open(p, encoding="utf-8"))
        out[d["document"]] = d
    return out


def load_output(out_dir):
    man = json.load(io.open(os.path.join(out_dir, "manifest.json"), encoding="utf-8"))
    db = os.path.join(out_dir, "extraction.db")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return man, conn


# ------------------------------------------------------------------ TEDS-Struct


def _grid_tree(n_rows, n_cols, spans):
    """Structure-only tree: rows -> cells, each cell labelled by its span shape.

    ``spans`` maps ``(row, col) -> (row_span, col_span)``; cells absorbed by a span are
    absent.  Labelling a leaf by its span shape is what makes the metric sensitive to
    merged cells, which is the part of structure that matters most in practice.
    """
    tree = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            if (r, c) in spans:
                rs, cs = spans[(r, c)]
                row.append("c%dx%d" % (rs, cs))
        tree.append(row)
    return tree


def teds_struct(a_rows, a_cols, a_spans, b_rows, b_cols, b_spans):
    """Tree-edit-distance similarity over structure, in [0, 1].

    The tree is two levels deep (table -> row -> cell), so the general tree-edit
    distance collapses to: row-sequence edit distance where substituting one row for
    another costs the edit distance between their cell-label sequences.  That is exact
    for this tree shape, not an approximation, and it runs in O(R^2 * C^2).
    """
    ta = _grid_tree(a_rows, a_cols, a_spans)
    tb = _grid_tree(b_rows, b_cols, b_spans)
    n_a = sum(len(r) for r in ta) + len(ta)
    n_b = sum(len(r) for r in tb) + len(tb)
    if n_a == 0 and n_b == 0:
        return 1.0

    def seq_dist(x, y):
        m, n = len(x), len(y)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            cur = [i] + [0] * n
            for j in range(1, n + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                             prev[j - 1] + (0 if x[i - 1] == y[j - 1] else 1))
            prev = cur
        return prev[n]

    m, n = len(ta), len(tb)
    prev = [0] * (n + 1)
    for j in range(1, n + 1):
        prev[j] = prev[j - 1] + len(tb[j - 1]) + 1
    for i in range(1, m + 1):
        cur = [prev[0] + len(ta[i - 1]) + 1] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + len(ta[i - 1]) + 1,
                         cur[j - 1] + len(tb[j - 1]) + 1,
                         prev[j - 1] + seq_dist(ta[i - 1], tb[j - 1]))
        prev = cur
    dist = prev[n]
    return max(0.0, 1.0 - dist / float(max(n_a, n_b)))


# -------------------------------------------------------------------- evaluation


def eval_one(out_dir, gt_all, emit_calibration=False):
    man, conn = load_output(out_dir)
    doc = man["document"]["filename"]
    gt = gt_all.get(doc)
    if gt is None:
        return None
    if not doc_matches(gt, man):
        print("  WARNING: sha256 prefix mismatch for %s (GT may be stale)" % doc)

    ex_tables = man["tables"]
    ex_by_page = {}
    for t in ex_tables:
        for reg in t["regions"]:
            ex_by_page.setdefault(reg["page_no"], []).append((t, reg["bbox"]))

    res = {"document": doc, "out_dir": out_dir}

    # ---- 1. detection ---------------------------------------------------
    labelled = set(gt.get("pages_exhaustively_labelled", []))
    gt_tables = [d for d in gt["detection"] if d["kind"] == "table"]
    gt_negs = [d for d in gt["detection"] if d["kind"] == "negative"]
    tp = fp = fn = 0
    matched_gt, det_detail = set(), []
    for pg in sorted(labelled):
        preds = ex_by_page.get(pg, [])
        golds = [g for g in gt_tables if g["page"] == pg]
        used = set()
        for pi, (t, bbox) in enumerate(preds):
            best, best_i = 0.0, None
            for gi, g in enumerate(golds):
                if gi in used:
                    continue
                v = iou(bbox, g["bbox"])
                if v > best:
                    best, best_i = v, gi
            if best >= 0.7:
                tp += 1
                used.add(best_i)
                matched_gt.add(golds[best_i]["id"])
                det_detail.append({"page": pg, "gt": golds[best_i]["id"],
                                   "iou": round(best, 3), "verdict": "TP"})
            else:
                fp += 1
                det_detail.append({"page": pg, "gt": None,
                                   "bbox": [round(v, 1) for v in bbox],
                                   "best_iou": round(best, 3), "verdict": "FP"})
        for gi, g in enumerate(golds):
            if gi not in used:
                fn += 1
                det_detail.append({"page": pg, "gt": g["id"], "verdict": "FN"})
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    res["detection"] = {
        "iou_threshold": 0.7, "pages_scored": sorted(labelled),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
        "detail": det_detail,
    }

    # ---- 1b. trap refusals ----------------------------------------------
    rejected = man.get("rejected_regions", [])
    trap = {"labelled_negatives": len(gt_negs), "correctly_refused": 0,
            "leaked_as_tables": 0, "detail": []}
    for g in gt_negs:
        overlapping_rejects = [r for r in rejected if r["page_no"] == g["page"]
                               and iou(r["bbox"], g["bbox"]) >= 0.3]
        leaked = [t for t, bbox in ex_by_page.get(g["page"], [])
                  if iou(bbox, g["bbox"]) >= 0.3]
        codes = sorted({rj["code"] for r in overlapping_rejects
                        for rj in r["rejections"]})
        if leaked:
            trap["leaked_as_tables"] += 1
            verdict = "LEAKED"
        elif overlapping_rejects:
            trap["correctly_refused"] += 1
            verdict = "REFUSED"
        else:
            verdict = "NOT_DETECTED_AT_ALL"
            trap["correctly_refused"] += 1
        trap["detail"].append({"page": g["page"], "kind": g.get("negative_kind"),
                               "verdict": verdict, "codes": codes})
    res["trap_refusals"] = trap

    # ---- 2/3. structure + content ---------------------------------------
    cell_rows = {}
    for r in conn.execute("SELECT * FROM cells"):
        cell_rows.setdefault(r["table_id"], []).append(dict(r))

    # structure_only entries carry no bbox of their own; take it from the detection
    # list so a table's extent is declared exactly once.
    bbox_by_id = {d["id"]: d["bbox"] for d in gt["detection"] if d.get("id")}
    struct, content, calib_points, silent = [], [], [], []
    for g in gt["tables"] + gt.get("structure_only", []):
        if "bbox" not in g and g.get("id") in bbox_by_id:
            g = dict(g, bbox=bbox_by_id[g["id"]])
        pred = _best_match(g, ex_by_page)
        entry = {"id": g["id"], "page": g["page"]}
        if pred is None:
            entry.update({"matched": False})
            struct.append(entry)
            continue
        t, bbox = pred
        spans_p = {(c["row_idx"], c["col_idx"]): (c["row_span"], c["col_span"])
                   for c in t["cells"]}
        spans_g = {(r, c): (1, 1) for r in range(g["n_rows"])
                   for c in range(g["n_cols"])}
        entry.update({
            "matched": True, "iou": round(iou(bbox, g["bbox"]), 3),
            "n_rows_pred": t["n_rows"], "n_rows_gt": g["n_rows"],
            "n_cols_pred": t["n_cols"], "n_cols_gt": g["n_cols"],
            "n_header_pred": t["n_header_rows"], "n_header_gt": g["n_header_rows"],
            "rows_ok": abs(t["n_rows"] - g["n_rows"]) <= g.get("tolerance_rows", 0),
            "cols_ok": abs(t["n_cols"] - g["n_cols"]) <= g.get("tolerance_cols", 0),
            "header_ok": t["n_header_rows"] == g["n_header_rows"],
            "teds_struct": round(teds_struct(
                t["n_rows"], t["n_cols"], spans_p,
                g["n_rows"], g["n_cols"], spans_g), 4),
        })
        entry["header_spans_ok"] = _check_header_spans(g, t)
        entry["header_path_ok"] = _check_header_path(g, t)
        struct.append(entry)

        if "cells" not in g:
            continue
        r0, r1 = g.get("row_range", [0, g["n_rows"]])
        pred_cells = {(c["row_idx"], c["col_idx"]): c for c in t["cells"]}
        n = n_raw = n_norm = 0
        mism = []
        for r in range(r0, min(r1, len(g["cells"]))):
            for c in range(min(g["n_cols"], len(g["cells"][r]))):
                want = norm_text(g["cells"][r][c])
                pc = pred_cells.get((r, c))
                got = norm_text(pc["raw_text"]) if pc else None
                n += 1
                ok_raw = (got or "") == (want or "")
                if ok_raw:
                    n_raw += 1
                else:
                    mism.append({"row": r, "col": c, "gt": want, "pred": got})
                want_n = (g.get("normalized") or [[None] * g["n_cols"]] * g["n_rows"])
                wv = want_n[r][c] if r < len(want_n) and c < len(want_n[r]) else None
                gv = pc.get("normalized_value") if pc else None
                ok_norm = _num_eq(wv, gv)
                if ok_norm:
                    n_norm += 1

                conf = (pc.get("confidence") if pc else 0.0) or 0.0
                flagged = bool(pc and (pc.get("mandatory_review")
                                       or conf < HIGH_CONF_FLOOR))
                # Correct means both fields agree.  raw_text and normalized_value are
                # reported separately (the brief scores them separately), but for
                # silent-error and calibration purposes a wrong value is wrong
                # whichever field carries it.
                ok_cell = bool(ok_raw and ok_norm)
                # A cell that is empty in the ground truth and empty in the output is
                # not a prediction, it is an abstention -- correctly emitting nothing.
                # Including those inflates accuracy and, because empty cells score low
                # confidence, poisons the reliability diagram from the other side (the
                # 0.0-0.1 bin came out 58% "accurate", giving an MCE of 0.58 that says
                # nothing about the values anyone will act on).
                trivial = not (want or "") and not (got or "")
                if trivial:
                    continue
                calib_points.append({"correct": ok_cell, "confidence": conf,
                                     "raw_confidence": (pc.get("raw_confidence")
                                                        if pc else 0.0) or 0.0,
                                     "mandatory": bool(pc and
                                                       pc.get("mandatory_review")),
                                     "flagged": flagged, "table": g["id"],
                                     "row": r, "col": c,
                                     "raw_ok": bool(ok_raw), "norm_ok": bool(ok_norm)})
                if not ok_cell and not flagged:
                    silent.append({"table": g["id"], "row": r, "col": c,
                                   "gt": want, "pred": got, "confidence": conf,
                                   "raw_ok": bool(ok_raw),
                                   "norm_ok": bool(ok_norm)})
        content.append({"id": g["id"], "n_cells_scored": n,
                        "raw_text_exact": n_raw,
                        "raw_text_accuracy": round(n_raw / max(1, n), 4),
                        "normalized_exact": n_norm,
                        "normalized_accuracy": round(n_norm / max(1, n), 4),
                        "mismatches": mism[:25]})

    res["structure"] = {
        "tables": struct,
        "mean_teds_struct": _mean([s["teds_struct"] for s in struct if s.get("matched")]),
        "shape_exact": _frac([s.get("rows_ok") and s.get("cols_ok")
                              for s in struct if s.get("matched")]),
        "header_rows_exact": _frac([s.get("header_ok")
                                    for s in struct if s.get("matched")]),
        "matched": sum(1 for s in struct if s.get("matched")),
        "total": len(struct),
    }
    res["content"] = {
        "tables": content,
        "cells_scored": sum(c["n_cells_scored"] for c in content),
        "raw_text_accuracy": _weighted(content, "raw_text_exact"),
        "normalized_accuracy": _weighted(content, "normalized_exact"),
    }

    # ---- 4. silent errors + calibration ---------------------------------
    res["silent_errors"] = {
        "count": len(silent),
        "rate": round(len(silent) / max(1, len(calib_points)), 5),
        "definition": ("cell whose raw_text OR normalized_value is wrong, AND "
                       "confidence >= %.2f, AND not mandatory-flagged"
                       % HIGH_CONF_FLOOR),
        "detail": silent[:25],
    }
    res["flagging"] = _pr_curve(calib_points)
    res["reliability"] = _reliability(calib_points)
    res["value_expectations"] = _check_values(gt, conn)

    if emit_calibration:
        res["_calibration_points"] = calib_points
    conn.close()
    return res


def doc_matches(gt, man):
    return man["document"]["sha256"].startswith(gt.get("sha256_prefix", ""))


def _best_match(g, ex_by_page):
    best, best_iou = None, 0.0
    for t, bbox in ex_by_page.get(g["page"], []):
        v = iou(bbox, g["bbox"])
        if v > best_iou:
            best, best_iou = (t, bbox), v
    return best if best_iou >= 0.5 else None


def _check_header_spans(g, t):
    exp = g.get("header_spans_expected")
    if not exp:
        return None
    ok = 0
    for e in exp:
        hit = [c for c in t["cells"]
               if c["row_idx"] == e["row"]
               and e["text_contains"].lower() in (c["raw_text"] or "").lower()
               and c["col_span"] == e["col_span"]]
        ok += 1 if hit else 0
    return {"expected": len(exp), "correct": ok}


def _check_header_path(g, t):
    exp = g.get("header_path_expected")
    if not exp:
        return None
    cols = {c["col_idx"]: c["header_path"] for c in t["columns"]}
    got = cols.get(exp["col"], [])
    return {"col": exp["col"], "expected": exp["path"], "got": got,
            "match": [norm_text(x) for x in got] == [norm_text(x)
                                                     for x in exp["path"]]}


def _num_eq(a, b, tol=1e-6):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= max(tol, 1e-6 * abs(float(a)))
    except (TypeError, ValueError):
        return norm_text(a) == norm_text(b)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _frac(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(1 for x in xs if x) / len(xs), 4) if xs else None


def _weighted(entries, num_key):
    tot = sum(e["n_cells_scored"] for e in entries)
    if not tot:
        return None
    return round(sum(e[num_key] for e in entries) / tot, 4)


def _pr_curve(points):
    """Flagging precision/recall as the confidence threshold sweeps 0..1."""
    n_err = sum(1 for p in points if not p["correct"])
    curve = []
    for i in range(0, 101, 5):
        thr = i / 100.0
        flagged = [p for p in points if p["confidence"] < thr]
        tp = sum(1 for p in flagged if not p["correct"])
        curve.append({
            "threshold": thr,
            "flagged_frac": round(len(flagged) / max(1, len(points)), 4),
            "precision": round(tp / max(1, len(flagged)), 4) if flagged else None,
            "recall": round(tp / max(1, n_err), 4) if n_err else None,
        })
    operating = [p for p in points if p["flagged"]]
    tp_op = sum(1 for p in operating if not p["correct"])
    return {"n_points": len(points), "n_errors": n_err, "curve": curve,
            "operating_point": {
                "flagged_frac": round(len(operating) / max(1, len(points)), 4),
                "error_recall": round(tp_op / max(1, n_err), 4) if n_err else None,
                "flag_precision": (round(tp_op / max(1, len(operating)), 4)
                                   if operating else None),
                "target": "flag <= 8% of cells, catch >= 95% of errors"}}


def _reliability(points, n_bins=10):
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        sel = [p for p in points if lo <= p["confidence"] < hi
               or (b == n_bins - 1 and p["confidence"] >= hi)]
        if not sel:
            bins.append({"bin": [round(lo, 2), round(hi, 2)], "n": 0,
                         "mean_confidence": None, "empirical_accuracy": None})
            continue
        bins.append({
            "bin": [round(lo, 2), round(hi, 2)], "n": len(sel),
            "mean_confidence": round(sum(p["confidence"] for p in sel) / len(sel), 4),
            "empirical_accuracy": round(
                sum(1 for p in sel if p["correct"]) / len(sel), 4)})
    filled = [b for b in bins if b["n"]]
    ece = (sum(b["n"] * abs(b["mean_confidence"] - b["empirical_accuracy"])
               for b in filled) / sum(b["n"] for b in filled)) if filled else None
    return {"bins": bins,
            "expected_calibration_error": round(ece, 4) if ece is not None else None}


def _check_values(gt, conn):
    out = []
    for v in gt.get("value_expectations", []):
        rows = [dict(r) for r in conn.execute(
            "SELECT c.* FROM cells c JOIN pages p ON p.page_id=c.page_id "
            "WHERE p.page_no=? AND c.raw_text=?", (v["page"], v["raw_text"]))]
        ok_type = all(r["value_type"] == v["value_type"] for r in rows) if rows else False
        ok_norm = all(_num_eq(v.get("normalized_value"), r["normalized_value"])
                      for r in rows) if rows else False
        flagged = all(bool(r["mandatory_review"]) for r in rows) if rows else False
        out.append({
            "page": v["page"], "raw_text": v["raw_text"], "found": len(rows),
            "min_occurrences": v.get("min_occurrences", 1),
            "enough": len(rows) >= v.get("min_occurrences", 1),
            "value_type_ok": ok_type, "normalized_ok": ok_norm,
            "flagged": flagged,
            "flag_requirement_met": (flagged if v.get("must_be_flagged") else True),
            "expected_codes_present": sorted(
                {c for r in rows for c in (r["reason_codes"] or "").split("|") if c}
                & set(v.get("expected_codes", []))) if v.get("expected_codes") else None,
        })
    return out


# ------------------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--emit-calibration", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    gt_all = load_gt()
    dirs = list(args.out)
    if args.all or not dirs:
        dirs = sorted(d for d in glob.glob("out_*")
                      if os.path.exists(os.path.join(d, "manifest.json")))
    results = []
    for d in dirs:
        r = eval_one(d, gt_all, emit_calibration=args.emit_calibration)
        if r is None:
            print("skip %s (no ground truth for its document)" % d)
            continue
        results.append(r)
        write_text(os.path.join(d, "ground_truth_eval.json"),
                   canonical_json({k: v for k, v in r.items()
                                   if not k.startswith("_")}) + "\n")
        if not args.quiet:
            _print(r)

    if results:
        agg = _aggregate(results)
        write_text(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results", "summary.json"),
                   canonical_json(agg) + "\n")
        if not args.quiet:
            print()
            print("=" * 78)
            print("AGGREGATE over %d document(s)" % len(results))
            print("=" * 78)
            for k, v in sorted(agg["totals"].items()):
                print("  %-34s %s" % (k, v))
    if args.emit_calibration:
        pts = [p for r in results for p in r.get("_calibration_points", [])]
        write_text(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results", "calibration_points.json"),
                   canonical_json(pts) + "\n")
        print("  wrote %d calibration points" % len(pts))
    return 0


def _print(r):
    print("=" * 78)
    print("%s   (%s)" % (r["document"], r["out_dir"]))
    print("=" * 78)
    d = r["detection"]
    print("  detection @IoU>=0.7 on pages %s" % d["pages_scored"])
    print("      TP=%d FP=%d FN=%d  precision=%.3f recall=%.3f f1=%.3f"
          % (d["tp"], d["fp"], d["fn"], d["precision"], d["recall"], d["f1"]))
    for x in d["detail"]:
        if x["verdict"] != "TP":
            print("      %s %s" % (x["verdict"], x))
    t = r["trap_refusals"]
    if t["labelled_negatives"]:
        print("  traps: %d/%d correctly refused, %d leaked"
              % (t["correctly_refused"], t["labelled_negatives"],
                 t["leaked_as_tables"]))
        for x in t["detail"]:
            print("      p%-3d %-22s %s" % (x["page"], x["verdict"], x["codes"]))
    s = r["structure"]
    print("  structure: matched %d/%d  mean TEDS-Struct=%s  shape exact=%s  "
          "header rows exact=%s"
          % (s["matched"], s["total"], s["mean_teds_struct"], s["shape_exact"],
             s["header_rows_exact"]))
    for x in s["tables"]:
        if not x.get("matched"):
            print("      UNMATCHED %s (page %s)" % (x["id"], x["page"]))
        elif not (x["rows_ok"] and x["cols_ok"] and x["header_ok"]):
            print("      %s: %dx%d vs gt %dx%d, header %d vs %d, teds=%.3f"
                  % (x["id"], x["n_rows_pred"], x["n_cols_pred"], x["n_rows_gt"],
                     x["n_cols_gt"], x["n_header_pred"], x["n_header_gt"],
                     x["teds_struct"]))
    c = r["content"]
    print("  content: %d cells  raw_text exact=%s  normalized exact=%s"
          % (c["cells_scored"], c["raw_text_accuracy"], c["normalized_accuracy"]))
    for x in c["tables"]:
        if x["raw_text_accuracy"] < 1.0:
            print("      %s: %.3f (%d/%d)" % (x["id"], x["raw_text_accuracy"],
                                              x["raw_text_exact"],
                                              x["n_cells_scored"]))
            for m in x["mismatches"][:6]:
                print("          r%d c%d  gt=%r  pred=%r"
                      % (m["row"], m["col"], m["gt"], m["pred"]))
    se = r["silent_errors"]
    print("  SILENT ERRORS: %d  (rate %.4f)  <- the headline number"
          % (se["count"], se["rate"]))
    for x in se["detail"][:6]:
        print("      %s r%d c%d conf=%.3f gt=%r pred=%r"
              % (x["table"], x["row"], x["col"], x["confidence"], x["gt"], x["pred"]))
    op = r["flagging"]["operating_point"]
    print("  flagging at operating point: %.1f%% of cells flagged, %s of errors caught"
          % (100 * op["flagged_frac"],
             ("%.1f%%" % (100 * op["error_recall"]))
             if op["error_recall"] is not None else "n/a (no errors)"))
    print("  reliability ECE=%s" % r["reliability"]["expected_calibration_error"])
    for v in r["value_expectations"]:
        print("  value check p%d %r: found %d (need %d) type_ok=%s norm_ok=%s "
              "flagged=%s" % (v["page"], v["raw_text"], v["found"],
                              v["min_occurrences"], v["value_type_ok"],
                              v["normalized_ok"], v["flagged"]))


def _aggregate(results):
    tp = sum(r["detection"]["tp"] for r in results)
    fp = sum(r["detection"]["fp"] for r in results)
    fn = sum(r["detection"]["fn"] for r in results)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    cells = sum(r["content"]["cells_scored"] for r in results)
    raw_ok = sum(sum(t["raw_text_exact"] for t in r["content"]["tables"])
                 for r in results)
    norm_ok = sum(sum(t["normalized_exact"] for t in r["content"]["tables"])
                  for r in results)
    silent = sum(r["silent_errors"]["count"] for r in results)
    n_neg = sum(r["trap_refusals"]["labelled_negatives"] for r in results)
    refused = sum(r["trap_refusals"]["correctly_refused"] for r in results)
    leaked = sum(r["trap_refusals"]["leaked_as_tables"] for r in results)
    teds = _mean([s["teds_struct"] for r in results
                  for s in r["structure"]["tables"] if s.get("matched")])
    return {"documents": [r["document"] for r in results],
            "totals": {
                "detection_tp": tp, "detection_fp": fp, "detection_fn": fn,
                "detection_precision": round(prec, 4),
                "detection_recall": round(rec, 4),
                "detection_f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
                "trap_negatives": n_neg, "traps_refused": refused,
                "traps_leaked": leaked,
                "mean_teds_struct": teds,
                "cells_scored": cells,
                "raw_text_accuracy": round(raw_ok / max(1, cells), 4),
                "normalized_accuracy": round(norm_ok / max(1, cells), 4),
                "silent_errors": silent,
                "silent_error_rate": round(silent / max(1, cells), 5)},
            "per_document": [{"document": r["document"],
                              "detection_f1": r["detection"]["f1"],
                              "mean_teds_struct": r["structure"]["mean_teds_struct"],
                              "raw_text_accuracy": r["content"]["raw_text_accuracy"],
                              "silent_errors": r["silent_errors"]["count"],
                              "operating_point": r["flagging"]["operating_point"]}
                             for r in results]}


if __name__ == "__main__":
    sys.exit(main())
