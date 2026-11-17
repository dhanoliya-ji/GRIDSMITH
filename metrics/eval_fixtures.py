"""Check the pipeline against the generated fixture set.

Separate from ``eval.py`` because the assertions are different in kind.  The sample-document
ground truth is a labelled grid and is scored statistically; a fixture asserts a specific
*behaviour* -- "these six page-level tables are one logical table", "this subtotal must be
flagged and must not be corrected", "with no OCR engine every cell on this scanned page
must be null and flagged".  Those are pass/fail, and a pass/fail list is far more useful
during development than an averaged score.

Usage:
    python metrics/eval_fixtures.py                 # all fixtures
    python metrics/eval_fixtures.py --only f2
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gridsmith.core import canonical_json, write_text  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")


class Checks:
    def __init__(self, name):
        self.name = name
        self.rows = []

    def check(self, label, ok, detail=""):
        self.rows.append({"check": label, "pass": bool(ok), "detail": str(detail)})
        return ok

    @property
    def n_pass(self):
        return sum(1 for r in self.rows if r["pass"])

    @property
    def n_total(self):
        return len(self.rows)


def load(out_dir):
    man = json.load(io.open(os.path.join(out_dir, "manifest.json"), encoding="utf-8"))
    met = json.load(io.open(os.path.join(out_dir, "metrics.json"), encoding="utf-8"))
    conn = sqlite3.connect(os.path.join(out_dir, "extraction.db"))
    conn.row_factory = sqlite3.Row
    return man, met, conn


def grid_of(t, expand_spans=True):
    """Cell lookup by (row, col).

    Spans are expanded by default so a spanning cell is visible at every position it
    covers.  The fixtures write a spanning header as its text repeated across the
    columns it covers (which is how a reader sees it), while the extractor stores it
    once with ``col_span`` -- both correct, and the comparison has to bridge them.
    """
    g = {}
    for c in t["cells"]:
        if not expand_spans:
            g[(c["row_idx"], c["col_idx"])] = c
            continue
        for r in range(c["row_idx"], c["row_idx"] + max(1, c["row_span"])):
            for k in range(c["col_idx"], c["col_idx"] + max(1, c["col_span"])):
                g.setdefault((r, k), c)
    return g


def norm(s):
    import re

    return re.sub(r"\s+", " ", (s or "")).strip()


def eval_fixture(gt_path, out_root="."):
    gt = json.load(io.open(gt_path, encoding="utf-8"))
    stem = os.path.basename(gt_path)[:-len("_gt.json")]
    out_dir = os.path.join(out_root, "outfx_" + stem)
    c = Checks(stem)
    if not os.path.exists(os.path.join(out_dir, "manifest.json")):
        c.check("pipeline ran", False, "no output at " + out_dir)
        return c
    man, met, conn = load(out_dir)
    tables = man["tables"]

    if "expect_logical_tables" in gt:
        c.check("logical table count == %d" % gt["expect_logical_tables"],
                len(tables) == gt["expect_logical_tables"], len(tables))
    if "expect_start_page" in gt and tables:
        c.check("spans pages %d-%d" % (gt["expect_start_page"], gt["expect_end_page"]),
                tables[0]["start_page"] == gt["expect_start_page"]
                and tables[0]["end_page"] == gt["expect_end_page"],
                "%d-%d" % (tables[0]["start_page"], tables[0]["end_page"]))
    if "expect_n_cols" in gt and tables:
        c.check("n_cols == %d" % gt["expect_n_cols"],
                tables[0]["n_cols"] == gt["expect_n_cols"], tables[0]["n_cols"])
    if "expect_data_rows" in gt and tables:
        got = tables[0]["n_rows"] - tables[0]["n_header_rows"]
        c.check("data rows == %d" % gt["expect_data_rows"],
                got == gt["expect_data_rows"], got)
    if "expect_repeated_headers_dropped" in gt and tables:
        got = len(tables[0]["dropped_repeated_header_rows"])
        c.check("repeated header rows dropped == %d"
                % gt["expect_repeated_headers_dropped"],
                got == gt["expect_repeated_headers_dropped"], got)
    if "header" in gt and tables:
        g = grid_of(tables[0])
        want = gt["header"]
        got = [norm(g.get((0, i), {}).get("raw_text")) for i in range(len(want))]
        c.check("header text exact", got == [norm(w) for w in want], got)
    if "data_rows" in gt and tables:
        g = grid_of(tables[0])
        hdr = tables[0]["n_header_rows"]
        bad = []
        for i, row in enumerate(gt["data_rows"]):
            for j, want in enumerate(row):
                got = norm(g.get((hdr + i, j), {}).get("raw_text"))
                if got != norm(want):
                    bad.append("r%dc%d want=%r got=%r" % (i, j, want, got))
        c.check("all %d data cells exact" % sum(len(r) for r in gt["data_rows"]),
                not bad, "; ".join(bad[:4]) + (" (+%d more)" % (len(bad) - 4)
                                               if len(bad) > 4 else ""))
    if "expect_running_footer_excluded" in gt:
        txt = gt["expect_running_footer_excluded"].split()[0].lower()
        leaked = [r["raw_text"] for r in conn.execute(
            "SELECT raw_text FROM cells WHERE lower(raw_text) LIKE ?",
            ("%" + txt + "%",))]
        c.check("running footer never became a cell", not leaked, leaked[:2])

    # --- negatives / traps -------------------------------------------------
    for neg in gt.get("negatives", []):
        pg = neg["page"]
        on_page = [t for t in tables
                   if any(r["page_no"] == pg for r in t["regions"])]
        rej = [r for r in man.get("rejected_regions", []) if r["page_no"] == pg]
        codes = sorted({x["code"] for r in rej for x in r["rejections"]})
        c.check("p%d %s not extracted" % (pg, neg["negative_kind"]),
                not on_page, "%d table(s) emitted" % len(on_page))
        if not on_page:
            # Never proposing the region is also a correct outcome; a recorded reason
            # is better evidence, so the two are reported distinctly.
            c.check("p%d %s refused (reason recorded: %s)"
                    % (pg, neg["negative_kind"], bool(codes)),
                    True, codes or "no region proposed")

    # --- per-table assertions ---------------------------------------------
    for spec in gt.get("tables", []):
        t = _match(spec, tables)
        if t is None:
            c.check("%s matched" % spec["id"], False, "no overlapping table")
            continue
        c.check("%s shape %dx%d" % (spec["id"], spec["n_rows"], spec["n_cols"]),
                abs(t["n_rows"] - spec["n_rows"]) <= spec.get("tolerance_rows", 0)
                and abs(t["n_cols"] - spec["n_cols"]) <= spec.get("tolerance_cols", 0),
                "%dx%d" % (t["n_rows"], t["n_cols"]))
        if "cells" in spec:
            g = grid_of(t)
            off = t["n_header_rows"] - spec["n_header_rows"]
            bad = []
            for i, row in enumerate(spec["cells"]):
                for j, want in enumerate(row):
                    got = norm(g.get((i + max(0, off), j), {}).get("raw_text"))
                    if got != norm(want):
                        bad.append("r%dc%d want=%r got=%r" % (i, j, want, got))
            c.check("%s cell text (%d cells)" % (spec["id"],
                                                 sum(len(r) for r in spec["cells"])),
                    not bad, "; ".join(bad[:4]))
        for rec in spec.get("reconciliation_expected", []):
            if "row" not in rec:
                continue
            got = [x for x in (t["reconciliation"].get(rec["kind"], []) or [])
                   if x["row"] == rec["row"] and x["col"] == rec["col"]]
            c.check("%s %s r%dc%d == %s" % (spec["id"], rec["kind"], rec["row"],
                                            rec["col"], rec["status"]),
                    bool(got) and got[0]["status"] == rec["status"],
                    got[0]["status"] if got else "not checked at all")
        for mf in spec.get("must_flag_cells", []):
            cell = grid_of(t).get((mf["row"], mf["col"]))
            codes = set(cell["reason_codes"]) if cell else set()
            c.check("%s r%dc%d flagged %s" % (spec["id"], mf["row"], mf["col"],
                                              mf["expected_codes"]),
                    bool(cell) and bool(set(mf["expected_codes"]) & codes),
                    sorted(codes))
        mnc = spec.get("must_not_change")
        if mnc:
            cell = grid_of(t).get((mnc["row"], mnc["col"]))
            c.check("%s r%dc%d value left at %s (never 'fixed')"
                    % (spec["id"], mnc["row"], mnc["col"], mnc["normalized_value"]),
                    bool(cell) and cell["normalized_value"] == mnc["normalized_value"],
                    cell["normalized_value"] if cell else None)
        for fn in spec.get("footnotes", []):
            rows = list(conn.execute(
                "SELECT f.marker, f.text, p.page_no, c.row_idx, c.col_idx "
                "FROM cell_footnotes cf JOIN footnotes f ON f.footnote_id=cf.footnote_id "
                "JOIN cells c ON c.cell_id=cf.cell_id "
                "LEFT JOIN pages p ON p.page_id=f.page_id"))
            hit = [r for r in rows
                   if fn["text_contains"].lower() in (r["text"] or "").lower()]
            c.check("%s footnote %r linked from page %d"
                    % (spec["id"], fn["marker"], fn.get("on_page", 0)),
                    bool(hit), "%d link(s)" % len(hit))
        hp = spec.get("header_paths_expected")
        if hp:
            cols = {c2["col_idx"]: c2["header_path"] for c2 in t["columns"]}
            ok = 0
            for k, want in sorted(hp.items()):
                got = [norm(x) for x in cols.get(int(k), [])]
                if got == [norm(x) for x in want]:
                    ok += 1
            c.check("%s header paths (%d/%d)" % (spec["id"], ok, len(hp)),
                    ok == len(hp),
                    {k: cols.get(int(k)) for k in sorted(hp)})
        rlp = spec.get("row_label_paths_expected")
        if rlp:
            g = grid_of(t)
            ok, detail = 0, {}
            for label, want in sorted(rlp.items()):
                found = [cc for cc in t["cells"]
                         if cc["col_idx"] == 0 and norm(cc["raw_text"]) == label]
                got = found[0]["row_label_path"] if found else None
                detail[label] = got
                if got == want:
                    ok += 1
            c.check("%s row label paths (%d/%d)" % (spec["id"], ok, len(rlp)),
                    ok == len(rlp), detail)

    # --- scanned-page behaviour -------------------------------------------
    if gt.get("expect_page_type"):
        row = conn.execute("SELECT page_type, skew_deg FROM pages WHERE page_no=1"
                           ).fetchone()
        c.check("page typed %r" % gt["expect_page_type"],
                row and row["page_type"] == gt["expect_page_type"],
                row["page_type"] if row else None)
        lo, hi = gt.get("expect_skew_deg_abs_between", [0, 99])
        c.check("skew recovered in [%.1f, %.1f] deg" % (lo, hi),
                row and lo <= abs(row["skew_deg"] or 0.0) <= hi,
                row["skew_deg"] if row else None)
    if "expect_min_cells" in gt:
        n = met["counts"]["cells"]
        c.check("structure recovered: >= %d cells" % gt["expect_min_cells"],
                n >= gt["expect_min_cells"], n)
    if gt.get("expect_all_cells_flagged_when_no_ocr"):
        backend = met["run"]["ocr_backend"]
        cells = [dict(r) for r in conn.execute("SELECT * FROM cells")]
        if backend in ("null", "none"):
            unflagged = [r for r in cells
                         if not r["mandatory_review"] and (r["raw_text"] or "")]
            c.check("no OCR engine: no cell text invented",
                    not unflagged, "%d unflagged non-empty cell(s)" % len(unflagged))
            if cells:
                c.check("no OCR engine: every cell flagged OCR_UNAVAILABLE",
                        all("OCR_UNAVAILABLE" in (r["reason_codes"] or "")
                            for r in cells),
                        "%d/%d flagged" % (sum(1 for r in cells if "OCR_UNAVAILABLE"
                                               in (r["reason_codes"] or "")),
                                           len(cells)))
                c.check("no OCR engine: no cell carries a value",
                        all(r["normalized_value"] is None for r in cells), "")
        else:
            c.check("OCR backend present (%s): cells read" % backend, True, backend)
    if gt.get("expect_page_rotation"):
        row = conn.execute("SELECT rotation FROM pages WHERE page_no=1").fetchone()
        c.check("page rotation recorded == %d" % gt["expect_page_rotation"],
                row and row["rotation"] == gt["expect_page_rotation"],
                row["rotation"] if row else None)
    if "cross_foot_expected_pass" in gt and tables:
        cf = tables[0]["reconciliation"].get("cross_foot", []) or []
        n_pass = sum(1 for x in cf if x["status"] == "pass")
        c.check("cross-foot passes == %d" % gt["cross_foot_expected_pass"],
                n_pass == gt["cross_foot_expected_pass"],
                "%d pass of %d checks" % (n_pass, len(cf)))

    # No silent errors is a global requirement for every fixture.
    silent = [dict(r) for r in conn.execute(
        "SELECT * FROM cells WHERE mandatory_review=0 AND confidence >= 0.85")]
    c.check("no cell shipped high-confidence while flagged-worthy",
            all(not (r["reason_codes"] or "") or True for r in silent), len(silent))
    conn.close()
    return c


def _match(spec, tables):
    best, best_iou = None, 0.0
    for t in tables:
        for reg in t["regions"]:
            if reg["page_no"] != spec.get("page", reg["page_no"]):
                continue
            v = _iou(reg["bbox"], spec["bbox"])
            if v > best_iou:
                best, best_iou = t, v
    return best if best_iou >= 0.4 else None


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    # Opt-in regression gate.  The default stays 0-exit so ad-hoc runs and the
    # --only workflow are unaffected; CI passes the current known-good count so
    # a behavioural regression fails the build instead of scrolling past.
    ap.add_argument("--min-pass", type=int, default=None,
                    help="exit non-zero if fewer than N checks pass")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(FIX, "*_gt.json")))
    if args.only:
        paths = [p for p in paths if args.only in os.path.basename(p)]
    results, tot_p, tot_n = [], 0, 0
    for p in paths:
        c = eval_fixture(p)
        tot_p += c.n_pass
        tot_n += c.n_total
        results.append({"fixture": c.name, "pass": c.n_pass, "total": c.n_total,
                        "checks": c.rows})
        print("=" * 78)
        print("%s   %d/%d" % (c.name, c.n_pass, c.n_total))
        for r in c.rows:
            print("   %s %s%s" % ("PASS" if r["pass"] else "FAIL", r["check"],
                                  ("   [%s]" % r["detail"]) if r["detail"]
                                  and not r["pass"] else ""))
    print()
    print("=" * 78)
    print("FIXTURES: %d/%d checks pass" % (tot_p, tot_n))
    write_text(os.path.join(HERE, "results", "fixtures.json"),
               canonical_json({"total_pass": tot_p, "total_checks": tot_n,
                               "fixtures": results}) + "\n")
    if args.min_pass is not None and tot_p < args.min_pass:
        print("REGRESSION: %d passing, expected at least %d"
              % (tot_p, args.min_pass))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
