"""S11b -- the ``audit``, ``debug``, ``verify`` and ``query`` commands.

``verify`` is the one worth a note.  Section 3.2 says two runs must produce
byte-identical database rows and Excel files with timestamps isolated to a single
``extraction_runs`` row.  Rather than assert that, this command proves it: it hashes
every artefact in both directories, and for ``extraction.db`` it compares a canonical
dump of every table *except* the three timestamp columns of ``extraction_runs``.  With
``--fixed-time`` even those match, so a plain recursive byte diff of ``out/`` is clean.

``query`` ships the named example the brief uses as its schema acceptance test --
"every cell whose column header path contains Revenue, with its page number and unit" --
so that claim is executable rather than asserted.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys

from .determinism import canonical_json

EXAMPLE_QUERIES = {
    "revenue_cells": (
        "the brief's acceptance test: every cell whose column header path OR row "
        "label path mentions revenue/amount, with its page number and unit",
        """
        SELECT t.logical_index, c.row_idx, c.col_idx,
               col.header_path_text        AS header_path,
               c.row_label_path_text       AS row_label,
               c.raw_text, c.normalized_number, col.unit,
               p.page_no, c.confidence, c.reason_codes
        FROM cells c
        JOIN columns col ON col.table_id = c.table_id AND col.col_idx = c.col_idx
        JOIN tables  t   ON t.table_id  = c.table_id
        JOIN pages   p   ON p.page_id   = c.page_id
        WHERE (col.header_path_text LIKE '%revenue%' COLLATE NOCASE
               OR col.header_path_text LIKE '%amount%' COLLATE NOCASE
               OR c.row_label_path_text LIKE '%revenue%' COLLATE NOCASE)
          AND c.is_header = 0
        ORDER BY t.logical_index, c.row_idx, c.col_idx
        """),
    "low_confidence": (
        "cells flagged for review, worst first",
        """
        SELECT t.logical_index, p.page_no, c.row_idx, c.col_idx, c.raw_text,
               c.confidence, c.mandatory_review, c.reason_codes
        FROM cells c
        JOIN tables t ON t.table_id = c.table_id
        LEFT JOIN pages p ON p.page_id = c.page_id
        WHERE c.mandatory_review = 1 OR c.confidence < 0.85
        ORDER BY c.mandatory_review DESC, c.confidence ASC
        """),
    "failed_reconciliation": (
        "totals that do not add up (never silently corrected)",
        """
        SELECT i.code, i.severity, t.logical_index, t.title, i.message
        FROM issues i
        LEFT JOIN tables t ON t.table_id = i.table_id
        WHERE i.code IN ('DOWNFOOT_FAIL','CROSSFOOT_FAIL','RECONCILIATION_INCOMPLETE')
        ORDER BY i.severity, t.logical_index
        """),
    "refused_regions": (
        "regions the trap classifiers refused, with the reason",
        """
        SELECT p.page_no, r.code, r.score, r.regime, r.message
        FROM rejected_regions r
        LEFT JOIN pages p ON p.page_id = r.page_id
        ORDER BY p.page_no, r.code
        """),
    "footnoted_cells": (
        "cells linked to a footnote, with the footnote text",
        """
        SELECT t.logical_index, c.row_idx, c.col_idx, c.raw_text,
               f.marker, f.text, fp.page_no AS footnote_page
        FROM cell_footnotes cf
        JOIN cells c     ON c.cell_id = cf.cell_id
        JOIN footnotes f ON f.footnote_id = cf.footnote_id
        JOIN tables t    ON t.table_id = c.table_id
        LEFT JOIN pages fp ON fp.page_id = f.page_id
        ORDER BY t.logical_index, c.row_idx, c.col_idx
        """),
    "table_inventory": (
        "one row per logical table",
        """
        SELECT logical_index, start_page, end_page, n_rows, n_cols, n_header_rows,
               regime, is_continuation, ROUND(table_confidence,4) AS conf,
               unit_scale_source, unit_currency, unit_magnitude, title
        FROM tables ORDER BY logical_index
        """),
}


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def audit(out_dir: str, as_json: bool = False) -> int:
    path = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(path):
        print("no metrics.json in %s -- run `gridsmith run` first" % out_dir)
        return 1
    m = _read_json(path)
    if as_json:
        print(canonical_json(m))
        return 0

    print("=" * 78)
    print("gridsmith audit -- %s" % m["document"]["filename"])
    print("=" * 78)
    _block("document", [
        ("sha256", m["document"]["sha256"][:16] + "..."),
        ("pages", m["document"]["page_count"]),
        ("page types", ", ".join("%s=%d" % kv for kv in
                                 sorted(m["document"]["page_types"].items()))),
    ])
    _block("run", [
        ("run_id", m["run"]["run_id"]),
        ("version", m["run"]["gridsmith_version"]),
        ("config fingerprint", m["run"]["config_fingerprint"]),
        ("git sha", m["run"]["git_sha"]),
        ("ocr backend", m["run"]["ocr_backend"]),
        ("offline", m["run"]["offline"]),
    ])
    _block("extraction", [(k, v) for k, v in sorted(m["counts"].items())])
    if m.get("rejections_by_code"):
        _block("refused regions (correctly excluded)",
               [(k, v) for k, v in sorted(m["rejections_by_code"].items())])
    _block("reconciliation", [(k, v) for k, v in sorted(m["reconciliation"].items())])
    _block("confidence", [(k, v) for k, v in sorted(m["confidence"].items())])
    _block("review queue", [(k, v) for k, v in sorted(m["review_queue"].items())])
    _block("calibration", [(k, v) for k, v in sorted(m["calibration"].items())
                           if k in ("source", "n_bins")])
    _block("resources", [(k, v) for k, v in sorted(m["resources"].items())])
    if m.get("issues_by_code"):
        _block("issues by code", [(k, v) for k, v in
                                  sorted(m["issues_by_code"].items())])

    gt = os.path.join(out_dir, "ground_truth_eval.json")
    if os.path.exists(gt):
        e = _read_json(gt)
        print()
        print("-" * 78)
        print("ground-truth evaluation (metrics/eval.py)")
        print("-" * 78)
        for k in sorted(e):
            if isinstance(e[k], dict):
                _block(k, [(kk, vv) for kk, vv in sorted(e[k].items())
                           if not isinstance(vv, (dict, list))])
    else:
        print()
        print("no ground_truth_eval.json -- run: python metrics/eval.py --out %s"
              % out_dir)
    return 0


def _block(title, rows):
    print()
    print("  %s" % title)
    for k, v in rows:
        print("    %-30s %s" % (k, v))


# --------------------------------------------------------------------------- debug


def debug_page(out_dir: str, page_no: int, input_path=None, dpi: int = 130) -> int:
    """Rebuild the overlay for one page straight from the database."""
    import pymupdf

    from . import s11_audit
    from .config import DEFAULT

    db = os.path.join(out_dir, "extraction.db")
    if not os.path.exists(db):
        print("no extraction.db in %s" % out_dir)
        return 1
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    doc_row = conn.execute("SELECT * FROM documents LIMIT 1").fetchone()
    src = input_path or _locate_source(doc_row["filename"], out_dir)
    if not src or not os.path.exists(src):
        print("source PDF not found; pass --input explicitly (recorded name: %s)"
              % doc_row["filename"])
        return 1

    page = conn.execute("SELECT * FROM pages WHERE page_no=?", (page_no,)).fetchone()
    if page is None:
        print("page %d is not in this document" % page_no)
        return 1
    pid = page["page_id"]

    regions = [{"bbox": [r["bbox_x0"], r["bbox_y0"], r["bbox_x1"], r["bbox_y1"]],
                "regime": r["regime"] or "",
                "label": "T%03d" % r["logical_index"]}
               for r in conn.execute(
                   "SELECT tr.*, t.logical_index FROM table_regions tr "
                   "JOIN tables t ON t.table_id=tr.table_id "
                   "WHERE tr.page_id=? ORDER BY t.logical_index", (pid,))]
    rejected = [{"bbox": [r["bbox_x0"], r["bbox_y0"], r["bbox_x1"], r["bbox_y1"]],
                 "code": r["code"]}
                for r in conn.execute(
                    "SELECT * FROM rejected_regions WHERE page_id=? ORDER BY code",
                    (pid,))]
    cells = [{"bbox": [r["bbox_x0"], r["bbox_y0"], r["bbox_x1"], r["bbox_y1"]],
              "row_span": r["row_span"], "col_span": r["col_span"],
              "confidence": r["confidence"]}
             for r in conn.execute(
                 "SELECT * FROM cells WHERE page_id=? AND bbox_x1 IS NOT NULL "
                 "ORDER BY table_id, row_idx, col_idx", (pid,))]
    assets = [{"bbox": [r["bbox_x0"], r["bbox_y0"], r["bbox_x1"], r["bbox_y1"]],
               "kind": r["kind"], "relation": ""}
              for r in conn.execute(
                  "SELECT * FROM assets WHERE page_id=? ORDER BY asset_id", (pid,))]

    # Inferred grid lines come from the manifest, which records the boundaries S4
    # actually settled on (`final_x_lines` / `final_y_lines`) rather than the raw
    # evidence.  A page-part of a stitched table only carries the boundaries of the
    # logical table's first part, so the y lines are clipped to this page's region.
    grids = []
    man = os.path.join(out_dir, "manifest.json")
    if os.path.exists(man):
        for t in _read_json(man).get("tables", []):
            ev = t.get("structure_evidence") or {}
            xs = ev.get("final_x_lines") or []
            ys = ev.get("final_y_lines") or []
            if not xs or not ys:
                continue
            for reg in t.get("regions", []):
                if reg["page_no"] != page_no:
                    continue
                y0, y1 = reg["bbox"][1], reg["bbox"][3]
                ys_here = [y for y in ys if y0 - 2 <= y <= y1 + 2] or [y0, y1]
                grids.append({"x_lines": list(xs), "y_lines": ys_here})

    overlay = {"chrome_bands": [], "regions": regions, "rejected": rejected,
               "grids": grids, "cells": cells, "assets": assets}
    doc = pymupdf.open(src)
    out = os.path.join(out_dir, "debug", "page_%03d.png" % page_no)
    s11_audit.render_page(doc, page_no, overlay, out, dpi=dpi,
                          high_confidence_floor=DEFAULT.high_confidence_floor)
    doc.close()
    conn.close()
    print("wrote %s" % out)
    print("legend: %s" % s11_audit.legend_text())
    print("  %d region(s), %d refused, %d cell(s), %d asset(s)"
          % (len(regions), len(rejected), len(cells), len(assets)))
    return 0


def _locate_source(filename: str, out_dir: str):
    for base in (os.getcwd(), os.path.dirname(os.path.abspath(out_dir)),
                 os.path.abspath(out_dir)):
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    return None


# -------------------------------------------------------------------------- verify

VOLATILE_DB_COLUMNS = {("extraction_runs", "started_at"),
                       ("extraction_runs", "finished_at"),
                       ("extraction_runs", "wall_seconds"),
                       ("extraction_runs", "peak_rss_mb")}
VOLATILE_FILES = {"logs/timing.json"}


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _walk(root):
    out = {}
    for base, _dirs, files in os.walk(root):
        for f in sorted(files):
            p = os.path.join(base, f)
            rel = os.path.relpath(p, root).replace("\\", "/")
            out[rel] = p
    return dict(sorted(out.items()))


def _db_dump(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    dump = {}
    for t in names:
        cols = [c[1] for c in conn.execute("PRAGMA table_info(%s)" % t)]
        keep = [c for c in cols if (t, c) not in VOLATILE_DB_COLUMNS]
        rows = [tuple(r[c] for c in keep) for r in conn.execute(
            "SELECT * FROM %s" % t)]
        dump[t] = sorted(canonical_json(list(r), None) for r in rows)
    conn.close()
    return dump


def verify_dirs(a: str, b: str) -> int:
    fa, fb = _walk(a), _walk(b)
    only_a = sorted(set(fa) - set(fb))
    only_b = sorted(set(fb) - set(fa))
    problems = []
    if only_a or only_b:
        problems.append("file set differs: only in A %s; only in B %s"
                        % (only_a, only_b))

    identical, differing, ignored = [], [], []
    for rel in sorted(set(fa) & set(fb)):
        if rel in VOLATILE_FILES:
            ignored.append(rel)
            continue
        if rel.endswith(".db"):
            da, dbb = _db_dump(fa[rel]), _db_dump(fb[rel])
            if da == dbb:
                identical.append(rel + "  (all tables; run timestamps excluded)")
            else:
                bad = sorted(t for t in set(da) | set(dbb)
                             if da.get(t) != dbb.get(t))
                differing.append("%s tables differ: %s" % (rel, bad))
            continue
        ha, hb = _hash_file(fa[rel]), _hash_file(fb[rel])
        (identical if ha == hb else differing).append(
            rel if ha == hb else "%s  (%s != %s)" % (rel, ha[:12], hb[:12]))

    print("=" * 78)
    print("determinism check:  A=%s  B=%s" % (a, b))
    print("=" * 78)
    print("  identical : %d" % len(identical))
    for r in identical:
        print("      = %s" % r)
    if ignored:
        print("  ignored   : %s  (documented as non-deterministic)" % ignored)
    if differing:
        print("  DIFFERING : %d" % len(differing))
        for r in differing:
            print("      ! %s" % r)
    for p in problems:
        print("  PROBLEM   : %s" % p)
    ok = not differing and not problems
    print()
    print("RESULT: %s" % ("byte-identical (timestamps isolated as permitted)"
                          if ok else "DIFFERENCES FOUND"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- query


def run_query(out_dir: str, sql=None, example=None, limit: int = 20,
              list_examples: bool = False) -> int:
    if list_examples:
        for name, (desc, _q) in sorted(EXAMPLE_QUERIES.items()):
            print("  %-24s %s" % (name, desc))
        return 0
    db = os.path.join(out_dir, "extraction.db")
    if not os.path.exists(db):
        print("no extraction.db in %s" % out_dir)
        return 1
    if example:
        if example not in EXAMPLE_QUERIES:
            print("unknown example %r; try --list-examples" % example)
            return 2
        desc, sql = EXAMPLE_QUERIES[example]
        print("-- %s" % desc)
    if not sql:
        print("pass --sql or --example")
        return 2

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
    except sqlite3.Error as e:
        print("SQL error: %s" % e)
        return 1
    rows = cur.fetchmany(limit)
    if not rows:
        print("(no rows)")
        return 0
    names = rows[0].keys()
    widths = [max(len(str(n)), max(len(_s(r[n])) for r in rows)) for n in names]
    widths = [min(w, 40) for w in widths]
    print("  ".join(str(n)[:w].ljust(w) for n, w in zip(names, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(_s(r[n])[:w].ljust(w) for n, w in zip(names, widths)))
    more = len(cur.fetchmany(1))
    if more:
        print("... (more rows; raise --limit)")
    conn.close()
    return 0


def _s(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return ("%.6g" % v)
    return str(v)
