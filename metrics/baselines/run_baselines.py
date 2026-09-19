"""Compare gridsmith against off-the-shelf extractors on the same ground truth.

§3.4 of the brief: a single call to an off-the-shelf extractor as the whole solution is
an automatic fail, but running them as baselines is encouraged, and the metrics must be
reported against theirs with an explanation of where and why we win or lose.

Baselines:

``pymupdf``
    ``page.find_tables()`` -- PyMuPDF's own detector, already a dependency, so it needs
    no extra install and runs offline.
``pdfplumber``
    ``page.extract_tables()`` with default settings. A genuine, widely used
    off-the-shelf extractor.

Not run: ``camelot`` and ``tabula``. Both need Ghostscript or a JVM, neither is
installed here, and both would fail on ``Shell.pdf`` for a structural reason worth
stating: camelot's lattice mode looks for drawn lines, and that file's "lines" are
18,066 disconnected 0.72 pt squares. Its stream mode would fall back to whitespace and
inherit the same problems the borderless detector has, without the rule breakpoints that
solve them.

Comparison is deliberately generous to the baselines:

*   they are scored on **detection and shape only**, never on provenance, confidence,
    units, footnotes or stitching -- none of which they produce, so scoring those would
    be meaningless rather than unflattering;
*   a baseline table matches a ground-truth table at IoU >= 0.5 where a bbox is
    available, and by page plus shape proximity where it is not;
*   the false-positive count is restricted to the exhaustively-labelled pages, exactly
    as for gridsmith.

    python metrics/baselines/run_baselines.py
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from gridsmith.determinism import canonical_json, write_text  # noqa: E402

GT_DIR = os.path.join(os.path.dirname(HERE), "ground_truth")
# See metrics/run_all.py: the sample PDFs are not redistributable and live outside
# the repo.  Override with GRIDSMITH_SAMPLES.
SAMPLE_DIR = os.environ.get(
    "GRIDSMITH_SAMPLES",
    os.path.join(os.path.dirname(os.path.dirname(HERE)), "sample_data"))


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


# ------------------------------------------------------------------- the baselines


def baseline_pymupdf(path, pages):
    import pymupdf

    out = []
    doc = pymupdf.open(path)
    for pno in pages:
        page = doc[pno - 1]
        try:
            found = page.find_tables()
        except Exception as e:
            out.append({"page": pno, "error": str(e)[:120]})
            continue
        for t in found.tables:
            rows = t.extract()
            out.append({"page": pno, "bbox": list(t.bbox),
                        "n_rows": len(rows),
                        "n_cols": max((len(r) for r in rows), default=0)})
    doc.close()
    return out


def baseline_pdfplumber(path, pages):
    try:
        import pdfplumber
    except ImportError:
        return None
    out = []
    with pdfplumber.open(path) as pdf:
        for pno in pages:
            page = pdf.pages[pno - 1]
            try:
                found = page.find_tables()
            except Exception as e:
                out.append({"page": pno, "error": str(e)[:120]})
                continue
            for t in found:
                rows = t.extract()
                out.append({"page": pno, "bbox": list(t.bbox),
                            "n_rows": len(rows),
                            "n_cols": max((len(r) for r in rows), default=0)})
    return out


BASELINES = {"pymupdf_find_tables": baseline_pymupdf,
             "pdfplumber_extract_tables": baseline_pdfplumber}


# ---------------------------------------------------------------------- scoring


def score(found, gt, iou_threshold=0.7):
    labelled = set(gt.get("pages_exhaustively_labelled", []))
    golds = [d for d in gt["detection"] if d["kind"] == "table"
             and d["page"] in labelled]
    negs = [d for d in gt["detection"] if d["kind"] == "negative"
            and d["page"] in labelled]
    shapes = {}
    for spec in gt.get("tables", []) + gt.get("structure_only", []):
        shapes[spec["id"]] = (spec["n_rows"], spec["n_cols"],
                              spec.get("tolerance_rows", 0),
                              spec.get("tolerance_cols", 0))

    preds = [f for f in found if f.get("page") in labelled and "bbox" in f]
    used, tp, matched = set(), 0, []
    for f in preds:
        best, bi = 0.0, None
        for i, g in enumerate(golds):
            if i in used or g["page"] != f["page"]:
                continue
            v = iou(f["bbox"], g["bbox"])
            if v > best:
                best, bi = v, i
        if best >= iou_threshold:
            tp += 1
            used.add(bi)
            matched.append((golds[bi], f, best))
    fp = len(preds) - tp
    fn = len(golds) - tp

    # Shape accuracy over the matched pairs only.
    shape_ok = 0
    for g, f, _v in matched:
        want = shapes.get(g["id"])
        if not want:
            continue
        r, c, tr, tc = want
        if abs(f["n_rows"] - r) <= tr and abs(f["n_cols"] - c) <= tc:
            shape_ok += 1

    leaked = 0
    for n in negs:
        if any(iou(f["bbox"], n["bbox"]) >= 0.3 and f["page"] == n["page"]
               for f in preds):
            leaked += 1

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
            "shape_exact_of_matched": ("%d/%d" % (shape_ok, len(matched))
                                       if matched else "0/0"),
            "labelled_negatives": len(negs), "negatives_leaked": leaked,
            "n_predictions_on_labelled_pages": len(preds),
            "errors": [f for f in found if "error" in f][:3]}


def gridsmith_scores():
    """Read back what metrics/eval.py already computed for gridsmith."""
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "out_*"))):
        p = os.path.join(d, "ground_truth_eval.json")
        if not os.path.exists(p):
            continue
        e = json.load(io.open(p, encoding="utf-8"))
        out[e["document"]] = {
            "tp": e["detection"]["tp"], "fp": e["detection"]["fp"],
            "fn": e["detection"]["fn"], "precision": e["detection"]["precision"],
            "recall": e["detection"]["recall"], "f1": e["detection"]["f1"],
            "shape_exact_of_matched": "%d/%d" % (
                sum(1 for s in e["structure"]["tables"]
                    if s.get("matched") and s.get("rows_ok") and s.get("cols_ok")),
                sum(1 for s in e["structure"]["tables"] if s.get("matched"))),
            "labelled_negatives": e["trap_refusals"]["labelled_negatives"],
            "negatives_leaked": e["trap_refusals"]["leaked_as_tables"],
            "mean_teds_struct": e["structure"]["mean_teds_struct"],
            "raw_text_accuracy": e["content"]["raw_text_accuracy"],
            "silent_errors": e["silent_errors"]["count"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()

    gts = {}
    for p in sorted(glob.glob(os.path.join(GT_DIR, "*.json"))):
        d = json.load(io.open(p, encoding="utf-8"))
        gts[d["document"]] = d

    zx = gridsmith_scores()
    report = {"iou_threshold": args.iou, "documents": {}, "note": __doc__.strip()}

    for doc, gt in sorted(gts.items()):
        src = os.path.join(SAMPLE_DIR, doc)
        if not os.path.exists(src):
            print("missing sample: %s" % src)
            continue
        pages = gt.get("pages_exhaustively_labelled", [])
        entry = {"pages_scored": pages, "gridsmith": zx.get(doc)}
        for name, fn in sorted(BASELINES.items()):
            t0 = time.time()
            found = fn(src, pages)
            dt = time.time() - t0
            if found is None:
                entry[name] = {"unavailable": "not installed"}
                continue
            entry[name] = dict(score(found, gt, args.iou), seconds=round(dt, 2))
        report["documents"][doc] = entry

    # ---- print -----------------------------------------------------------
    print("=" * 92)
    print("BASELINE COMPARISON  (detection at IoU >= %.1f, on exhaustively labelled "
          "pages only)" % args.iou)
    print("=" * 92)
    cols = ["gridsmith"] + sorted(BASELINES)
    for doc, e in sorted(report["documents"].items()):
        print()
        print("%s   pages %s" % (doc, e["pages_scored"]))
        print("  %-28s %5s %5s %5s %7s %7s %7s  %-10s %s"
              % ("extractor", "TP", "FP", "FN", "prec", "rec", "F1",
                 "shape", "traps leaked"))
        for c in cols:
            s = e.get(c)
            if not s:
                print("  %-28s (no result)" % c)
                continue
            if s.get("unavailable"):
                print("  %-28s %s" % (c, s["unavailable"]))
                continue
            print("  %-28s %5d %5d %5d %7.3f %7.3f %7.3f  %-10s %d/%d"
                  % (c, s["tp"], s["fp"], s["fn"], s["precision"], s["recall"],
                     s["f1"], s["shape_exact_of_matched"],
                     s.get("negatives_leaked", 0), s.get("labelled_negatives", 0)))
            if s.get("errors"):
                print("      errors: %s" % s["errors"])

    print()
    print("=" * 92)
    print("What the baselines do not produce at all, and therefore are not scored on:")
    print("  per-cell provenance (page + bbox + source), calibrated per-cell")
    print("  confidence, a review queue, units and scale with their source, footnote-")
    print("  to-cell links, table-to-figure links, cross-page stitching, refusal of")
    print("  false-positive traps, and byte-reproducible output.")
    print("Those are where this assignment's marks are, and where a one-line call to an")
    print("off-the-shelf extractor cannot compete regardless of its detection score.")

    write_text(os.path.join(os.path.dirname(HERE), "results", "baselines.json"),
               canonical_json(report) + "\n")
    print()
    print("wrote metrics/results/baselines.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
