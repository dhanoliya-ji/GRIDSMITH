"""Pooled reliability diagram and flagging precision-recall curve.

The per-document numbers in ``eval.py`` cannot show a PR curve for the default
configuration, because on the labelled sample set the default configuration makes no
errors -- a curve needs negatives.  The negatives come from the ablated runs
(``metrics/run_all.py``), which extract the *same* documents against the *same* ground
truth with a degraded configuration.

Pooling them is the honest way to answer the question the brief actually asks: "when
this pipeline says 0.7, how often is it right, and if I review everything below a
threshold, what share of the errors do I catch?"  Both answers are properties of the
confidence function, not of one document.

    python metrics/calibration_report.py
"""
from __future__ import annotations

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from gridsmith import canonical_json, write_text  # noqa: E402

POINTS = os.path.join(HERE, "results", "calibration_points.json")
HIGH_CONF_FLOOR = 0.85
BUDGET = 0.08


def reliability(points, n_bins=10):
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        sel = [p for p in points
               if (lo <= p["confidence"] < hi)
               or (b == n_bins - 1 and p["confidence"] >= hi)]
        if not sel:
            bins.append({"lo": lo, "hi": hi, "n": 0, "mean_conf": None, "acc": None})
            continue
        bins.append({
            "lo": round(lo, 2), "hi": round(hi, 2), "n": len(sel),
            "mean_conf": round(sum(p["confidence"] for p in sel) / len(sel), 4),
            "acc": round(sum(1 for p in sel if p["correct"]) / len(sel), 4)})
    filled = [b for b in bins if b["n"]]
    ece = (sum(b["n"] * abs(b["mean_conf"] - b["acc"]) for b in filled)
           / sum(b["n"] for b in filled)) if filled else None
    mce = max((abs(b["mean_conf"] - b["acc"]) for b in filled), default=None)
    return bins, ece, mce


def pr_curve(points):
    n_err = sum(1 for p in points if not p["correct"])
    rows = []
    for i in range(0, 101, 2):
        thr = i / 100.0
        flagged = [p for p in points if p["confidence"] < thr or p.get("mandatory")]
        tp = sum(1 for p in flagged if not p["correct"])
        rows.append({
            "threshold": thr,
            "flagged_frac": round(len(flagged) / max(1, len(points)), 4),
            "error_recall": round(tp / max(1, n_err), 4),
            "flag_precision": round(tp / max(1, len(flagged)), 4) if flagged else None,
        })
    return rows, n_err


def bar(v, width=34, ch="#"):
    if v is None:
        return ""
    return ch * int(round(v * width))


def main():
    if not os.path.exists(POINTS):
        print("no %s -- run metrics/run_all.py first" % POINTS)
        return 1
    pts = json.load(io.open(POINTS, encoding="utf-8"))
    pts = [p for p in pts if p.get("confidence") is not None]
    n_err = sum(1 for p in pts if not p["correct"])

    print("=" * 78)
    print("POOLED CALIBRATION REPORT")
    print("=" * 78)
    by_cfg = {}
    for p in pts:
        by_cfg.setdefault(p.get("config", "?"), []).append(p)
    print("  points: %d   errors: %d   accuracy: %.3f"
          % (len(pts), n_err, 1 - n_err / max(1, len(pts))))
    for cfg in sorted(by_cfg):
        sub = by_cfg[cfg]
        e = sum(1 for p in sub if not p["correct"])
        print("      %-16s %4d points, %3d errors (%.1f%% accurate)"
              % (cfg, len(sub), e, 100 * (1 - e / max(1, len(sub)))))

    bins, ece, mce = reliability(pts)
    print()
    print("  RELIABILITY DIAGRAM  (perfect calibration: accuracy == confidence)")
    print("  %-12s %6s  %-36s %-36s" % ("bin", "n", "stated confidence",
                                        "empirical accuracy"))
    for b in bins:
        if not b["n"]:
            continue
        print("  %.1f-%.1f      %6d  %-36s %-36s"
              % (b["lo"], b["hi"], b["n"],
                 "%s %.3f" % (bar(b["mean_conf"], 26), b["mean_conf"]),
                 "%s %.3f" % (bar(b["acc"], 26, "="), b["acc"])))
    print("  expected calibration error (ECE): %.4f" % (ece or 0.0))
    print("  maximum calibration error  (MCE): %.4f" % (mce or 0.0))

    rows, _ = pr_curve(pts)
    print()
    print("  FLAGGING PRECISION-RECALL  (target: flag <= 8%% of cells, catch >= 95%%)")
    print("  %-10s %-13s %-14s %s" % ("threshold", "cells flagged", "errors caught",
                                      "flag precision"))
    shown = [r for r in rows if r["threshold"] in
             (0.0, 0.5, 0.6, 0.7, 0.8, 0.84, 0.86, 0.9, 0.94, 0.98, 1.0)]
    for r in shown:
        print("  %-10.2f %-13s %-14s %s"
              % (r["threshold"], "%.1f%%" % (100 * r["flagged_frac"]),
                 "%.1f%%" % (100 * r["error_recall"]),
                 ("%.3f" % r["flag_precision"]) if r["flag_precision"] else "-"))

    # The operating point the pipeline actually ships.
    op = [p for p in pts if p.get("mandatory") or p["confidence"] < HIGH_CONF_FLOOR]
    tp = sum(1 for p in op if not p["correct"])
    print()
    print("  SHIPPED OPERATING POINT (mandatory codes + confidence < %.2f)"
          % HIGH_CONF_FLOOR)
    print("      flagged        : %d cells (%.1f%%)"
          % (len(op), 100 * len(op) / max(1, len(pts))))
    print("      errors caught  : %d of %d (%.1f%%)"
          % (tp, n_err, 100 * tp / max(1, n_err)))
    print("      flag precision : %.3f" % (tp / max(1, len(op))))
    silent = [p for p in pts if not p["correct"]
              and not (p.get("mandatory") or p["confidence"] < HIGH_CONF_FLOOR)]
    print("      SILENT ERRORS  : %d (%.2f%% of all cells)"
          % (len(silent), 100 * len(silent) / max(1, len(pts))))

    # What the 8% budget alone would catch, ranked by confidence.
    ranked = sorted(pts, key=lambda p: (0 if p.get("mandatory") else 1,
                                        p["confidence"]))
    keep = max(int(round(BUDGET * len(pts))),
               sum(1 for p in pts if p.get("mandatory")))
    caught = sum(1 for p in ranked[:keep] if not p["correct"])
    print()
    print("  AT THE 8%% REVIEW BUDGET (ranked queue, mandatory first)")
    print("      queue length   : %d cells (%.1f%%)"
          % (keep, 100 * keep / max(1, len(pts))))
    print("      errors caught  : %d of %d (%.1f%%)"
          % (caught, n_err, 100 * caught / max(1, n_err)))

    payload = {"n_points": len(pts), "n_errors": n_err,
               "per_config": {k: {"n": len(v),
                                  "errors": sum(1 for p in v if not p["correct"])}
                              for k, v in sorted(by_cfg.items())},
               "reliability": {"bins": bins, "ece": round(ece or 0.0, 4),
                               "mce": round(mce or 0.0, 4)},
               "pr_curve": rows,
               "operating_point": {
                   "floor": HIGH_CONF_FLOOR, "flagged": len(op),
                   "flagged_frac": round(len(op) / max(1, len(pts)), 4),
                   "errors_caught": tp,
                   "error_recall": round(tp / max(1, n_err), 4),
                   "flag_precision": round(tp / max(1, len(op)), 4),
                   "silent_errors": len(silent)},
               "budget_point": {"budget_frac": BUDGET, "queue_length": keep,
                                "errors_caught": caught,
                                "error_recall": round(caught / max(1, n_err), 4)}}
    write_text(os.path.join(HERE, "results", "calibration_report.json"),
               canonical_json(payload) + "\n")
    print()
    print("wrote metrics/results/calibration_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
