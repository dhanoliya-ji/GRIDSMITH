"""Fit the confidence calibration curve offline and vendor it into the package.

Run after ``metrics/run_all.py``:

    python metrics/fit_calibration.py
    python metrics/fit_calibration.py --dry-run        # print, do not write

Why offline.  §3.2 requires byte-identical reruns, so nothing may be fitted while
extracting.  The logistic weights in ``gridsmith.config`` are fixed by hand and
documented; the only *learned* object is the monotone map from that logistic score to a
probability, and it is fitted here, once, and shipped as ``gridsmith/calibration.json``
-- a lookup table the runtime reads and never modifies.

Method: **isotonic regression by pool-adjacent-violators (PAV)**.

Ordinary equal-width binning gives a reliability diagram straight away but a
non-monotone one: with a few hundred points some bin will, by sampling noise, report
lower accuracy than the bin below it, and a confidence function that is not monotone in
its own score is indefensible ("0.7 means less than 0.6 here"). PAV is the standard fix
and it is about fifteen lines: bin, then repeatedly merge any adjacent pair that
violates monotonicity, replacing both with their pooled accuracy. The result is the
closest monotone fit to the data in least squares, and monotone by construction.

Guard rails, because the labelled set is small:

*   bins with fewer than ``--min-bin`` points are merged into their neighbour before
    fitting, so a 3-point bin cannot dictate a plateau;
*   fitted values are shrunk toward the pooled base rate by a factor that decays with
    bin size (a 6-point bin reporting 100% is not evidence of 100%);
*   the top bin is capped at ``--max-confidence`` so no cell is ever emitted at 1.00,
    which would claim certainty the method cannot support.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from gridsmith.core import canonical_json, write_text  # noqa: E402

POINTS = os.path.join(HERE, "results", "calibration_points.json")
OUT = os.path.join(ROOT, "gridsmith", "calibration.json")


def pav(bins):
    """Pool adjacent violators.  ``bins`` is a list of ``[n_total, n_correct]``."""
    stack = []
    for n, k in bins:
        stack.append([n, k])
        while len(stack) > 1 and _rate(stack[-2]) > _rate(stack[-1]) + 1e-12:
            n2, k2 = stack.pop()
            n1, k1 = stack.pop()
            stack.append([n1 + n2, k1 + k2])
    return stack


def _rate(b):
    return b[1] / b[0] if b[0] else 0.0


def fit(points, n_bins=12, min_bin=8, max_confidence=0.99, shrink_k=12.0):
    pts = [(float(p["raw_confidence"]), bool(p["correct"])) for p in points
           if p.get("raw_confidence") is not None]
    pts.sort()
    if len(pts) < 20:
        return None, {"error": "too few points (%d)" % len(pts)}
    base = sum(1 for _v, ok in pts if ok) / len(pts)

    # Equal-count bins: with a skewed score distribution, equal-width bins leave most
    # of the range empty and one bin holding everything.
    per = max(min_bin, len(pts) // n_bins)
    raw_bins, edges = [], []
    i = 0
    while i < len(pts):
        chunk = pts[i:i + per]
        if len(chunk) < min_bin and raw_bins:
            prev_n, prev_k = raw_bins[-1]
            raw_bins[-1] = [prev_n + len(chunk),
                            prev_k + sum(1 for _v, ok in chunk if ok)]
            edges[-1] = chunk[-1][0]
        else:
            raw_bins.append([len(chunk), sum(1 for _v, ok in chunk if ok)])
            edges.append(chunk[-1][0])
        i += per

    pooled = pav(raw_bins)
    # Re-derive edges after pooling: each pooled bin absorbs consecutive raw bins.
    pooled_edges, idx = [], 0
    for b in pooled:
        n_acc = 0
        while idx < len(raw_bins) and n_acc < b[0]:
            n_acc += raw_bins[idx][0]
            idx += 1
        pooled_edges.append(edges[min(idx, len(edges)) - 1])

    values = []
    for b in pooled:
        n, k = b
        rate = k / n
        # Shrink toward the base rate: weight = n / (n + shrink_k).
        w = n / (n + shrink_k)
        v = w * rate + (1 - w) * base
        values.append(min(max_confidence, max(0.01, round(v, 4))))
    # Monotone after shrinkage too (shrinkage is monotone in rate, but round can tie).
    for j in range(1, len(values)):
        values[j] = max(values[j], values[j - 1])

    diag = {
        "n_points": len(pts), "n_correct": sum(1 for _v, ok in pts if ok),
        "base_rate": round(base, 4),
        "n_raw_bins": len(raw_bins), "n_pooled_bins": len(pooled),
        "bins": [{"upper_edge": round(e, 4), "n": b[0], "n_correct": b[1],
                  "empirical": round(_rate(b), 4), "fitted": v}
                 for e, b, v in zip(pooled_edges, pooled, values)],
    }
    payload = {
        "source": "metrics/fit_calibration.py (isotonic PAV over %d labelled cells)"
                  % len(pts),
        "method": "equal-count binning -> pool-adjacent-violators -> shrink to base "
                  "rate (weight n/(n+%g)) -> cap at %g" % (shrink_k, max_confidence),
        "fitted_on": "3 sample documents + 8 generated fixtures, default configuration "
                     "and three ablations (metrics/run_all.py)",
        "edges": [round(e, 4) for e in pooled_edges],
        "values": values,
        "diagnostics": diag,
    }
    return payload, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--min-bin", type=int, default=8)
    ap.add_argument("--max-confidence", type=float, default=0.99)
    args = ap.parse_args()

    if not os.path.exists(POINTS):
        print("no %s -- run metrics/run_all.py first" % POINTS)
        return 1
    points = json.load(io.open(POINTS, encoding="utf-8"))
    payload, diag = fit(points, n_bins=args.bins, min_bin=args.min_bin,
                        max_confidence=args.max_confidence)
    if payload is None:
        print("fit failed: %s" % diag)
        return 1

    print("labelled cells: %d   correct: %d   base rate: %.3f"
          % (diag["n_points"], diag["n_correct"], diag["base_rate"]))
    print("raw bins: %d   after pooling: %d" % (diag["n_raw_bins"],
                                                diag["n_pooled_bins"]))
    print()
    print("  %-12s %6s %9s %9s %9s" % ("score <=", "n", "correct", "empirical",
                                       "fitted"))
    print("  " + "-" * 50)
    for b in diag["bins"]:
        print("  %-12.4f %6d %9d %9.3f %9.3f"
              % (b["upper_edge"], b["n"], b["n_correct"], b["empirical"],
                 b["fitted"]))
    if args.dry_run:
        print("\n(dry run; %s not written)" % OUT)
        return 0
    write_text(OUT, canonical_json(payload) + "\n")
    print("\nwrote %s" % OUT)
    print("re-run metrics/run_all.py to measure the calibrated pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
