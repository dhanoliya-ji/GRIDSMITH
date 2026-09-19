"""Run every configuration over every document, evaluate, and collect the results.

This does three jobs at once, which is why it exists as one script:

1.  **The headline metrics** -- the default configuration over the three sample
    documents and the eight fixtures.
2.  **The ablations** the brief asks for.  §3.1 says any model must be shown "with it
    switched off"; we have no learned model, but the same obligation applies to every
    component we *do* rely on, so each detector and the stitcher are switched off in
    turn and re-measured.  A component whose removal costs nothing was not earning its
    place.
3.  **Calibration data.**  With the default configuration the labelled set contains
    almost no errors, which is good for accuracy and useless for calibration: a curve
    fitted on a set with no negatives cannot say anything about the low-confidence end.
    The ablated runs extract the *same* documents against the *same* ground truth with
    a degraded configuration, so they generate genuine errors with genuine confidence
    scores -- exactly the (prediction, outcome) pairs a reliability diagram needs.

    This is legitimate because the confidence model is not being fitted to the ablation
    *labels*; it is being asked, across a wide range of extraction quality, "when you
    said 0.7, how often were you right?"

Usage:
    python metrics/run_all.py                  # everything
    python metrics/run_all.py --no-ablations   # default config only
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from gridsmith.determinism import canonical_json, write_text  # noqa: E402

# The three assignment PDFs are not redistributable, so they are not in the repo.
# Point GRIDSMITH_SAMPLES at a directory holding them; absent that, the sample
# runs are skipped with a notice and the fixture suite still runs in full.
SAMPLE_DIR = os.environ.get("GRIDSMITH_SAMPLES",
                            os.path.join(ROOT, "sample_data"))

SAMPLES = [
    ("mrpl", "Shell MRPL Aviation Fuels and Services Limited.pdf"),
    ("shell", "Shell.pdf"),
    ("apple", "Apple Form 10-K.pdf"),
]

CONFIGS = [
    ("default", []),
    ("no_ruled", ["--no-ruled"]),
    ("no_borderless", ["--no-borderless"]),
    ("no_stitch", ["--no-stitch"]),
]


def run(cmd, quiet=True):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, cwd=ROOT)
    if r.returncode != 0 and not quiet:
        sys.stderr.write(r.stderr.decode("utf-8", "replace")[-2000:])
    return r.returncode, time.time() - t0, r.stdout.decode("utf-8", "replace")


def env():
    e = dict(os.environ)
    e["PYTHONPATH"] = os.path.join(ROOT, "src")
    e["PYTHONIOENCODING"] = "utf-8"
    return e


def gridsmith(input_path, out_dir, extra):
    cmd = [sys.executable, "-m", "gridsmith", "run", "--input", input_path,
           "--out", out_dir, "--offline", "--quiet"] + extra
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, cwd=ROOT, env=env())
    dt = time.time() - t0
    if r.returncode != 0:
        print("   FAILED %s: %s" % (out_dir,
                                    r.stderr.decode("utf-8", "replace")[-400:]))
    return r.returncode == 0, dt


def evaluate(out_dirs, emit_calibration=False):
    cmd = [sys.executable, os.path.join(HERE, "eval.py"), "--quiet"]
    for d in out_dirs:
        cmd += ["--out", d]
    if emit_calibration:
        cmd += ["--emit-calibration"]
    subprocess.run(cmd, capture_output=True, cwd=ROOT, env=env())
    results = []
    for d in out_dirs:
        p = os.path.join(ROOT, d, "ground_truth_eval.json")
        if os.path.exists(p):
            results.append(json.load(io.open(p, encoding="utf-8")))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--skip-fixtures", action="store_true")
    args = ap.parse_args()

    summary = {"configs": {}, "resources": {}, "fixtures": None}
    calib_points = []

    for cfg_name, extra in CONFIGS:
        if args.no_ablations and cfg_name != "default":
            continue
        print("=" * 78)
        print("configuration: %s   %s" % (cfg_name, " ".join(extra) or "(defaults)"))
        print("=" * 78)
        out_dirs, timings = [], {}
        for tag, fn in SAMPLES:
            src = os.path.join(SAMPLE_DIR, fn)
            if not os.path.exists(src):
                print("   missing sample: %s" % src)
                continue
            out = "out_%s" % tag if cfg_name == "default" else \
                  "abl_%s_%s" % (cfg_name, tag)
            ok, dt = gridsmith(src, out, extra)
            if ok:
                out_dirs.append(out)
                timings[tag] = round(dt, 2)
                print("   %-8s %-34s %6.2fs" % (tag, fn[:34], dt))
        res = evaluate(out_dirs, emit_calibration=True)
        agg = _aggregate(res)
        summary["configs"][cfg_name] = agg
        summary["resources"][cfg_name] = timings
        print("   detection F1=%.3f  TEDS=%.3f  raw_text=%.3f  normalized=%.3f  "
              "silent=%d" % (agg["detection_f1"], agg["mean_teds_struct"] or 0.0,
                             agg["raw_text_accuracy"], agg["normalized_accuracy"],
                             agg["silent_errors"]))
        pts = os.path.join(HERE, "results", "calibration_points.json")
        if os.path.exists(pts):
            for p in json.load(io.open(pts, encoding="utf-8")):
                calib_points.append(dict(p, config=cfg_name))

    if not args.skip_fixtures:
        print("=" * 78)
        print("fixtures (default configuration)")
        print("=" * 78)
        for f in sorted(glob.glob(os.path.join(HERE, "fixtures", "f*.pdf"))):
            stem = os.path.basename(f)[:-4]
            ok, dt = gridsmith(f, "outfx_" + stem, [])
            print("   %-26s %s %5.2fs" % (stem, "ok" if ok else "FAILED", dt))
        rc, _dt, out = run([sys.executable, os.path.join(HERE, "eval_fixtures.py")])
        tail = [l for l in out.splitlines() if l.startswith("FIXTURES")]
        print("   " + (tail[0] if tail else "see metrics/results/fixtures.json"))
        fp = os.path.join(HERE, "results", "fixtures.json")
        if os.path.exists(fp):
            fx = json.load(io.open(fp, encoding="utf-8"))
            summary["fixtures"] = {"total_pass": fx["total_pass"],
                                   "total_checks": fx["total_checks"],
                                   "per_fixture": {f["fixture"]:
                                                   "%d/%d" % (f["pass"], f["total"])
                                                   for f in fx["fixtures"]}}

    write_text(os.path.join(HERE, "results", "all_configs.json"),
               canonical_json(summary) + "\n")
    write_text(os.path.join(HERE, "results", "calibration_points.json"),
               canonical_json(calib_points) + "\n")

    print()
    print("=" * 78)
    print("ABLATION SUMMARY   (delta vs default; a component that costs nothing")
    print("                    when removed was not earning its place)")
    print("=" * 78)
    base = summary["configs"].get("default")
    hdr = ("%-16s %8s %8s %9s %11s %8s"
           % ("config", "det F1", "TEDS", "raw_text", "normalized", "silent"))
    print(hdr)
    print("-" * len(hdr))
    for name, a in sorted(summary["configs"].items()):
        d = ""
        if base and name != "default":
            d = "   (F1 %+.3f, raw %+.3f)" % (a["detection_f1"] - base["detection_f1"],
                                              a["raw_text_accuracy"]
                                              - base["raw_text_accuracy"])
        print("%-16s %8.3f %8.3f %9.3f %11.3f %8d%s"
              % (name, a["detection_f1"], a["mean_teds_struct"] or 0.0,
                 a["raw_text_accuracy"], a["normalized_accuracy"],
                 a["silent_errors"], d))
    print()
    print("calibration points collected: %d (%d wrong)"
          % (len(calib_points), sum(1 for p in calib_points if not p["correct"])))
    print("wrote metrics/results/all_configs.json and calibration_points.json")
    return 0


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
    teds = [s["teds_struct"] for r in results for s in r["structure"]["tables"]
            if s.get("matched")]
    return {
        "documents": len(results),
        "detection_tp": tp, "detection_fp": fp, "detection_fn": fn,
        "detection_precision": round(prec, 4), "detection_recall": round(rec, 4),
        "detection_f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
        "mean_teds_struct": round(sum(teds) / len(teds), 4) if teds else None,
        "cells_scored": cells,
        "raw_text_accuracy": round(raw_ok / max(1, cells), 4),
        "normalized_accuracy": round(norm_ok / max(1, cells), 4),
        "silent_errors": sum(r["silent_errors"]["count"] for r in results),
        "traps_refused": sum(r["trap_refusals"]["correctly_refused"] for r in results),
        "traps_leaked": sum(r["trap_refusals"]["leaked_as_tables"] for r in results),
    }


if __name__ == "__main__":
    sys.exit(main())
