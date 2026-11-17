"""Reproduce every headline number in one command.

    python reproduce.py            # fixtures, all configurations + ablations, baselines, summary
    python reproduce.py --refit    # also refit the isotonic confidence calibration first

Put the three sample PDFs in sample_data/ first (they are not redistributable, so they are not
in the repo):
    sample_data/Apple Form 10-K.pdf
    sample_data/Shell.pdf
    sample_data/Shell MRPL Aviation Fuels and Services Limited.pdf
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(ROOT, "metrics")
RES = os.path.join(M, "results")
SAMPLES = ["Apple Form 10-K.pdf", "Shell.pdf", "Shell MRPL Aviation Fuels and Services Limited.pdf"]


def step(title, script):
    print(f"\n== {title}", flush=True)
    if subprocess.run([sys.executable, os.path.join(M, script)], cwd=ROOT).returncode != 0:
        sys.exit(f"{script} failed")


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def main():
    sample_dir = os.environ.get("GRIDSMITH_SAMPLES", os.path.join(ROOT, "sample_data"))
    missing = [s for s in SAMPLES if not os.path.exists(os.path.join(sample_dir, s))]
    if missing:
        sys.exit("missing sample PDFs in %s:\n  %s" % (sample_dir, "\n  ".join(missing)))

    step("1/4 generate the 8 adversarial fixture PDFs and their exact ground truth", "fixtures/make_fixtures.py")
    if "--refit" in sys.argv:
        step("refit: extract once to collect (confidence, correct) pairs", "run_all.py")
        step("refit: fit the isotonic calibration curve", "fit_calibration.py")
    step("2/4 run the default configuration and the three ablations on samples and fixtures", "run_all.py")
    step("3/4 calibration report", "calibration_report.py")
    step("4/4 pdfplumber and PyMuPDF baselines on the same ground truth", "run_baselines.py")

    cfg = load("all_configs.json").get("configs", {})
    d, abl = cfg.get("default", {}), cfg.get("no_ruled", {})
    docs = load("baselines.json").get("documents", {})
    base = [v[k]["f1"] for v in docs.values() for k in ("pdfplumber_extract_tables", "pymupdf_find_tables") if k in v]

    print("\n================  RESULTS  ================")
    print(f"hand-labelled cells scored        {d.get('cells_scored')}")
    print(f"silent errors                     {d.get('silent_errors')}")
    print(f"raw text accuracy                 {d.get('raw_text_accuracy')}")
    print(f"mean TEDS-Struct                  {d.get('mean_teds_struct')}")
    print(f"detection F1 at IoU 0.7           {d.get('detection_f1')}")
    if base:
        print(f"baselines detection F1            {min(base):.2f} to {max(base):.2f} (pdfplumber, PyMuPDF)")
    print(f"ablation: no ruled-line detector  F1 {abl.get('detection_f1')}, silent errors {abl.get('silent_errors')}")
    print("all numbers are in metrics/results/")


if __name__ == "__main__":
    main()
