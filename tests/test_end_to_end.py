"""End-to-end tests: the output contract, and the two guarantees.

These run the real pipeline over a generated fixture, so they need no sample document
and work inside the Docker image. The fixture is generated on demand if absent.

Two of these are the tests that matter most, because they check the two promises the
brief treats as non-negotiable:

* ``test_two_runs_are_byte_identical`` -- §3.2.
* ``test_no_cell_is_both_wrong_and_confident`` -- §4, on the one fixture that contains a
  deliberate error.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

FIXTURE_DIR = os.path.join(ROOT, "metrics", "fixtures")
FIXTURE = "f5_watermark_typo"          # small, and contains a planted arithmetic error


def _env():
    e = dict(os.environ)
    e["PYTHONPATH"] = SRC
    e["PYTHONIOENCODING"] = "utf-8"
    return e


def _ensure_fixtures():
    pdf = os.path.join(FIXTURE_DIR, FIXTURE + ".pdf")
    if not os.path.exists(pdf):
        r = subprocess.run([sys.executable,
                            os.path.join(FIXTURE_DIR, "make_fixtures.py")],
                           capture_output=True, cwd=ROOT, env=_env())
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-2000:]
    return pdf


def _run(out_dir, extra=()):
    pdf = _ensure_fixtures()
    cmd = [sys.executable, "-m", "gridsmith", "run", "--input", pdf,
           "--out", out_dir, "--offline", "--quiet"] + list(extra)
    r = subprocess.run(cmd, capture_output=True, cwd=ROOT, env=_env())
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-3000:]
    return out_dir


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    return _run(str(tmp_path_factory.mktemp("out")))


# ------------------------------------------------------------- output contract


def test_every_required_artefact_exists(out):
    for rel in ("extraction.db", "manifest.json", "metrics.json",
                "review_queue.csv", "logs/run.jsonl"):
        assert os.path.exists(os.path.join(out, rel)), rel
    assert os.listdir(os.path.join(out, "tables")), "no .xlsx written"
    assert os.listdir(os.path.join(out, "debug")), "no debug render written"


def test_schema_has_every_required_table_and_column(out):
    conn = sqlite3.connect(os.path.join(out, "extraction.db"))
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"documents", "pages", "extraction_runs", "tables", "table_regions",
                "columns", "cells", "footnotes", "cell_footnotes", "assets",
                "table_assets", "issues"}
    assert required <= names, required - names
    # The §6.1 minimum columns on `cells`, verbatim.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(cells)")}
    assert {"cell_id", "table_id", "row_idx", "col_idx", "row_span", "col_span",
            "is_header", "raw_text", "normalized_value", "value_type",
            "row_label_path", "page_id", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
            "source", "confidence"} <= cols
    conn.close()


def test_excel_has_the_four_required_sheets(out):
    tables = sorted(os.listdir(os.path.join(out, "tables")))
    with zipfile.ZipFile(os.path.join(out, "tables", tables[0])) as z:
        wb = z.read("xl/workbook.xml").decode("utf-8")
    for sheet in ("Data", "Context", "Provenance", "Issues"):
        assert 'name="%s"' % sheet in wb, sheet


def test_every_cell_traces_to_a_page_and_a_bbox(out):
    conn = sqlite3.connect(os.path.join(out, "extraction.db"))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM cells")]
    assert rows
    for r in rows:
        assert r["page_id"], "cell with no page"
        assert r["source"] in ("text-layer", "ocr", "none")
        for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"):
            assert r[k] is not None, k
        assert r["bbox_x1"] >= r["bbox_x0"] and r["bbox_y1"] >= r["bbox_y0"]
    conn.close()


def test_run_log_carries_no_timestamps(out):
    with open(os.path.join(out, "logs", "run.jsonl"), encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    assert lines
    assert [e["seq"] for e in lines] == list(range(len(lines)))
    blob = json.dumps(lines)
    for banned in ("started_at", "finished_at", "wall_seconds", "elapsed"):
        assert banned not in blob, banned


# ------------------------------------------------------------- the guarantees


def test_two_runs_are_byte_identical(tmp_path):
    a = _run(str(tmp_path / "a"), ["--fixed-time"])
    b = _run(str(tmp_path / "b"), ["--fixed-time"])
    r = subprocess.run([sys.executable, "-m", "gridsmith", "verify", "--a", a,
                        "--b", b], capture_output=True, cwd=ROOT, env=_env())
    out = r.stdout.decode("utf-8", "replace")
    assert r.returncode == 0, out[-3000:]
    assert "byte-identical" in out


def test_no_cell_is_both_wrong_and_confident(out):
    """The planted typo must be extracted as printed, flagged, and never corrected."""
    conn = sqlite3.connect(os.path.join(out, "extraction.db"))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM cells WHERE raw_text = '3,505'").fetchone()
    assert row is not None, "the subtotal cell was not extracted"
    # Extracted as printed -- not silently 'fixed' to the arithmetic answer 3,605.
    assert row["normalized_number"] == pytest.approx(3505.0)
    # And flagged, so it can never be shipped as high confidence.
    assert row["mandatory_review"] == 1
    assert "DOWNFOOT_FAIL" in (row["reason_codes"] or "")
    assert row["confidence"] < 0.85

    # The column that *does* reconcile is left alone.
    ok = conn.execute("SELECT * FROM cells WHERE raw_text = '2,805'").fetchone()
    assert ok is not None
    assert "DOWNFOOT_FAIL" not in (ok["reason_codes"] or "")
    conn.close()


def test_reconciliation_failure_raises_an_issue(out):
    conn = sqlite3.connect(os.path.join(out, "extraction.db"))
    codes = [r[0] for r in conn.execute("SELECT code FROM issues")]
    assert "DOWNFOOT_FAIL" in codes
    conn.close()


def test_review_queue_is_ranked_and_labels_its_priority(out):
    import csv

    with open(os.path.join(out, "review_queue.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "empty review queue on a fixture with a planted error"
    assert [int(r["rank"]) for r in rows] == list(range(1, len(rows) + 1))
    assert all(r["priority"] in ("mandatory", "discretionary") for r in rows)
    # Mandatory rows come first, then ascending confidence.
    mand = [i for i, r in enumerate(rows) if r["priority"] == "mandatory"]
    assert mand == list(range(len(mand))), "mandatory rows are not ranked first"
    assert all(r["reason"] for r in rows), "a queue row with no reason is useless"


def test_watermark_is_detected_and_kept_out_of_the_value(out):
    conn = sqlite3.connect(os.path.join(out, "extraction.db"))
    n = conn.execute("SELECT COUNT(*) FROM cells WHERE raw_text LIKE '%DRAFT%'"
                     ).fetchone()[0]
    assert n == 0, "watermark glyphs leaked into a cell value"
    conn.close()


def test_offline_guard_blocks_the_network():
    from gridsmith.cli import install_offline_guard

    code = ("import sys; sys.path.insert(0, %r);"
            "from gridsmith.cli import install_offline_guard;"
            "install_offline_guard();"
            "import socket;"
            "\ntry:\n    socket.socket()\n"
            "    print('NOT BLOCKED')\nexcept Exception as e:\n"
            "    print('blocked')" % SRC)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, env=_env())
    assert b"blocked" in r.stdout, r.stdout + r.stderr
    assert b"NOT BLOCKED" not in r.stdout


def test_config_fingerprint_is_recorded_with_the_run(out):
    conn = sqlite3.connect(os.path.join(out, "extraction.db"))
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT * FROM extraction_runs").fetchone()
    cfg = json.loads(run["config_json"])
    assert run["config_fingerprint"] and len(run["config_fingerprint"]) == 16
    assert cfg["offline"] is True
    assert run["ocr_backend"] in ("tesseract", "null")
    conn.close()
