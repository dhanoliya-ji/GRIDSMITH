"""Web front end for gridsmith.

A thin HTTP layer over the same pipeline the CLI runs. It uploads or picks a
PDF, extracts it into a temporary directory, and returns the manifest so the
browser can show what the extractor decided -- including the cells it refused
to ship confidently and the regions it declined to call tables at all.

Nothing here weakens the extraction contract: the pipeline runs with the same
defaults, with ``offline`` forced on, and the response is derived from
``manifest.json`` rather than recomputed.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from gridsmith.config import DEFAULT
from gridsmith import pipeline

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIXTURES = REPO / "metrics" / "fixtures"

# Free-tier containers have little memory and a request budget, so the web path
# caps pages. The CLI is unrestricted; this limit belongs to the demo, not the
# extractor, and the response says so.
MAX_PAGES = int(os.environ.get("GRIDSMITH_WEB_MAX_PAGES", "8"))
MAX_UPLOAD_BYTES = int(os.environ.get("GRIDSMITH_WEB_MAX_BYTES", str(8 * 1024 * 1024)))

# Extraction output is kept only long enough to serve the overlay renders.
RUNS: dict[str, Path] = {}
MAX_RUNS = 12

SAMPLES = [
    ("f4_traps", "Four traps", "A contents page, an index, two-column prose and a bordered callout. Every one is geometrically a flawless table. None of them is data."),
    ("f1_indent_hierarchy", "Indented hierarchy", "Two-level header with spans, indentation-only row hierarchy, Indian digit grouping, parenthesised negatives, and the scale declared in a sentence above the table."),
    ("f2_continuation_6pages", "One table, six pages", "Header repeated on three pages, a (Contd.) marker, ±2 pt column drift and a running footer in the way."),
    ("f5_watermark_typo", "Watermark and a bad subtotal", "A 45° watermark across the table, and a subtotal that genuinely does not add up."),
    ("f6_footnote_next_page", "Footnote overleaf", "The marker is on one page and the footnote text is on the next."),
    ("f7_rotated_landscape", "Rotated landscape", "A landscape exhibit with a self-declared total column."),
    ("f8_scanned_skew", "Scanned and skewed", "An image-only page at −1.6°, with no OCR engine guaranteed. It fails closed rather than guessing."),
    ("f3_side_by_side", "Side by side", "Two independent tables sharing one page."),
]

app = FastAPI(title="gridsmith", docs_url="/api/docs", openapi_url="/api/openapi.json")


def _prune_runs() -> None:
    while len(RUNS) > MAX_RUNS:
        _, path = next(iter(RUNS.items()))
        RUNS.pop(_, None)
        shutil.rmtree(path, ignore_errors=True)


def _extract(pdf_path: Path) -> dict:
    out_dir = Path(tempfile.mkdtemp(prefix="gridsmith-web-"))
    # Same defaults the CLI uses; only the page cap is specific to this demo.
    cfg = DEFAULT.replace(
        offline=True,
        max_pages=MAX_PAGES,
        write_debug_pages="tables",
    )
    pipeline.run(str(pdf_path), str(out_dir), cfg)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))

    token = uuid.uuid4().hex[:12]
    RUNS[token] = out_dir
    _prune_runs()
    return {"token": token, "manifest": manifest, "metrics": metrics}


def _summarise(payload: dict) -> dict:
    """Reduce the manifest to what the page draws, keeping the vocabulary intact."""
    manifest = payload["manifest"]
    tables = []
    for t in manifest.get("tables", []):
        cells = t.get("cells", [])
        flagged = [c for c in cells if c.get("reason_codes") or c.get("mandatory_review")]
        tables.append({
            "table_id": t.get("table_id"),
            "title": t.get("title") or t.get("caption") or "",
            "regime": t.get("regime"),
            "start_page": t.get("start_page"),
            "end_page": t.get("end_page"),
            "n_rows": t.get("n_rows"),
            "n_cols": t.get("n_cols"),
            "n_header_rows": t.get("n_header_rows"),
            "table_confidence": t.get("table_confidence"),
            "unit_scale": t.get("unit_scale"),
            "reconciliation": t.get("reconciliation"),
            "footnotes": t.get("footnotes", []),
            "columns": t.get("columns", []),
            "n_flagged": len(flagged),
            "cells": [
                {
                    "row": c["row_idx"], "col": c["col_idx"],
                    "row_span": c.get("row_span", 1), "col_span": c.get("col_span", 1),
                    "is_header": c.get("is_header", False),
                    "raw_text": c.get("raw_text", ""),
                    "normalized_value": c.get("normalized_value"),
                    "value_type": c.get("value_type"),
                    "confidence": c.get("confidence"),
                    "reason_codes": c.get("reason_codes", []),
                    "mandatory_review": c.get("mandatory_review", False),
                    "page_no": c.get("page_no"),
                }
                for c in cells
            ],
        })

    rejected = [
        {
            "page_no": r.get("page_no"),
            "regime": r.get("regime"),
            "rejections": [
                {"code": x.get("code"), "message": x.get("message")}
                for x in r.get("rejections", [])
            ],
        }
        for r in manifest.get("rejected_regions", [])
    ]

    return {
        "token": payload["token"],
        "document": manifest.get("document", {}),
        "counts": manifest.get("counts", {}),
        "debug_pages": manifest.get("debug_pages", []),
        "tables": tables,
        "rejected_regions": rejected,
        "max_pages": MAX_PAGES,
        "resources": payload["metrics"].get("resources", {}),
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/samples")
def samples() -> dict:
    out = []
    for name, title, note in SAMPLES:
        if (FIXTURES / f"{name}.pdf").exists():
            out.append({"name": name, "title": title, "note": note})
    return {"samples": out, "max_pages": MAX_PAGES}


@app.post("/api/extract")
async def extract(file: UploadFile | None = File(default=None),
                  sample: str | None = Form(default=None)) -> JSONResponse:
    tmp_pdf: Path | None = None
    try:
        if sample:
            if not any(sample == s[0] for s in SAMPLES):
                raise HTTPException(400, "unknown sample")
            pdf = FIXTURES / f"{sample}.pdf"
            if not pdf.exists():
                raise HTTPException(404, "sample not generated in this image")
        elif file is not None:
            data = await file.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"PDF larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            if not data.startswith(b"%PDF"):
                raise HTTPException(400, "that does not look like a PDF")
            fd, name = tempfile.mkstemp(suffix=".pdf")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            tmp_pdf = pdf = Path(name)
        else:
            raise HTTPException(400, "send a PDF or pick a sample")

        payload = _extract(pdf)
        return JSONResponse(_summarise(payload))
    except HTTPException:
        raise
    except Exception as exc:  # surfaced to the page rather than a bare 500
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
    finally:
        if tmp_pdf is not None:
            tmp_pdf.unlink(missing_ok=True)


@app.get("/api/render/{token}/{page_no}")
def render(token: str, page_no: int) -> FileResponse:
    out_dir = RUNS.get(token)
    if out_dir is None:
        raise HTTPException(404, "run expired")
    png = out_dir / "debug" / f"page_{page_no:03d}.png"
    if not png.exists():
        raise HTTPException(404, "no overlay for that page")
    return FileResponse(png, media_type="image/png")


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")
