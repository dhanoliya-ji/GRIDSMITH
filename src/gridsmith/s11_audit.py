"""S11 -- auditability: the debug overlay renderer.

"We will use it during the live review, and so will you at 2 a.m. on day four."  So this
draws what the pipeline *believed*, not a prettier version of the page:

* accepted table regions in blue, with the regime and table id;
* **rejected** regions in red with the trap code that refused them -- the refusals are
  half the score on detection and invisible otherwise;
* inferred column boundaries (solid green) and row boundaries (dashed green);
* every cell box, tinted by confidence: green above the high-confidence floor, amber
  between 0.6 and the floor, red below -- so a bad column lights up as a red stripe;
* merged cells outlined thicker with their span printed;
* running header/footer bands greyed out, so "why did it miss that row" is answerable;
* detected assets in purple with their link confidence.

Rendering is pure PyMuPDF drawing onto a rasterised page, which keeps the dependency
list unchanged and works identically for digital and scanned pages.
"""
from __future__ import annotations

import os

import pymupdf

COL_REGION = (0.10, 0.35, 0.85)
COL_REJECT = (0.85, 0.15, 0.15)
COL_GRID = (0.05, 0.55, 0.20)
COL_ASSET = (0.55, 0.20, 0.70)
COL_CHROME = (0.55, 0.55, 0.55)
COL_HIGH = (0.15, 0.65, 0.25)
COL_MID = (0.95, 0.65, 0.10)
COL_LOW = (0.90, 0.15, 0.15)


def confidence_colour(conf, floor: float):
    if conf is None:
        return COL_LOW
    if conf >= floor:
        return COL_HIGH
    if conf >= 0.60:
        return COL_MID
    return COL_LOW


def render_page(doc, page_no: int, overlay, out_path: str, dpi: int = 130,
                high_confidence_floor: float = 0.85) -> str:
    """Draw ``overlay`` onto page ``page_no`` and write a PNG.

    ``overlay`` is a plain dict so the renderer stays decoupled from pipeline objects
    and ``gridsmith debug`` can rebuild it straight from the database.
    """
    page = doc[page_no - 1]
    shape = page.new_shape()

    for band in overlay.get("chrome_bands", []):
        shape.draw_rect(pymupdf.Rect(*band))
        shape.finish(color=COL_CHROME, fill=COL_CHROME, fill_opacity=0.10, width=0.4)

    for reg in overlay.get("regions", []):
        r = pymupdf.Rect(*reg["bbox"])
        shape.draw_rect(r)
        shape.finish(color=COL_REGION, width=1.6, fill=COL_REGION, fill_opacity=0.04)
        _label(shape, r.x0, r.y0 - 3.0, "%s  %s" % (reg.get("label", ""),
                                                    reg.get("regime", "")), COL_REGION)

    for reg in overlay.get("rejected", []):
        r = pymupdf.Rect(*reg["bbox"])
        shape.draw_rect(r)
        shape.finish(color=COL_REJECT, width=1.4, dashes="[3 3] 0",
                     fill=COL_REJECT, fill_opacity=0.05)
        _label(shape, r.x0, r.y0 - 3.0, "REFUSED " + reg.get("code", ""), COL_REJECT)

    for g in overlay.get("grids", []):
        y0, y1 = g["y_lines"][0], g["y_lines"][-1]
        for x in g["x_lines"]:
            shape.draw_line(pymupdf.Point(x, y0), pymupdf.Point(x, y1))
        shape.finish(color=COL_GRID, width=0.7)
        x0, x1 = g["x_lines"][0], g["x_lines"][-1]
        for y in g["y_lines"]:
            shape.draw_line(pymupdf.Point(x0, y), pymupdf.Point(x1, y))
        shape.finish(color=COL_GRID, width=0.5, dashes="[2 2] 0")

    for cell in overlay.get("cells", []):
        col = confidence_colour(cell.get("confidence"), high_confidence_floor)
        r = pymupdf.Rect(*cell["bbox"])
        if r.is_empty or r.width <= 0 or r.height <= 0:
            continue
        shape.draw_rect(r)
        merged = cell.get("row_span", 1) > 1 or cell.get("col_span", 1) > 1
        shape.finish(color=col, width=1.1 if merged else 0.5,
                     fill=col, fill_opacity=0.13 if merged else 0.07)
        if merged:
            _label(shape, r.x0 + 1, r.y0 + 6,
                   "%dx%d" % (cell.get("row_span", 1), cell.get("col_span", 1)), col)

    for a in overlay.get("assets", []):
        r = pymupdf.Rect(*a["bbox"])
        shape.draw_rect(r)
        shape.finish(color=COL_ASSET, width=1.3)
        _label(shape, r.x0, r.y0 - 3.0,
               "%s %s" % (a.get("kind", "asset"), a.get("relation", "")), COL_ASSET)

    shape.commit()
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    pix.save(out_path)
    del pix
    return out_path


def _label(shape, x, y, text, colour):
    if not text:
        return
    try:
        shape.insert_text(pymupdf.Point(x, max(6.0, y)), text[:64], fontsize=5.2,
                          color=colour, fontname="helv")
    except Exception:
        pass


def legend_text() -> str:
    return ("blue = accepted region | red dashed = refused region (trap code) | "
            "green solid = column boundary, dashed = row boundary | "
            "cell tint: green >= high-confidence floor, amber >= 0.60, red below | "
            "thick cell = merged, labelled with its span | "
            "grey = running header/footer | purple = linked asset")
