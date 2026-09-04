"""Dev harness: run S1-S5 + S7 on selected pages and print the reconstructed grids.

Not part of the pipeline.  This is the tool used while tuning boundary resolution --
``python tools/dump_tables.py "path.pdf" 25 32`` prints each table as a fixed-width
grid with header rows marked, so a structure regression is visible immediately.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gridsmith import s1_intake as s1, s2_layout as s2, s3_arbitrate as s3a  # noqa: E402
from gridsmith import s3_borderless as s3b, s3_ruled as s3r, s4_structure as s4  # noqa: E402
from gridsmith import s5_cells as s5  # noqa: E402
from gridsmith.config import DEFAULT  # noqa: E402
from gridsmith.s7_normalize import DateOrderModel, infer_column_type, normalize  # noqa: E402
from gridsmith.textlayer import extract_page_text  # noqa: E402


def render(cells, n_rows, n_cols, n_header, width=17):
    by = {}
    for c in cells:
        by[(c.row, c.col)] = c
    lines = []
    for r in range(n_rows):
        parts = []
        for c in range(n_cols):
            cell = by.get((r, c))
            txt = "" if cell is None else (cell.raw_text or "<NULL>")
            span = ""
            if cell is not None and (cell.col_span > 1 or cell.row_span > 1):
                span = "<%dx%d>" % (cell.row_span, cell.col_span)
            s = (txt + span)[:width]
            parts.append(s.ljust(width))
        mark = "H|" if r < n_header else " |"
        lines.append(mark + "|".join(parts))
    return "\n".join(lines)


def main(path, pages):
    doc = s1.open_document(path)
    pts = [extract_page_text(doc[i], i + 1) for i in range(doc.page_count)]
    keys, _ = s2.find_repeating_chrome(pts, DEFAULT)
    layouts = [s2.segment_page(pt, keys, DEFAULT) for pt in pts]
    s2.assign_heading_levels(pts, layouts, DEFAULT)
    order = DateOrderModel()
    for pt in pts:
        for ln in pt.lines:
            for t in ln.tokens:
                order.observe(t.text)

    for pno in pages:
        page, pt, lay = doc[pno - 1], pts[pno - 1], layouts[pno - 1]
        rev = s3r.detect_ruled(page, pt, DEFAULT)
        bl, _ = s3b.detect_borderless(pt, lay, DEFAULT)
        acc, rej = s3a.arbitrate(pno, pt, lay, rev, bl, DEFAULT)
        print("=" * 100)
        print("PAGE %d   %d accepted   %d rejected" % (pno, len(acc), len(rej)))
        for c in rej:
            print("   rejected %s -> %s" % (
                tuple(round(v) for v in c.bbox),
                [r.code for r in c.rejections]))
        for ci, cand in enumerate(acc):
            grid = s4.build_grid(cand, pt, DEFAULT)
            cells = s5.extract_text_cells(grid, pt, DEFAULT)
            print("-" * 100)
            print("table %d  regime=%s  %dr x %dc  header_rows=%d" % (
                ci, cand.regime, grid.n_rows, grid.n_cols, grid.n_header_rows))
            print("  x_lines: %s" % [round(v, 1) for v in grid.x_lines])
            print("  col evidence: %s  rows: %s  grow: %s  header: %s" % (
                grid.evidence["columns"]["sources"], grid.evidence["rows"],
                grid.evidence["grow"], grid.evidence["header"]))
            if grid.evidence["folds"]:
                print("  folds: %s" % grid.evidence["folds"][:6])
            if grid.evidence["columns"]["vetoed"]:
                print("  vetoed: %s" % grid.evidence["columns"]["vetoed"][:4])
            print("  header_paths: %s" % grid.header_paths)
            print("  units: %s" % grid.col_units)
            print(render(cells, grid.n_rows, grid.n_cols, grid.n_header_rows))
            # column types
            by_col = {}
            for cl in cells:
                if cl.row >= grid.n_header_rows:
                    by_col.setdefault(cl.col, []).append(
                        normalize(cl.raw_text or "", order_model=order))
            for col in sorted(by_col):
                ty, conf, st = infer_column_type(
                    by_col[col], grid.header_paths[col] if col < len(grid.header_paths)
                    else [])
                print("   col %d: %-10s conflicts=%d %s" % (col, ty, len(conf), st))
            print("  indent levels: %s" % grid.indent_levels)
            print("  row label paths: %s" % [p for p in grid.row_label_paths if p][:8])


if __name__ == "__main__":
    main(sys.argv[1], [int(p) for p in sys.argv[2:]])
