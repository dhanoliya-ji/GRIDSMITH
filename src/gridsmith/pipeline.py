"""End-to-end orchestration: S1 through S11.

Stage order here is not quite the numbering in the brief, and the two departures are
deliberate:

*   **S2 runs twice.**  Layout segmentation has to happen before detection (the
    detectors need running headers excluded) but it cannot finish before detection,
    because a bold table header row looks exactly like a section heading until you know
    a table is there.  So S2 assigns provisional roles, S3 detects, then
    ``demote_in_regions`` corrects the roles and the section tree is built from the
    corrected headings.  Without the second pass the ICRA rating table's
    ``section_path`` ended with ``"Others"`` -- half of a cell.

*   **S7 runs before S9 but is fed by a document-level pre-pass.**  Day/month order is
    a property of the document, not of a cell, so every candidate date is observed
    before any cell is normalised.  That also keeps normalisation deterministic:
    the answer cannot depend on which cell is processed first.

Memory: pages are rasterised one at a time and the pixmap is dropped immediately; only
the token model is held for the whole document, which is what the cross-page stages
(chrome detection, scale inheritance, stitching) need.  Measured figures are in the
README.
"""
from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass, field

from . import s1_intake as s1
from . import s2_layout as s2
from . import s3_arbitrate as s3a
from . import s3_borderless as s3b
from . import s3_ruled as s3r
from . import s4_structure as s4
from . import s5_cells as s5
from . import s6_stitch as s6
from . import s7_normalize as s7
from . import s8_context as s8
from . import s9_confidence as s9
from . import s9_review as s9r
from . import s10_persist as s10
from . import s11_audit as s11
from .config import Config
from .determinism import (canonical_json, doc_sha256, q, seed_everything, sid,
                          write_text, FIXED_ISO)
from .geometry import union
from .logging_utils import RunLog

VERSION = "0.1.0"


@dataclass
class PageTable:
    """One arbitrated table on one page, after structure and cell extraction."""

    page_no: int
    cand: object
    grid: object
    cells: list
    title: str = ""
    caption: str = ""
    section_path: list = field(default_factory=list)
    section_level: int = 1
    context_evidence: dict = field(default_factory=dict)


@dataclass
class LogicalAssembly:
    logical_index: int
    parts: list
    cells: list                      # re-indexed across parts
    n_rows: int
    n_cols: int
    n_header_rows: int
    header_paths: list
    indent_levels: list
    row_label_paths: list
    col_units: list
    dropped_header_rows: list
    votes: list
    row_page: dict = field(default_factory=dict)   # merged row -> page_no


def git_sha() -> str:
    """Best-effort git revision.  Never fails and never needs the network."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=os.path.dirname(os.path.dirname(
                                 os.path.dirname(os.path.abspath(__file__)))),
                             capture_output=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.decode().strip()[:40]
    except Exception:
        pass
    return os.environ.get("GRIDSMITH_GIT_SHA", "unknown")


def peak_rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process().memory_info().peak_wset / (1024 ** 2), 1) \
            if hasattr(psutil.Process().memory_info(), "peak_wset") else \
            round(psutil.Process().memory_info().rss / (1024 ** 2), 1)
    except Exception:
        try:
            import resource

            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        except Exception:
            return 0.0


# --------------------------------------------------------------------------- stages


def _detect_page(doc, pt, info, lay, cfg, log):
    """S3 for one page: ruled + borderless + arbitration, raster path included."""
    page = doc[pt.page_no - 1]
    raster = None
    if info.page_type in (s1.PAGE_SCANNED, s1.PAGE_HYBRID):
        raster = s1.prepare_raster(page, info, cfg)
        ruled_ev = s3r.detect_ruled_raster(raster, cfg)
        vec = s3r.detect_ruled(page, pt, cfg)
        ruled_ev.regions = list(ruled_ev.regions) + list(vec.regions)
        ruled_ev.h_rules += vec.h_rules
        ruled_ev.v_rules += vec.v_rules
        ruled_ev.breakpoints = sorted(set(ruled_ev.breakpoints) | set(vec.breakpoints))
        ruled_ev.stats["vector_also"] = vec.stats
    else:
        ruled_ev = s3r.detect_ruled(page, pt, cfg)
    borderless, bl_stats = s3b.detect_borderless(pt, lay, cfg)
    text_available = pt.n_chars >= cfg.digital_min_chars
    accepted, rejected = s3a.arbitrate(pt.page_no, pt, lay, ruled_ev,
                                       borderless, cfg,
                                       text_available=text_available)
    log.event("detect", page=pt.page_no, page_type=info.page_type,
              ruled=ruled_ev.stats, borderless=bl_stats,
              n_accepted=len(accepted), n_rejected=len(rejected),
              rejected_codes=sorted({r.code for c in rejected for r in c.rejections}))
    return accepted, rejected, raster


def _assemble(logical, li: int) -> LogicalAssembly:
    """Concatenate the page parts of one logical table, dropping repeated headers."""
    parts = logical.parts
    base = parts[0]
    drop = {}
    for part_idx, row in logical.dropped_header_rows:
        drop.setdefault(part_idx, set()).add(row)

    cells, row_page = [], {}
    offset = 0
    indent, labels = [], []
    for pi, part in enumerate(parts):
        skip = drop.get(pi, set())
        keep_rows = [r for r in range(part.grid.n_rows) if r not in skip]
        remap = {r: offset + k for k, r in enumerate(keep_rows)}
        for c in part.cells:
            if c.row in skip:
                continue
            new = _clone_cell(c, remap[c.row])
            cells.append(new)
            row_page[new.row] = part.page_no
        for r in keep_rows:
            indent.append(part.grid.indent_levels[r]
                          if r < len(part.grid.indent_levels) else 0)
        offset += len(keep_rows)

    n_rows = offset
    n_cols = base.grid.n_cols
    n_header = base.grid.n_header_rows
    # Row-label paths are rebuilt over the *merged* row sequence so an indented
    # hierarchy continues across a page break instead of restarting.
    class _Shim:
        pass

    shim = [type("C", (), {"row": c.row, "col": c.col, "tokens": c.tokens})()
            for c in cells]
    paths = s4.build_row_label_paths(shim, indent, n_header, n_rows)
    return LogicalAssembly(
        logical_index=li, parts=parts, cells=cells, n_rows=n_rows, n_cols=n_cols,
        n_header_rows=n_header, header_paths=base.grid.header_paths,
        indent_levels=indent, row_label_paths=paths,
        col_units=base.grid.col_units,
        dropped_header_rows=list(logical.dropped_header_rows),
        votes=list(logical.votes), row_page=row_page)


def _clone_cell(c, new_row):
    from .s5_cells import ExtractedCell

    return ExtractedCell(
        row=new_row, col=c.col, row_span=c.row_span, col_span=c.col_span,
        is_header=c.is_header, raw_text=c.raw_text, bbox=c.bbox, source=c.source,
        ocr_conf=c.ocr_conf, footnote_markers=list(c.footnote_markers),
        flags=list(c.flags), n_lines=c.n_lines, tokens=list(c.tokens))


# ------------------------------------------------------------------------- driver


def run(input_path: str, out_dir: str, cfg: Config) -> dict:
    t_start = time.time()
    seed_everything(cfg.seed)
    log = RunLog()
    os.makedirs(out_dir, exist_ok=True)
    for sub in ("tables", "assets", "debug", "logs"):
        s10.reset_dir(os.path.join(out_dir, sub))

    doc = s1.open_document(input_path)
    sha = doc_sha256(input_path)
    n_pages = doc.page_count if not cfg.max_pages else min(doc.page_count,
                                                           cfg.max_pages)
    doc_id = sid("doc", sha)
    run_id = sid("run", sha, cfg.fingerprint(), VERSION)
    log.event("open", filename=os.path.basename(input_path), sha256=sha,
              page_count=n_pages, run_id=run_id, config=cfg.fingerprint())

    # ---- S1 + S2 (provisional) ---------------------------------------------
    from .textlayer import extract_page_text

    pts, infos = [], []
    for i in range(n_pages):
        pt = extract_page_text(doc[i], i + 1, cfg.vertical_dir_tol)
        pts.append(pt)
        infos.append(s1.classify_page(doc[i], pt, cfg))
    chrome_keys, chrome_stats = s2.find_repeating_chrome(pts, cfg)
    layouts = [s2.segment_page(pt, chrome_keys, cfg) for pt in pts]
    size_rank = s2.assign_heading_levels(pts, layouts, cfg)
    log.event("intake", page_types={t: sum(1 for i in infos if i.page_type == t)
                                    for t in sorted({i.page_type for i in infos})},
              chrome=chrome_stats, heading_size_rank={str(k): v for k, v
                                                      in sorted(size_rank.items())})

    # ---- document-level date order pre-pass (S7) ---------------------------
    order = s7.DateOrderModel()
    for pt in pts:
        for ln in pt.lines:
            order.observe(ln.text.strip())
            for t in ln.tokens:
                order.observe(t.text)
    log.event("date_order", **order.to_dict())

    # ---- S3 ----------------------------------------------------------------
    ocr = s5.get_backend(cfg)
    log.event("ocr_backend", name=ocr.name, available=ocr.available)
    chrome_issues, all_rejected, per_page_accepted = [], [], {}
    for pt, info, lay in zip(pts, infos, layouts):
        accepted, rejected, raster = _detect_page(doc, pt, info, lay, cfg, log)
        s2.demote_in_regions(pt, lay, [c.bbox for c in accepted], chrome_issues)
        per_page_accepted[pt.page_no] = (accepted, raster)
        all_rejected.extend(rejected)

    # Section tree AFTER demotion, so table header rows are not headings.
    section_events = s2.build_section_events(pts, layouts)

    # ---- S4 + S5 -----------------------------------------------------------
    page_tables = []
    for pt, info, lay in zip(pts, infos, layouts):
        accepted, raster = per_page_accepted[pt.page_no]
        for cand in accepted:
            grid = s4.build_grid(cand, pt, cfg, layout=lay)
            if grid.n_rows == 0 or grid.n_cols == 0:
                continue
            if info.page_type == s1.PAGE_SCANNED and raster is not None:
                cells = s5.extract_ocr_cells(grid, doc[pt.page_no - 1], raster, cfg, ocr)
            else:
                cells = s5.extract_text_cells(grid, pt, cfg)
            # The detector's proposal is a search region; the *table* is what S4
            # reconstructed inside it.  Tighten the candidate box to the grid extent
            # before anything downstream records it, otherwise a rule stack that
            # reached 70 pt past the last row is persisted as the table's bbox and
            # detection IoU collapses (the ICRA complexity table scored 0.41).
            if grid.x_lines and grid.y_lines:
                cand.bbox = (min(grid.x_lines), min(grid.y_lines),
                             max(grid.x_lines), max(grid.y_lines))
            title, caption, cev = s8.find_title_caption(pt, lay, cand.bbox, cfg)
            spath = s2.section_path_at(section_events, pt.page_no, cand.bbox[1])
            lvl = 1
            for pno, y, l, txt in section_events:
                if (pno, y) <= (pt.page_no, cand.bbox[1]) and spath and txt == spath[-1]:
                    lvl = l
            page_tables.append(PageTable(
                page_no=pt.page_no, cand=cand, grid=grid, cells=cells, title=title,
                caption=caption, section_path=spath, section_level=lvl,
                context_evidence=cev))
    log.event("structure", n_page_tables=len(page_tables),
              regimes={r: sum(1 for t in page_tables if t.cand.regime == r)
                       for r in sorted({t.cand.regime for t in page_tables})})

    # ---- S6 ----------------------------------------------------------------
    logical = s6.stitch(page_tables, pts, cfg, headings=section_events)
    log.event("stitch", n_logical=len(logical),
              n_multi_page=sum(1 for lt in logical if lt.is_continuation),
              joins=[v for lt in logical for v in lt.votes])

    # ---- S8 assets ---------------------------------------------------------
    table_bboxes = {}
    for t in page_tables:
        table_bboxes.setdefault(t.page_no, []).append(t.cand.bbox)
    assets, asset_rows = [], []
    raw_assets = []
    for pt, lay in zip(pts, layouts):
        raw_assets.extend(s8.find_assets(doc[pt.page_no - 1], pt, lay,
                                         table_bboxes.get(pt.page_no, []), cfg))
    # Page furniture masquerades as a figure: the ICRA logo is an image block on all
    # seven pages at the same coordinates.  The same repetition test that finds running
    # headers finds these -- an asset whose position and kind recur on three or more
    # pages is branding, not a figure linked to a table.
    pos_count = {}
    for a in raw_assets:
        pos_count[_asset_key(a)] = pos_count.get(_asset_key(a), 0) + 1
    n_repeat_dropped = 0
    for a in raw_assets:
        if pos_count[_asset_key(a)] >= 3:
            n_repeat_dropped += 1
            continue
        assets.append(a)
    for idx, a in enumerate(assets):
        name = "%s_p%03d_%03d.png" % (a.kind, a.page_no, idx)
        a.file_path = os.path.join("assets", name)
        _crop_asset(doc, a, os.path.join(out_dir, a.file_path), cfg)
    log.event("assets", n_assets=len(assets),
              n_repeating_dropped=n_repeat_dropped,
              kinds={k: sum(1 for a in assets if a.kind == k)
                     for k in sorted({a.kind for a in assets})})

    # ---- assemble, normalise, score ---------------------------------------
    calib = s9.Calibration.load(os.path.join(os.path.dirname(__file__),
                                             "calibration.json"))
    log.event("calibration", **calib.to_dict())

    conn = s10.open_db(os.path.join(out_dir, "extraction.db"))
    started = FIXED_ISO if cfg.fixed_time else _now_iso()
    page_ids = {}
    doc_rows = [{"doc_id": doc_id, "filename": os.path.basename(input_path),
                 "sha256": sha, "page_count": n_pages,
                 "ingested_at": started}]
    page_rows = []
    for pt, info in zip(pts, infos):
        pid = sid("pag", doc_id, pt.page_no)
        page_ids[pt.page_no] = pid
        page_rows.append({
            "page_id": pid, "doc_id": doc_id, "page_no": pt.page_no,
            "width": q(pt.width), "height": q(pt.height),
            "page_type": info.page_type, "rotation": info.rotation,
            "content_rotation": info.content_rotation, "skew_deg": q(info.skew_deg),
            "image_area_frac": q(info.image_area_frac),
            "char_density": q(info.char_density),
            "notes_json": canonical_json(sorted(info.notes), None)})
    s10.insert(conn, "documents", doc_rows)
    s10.insert(conn, "pages", page_rows)
    # The run row must exist before any table references it.  Completed below.
    s10.insert(conn, "extraction_runs", [{
        "run_id": run_id, "doc_id": doc_id, "started_at": started,
        "finished_at": None, "git_sha": git_sha(),
        "config_json": cfg.to_json(), "gridsmith_version": VERSION,
        "config_fingerprint": cfg.fingerprint(), "wall_seconds": None,
        "peak_rss_mb": None, "offline": int(cfg.offline),
        "ocr_backend": ocr.name}])

    manifest_tables, all_review_items, all_issues = [], [], []
    table_rows, region_rows, column_rows, cell_rows = [], [], [], []
    footnote_rows, cellfn_rows, tasset_rows, issue_rows = [], [], [], []
    n_cells_total = 0
    seen_footnotes = {}

    for li, lt in enumerate(sorted(logical, key=lambda l: (l.start_page,
                                                           l.parts[0].grid.y_lines[0]
                                                           if l.parts[0].grid.y_lines
                                                           else 0)), start=1):
        asm = _assemble(lt, li)
        base = lt.parts[0]
        table_id = sid("tab", doc_id, run_id, li)

        # normalise, then infer column types, then re-normalise with the type known
        header_texts = [c.raw_text for c in asm.cells
                        if c.row < asm.n_header_rows and (c.raw_text or "").strip()]
        scale = s8.resolve_scale(base.page_no, base.cand.bbox, pts, layouts,
                                 base.section_path, base.caption, cfg,
                                 header_texts=header_texts)
        norms = {}
        for c in asm.cells:
            hdr = asm.header_paths[c.col] if c.col < len(asm.header_paths) else []
            n = s7.normalize(c.raw_text or "", column_header=" ".join(hdr),
                             order_model=order)
            if c.is_header:
                # A header cell is a *label*, so it keeps raw_text and a value_type but
                # carries no normalized_value.  Emitting 2024.0 for the column headed
                # "2024" is not wrong so much as an invitation to sum a column label,
                # and the meaning of a header belongs in columns.header_path.
                n.number = None
                n.date_iso = None
            norms[(c.row, c.col)] = n
        col_types, col_type_ev, col_conflicts = {}, {}, {}
        for col in range(asm.n_cols):
            vals = [norms[(r, col)] for r in range(asm.n_header_rows, asm.n_rows)
                    if (r, col) in norms]
            hdr = asm.header_paths[col] if col < len(asm.header_paths) else []
            ty, conf_idx, ev = s7.infer_column_type(vals, hdr, cfg.type_majority_frac)
            col_types[col], col_type_ev[col] = ty, ev
            col_conflicts[col] = len(conf_idx)

        source_of = {(c.row, c.col): c.source for c in asm.cells}
        ocr_conf_of = {(c.row, c.col): c.ocr_conf for c in asm.cells}
        grid_support = _grid_support(base.grid)
        score = s9.score_table(asm.cells, norms, _grid_view(asm, base.grid), cfg,
                               calib, col_types, source_of, ocr_conf_of, grid_support)

        # ---- footnotes -----------------------------------------------------
        fns = s8.collect_footnotes(pts, layouts, base.page_no, look_ahead=1)
        links, unmatched = s8.link_footnotes(asm.cells, fns)
        fn_ids = {}
        for fi, fn in enumerate(fns):
            key = (fn.page_no, fn.marker, fn.text)
            if key not in seen_footnotes:
                fid = sid("fnt", doc_id, fn.page_no, fn.marker, fn.text[:80])
                seen_footnotes[key] = fid
                footnote_rows.append({
                    "footnote_id": fid, "doc_id": doc_id,
                    "page_id": page_ids.get(fn.page_no), "marker": fn.marker,
                    "text": fn.text})
            fn_ids[fi] = seen_footnotes[key]

        # ---- asset links ---------------------------------------------------
        next_h = None
        for pno, y, lvl, txt in section_events:
            if pno == base.page_no and y > base.cand.bbox[3]:
                next_h = y if next_h is None else min(next_h, y)
        page_assets = [(i, a) for i, a in enumerate(assets)
                       if a.page_no in range(base.page_no, lt.end_page + 1)]
        alinks = s8.associate_assets(
            base.caption, base.title, base.cand.bbox, base.page_no,
            base.section_path, [a for _i, a in page_assets], next_h, cfg,
            nearby_text=" ".join(ln.text for ln in pts[base.page_no - 1].lines
                                 if abs(ln.y0 - base.cand.bbox[3]) < 80))

        # ---- rows ----------------------------------------------------------
        slug = s10.table_slug(base.title or base.caption or "table")
        xlsx_name = s10.excel_filename(li, lt.start_page, lt.end_page, slug)
        table_rows.append({
            "table_id": table_id, "doc_id": doc_id, "run_id": run_id,
            "logical_index": li, "start_page": lt.start_page, "end_page": lt.end_page,
            "n_rows": asm.n_rows, "n_cols": asm.n_cols,
            "title": base.title, "caption": base.caption,
            "section_path": canonical_json(base.section_path, None),
            "unit_scale_note": scale.note, "is_continuation": int(lt.is_continuation),
            "detector_votes_json": canonical_json(base.cand.votes, None),
            "table_confidence": q(score.table_confidence),
            "n_header_rows": asm.n_header_rows, "regime": base.cand.regime,
            "unit_scale_source": scale.source, "unit_scale_page": scale.page_no,
            "unit_scale_confidence": q(scale.confidence),
            "unit_currency": scale.currency, "unit_magnitude": q(scale.magnitude),
            "stitch_votes_json": canonical_json(asm.votes, None),
            "structure_evidence_json": canonical_json(base.grid.evidence, None),
            "reconciliation_json": canonical_json(score.reconciliation, None),
            "excel_path": os.path.join("tables", xlsx_name)})

        for pi, part in enumerate(lt.parts):
            region_rows.append({
                "region_id": s10.region_id(table_id, part.page_no, pi),
                "table_id": table_id, "page_id": page_ids[part.page_no],
                "bbox_x0": q(part.cand.bbox[0]), "bbox_y0": q(part.cand.bbox[1]),
                "bbox_x1": q(part.cand.bbox[2]), "bbox_y1": q(part.cand.bbox[3]),
                "part_index": pi, "regime": part.cand.regime})

        for col in range(asm.n_cols):
            hdr = asm.header_paths[col] if col < len(asm.header_paths) else []
            unit = asm.col_units[col] if col < len(asm.col_units) else ""
            if not unit and scale.currency:
                unit = scale.currency
            if unit and scale.magnitude != 1.0:
                unit = "%s %s" % (unit, _magnitude_word(scale.magnitude))
            column_rows.append({
                "column_id": sid("col", table_id, col), "table_id": table_id,
                "col_idx": col, "header_path": canonical_json(hdr, None),
                "header_path_text": " > ".join(hdr),
                "inferred_type": col_types.get(col), "unit": unit,
                "nullable": int(any(not (c.raw_text or "").strip()
                                    for c in asm.cells if c.col == col)),
                "type_evidence_json": canonical_json(col_type_ev.get(col, {}), None),
                "n_conflicts": col_conflicts.get(col, 0)})

        by_rc = {(c.row, c.col): c for c in score.cells}
        cell_id_of = {}
        for c in sorted(asm.cells, key=lambda c: (c.row, c.col)):
            cid = sid("cel", table_id, c.row, c.col)
            cell_id_of[(c.row, c.col)] = cid
            nrm = norms[(c.row, c.col)]
            cs = by_rc.get((c.row, c.col))
            pno = asm.row_page.get(c.row, base.page_no)
            rlp = asm.row_label_paths[c.row] if c.row < len(asm.row_label_paths) else []
            n_cells_total += 1
            cell_rows.append({
                "cell_id": cid, "table_id": table_id, "row_idx": c.row,
                "col_idx": c.col, "row_span": c.row_span, "col_span": c.col_span,
                "is_header": int(c.is_header), "raw_text": c.raw_text,
                "normalized_value": nrm.value, "value_type": nrm.value_type,
                "row_label_path": canonical_json(rlp, None),
                "row_label_path_text": " > ".join(rlp),
                "page_id": page_ids.get(pno),
                "bbox_x0": q(c.bbox[0]), "bbox_y0": q(c.bbox[1]),
                "bbox_x1": q(c.bbox[2]), "bbox_y1": q(c.bbox[3]),
                "source": c.source,
                "confidence": q(cs.confidence if cs else 0.0),
                "raw_confidence": q(cs.raw_confidence if cs else 0.0),
                "normalized_number": q(nrm.number) if nrm.number is not None else None,
                "normalized_date": nrm.date_iso,
                "currency": nrm.currency or (scale.currency or ""),
                "indent_level": (asm.indent_levels[c.row]
                                 if c.row < len(asm.indent_levels) else 0),
                "footnote_markers": "|".join(c.footnote_markers),
                "ocr_conf": q(c.ocr_conf) if c.ocr_conf is not None else None,
                "reason_codes": "|".join(cs.codes if cs else []),
                "mandatory_review": int(bool(cs and cs.mandatory)),
                "features_json": canonical_json(cs.features if cs else {}, None)})

            if cs:
                hp = asm.header_paths[c.col] if c.col < len(asm.header_paths) else []
                item = s9r.ReviewItem(
                    table_id=table_id, logical_index=li, page_no=pno,
                    row_idx=c.row, col_idx=c.col, header_path=" > ".join(hp),
                    row_label_path=" > ".join(rlp), raw_text=c.raw_text,
                    normalized_value=nrm.value, value_type=nrm.value_type,
                    confidence=cs.confidence, codes=cs.codes,
                    severity=s9r.severity_of(cs.codes), bbox=c.bbox,
                    source=c.source, mandatory=cs.mandatory)
                all_review_items.append(item)

        for (r, c, fi) in links:
            if (r, c) in cell_id_of and fi in fn_ids:
                cellfn_rows.append({"cell_id": cell_id_of[(r, c)],
                                    "footnote_id": fn_ids[fi]})
        for link in alinks:
            gi, a = page_assets[link["asset_index"]]
            tasset_rows.append({
                "table_id": table_id, "asset_id": sid("ast", doc_id, gi),
                "relation": link["relation"],
                "relation_confidence": q(link["relation_confidence"]),
                "components_json": canonical_json(link["components"], None)})

        tissues = s9.issues_for(score, table_id, cell_id_of)
        for it in tissues:
            it["issue_id"] = sid("iss", table_id, it["code"], it.get("cell_id") or "",
                                 it["message"][:64])
        issue_rows.extend(tissues)
        all_issues.extend(tissues)

        # ---- Excel ---------------------------------------------------------
        _write_excel(os.path.join(out_dir, "tables", xlsx_name), asm, norms, score,
                     base, scale, fns, alinks, [a for _i, a in page_assets], tissues,
                     cfg, col_types)

        manifest_tables.append(_manifest_table(
            table_id, li, lt, asm, base, scale, score, col_types, col_type_ev,
            fns, alinks, [a for _i, a in page_assets], xlsx_name, norms))

    # ---- assets + rejections ----------------------------------------------
    s10.insert(conn, "assets", [{
        "asset_id": sid("ast", doc_id, i), "doc_id": doc_id,
        "page_id": page_ids.get(a.page_no), "kind": a.kind,
        "file_path": a.file_path.replace("\\", "/"), "caption": a.caption,
        "bbox_x0": q(a.bbox[0]), "bbox_y0": q(a.bbox[1]),
        "bbox_x1": q(a.bbox[2]), "bbox_y1": q(a.bbox[3]),
        "series_json": None} for i, a in enumerate(assets)])

    rej_rows = []
    for cnd in all_rejected:
        for rj in cnd.rejections:
            rej_rows.append({
                "rejection_id": sid("rej", doc_id, cnd.page_no, rj.code,
                                    tuple(round(v, 1) for v in cnd.bbox)),
                "doc_id": doc_id, "page_id": page_ids.get(cnd.page_no),
                "bbox_x0": q(cnd.bbox[0]), "bbox_y0": q(cnd.bbox[1]),
                "bbox_x1": q(cnd.bbox[2]), "bbox_y1": q(cnd.bbox[3]),
                "regime": cnd.regime, "code": rj.code, "score": q(rj.score),
                "message": rj.message,
                "evidence_json": canonical_json(rj.evidence, None)})
    for ci in chrome_issues:
        issue_rows.append({
            "issue_id": sid("iss", doc_id, ci["code"], ci["message"][:64]),
            "table_id": None, "cell_id": None, "severity": ci["severity"],
            "code": ci["code"], "message": ci["message"]})

    s10.insert(conn, "tables", table_rows)
    s10.insert(conn, "table_regions", region_rows)
    s10.insert(conn, "columns", column_rows)
    s10.insert(conn, "cells", cell_rows)
    s10.insert(conn, "footnotes", footnote_rows)
    s10.insert(conn, "cell_footnotes",
               [{"cell_id": a, "footnote_id": b} for a, b in
                sorted({(r["cell_id"], r["footnote_id"]) for r in cellfn_rows})])
    s10.insert(conn, "table_assets", tasset_rows)
    s10.insert(conn, "issues", issue_rows)
    s10.insert(conn, "rejected_regions", rej_rows)

    # ---- review queue ------------------------------------------------------
    queued, ranked, qstats = s9r.build_queue(all_review_items, cfg, n_cells_total)
    write_text(os.path.join(out_dir, "review_queue.csv"), s9r.to_csv(queued, cfg))
    write_text(os.path.join(out_dir, "review_queue_full.csv"),
               s9r.to_csv(ranked, cfg))

    # ---- debug renders -----------------------------------------------------
    debug_pages = _debug_page_list(page_tables, all_rejected, cfg, n_pages)
    for pno in debug_pages:
        ov = build_overlay(pno, pts, layouts, page_tables, all_rejected, assets,
                           cell_conf={(t.page_no, id(t)): None for t in page_tables},
                           table_ids={}, cell_rows=cell_rows, page_ids=page_ids)
        s11.render_page(doc, pno, ov,
                        os.path.join(out_dir, "debug", "page_%03d.png" % pno),
                        high_confidence_floor=cfg.high_confidence_floor)

    # ---- run row + manifest + metrics -------------------------------------
    wall = time.time() - t_start
    finished = FIXED_ISO if cfg.fixed_time else _now_iso()
    conn.execute("UPDATE extraction_runs SET finished_at=?, wall_seconds=?, "
                 "peak_rss_mb=? WHERE run_id=?",
                 (finished,
                  0.0 if cfg.fixed_time else round(wall, 3),
                  0.0 if cfg.fixed_time else peak_rss_mb(),
                  run_id))
    conn.commit()
    conn.close()

    summary = {
        "document": {"filename": os.path.basename(input_path), "sha256": sha,
                     "page_count": n_pages,
                     "page_types": {t: sum(1 for i in infos if i.page_type == t)
                                    for t in sorted({i.page_type for i in infos})}},
        "run": {"run_id": run_id, "gridsmith_version": VERSION,
                "config_fingerprint": cfg.fingerprint(),
                "git_sha": git_sha(), "ocr_backend": ocr.name,
                "offline": cfg.offline, "python": platform.python_version()},
        "counts": {"logical_tables": len(manifest_tables),
                   "page_level_tables": len(page_tables),
                   "multi_page_tables": sum(1 for lt in logical
                                            if lt.is_continuation),
                   "cells": n_cells_total,
                   "assets": len(assets),
                   "footnotes": len(footnote_rows),
                   "cell_footnote_links": len(set(
                       (r["cell_id"], r["footnote_id"]) for r in cellfn_rows)),
                   "rejected_regions": len(all_rejected),
                   "issues": len(issue_rows)},
        "rejections_by_code": _count_by(
            [rj.code for c in all_rejected for rj in c.rejections]),
        "issues_by_code": _count_by([i["code"] for i in issue_rows]),
        "review_queue": qstats,
        "confidence": _confidence_stats(cell_rows, cfg),
        "reconciliation": _recon_stats(manifest_tables),
        "calibration": calib.to_dict(),
        # Under --fixed-time even metrics.json must be byte-identical, so the two
        # genuinely time-varying values are zeroed rather than omitted (a reader can
        # then tell "not measured" from "fast").  logs/timing.json always carries the
        # real figures and is excluded from the determinism claim.
        "resources": ({"wall_seconds": 0.0, "peak_rss_mb": 0.0,
                       "seconds_per_page": 0.0, "fixed_time": True}
                      if cfg.fixed_time else
                      {"wall_seconds": round(wall, 2),
                       "peak_rss_mb": peak_rss_mb(),
                       "seconds_per_page": round(wall / max(1, n_pages), 3)}),
    }
    manifest = {"schema_version": 1, "document": summary["document"],
                "run": summary["run"], "tables": manifest_tables,
                "assets": [{"index": i, "kind": a.kind, "page_no": a.page_no,
                            "bbox": [q(v) for v in a.bbox], "caption": a.caption,
                            "file_path": a.file_path.replace("\\", "/")}
                           for i, a in enumerate(assets)],
                "rejected_regions": [
                    {"page_no": c.page_no, "bbox": [q(v) for v in c.bbox],
                     "regime": c.regime,
                     "rejections": [{"code": r.code, "score": q(r.score),
                                     "message": r.message, "evidence": r.evidence}
                                    for r in c.rejections]}
                    for c in all_rejected],
                "chrome": chrome_stats,
                "debug_pages": debug_pages,
                "counts": summary["counts"]}
    s10.write_manifest(os.path.join(out_dir, "manifest.json"), manifest)
    write_text(os.path.join(out_dir, "metrics.json"), canonical_json(summary) + "\n")
    log.event("done", **summary["counts"])
    log.write(os.path.join(out_dir, "logs", "run.jsonl"))
    if not cfg.fixed_time:
        write_text(os.path.join(out_dir, "logs", "timing.json"),
                   canonical_json({"wall_seconds": round(wall, 3),
                                   "peak_rss_mb": peak_rss_mb(),
                                   "note": "excluded from the determinism claim"}) + "\n")
    doc.close()
    return summary


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _count_by(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def _magnitude_word(mag: float) -> str:
    for word, val in (("thousand", 1e3), ("lakh", 1e5), ("million", 1e6),
                      ("crore", 1e7), ("billion", 1e9), ("trillion", 1e12)):
        if abs(mag - val) < 1e-6:
            return word
    return ""


def _grid_support(grid) -> float:
    """0..1 measure of how well-evidenced this grid's boundaries are."""
    ev = (grid.evidence or {}).get("columns", {})
    srcs = ev.get("sources", {}) or {}
    strong = srcs.get("vrule", 0) + srcs.get("breakpoint", 0) + srcs.get("fill_edge", 0)
    weak = srcs.get("gutter", 0)
    total = strong + weak
    if total == 0:
        return 0.35
    return min(1.0, 0.45 + 0.55 * (strong / total))


def _grid_view(asm, base_grid):
    """A grid-shaped object describing the *merged* logical table."""
    class _G:
        pass

    g = _G()
    g.x_lines = base_grid.x_lines
    g.y_lines = base_grid.y_lines if len(base_grid.y_lines) >= asm.n_rows + 1 else \
        list(base_grid.y_lines) + [base_grid.y_lines[-1]] * (
            asm.n_rows + 1 - len(base_grid.y_lines))
    g.n_rows, g.n_cols = asm.n_rows, asm.n_cols
    g.n_header_rows = asm.n_header_rows
    g.header_paths = asm.header_paths
    g.indent_levels = asm.indent_levels
    return g


def _confidence_stats(cell_rows, cfg: Config):
    body = [r for r in cell_rows if not r["is_header"]]
    pool = body or cell_rows
    if not pool:
        return {}
    confs = sorted(float(r["confidence"] or 0.0) for r in pool)
    n = len(confs)
    return {"n_body_cells": n,
            "mean": round(sum(confs) / n, 4),
            "p05": round(confs[int(0.05 * (n - 1))], 4),
            "median": round(confs[int(0.50 * (n - 1))], 4),
            "p95": round(confs[int(0.95 * (n - 1))], 4),
            "n_high_confidence": sum(1 for c in confs
                                     if c >= cfg.high_confidence_floor),
            "frac_high_confidence": round(
                sum(1 for c in confs if c >= cfg.high_confidence_floor) / n, 4),
            "n_mandatory": sum(1 for r in pool if r["mandatory_review"])}


def _recon_stats(manifest_tables):
    p = f_ = i = 0
    for t in manifest_tables:
        r = t.get("reconciliation", {})
        p += r.get("n_pass", 0)
        f_ += r.get("n_fail", 0)
        i += r.get("n_incomplete", 0)
    return {"checks_pass": p, "checks_fail": f_, "checks_incomplete": i,
            "pass_rate": round(p / max(1, p + f_), 4)}


def _debug_page_list(page_tables, rejected, cfg: Config, n_pages: int):
    if cfg.write_debug_pages == "none":
        return []
    if cfg.write_debug_pages == "all":
        return list(range(1, n_pages + 1))
    pages = {t.page_no for t in page_tables} | {c.page_no for c in rejected}
    return sorted(pages)


def build_overlay(page_no, pts, layouts, page_tables, rejected, assets,
                  cell_conf=None, table_ids=None, cell_rows=None, page_ids=None):
    """Assemble the plain-dict overlay description for one page."""
    lay = layouts[page_no - 1]
    pt = pts[page_no - 1]
    chrome = []
    for i, ln in enumerate(pt.lines):
        if lay.is_chrome(i):
            chrome.append([0.0, ln.y0 - 1.0, pt.width, ln.y1 + 1.0])

    regions, grids, cells = [], [], []
    for t in page_tables:
        if t.page_no != page_no:
            continue
        regions.append({"bbox": list(t.cand.bbox), "regime": t.cand.regime,
                        "label": "T"})
        if t.grid.x_lines and t.grid.y_lines:
            grids.append({"x_lines": list(t.grid.x_lines),
                          "y_lines": list(t.grid.y_lines)})

    # Cells come from the persisted rows, so the confidence tint is the confidence that
    # was actually written.  Building them from the page-level tables instead meant
    # passing confidence=None for every cell -- and None renders as "below the floor",
    # so every cell in every pipeline-generated overlay came out red regardless of how
    # good it was.  The tinting only worked through `gridsmith debug`, which reads the
    # database.  A debug renderer that lies about confidence is worse than none.
    pid = (page_ids or {}).get(page_no)
    for r in (cell_rows or []):
        if pid is None or r.get("page_id") != pid:
            continue
        if r.get("bbox_x1") is None:
            continue
        cells.append({"bbox": [r["bbox_x0"], r["bbox_y0"], r["bbox_x1"],
                               r["bbox_y1"]],
                      "row_span": r.get("row_span", 1),
                      "col_span": r.get("col_span", 1),
                      "confidence": r.get("confidence")})
    rej = [{"bbox": list(c.bbox),
            "code": (max(c.rejections, key=lambda r: r.score).code
                     if c.rejections else "REFUSED")}
           for c in rejected if c.page_no == page_no]
    ast = [{"bbox": list(a.bbox), "kind": a.kind, "relation": ""}
           for a in assets if a.page_no == page_no]
    return {"chrome_bands": chrome, "regions": regions, "rejected": rej,
            "grids": [g for g in grids if g["x_lines"]], "cells": cells,
            "assets": ast}


def _asset_key(a):
    """Kind plus a coarse position, for the repetition test."""
    return (a.kind, tuple(round(v / 6.0) for v in a.bbox))


def _crop_asset(doc, asset, out_path: str, cfg: Config):
    import pymupdf

    page = doc[asset.page_no - 1]
    clip = pymupdf.Rect(*asset.bbox) & page.rect
    if clip.is_empty:
        return
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0), clip=clip, alpha=False)
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    pix.save(out_path)
    del pix


def _write_excel(path, asm, norms, score, base, scale, fns, alinks, page_assets,
                 tissues, cfg: Config, col_types):
    by_rc = {(c.row, c.col): c for c in asm.cells}
    sc_by = {(c.row, c.col): c for c in score.cells}
    data_rows, merges = [], []
    for r in range(asm.n_rows):
        row = []
        for c in range(asm.n_cols):
            cell = by_rc.get((r, c))
            if cell is None:
                row.append(None)
                continue
            nrm = norms.get((r, c))
            cs = sc_by.get((r, c))
            trusted = bool(cs and not cs.mandatory
                           and cs.confidence >= cfg.high_confidence_floor)
            # Native types only for values we trust; anything flagged stays text so a
            # spreadsheet SUM cannot quietly absorb a value we do not stand behind.
            if trusted and nrm is not None and nrm.value_type in s7.NUMERIC_TYPES \
                    and nrm.number is not None:
                row.append(nrm.number)
            else:
                row.append(cell.raw_text)
            if cell.row_span > 1 or cell.col_span > 1:
                merges.append((r, c, r + cell.row_span - 1, c + cell.col_span - 1))
        data_rows.append(row)

    ctx = [
        ("title", base.title), ("caption", base.caption),
        ("section_path", " > ".join(base.section_path)),
        ("page_range", "%d-%d" % (asm.parts[0].page_no, asm.parts[-1].page_no)),
        ("is_continuation", str(len(asm.parts) > 1)),
        ("n_page_parts", str(len(asm.parts))),
        ("unit_scale_note", scale.note),
        ("unit_scale_source", scale.source),
        ("unit_scale_page", str(scale.page_no)),
        ("unit_scale_confidence", "%.2f" % scale.confidence),
        ("unit_currency", scale.currency),
        ("unit_magnitude", str(scale.magnitude)),
        ("table_confidence", "%.4f" % score.table_confidence),
        ("detector_regime", base.cand.regime),
        ("detector_reason", base.cand.votes.get("reason", "")),
        ("n_header_rows", str(asm.n_header_rows)),
        ("column_types", ", ".join("%d:%s" % (k, v)
                                   for k, v in sorted(col_types.items()))),
        ("header_paths", " | ".join(" > ".join(h) for h in asm.header_paths)),
        ("reconciliation", "pass=%d fail=%d incomplete=%d" % (
            score.reconciliation.get("n_pass", 0),
            score.reconciliation.get("n_fail", 0),
            score.reconciliation.get("n_incomplete", 0))),
        ("dropped_repeated_header_rows", str(asm.dropped_header_rows)),
    ]
    for i, fn in enumerate(fns):
        ctx.append(("footnote[%s]" % (fn.marker or i), "p%d: %s" % (fn.page_no, fn.text)))
    for link in alinks:
        a = page_assets[link["asset_index"]]
        ctx.append(("linked_asset", "%s (%s, confidence %.2f) %s"
                    % (a.file_path.replace("\\", "/"), link["relation"],
                       link["relation_confidence"], a.caption)))

    prov = []
    for c in sorted(asm.cells, key=lambda c: (c.row, c.col)):
        nrm = norms.get((c.row, c.col))
        cs = sc_by.get((c.row, c.col))
        prov.append({
            "row_idx": c.row, "col_idx": c.col, "raw_text": c.raw_text,
            "normalized_value": (nrm.value if nrm else None),
            "page_no": asm.row_page.get(c.row, base.page_no),
            "bbox": "%.2f,%.2f,%.2f,%.2f" % tuple(round(float(v), 2) for v in c.bbox),
            "source": c.source,
            "confidence": round(cs.confidence, 4) if cs else None,
            "reason_codes": "|".join(cs.codes) if cs else ""})

    iss = []
    for it in tissues:
        rc = _parse_rc(it["message"])
        iss.append({"severity": it["severity"], "code": it["code"],
                    "row_idx": rc[0], "col_idx": rc[1], "message": it["message"]})

    s10.write_table_xlsx(path, data_rows=data_rows, merges=merges,
                         n_header_rows=asm.n_header_rows, context_rows=ctx,
                         provenance_rows=prov, issue_rows=iss,
                         freeze_header=cfg.excel_freeze_header)


def _parse_rc(message: str):
    import re

    m = re.search(r"r(\d+) c(\d+)", message or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"at r(\d+) c(\d+)", message or "")
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _manifest_table(table_id, li, lt, asm, base, scale, score, col_types, col_type_ev,
                    fns, alinks, page_assets, xlsx_name, norms):
    by_rc = {(c.row, c.col): c for c in asm.cells}
    sc_by = {(c.row, c.col): c for c in score.cells}
    cells = []
    for c in sorted(asm.cells, key=lambda c: (c.row, c.col)):
        nrm = norms.get((c.row, c.col))
        cs = sc_by.get((c.row, c.col))
        cells.append({
            "row_idx": c.row, "col_idx": c.col, "row_span": c.row_span,
            "col_span": c.col_span, "is_header": c.is_header,
            "raw_text": c.raw_text,
            "normalized_value": (nrm.value if nrm else None),
            "value_type": (nrm.value_type if nrm else None),
            "row_label_path": (asm.row_label_paths[c.row]
                               if c.row < len(asm.row_label_paths) else []),
            "indent_level": (asm.indent_levels[c.row]
                             if c.row < len(asm.indent_levels) else 0),
            "page_no": asm.row_page.get(c.row, base.page_no),
            "bbox": [q(v) for v in c.bbox], "source": c.source,
            "confidence": q(cs.confidence) if cs else None,
            "raw_confidence": q(cs.raw_confidence) if cs else None,
            "reason_codes": (cs.codes if cs else []),
            "mandatory_review": bool(cs and cs.mandatory),
            "footnote_markers": c.footnote_markers})
    return {
        "table_id": table_id, "logical_index": li,
        "start_page": lt.start_page, "end_page": lt.end_page,
        "is_continuation": lt.is_continuation, "n_page_parts": len(lt.parts),
        "n_rows": asm.n_rows, "n_cols": asm.n_cols,
        "n_header_rows": asm.n_header_rows,
        "title": base.title, "caption": base.caption,
        "section_path": base.section_path,
        "regime": base.cand.regime,
        "detector_votes": base.cand.votes,
        "stitch_votes": asm.votes,
        "dropped_repeated_header_rows": asm.dropped_header_rows,
        "structure_evidence": base.grid.evidence,
        "unit_scale": {"note": scale.note, "source": scale.source,
                       "page_no": scale.page_no, "currency": scale.currency,
                       "magnitude": scale.magnitude,
                       "confidence": q(scale.confidence), "flags": scale.flags},
        "columns": [{"col_idx": i,
                     "header_path": (asm.header_paths[i]
                                     if i < len(asm.header_paths) else []),
                     "inferred_type": col_types.get(i),
                     "unit": (asm.col_units[i] if i < len(asm.col_units) else ""),
                     "type_evidence": col_type_ev.get(i, {})}
                    for i in range(asm.n_cols)],
        "regions": [{"page_no": p.page_no, "bbox": [q(v) for v in p.cand.bbox],
                     "regime": p.cand.regime} for p in lt.parts],
        "footnotes": [{"marker": f.marker, "text": f.text, "page_no": f.page_no}
                      for f in fns],
        "linked_assets": [{"file_path": page_assets[l["asset_index"]].file_path
                           .replace("\\", "/"),
                           "relation": l["relation"],
                           "relation_confidence": l["relation_confidence"],
                           "components": l["components"]} for l in alinks],
        "reconciliation": score.reconciliation,
        "table_confidence": q(score.table_confidence),
        "excel_path": os.path.join("tables", xlsx_name).replace("\\", "/"),
        "cells": cells,
    }
