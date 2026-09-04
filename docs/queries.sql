-- Worked SQL against out/extraction.db.
--
-- The schema's design target is the query the brief uses as its acceptance test: if
-- "give me every cell whose column header path contains 'Revenue', with its page number
-- and the unit it is expressed in" is awkward, the schema is wrong.  It is a three-table
-- join with no string surgery, because header_path lives on `columns` rather than being
-- smeared across cells, the unit sits beside it, and the page number reaches the cell
-- through `pages`.
--
-- Run any of these with:
--     python -m gridsmith query --out ./out --sql "<query>"
-- or the named ones with:
--     python -m gridsmith query --out ./out --example revenue_cells

-- ===========================================================================
-- 1. THE ACCEPTANCE TEST
-- ===========================================================================
SELECT t.logical_index,
       c.row_idx, c.col_idx,
       col.header_path_text     AS header_path,
       c.row_label_path_text    AS row_label,
       c.raw_text,
       c.normalized_number,
       col.unit,
       p.page_no,
       c.confidence,
       c.reason_codes
FROM cells c
JOIN columns col ON col.table_id = c.table_id AND col.col_idx = c.col_idx
JOIN tables  t   ON t.table_id  = c.table_id
JOIN pages   p   ON p.page_id   = c.page_id
WHERE col.header_path_text LIKE '%Revenue%' COLLATE NOCASE
  AND c.is_header = 0
ORDER BY t.logical_index, c.row_idx, c.col_idx;

-- The same thing through the convenience view, which pre-joins header path, unit and
-- page number -- the shape most downstream questions want.
SELECT page_no, header_path_text, row_label_path_text, raw_text, normalized_number,
       unit, confidence
FROM v_cells_full
WHERE header_path_text LIKE '%amount%' COLLATE NOCASE AND is_header = 0;

-- ===========================================================================
-- 2. TRUST: what must a human look at, and why
-- ===========================================================================
-- The review queue, worst first.  `mandatory_review` cells carry a condition that
-- means "we know we might be wrong" and are never budgeted out.
SELECT p.page_no, t.logical_index, c.row_idx, c.col_idx, c.raw_text,
       ROUND(c.confidence, 3) AS conf, c.mandatory_review, c.reason_codes
FROM cells c
JOIN tables t ON t.table_id = c.table_id
LEFT JOIN pages p ON p.page_id = c.page_id
WHERE c.mandatory_review = 1 OR c.confidence < 0.85
ORDER BY c.mandatory_review DESC, c.confidence ASC
LIMIT 40;

-- Totals that do not add up.  Never silently corrected -- the value stays as printed.
SELECT t.logical_index, t.title, i.code, i.severity, i.message
FROM issues i
LEFT JOIN tables t ON t.table_id = i.table_id
WHERE i.code IN ('DOWNFOOT_FAIL', 'CROSSFOOT_FAIL', 'RECONCILIATION_INCOMPLETE');

-- Every distinct reason code, with how often it fired.
SELECT code, severity, COUNT(*) AS n
FROM issues GROUP BY code, severity ORDER BY n DESC;

-- Confidence distribution per table: is any table systematically untrusted?
SELECT t.logical_index, t.title, t.n_rows, t.n_cols,
       ROUND(t.table_confidence, 3)      AS table_conf,
       COUNT(c.cell_id)                  AS cells,
       SUM(c.mandatory_review)           AS must_review,
       SUM(CASE WHEN c.confidence >= 0.85 THEN 1 ELSE 0 END) AS high_conf
FROM tables t JOIN cells c ON c.table_id = t.table_id
GROUP BY t.table_id ORDER BY table_conf ASC;

-- ===========================================================================
-- 3. REFUSALS: what the pipeline declined to extract, and on what evidence
-- ===========================================================================
SELECT p.page_no, r.code, ROUND(r.score, 2) AS score, r.regime, r.message
FROM rejected_regions r
LEFT JOIN pages p ON p.page_id = r.page_id
ORDER BY p.page_no, r.code;

-- ===========================================================================
-- 4. CONTEXT: units, scale and where they came from
-- ===========================================================================
-- A unit inherited from two pages back is a materially weaker claim than one in the
-- table's own caption, and the schema keeps the difference.
SELECT logical_index, start_page, end_page,
       unit_currency, unit_magnitude, unit_scale_source, unit_scale_page,
       ROUND(unit_scale_confidence, 2) AS unit_conf,
       unit_scale_note, title
FROM tables ORDER BY logical_index;

-- Cells linked to a footnote, with the footnote text and the page it was found on
-- (which may be the page *after* the table).
SELECT t.logical_index, c.row_idx, c.col_idx, c.raw_text,
       f.marker, f.text, fp.page_no AS footnote_page
FROM cell_footnotes cf
JOIN cells c     ON c.cell_id     = cf.cell_id
JOIN footnotes f ON f.footnote_id = cf.footnote_id
JOIN tables t    ON t.table_id    = c.table_id
LEFT JOIN pages fp ON fp.page_id  = f.page_id;

-- Figures linked to tables, with the per-signal breakdown of the score.  Proximity
-- alone caps at 0.30, deliberately below any threshold worth trusting blindly.
SELECT t.logical_index, a.kind, a.file_path, ta.relation,
       ROUND(ta.relation_confidence, 2) AS conf, ta.components_json
FROM table_assets ta
JOIN assets a ON a.asset_id = ta.asset_id
JOIN tables t ON t.table_id = ta.table_id
ORDER BY conf DESC;

-- ===========================================================================
-- 5. STRUCTURE AND PROVENANCE
-- ===========================================================================
-- Multi-page tables and the join evidence that merged them.
SELECT logical_index, start_page, end_page, n_rows, n_cols, is_continuation,
       stitch_votes_json
FROM tables WHERE is_continuation = 1;

-- Merged cells, with their spans.
SELECT t.logical_index, c.row_idx, c.col_idx, c.row_span, c.col_span, c.raw_text
FROM cells c JOIN tables t ON t.table_id = c.table_id
WHERE c.row_span > 1 OR c.col_span > 1
ORDER BY t.logical_index, c.row_idx, c.col_idx;

-- Full provenance for one cell: page, bounding box, source, confidence, features.
SELECT c.raw_text, c.normalized_value, c.value_type, c.source,
       p.page_no, c.bbox_x0, c.bbox_y0, c.bbox_x1, c.bbox_y1,
       c.confidence, c.raw_confidence, c.reason_codes, c.features_json
FROM cells c JOIN pages p ON p.page_id = c.page_id
WHERE c.cell_id = (SELECT cell_id FROM cells LIMIT 1);

-- Row-label hierarchy recovered from indentation alone.
SELECT t.logical_index, c.row_idx, c.indent_level, c.row_label_path_text
FROM cells c JOIN tables t ON t.table_id = c.table_id
WHERE c.col_idx = 0 AND c.is_header = 0
ORDER BY t.logical_index, c.row_idx;

-- How each page was typed, and what S1 did to it.
SELECT page_no, page_type, rotation, content_rotation, skew_deg,
       ROUND(image_area_frac, 3) AS img_frac, notes_json
FROM pages ORDER BY page_no;

-- ===========================================================================
-- 6. THE RUN ITSELF
-- ===========================================================================
-- The only wall-clock values in the database live in this one row (§3.2).
SELECT run_id, gridsmith_version, config_fingerprint, git_sha, ocr_backend, offline,
       started_at, finished_at, ROUND(wall_seconds, 2) AS secs,
       ROUND(peak_rss_mb, 1) AS rss_mb
FROM extraction_runs;
