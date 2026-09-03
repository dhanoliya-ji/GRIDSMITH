-- gridsmith output schema.
--
-- The minimum schema from section 6.1 of the brief is present verbatim; every addition
-- is additive and noted with a comment.  The design target is the query the brief uses
-- as its acceptance test:
--
--   "give me every cell in this document whose column header path contains 'Revenue',
--    together with its page number and the unit it is expressed in"
--
-- That resolves to a three-table join with no string surgery, because header_path lives
-- on `columns` (not smeared across cells), the unit lives beside it, and the page number
-- reaches the cell through `pages`.  See docs/queries.sql for the query itself.
--
-- Ids are content-addressed digests (see gridsmith.determinism.sid), not autoincrement
-- integers: identical input must produce identical rows, and a counter reorders whenever
-- anything upstream changes.

PRAGMA journal_mode = DELETE;   -- no -wal sidecar; the file must stand alone
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    page_count   INTEGER NOT NULL,
    ingested_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    page_id      TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    page_no      INTEGER NOT NULL,
    width        REAL NOT NULL,
    height       REAL NOT NULL,
    page_type    TEXT NOT NULL,          -- digital | scanned | hybrid | empty
    rotation     INTEGER NOT NULL,
    -- additions
    content_rotation INTEGER DEFAULT 0,  -- rotated content without a /Rotate entry
    skew_deg     REAL DEFAULT 0.0,       -- deskew actually applied
    image_area_frac REAL DEFAULT 0.0,
    char_density REAL DEFAULT 0.0,
    notes_json   TEXT DEFAULT '[]',
    UNIQUE (doc_id, page_no)
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    run_id       TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    started_at   TEXT NOT NULL,          -- the ONLY wall-clock values in the database
    finished_at  TEXT,
    git_sha      TEXT,
    config_json  TEXT NOT NULL,
    -- additions
    gridsmith_version TEXT,
    config_fingerprint TEXT,
    wall_seconds REAL,
    peak_rss_mb  REAL,
    offline      INTEGER DEFAULT 1,
    ocr_backend  TEXT
);

CREATE TABLE IF NOT EXISTS tables (
    table_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    run_id       TEXT NOT NULL REFERENCES extraction_runs(run_id),
    logical_index INTEGER NOT NULL,
    start_page   INTEGER NOT NULL,
    end_page     INTEGER NOT NULL,
    n_rows       INTEGER NOT NULL,
    n_cols       INTEGER NOT NULL,
    title        TEXT,
    caption      TEXT,
    section_path TEXT,                   -- JSON array
    unit_scale_note TEXT,
    is_continuation INTEGER NOT NULL DEFAULT 0,
    detector_votes_json TEXT,
    table_confidence REAL,
    -- additions
    n_header_rows INTEGER DEFAULT 1,
    regime       TEXT,                   -- grid | fills | hlines | vlines | borderless
    unit_scale_source TEXT,              -- caption | above_region | section_heading | ...
    unit_scale_page INTEGER,
    unit_scale_confidence REAL,
    unit_currency TEXT,
    unit_magnitude REAL DEFAULT 1.0,
    stitch_votes_json TEXT,
    structure_evidence_json TEXT,
    reconciliation_json TEXT,
    excel_path   TEXT
);

CREATE TABLE IF NOT EXISTS table_regions (
    region_id    TEXT PRIMARY KEY,
    table_id     TEXT NOT NULL REFERENCES tables(table_id),
    page_id      TEXT NOT NULL REFERENCES pages(page_id),
    bbox_x0      REAL NOT NULL,
    bbox_y0      REAL NOT NULL,
    bbox_x1      REAL NOT NULL,
    bbox_y1      REAL NOT NULL,
    -- additions
    part_index   INTEGER DEFAULT 0,      -- which page-part of a stitched table
    regime       TEXT
);

CREATE TABLE IF NOT EXISTS columns (
    column_id    TEXT PRIMARY KEY,
    table_id     TEXT NOT NULL REFERENCES tables(table_id),
    col_idx      INTEGER NOT NULL,
    header_path  TEXT,                   -- JSON array, outermost header level first
    inferred_type TEXT,
    unit         TEXT,
    nullable     INTEGER DEFAULT 1,
    -- additions
    header_path_text TEXT,               -- " > " joined, for LIKE queries
    type_evidence_json TEXT,
    n_conflicts  INTEGER DEFAULT 0,
    UNIQUE (table_id, col_idx)
);

CREATE TABLE IF NOT EXISTS cells (
    cell_id      TEXT PRIMARY KEY,
    table_id     TEXT NOT NULL REFERENCES tables(table_id),
    row_idx      INTEGER NOT NULL,
    col_idx      INTEGER NOT NULL,
    row_span     INTEGER NOT NULL DEFAULT 1,
    col_span     INTEGER NOT NULL DEFAULT 1,
    is_header    INTEGER NOT NULL DEFAULT 0,
    raw_text     TEXT,                   -- NULL means "could not be read", not "empty"
    normalized_value,                    -- no type: REAL for numbers, TEXT for dates
    value_type   TEXT,
    row_label_path TEXT,                 -- JSON array
    page_id      TEXT REFERENCES pages(page_id),
    bbox_x0      REAL,
    bbox_y0      REAL,
    bbox_x1      REAL,
    bbox_y1      REAL,
    source       TEXT,                   -- text-layer | ocr | none
    confidence   REAL,
    -- additions
    normalized_number REAL,              -- typed mirrors, so SQL needs no CAST
    normalized_date TEXT,
    currency     TEXT,
    indent_level INTEGER DEFAULT 0,
    row_label_path_text TEXT,
    footnote_markers TEXT,
    ocr_conf     REAL,
    raw_confidence REAL,               -- pre-calibration logistic output
    reason_codes TEXT,                   -- pipe-joined
    mandatory_review INTEGER DEFAULT 0,
    features_json TEXT,
    UNIQUE (table_id, row_idx, col_idx)
);

CREATE TABLE IF NOT EXISTS footnotes (
    footnote_id  TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    page_id      TEXT REFERENCES pages(page_id),
    marker       TEXT,
    text         TEXT
);

CREATE TABLE IF NOT EXISTS cell_footnotes (
    cell_id      TEXT NOT NULL REFERENCES cells(cell_id),
    footnote_id  TEXT NOT NULL REFERENCES footnotes(footnote_id),
    PRIMARY KEY (cell_id, footnote_id)
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    page_id      TEXT REFERENCES pages(page_id),
    kind         TEXT,                   -- image | chart
    file_path    TEXT,
    caption      TEXT,
    bbox_x0      REAL,
    bbox_y0      REAL,
    bbox_x1      REAL,
    bbox_y1      REAL,
    -- additions
    series_json  TEXT                    -- recovered chart series, when available
);

CREATE TABLE IF NOT EXISTS table_assets (
    table_id     TEXT NOT NULL REFERENCES tables(table_id),
    asset_id     TEXT NOT NULL REFERENCES assets(asset_id),
    relation     TEXT,
    relation_confidence REAL,
    -- additions
    components_json TEXT,                -- per-signal breakdown of the score
    PRIMARY KEY (table_id, asset_id)
);

CREATE TABLE IF NOT EXISTS issues (
    issue_id     TEXT PRIMARY KEY,
    table_id     TEXT REFERENCES tables(table_id),
    cell_id      TEXT REFERENCES cells(cell_id),
    severity     TEXT NOT NULL,          -- error | warn | info
    code         TEXT NOT NULL,
    message      TEXT
);

-- Regions the detectors proposed and the trap rules refused.  Not in the minimum
-- schema, but "we correctly excluded the table of contents" is only a checkable claim
-- if the refusal is recorded.
CREATE TABLE IF NOT EXISTS rejected_regions (
    rejection_id TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    page_id      TEXT REFERENCES pages(page_id),
    bbox_x0      REAL, bbox_y0 REAL, bbox_x1 REAL, bbox_y1 REAL,
    regime       TEXT,
    code         TEXT NOT NULL,
    score        REAL,
    message      TEXT,
    evidence_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_cells_table ON cells(table_id, row_idx, col_idx);
CREATE INDEX IF NOT EXISTS idx_cells_conf ON cells(confidence);
CREATE INDEX IF NOT EXISTS idx_cells_mandatory ON cells(mandatory_review);
CREATE INDEX IF NOT EXISTS idx_columns_header ON columns(header_path_text);
CREATE INDEX IF NOT EXISTS idx_issues_code ON issues(code, severity);
CREATE INDEX IF NOT EXISTS idx_pages_no ON pages(doc_id, page_no);
CREATE INDEX IF NOT EXISTS idx_regions_table ON table_regions(table_id);

-- Convenience view: one row per cell with its header path, unit and page number
-- already joined.  This is the shape most downstream questions want.
CREATE VIEW IF NOT EXISTS v_cells_full AS
SELECT
    c.cell_id, c.table_id, t.logical_index, t.title, t.caption,
    t.section_path, t.unit_scale_note, t.unit_currency, t.unit_magnitude,
    col.header_path, col.header_path_text, col.inferred_type, col.unit,
    c.row_idx, c.col_idx, c.row_span, c.col_span, c.is_header,
    c.raw_text, c.normalized_value, c.normalized_number, c.normalized_date,
    c.value_type, c.currency, c.row_label_path, c.row_label_path_text,
    c.indent_level, p.page_no, c.bbox_x0, c.bbox_y0, c.bbox_x1, c.bbox_y1,
    c.source, c.confidence, c.mandatory_review, c.reason_codes
FROM cells c
JOIN tables t   ON t.table_id = c.table_id
LEFT JOIN columns col ON col.table_id = c.table_id AND col.col_idx = c.col_idx
LEFT JOIN pages p ON p.page_id = c.page_id;
