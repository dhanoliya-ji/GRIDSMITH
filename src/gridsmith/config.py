"""Every tunable number in the pipeline lives here.

Two reasons this file exists as a frozen dataclass rather than scattered literals:

1.  Determinism.  ``Config.fingerprint()`` is hashed into ``run_id`` and stored in
    ``extraction_runs.config_json``, so a run's output can always be tied back to the
    exact thresholds that produced it.
2.  Ablation.  ``metrics/eval.py`` flips single fields (e.g. ``enable_borderless``) to
    measure what each detector contributes.  §3.1 of the brief demands we can show the
    pipeline with a component switched off; the same machinery answers that for every
    stage, not just a learned model (of which we have none -- see docs/DECISIONS.md D1).

Units: all lengths are PDF points (1/72 inch) unless the name says ``px`` or ``frac``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Config:
    # ---------------------------------------------------------------- S1 intake
    # A page is "digital" if its text layer carries real glyphs.  Thresholds are
    # deliberately generous: a page with 40 chars is a title page, not a scan.
    digital_min_chars: int = 40
    # Fraction of page area covered by raster images above which we suspect a scan.
    scan_image_area_frac: float = 0.55
    # Chars per 1000 pt^2 of page area below which a page with a big image is a scan.
    hybrid_min_char_density: float = 0.02
    raster_dpi: int = 200          # working DPI for scanned page analysis
    ocr_cell_dpi: int = 300        # per-cell crop DPI (S5); higher = better OCR
    deskew_max_deg: float = 8.0    # refuse to "fix" more than this; flag instead
    deskew_step_deg: float = 0.1
    # A text line is treated as vertical (90/270 rotated content) if its writing
    # direction deviates from horizontal by more than this.
    vertical_dir_tol: float = 0.35

    # ---------------------------------------------------------------- S2 layout
    header_band_frac: float = 0.14   # top fraction of page scanned for running header
    footer_band_frac: float = 0.14   # bottom fraction likewise
    # A normalised line must recur on at least this many pages AND this fraction of
    # pages to be called a running header/footer.
    repeat_min_pages: int = 3
    repeat_min_frac: float = 0.25
    repeat_y_tol: float = 12.0       # same-band vertical tolerance across pages
    heading_size_ratio: float = 1.12  # font size / body median above which -> heading
    heading_max_words: int = 18
    body_line_gap_mult: float = 1.8  # gap > mult * median line height => block break

    # ------------------------------------------------------------ S3 detection
    enable_ruled: bool = True
    enable_borderless: bool = True
    # Vector "lines": a filled rect thinner than this in one axis is a rule, not a box.
    rule_max_thickness: float = 2.6
    rule_min_length: float = 24.0
    # Snapping tolerance when clustering collinear rules (JPEG/hairline jitter).
    rule_snap_tol: float = 2.2
    # Two rules on the same axis merge if their gap is under this (dotted/broken rules).
    rule_join_gap: float = 6.0
    ruled_min_rows: int = 2
    ruled_min_cols: int = 2
    # Borderless: gutter must be whitespace in >= this fraction of candidate rows.
    gutter_persist_frac: float = 0.80
    gutter_min_width: float = 3.6
    borderless_min_rows: int = 3
    borderless_min_cols: int = 2
    # A row band is part of the same candidate block if the vertical gap to the next
    # line is under mult * median gap of the block so far.
    row_gap_mult: float = 2.1
    # Arbitration
    merge_iou: float = 0.30          # regions above this IoU are the same table
    ruled_evidence_min_cells: int = 4

    # ------------------------------------------------------------ S4 structure
    col_align_tol: float = 3.0       # token edge clustering tolerance
    row_align_tol: float = 2.4
    # Adornment columns ($ / % / () only) are folded into their data neighbour.
    fold_adornment_columns: bool = True
    adornment_max_width: float = 26.0
    indent_step_min: float = 5.0     # x0 delta that counts as one indent level
    max_header_rows: int = 4

    # -------------------------------------------------------------- S6 stitch
    enable_stitch: bool = True
    stitch_col_iou_min: float = 0.85   # 1-D overlap of column bands, page to page
    stitch_max_page_gap: int = 1       # only consecutive pages (see DECISIONS D6)
    stitch_header_sim_min: float = 0.86  # header repeat detection similarity
    stitch_score_min: float = 0.62

    # ----------------------------------------------------------- S7 normalise
    # Column type is assigned when this share of non-empty cells agree.
    type_majority_frac: float = 0.70

    # ------------------------------------------------------------- S8 context
    caption_max_gap: float = 46.0      # points above/below table to look for a caption
    scale_search_pages: int = 3        # how far back a scale declaration may govern
    asset_min_area: float = 2400.0     # ignore logos/rules smaller than this
    asset_link_max_gap: float = 260.0

    # ---------------------------------------------------------- S9 confidence
    # Logistic weights.  Fixed and documented, not fitted at runtime (determinism);
    # the mapping from this score to a probability IS fitted offline -> calibration.json.
    w_bias: float = 2.35
    w_source_ocr: float = -1.55
    w_ocr_conf: float = 2.10
    w_snap_residual: float = -2.60
    w_straddle: float = -2.90
    w_type_conflict: float = -2.20
    w_digit_anomaly: float = -1.35
    w_multiline: float = -0.45
    w_crossfoot_fail: float = -2.05
    w_downfoot_fail: float = -1.85
    w_colcount_mismatch: float = -1.10
    w_dup_row: float = -0.55
    w_watermark: float = -1.20
    w_empty_in_dense_row: float = -0.90
    w_grid_low_support: float = -1.15
    review_budget_frac: float = 0.08   # §4 target: flag <= 8% of cells
    high_confidence_floor: float = 0.85  # "shipped with high confidence" boundary

    # ------------------------------------------------------------- S10 persist
    excel_freeze_header: bool = True
    write_debug_pages: str = "tables"  # "none" | "tables" | "all"

    # ------------------------------------------------------------------ misc
    seed: int = 20240928
    offline: bool = True
    fixed_time: bool = False           # make extraction_runs timestamps constant too
    max_pages: int = 0                 # 0 = all; used by tests
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    def replace(self, **kw: Any) -> "Config":
        return dataclasses.replace(self, **kw)


DEFAULT = Config()
