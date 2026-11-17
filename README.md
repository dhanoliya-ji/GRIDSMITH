# Gridsmith

**Deterministic table extraction from large, messy PDFs. Every cell is either correct or flagged as uncertain.**

No language model touches the document. Gridsmith uses classical computer vision on the PDF's own
geometry (ruled lines, cell fills, text positions), scores every cell with **14 confidence signals**
calibrated by **isotonic regression**, and sends uncertain cells to a review queue instead of guessing.
It runs offline and its output is byte-reproducible.

The key metric is the **silent error**: a cell that is wrong, shipped with confidence ≥ 0.85, and not
flagged. On the labelled set that count is **0**.

## Results

Measured on 3 real financial PDFs against **202 hand-labelled cells**, plus 8 generated adversarial PDFs
whose ground truth is exact by construction.

| Metric | Result |
|---|---|
| Table detection F1 at IoU ≥ 0.7 | **1.000** (10 TP, 0 FP, 0 FN) |
| Cell text exact match | **1.000** (202 cells) |
| TEDS-Struct (table structure) | **0.927** |
| **Silent errors** | **0** |
| Lookalike tables refused (contents page) | 1 / 1 |
| Adversarial fixture checks | 56 / 57 |
| Expected calibration error (pooled) | 0.029 |

**Against off-the-shelf extractors** (detection F1 at IoU ≥ 0.7, same pages):

| Document | Gridsmith | pdfplumber | PyMuPDF `find_tables` |
|---|---|---|---|
| Apple 10-K | **1.000** | 0.500 | 0.500 |
| Shell MRPL report | **1.000** | 0.533 | 0.571 |
| Shell financials | **1.000** | 0.500 | 0.667 |

Both baselines extract the table of contents as a table; Gridsmith rejects it.

**Ablation**: the ruled-line detector does the heavy lifting.

| Configuration | Detection F1 | TEDS | Cell text | Silent errors |
|---|---|---|---|---|
| Default | 1.000 | 0.927 | 1.000 | 0 |
| No ruled-line detector | **0.444** | 0.857 | 0.287 | **17** |
| No borderless detector | 1.000 | 0.927 | 1.000 | 0 |
| No cross-page stitching | 1.000 | 0.927 | 1.000 | 0 |

**Caveat.** The perfect scores come from the documents Gridsmith was developed against. The independent
evidence is the adversarial fixtures (multi-page tables, side-by-side tables, watermarks, rotated and
scanned pages, contents and index traps) and the ablation table.

## How it works

```
PDF -> reading -> detection -> structure -> values -> confidence -> output
       S1, S2      S3           S4, S5, S6   S7, S8    S9            S10, S11
```

| File | Stage | What it computes | Key functions |
|---|---|---|---|
| `gridsmith/reading.py` | S1, S2 | PDF text layer, page type, deskew and rotation, running headers, headings, captions | `extract_page_text`, `classify_page`, `deskew`, `segment_page` |
| `gridsmith/detection.py` | S3 | Table candidates from ruled lines and fills, from whitespace gutters, and lookalike traps (contents, index, prose); arbitration between them | `detect_ruled`, `detect_borderless`, `classify`, `arbitrate` |
| `gridsmith/structure.py` | S4, S5, S6 | Column and row boundaries, spanning headers, indent hierarchy, cell text or OCR, tables continued across pages | `build_grid`, `detect_header_rows`, `extract_text_cells`, `stitch` |
| `gridsmith/values.py` | S7, S8 | Numbers, negatives, currency, dates and scale; table title, units and footnote links | `parse_number`, `normalize`, `resolve_scale`, `link_footnotes` |
| `gridsmith/confidence.py` | S9 | 14 confidence signals per cell, isotonic calibration, subtotal checks, review queue | `score_table`, `Calibration`, `down_foot`, `build_queue` |
| `gridsmith/output.py` | S10, S11 | SQLite database, one xlsx per table, manifest, audit and debug overlays | `open_db`, `write_table_xlsx`, `write_manifest`, `audit` |
| `gridsmith/pipeline.py` | all | Runs every stage on one PDF; command line | `run`, `main` |
| `gridsmith/core.py` | shared | Configuration, deterministic ids and JSON, box geometry, run log | `Config`, `canonical_json`, `iou`, `RunLog` |

| Evaluation file | What it computes |
|---|---|
| `metrics/eval.py` | Detection precision, recall and F1 at IoU 0.7, TEDS-Struct, cell accuracy, silent errors |
| `metrics/run_all.py` | Runs the default configuration and the 3 ablations, then evaluates each |
| `metrics/fit_calibration.py` | Fits the isotonic calibration curve (pool adjacent violators) |
| `metrics/calibration_report.py` | Reliability bins, expected calibration error, flagging precision and recall |
| `metrics/run_baselines.py` | The same detection scoring for pdfplumber and PyMuPDF |
| `metrics/eval_fixtures.py` | Pass or fail checks on the 8 adversarial PDFs |
| `metrics/fixtures/make_fixtures.py` | Generates those 8 PDFs and their exact ground truth |
| `metrics/ground_truth/` | 202 hand-labelled cells for the 3 sample PDFs |

## Reproduce

```bash
pip install -r requirements.txt
# put the 3 sample PDFs in sample_data/ (not redistributable, so not in the repo):
#   Apple Form 10-K.pdf, Shell.pdf, Shell MRPL Aviation Fuels and Services Limited.pdf
python reproduce.py            # a few minutes; prints every number above
python reproduce.py --refit    # also refits the isotonic calibration curve
```

Extract tables from any PDF:

```bash
python -m gridsmith run --input your.pdf --out out_your --offline
```
