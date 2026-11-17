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
S1  intake         page typing, deskew, rotation
S2  layout         running headers and footers, headings, captions, footnotes, scale notes
S3  detection      ruled lines and fills | whitespace gutters | trap detector (TOC, index, prose)
                   proposals are arbitrated, so refusing a lookalike is a real outcome
S4  structure      column boundaries, row sources, spanning headers, indent hierarchy
S5  cells          text layer, or per-cell OCR that fails closed
S6  stitch         joins tables that continue across pages
S7  normalise      types, scale, negatives, Indian and Western digit grouping
S8  context        title, section, units, footnote links
S9  confidence     14 signals -> isotonic calibration -> review queue
S10 persist        SQLite, xlsx, manifest
S11 audit          metrics and determinism check
```

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

## Layout

```
gridsmith.py                 the whole pipeline in one file, one section per stage S1 to S11
calibration.json             fitted isotonic calibration curve
reproduce.py                 one command for all metrics
metrics/eval.py              detection P/R/F1, TEDS-Struct, cell accuracy, silent errors
metrics/run_all.py           default configuration and 3 ablations
metrics/fit_calibration.py   isotonic calibration fit
metrics/run_baselines.py     pdfplumber and PyMuPDF comparison
metrics/fixtures/            adversarial PDF generator
metrics/ground_truth/        hand-labelled cells for the 3 sample PDFs
```
