<div align="center">

# gridsmith

**Deterministic table and context extraction from large, messy PDFs.**

*Every cell is either correct, or flagged as uncertain.*

[![ci](https://github.com/dhanoliya-ji/gridsmith/actions/workflows/ci.yml/badge.svg)](https://github.com/dhanoliya-ji/gridsmith/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![llm](https://img.shields.io/badge/LLM-none-informational)
![offline](https://img.shields.io/badge/network-enforced%20offline-informational)
![determinism](https://img.shields.io/badge/output-byte--reproducible-informational)

</div>

---

No language model touches the document. The pipeline is pure classical computer vision,
PDF text-layer geometry and signal processing, and it runs with no network access and no
API keys.

The governing rule — **every cell is either correct or flagged as uncertain** — is
treated as a structural property of the output rather than an aspiration. The metric that
follows from it is not accuracy but the **silent error**: a cell that is wrong, shipped
confidently, and unflagged. That count is **0**.

📄 **[Problem statement and objectives](PROBLEM_STATEMENT.md)** — what this solves and why it is hard
📊 **[Results notebook](notebooks/01_results_and_milestones.ipynb)** — milestones, reliability diagrams, PR curves, ablation charts
⚠️ **[Limitations](docs/LIMITATIONS.md)** — read this before the numbers below

---

## Table of contents

- [Why this is hard](#why-this-is-hard)
- [Quickstart](#quickstart)
- [Command line](#command-line)
- [Results](#results)
- [Offline and determinism](#offline-and-determinism)
- [Output contract](#output-contract)
- [How it works](#how-it-works)
- [What works, what does not](#what-works-what-does-not)
- [Repository layout](#repository-layout)
- [Reproducing the numbers](#reproducing-the-numbers)

---

## Why this is hard

A PDF has no notion of a table. It has glyphs at coordinates and, sometimes, line
segments at other coordinates. Everything else is inference. The sample corpus exercises
most of the ways that inference goes wrong, at once:

- One document draws full rules, one draws dotted rules, one uses per-cell background
  fills. `Shell.pdf` does not draw rules at all — it draws **18,066 disconnected 0.72 pt
  squares** that a human eye resolves into lines.
- A table of contents with alternating row shading is, geometrically, a **flawless
  seven-column table**. So are indexes, two-column prose and bordered callouts. Both
  off-the-shelf baselines extract the TOC as data.
- One logical table spans six pages, repeats its header on three of them, drifts ±2 pt,
  and is interrupted by running footers.
- The scale (`₹ in crore`) is declared in a sentence *above* the table. The footnote
  marker is on page 1 and its text is on page 2.
- `(1,234)` is negative. `1,23,456` is Indian grouping. Empty, `-`, `–`, `Nil`, `NA` and
  `0` are five different things.
- Some subtotals in real documents genuinely do not add up.

The full treatment is in **[PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md)**.

---

## Quickstart

```bash
python -m venv .venv && . .venv/Scripts/activate   # Linux/macOS: . .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m gridsmith run   --input "path/to/doc.pdf" --out ./out --offline
python -m gridsmith audit --out ./out
```

Python 3.10+. `tesseract-ocr` is optional and only affects scanned pages — without it
they are flagged rather than guessed (see [LIMITATIONS L1](docs/LIMITATIONS.md)).

Nothing is downloaded at run time and nothing is fitted at run time: the confidence curve
in `calibration.json` is vendored, already fitted.

<details>
<summary><b>Docker</b> — note that this image has never been built</summary>

Docker was unavailable on the development machine, so the Dockerfile is **unverified**:
every command inside it is one that passes locally, but layer ordering, the
`tesseract-ocr` apt install and the editable-install fallback are untested. Expect to fix
something on the first build. See [LIMITATIONS L8](docs/LIMITATIONS.md).

```bash
docker build -t gridsmith .
docker run --rm --network none -v "$PWD:/data" gridsmith \
    run --input /data/doc.pdf --out /data/out --offline
```
</details>

---

## Command line

```
python -m gridsmith run    --input doc.pdf --out ./out --offline   # extract
python -m gridsmith audit  --out ./out                             # metrics summary
python -m gridsmith debug  --out ./out --page 42                   # overlay render
python -m gridsmith verify --a ./out1 --b ./out2                   # two-run byte diff
python -m gridsmith query  --out ./out --example revenue_cells     # SQL against the db
```

Useful flags on `run`:

| flag | effect |
|---|---|
| `--offline` | installs a socket guard that **raises** on any connection attempt |
| `--fixed-time` | pins the `extraction_runs` timestamps so even the database is byte-identical |
| `--no-ruled` / `--no-borderless` / `--no-stitch` | ablations — switch a detector off |
| `--debug-pages all` | write an overlay render for every page |
| `--review-budget 0.08` | truncate the review queue at a fraction of cells |
| `--max-pages N` | stop after N pages |

`query --list-examples` shows the built-in queries, including the acceptance test the
brief specifies.

---

## Results

Measured on the three supplied documents against 202 hand-labelled cells
(`metrics/ground_truth/`), plus 8 generated adversarial fixtures whose ground truth is
exact by construction (`metrics/fixtures/`).

> **Read the 1.000s with the caveat in [LIMITATIONS.md](docs/LIMITATIONS.md).** Those
> cells come from documents this was developed against, and every threshold in
> `config.py` was chosen with them in view. The uncontaminated evidence is the fixture
> suite and the ablation table.

### Sample documents, default configuration

| metric | result |
|---|---|
| detection precision / recall / F1 at IoU ≥ 0.7 | **1.000 / 1.000 / 1.000** (10 TP, 0 FP, 0 FN) |
| false-positive traps refused | **1 / 1**, 0 leaked |
| TEDS-Struct (structure only, merged cells included) | **0.927** |
| cell `raw_text` exact match | **1.000** (202 cells) |
| `normalized_value` exact match | **1.000** |
| **silent errors** (wrong + shipped ≥ 0.85 + unflagged) | **0** |
| review queue | 2.8 – 10.9% of cells per document |
| reliability, expected calibration error | 0.057 – 0.112 per document |

### Adversarial fixtures — 56 / 57 behavioural checks

| fixture | tests | result |
|---|---|---|
| `f1_indent_hierarchy` | 2-level header with spans, 3-level indentation-only hierarchy, Indian grouping, parenthesised negatives, scale declared above the table | 4/5 |
| `f2_continuation_6pages` | one table over 6 pages, header on 1/3/5 only, `(Contd.)` on p4, ±2 pt column drift, running footer | **9/9** |
| `f3_side_by_side` | two independent tables side by side | **5/5** |
| `f4_traps` | TOC, index, two-column prose, bordered callout | **10/10** |
| `f5_watermark_typo` | 45° watermark over the table; subtotal that deliberately fails | **7/7** |
| `f6_footnote_next_page` | footnote marker on p1, footnote text on p2 | **6/6** |
| `f7_rotated_landscape` | landscape exhibit, self-declared total column | **7/7** |
| `f8_scanned_skew` | image-only page skewed −1.6°, no OCR engine present | **8/8** |

The one failure is a spanning-header cell on `f1` ([LIMITATIONS L3](docs/LIMITATIONS.md)).
Notably `f5` passes in full: the failing subtotal is extracted as **3505**, flagged
`DOWNFOOT_FAIL`, and **never corrected** — while the column that reconciles correctly is
left alone.

### Ablations — §3.1 asks for performance with the component switched off

| configuration | detection F1 | TEDS | `raw_text` | silent errors |
|---|---|---|---|---|
| default | 1.000 | 0.927 | 1.000 | 0 |
| `--no-ruled` | **0.444** | 0.857 | **0.287** | 17 |
| `--no-borderless` | 1.000 | 0.927 | 1.000 | 0 |
| `--no-stitch` | 1.000 | 0.927 | 1.000 | 0 |

Author-drawn grid evidence is doing the heavy lifting: remove it and detection F1 more
than halves. `--no-borderless` costs nothing *on this corpus* because every sample table
carries rule or fill evidence — the borderless detector is insurance for the held-out
set, and there is **no measurement proving it earns its place**
([LIMITATIONS L8](docs/LIMITATIONS.md)). `--no-stitch` costs nothing on the samples
because none of the labelled tables spans pages; it is what makes `f2` work.

### Calibration, pooled over all configurations

698 labelled (prediction, outcome) pairs, 146 of them errors. Pooling the ablated runs is
what supplies the negatives — the default configuration makes no errors on the labelled
set, and a curve fitted on a set with no negatives says nothing.

```
RELIABILITY                stated confidence          empirical accuracy
  0.2-0.3   n=116          0.277                      0.259
  0.5-0.6   n= 12          0.536                      1.000
  0.9-1.0   n=525          0.950                      0.977
  expected calibration error 0.029
```

Shipped operating point (mandatory codes + confidence < 0.85): **24.8% of cells flagged,
91.8% of errors caught**, flag precision 0.775, 12 silent errors (1.7%).

On the brief's 8% / 95% target: the pooled set is 21% errors, and you cannot catch 95% of
errors by reviewing 8% of cells when a fifth of them are wrong — that is arithmetic. In
the regime the target describes (error rate well under 8%) the default configuration
flags 3–11% per document and has zero errors to catch. **The number to hold me to is the
silent-error count.**

### Baselines — §3.4

Detection at IoU ≥ 0.7 on the same labelled pages. Generous to the baselines: they are
scored on detection and shape only.

| document | gridsmith F1 | `pdfplumber` | `pymupdf.find_tables` |
|---|---|---|---|
| Apple 10-K | **1.000** | 0.500 (6 FP, leaks the TOC) | 0.500 (6 FP, leaks the TOC) |
| ICRA report | **1.000** | 0.533 | 0.571 |
| Shell financials | **1.000** | 0.500 | 0.667 |

Both baselines emit the table of contents as a table. Neither produces per-cell
provenance, calibrated confidence, a review queue, units with their source,
footnote-to-cell links, cross-page stitching or reproducible bytes — which is where this
assignment's marks are. `camelot` and `tabula` are not run (no Ghostscript/JVM here) and
would fail structurally on `Shell.pdf`, whose "rules" are 18,066 disconnected 0.72 pt
squares. See `metrics/baselines/run_baselines.py`.

### Resources — §3.3

| document | pages | wall | s/page | peak RSS |
|---|---|---|---|---|
| Apple Form 10-K | 121 | 15.1 s | 0.12 | 173 MB |
| Shell financials | 3 | 2.5 s | 0.83 | 122 MB |
| ICRA report | 7 | 2.2 s | 0.31 | 137 MB |

A 100-page document extrapolates to ~13 s and under 200 MB, against a budget of 20
minutes and 4 GB. **These timings are for runs with no OCR**; a scanned 100-page document
is unmeasured and could plausibly approach the budget
([LIMITATIONS L7](docs/LIMITATIONS.md)).

---

## Offline and determinism

**Offline is enforced, not asserted.** `--offline` installs a socket guard that raises on
any attempt to open a connection. The pipeline contains no network code, which is exactly
why the guard is worth installing: it converts "we believe it never connected" into "it
could not have".

**Determinism is proved, not claimed:**

```bash
python -m gridsmith run --input doc.pdf --out ./a --offline --fixed-time
python -m gridsmith run --input doc.pdf --out ./b --offline --fixed-time
python -m gridsmith verify --a ./a --b ./b
#   RESULT: byte-identical (timestamps isolated as permitted)
```

Verified on the ICRA report: all 7 `.xlsx` files, `extraction.db`, `manifest.json`,
`review_queue.csv`, `logs/run.jsonl`, every `debug/*.png` and every `assets/*.png` are
byte-identical. Without `--fixed-time` everything above still matches and only the three
timestamp columns of the single `extraction_runs` row differ, as §3.2 permits;
`logs/timing.json` carries the wall clock and is explicitly excluded from the claim. CI
asserts this on every push.

Three things had to be fixed to get there, none of them obvious: content-addressed ids
instead of `AUTOINCREMENT`; rewriting the `.xlsx` zip with fixed entry timestamps; and
rewriting `docProps/core.xml`, because `openpyxl` overwrites `dcterms:modified` with the
current clock during `save()` no matter what you set beforehand. The two-run diff found
the last one.

---

## Output contract

```
out/
  extraction.db          SQLite; §6.1 minimum schema verbatim plus additive columns
  tables/T001__p25-25__segment_operating_performance.xlsx
                         one per logical table; Data / Context / Provenance / Issues
  assets/chart_p023_000.png
  manifest.json          full structured dump, one object per table
  review_queue.csv       ranked, with reason codes and a plain-English reason
  review_queue_full.csv  the complete ranking, for the PR curve
  metrics.json           self-reported scores
  debug/page_025.png     overlay renders
  logs/run.jsonl         structured, deterministic (no timestamps by construction)
  logs/timing.json       wall clock and peak RSS (excluded from the determinism claim)
```

The schema is built around the query the brief uses as its acceptance test:

```bash
python -m gridsmith query --out ./out --example revenue_cells
```

which resolves to a three-table join with no string surgery, because `header_path` lives
on `columns` (not smeared across cells), the unit sits beside it, and the page number
reaches the cell through `pages`. More examples in [docs/queries.sql](docs/queries.sql).

---

## How it works

Eleven stages, one module each. Detection is a three-way split whose proposals are
*arbitrated* rather than merged, which is what makes refusal a first-class outcome rather
than an absence of output.

```
  doc.pdf
     │
     ▼
  S1  intake        page typing, deskew, rotation
  S2  layout        running header/footer, headings, captions, footnotes, scale notes
     │
     ├── S3a ruled        rules · dots · cell fills · rule breakpoints
     ├── S3b borderless   projection-profile gutters · side-by-side
     └── S3c traps        TOC · index · prose · callout
              │
  S3d arbitration   two-tier clustering ───────────► rejected_regions (with reasons)
     │
  S4  structure     weighted boundaries, row-source selection, spans, indent hierarchy
  S5  cells         text layer / per-cell OCR (fails closed)
  S6  stitch        2 gates + 5 weighted signals
     │
     ├── S7 normalise   types, scale, negatives, digit grouping
     ├── S8 context     title, section, units, footnote links
     └── S9 confidence  14 signals → isotonic calibration → review queue
     │
  S10 persist       SQLite · xlsx · manifest · assets
  S11 audit         metrics, determinism verification
```

Stage-by-stage detail, with the measurements that forced each design, is in
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**. Every non-obvious choice and the
alternatives rejected are in **[docs/DECISIONS.md](docs/DECISIONS.md)**.

---

## What works, what does not

**Works well.** Detection across three different grid regimes (drawn rules, dotted rules,
per-cell background fills) including the horizontally-ruled-only case the brief says
neither classical approach handles alone. Refusal of all four trap types with a recorded
reason. Multi-level headers with spanning cells and `header_path` per column.
Indentation-only row hierarchies. Cross-page stitching with header de-duplication. Indian
and Western digit grouping, parenthesised negatives, magnitude suffixes, six-plus date
formats, and the semantic distinction between empty, `-`, `–`, `Nil`, `NA` and `0`.
Reconciliation that flags a genuine typo and never repairs it. Byte-reproducible output.

**Does not work.** The Docker image is unbuilt and unverified. The OCR path has never
read a real character (no Tesseract available during development, and the image that
would have provided it was never built) — it is implemented and fails closed, but
unvalidated. Content rotated without a `/Rotate` entry is detected and then ignored. The
Shell cash-flow statement recovers 2 columns against a true 3–4. Chart data recovery and
the active-learning loop are not implemented. Row-span detection is deliberately
conservative.

All of it, with severities and what I would fix first, is in
**[docs/LIMITATIONS.md](docs/LIMITATIONS.md)** — which is worth reading before the
numbers above.

---

## Repository layout

```
PROBLEM_STATEMENT.md       the problem, the objectives, the success criteria
README.md                  this file
notebooks/
  01_results_and_milestones.ipynb  milestones, reliability diagrams, PR curves, ablations
src/gridsmith/             the pipeline, one module per stage (s1_intake … s11_metrics)
  config.py                every tunable threshold, hashed into run_id
  schema.sql               the output schema, with the reasoning in comments
  calibration.json         the fitted confidence curve (vendored, never fitted at runtime)
metrics/
  ground_truth/*.json      hand-labelled cells, with the labelling method and corrections
  fixtures/                adversarial fixture generator + exact ground truth
  eval.py                  detection / TEDS-Struct / cell accuracy / silent errors / PR curve
  eval_fixtures.py         behavioural pass-fail checks
  run_all.py               every configuration over every document
  fit_calibration.py       isotonic (PAV) calibration fit
  calibration_report.py    pooled reliability diagram and PR curve
  baselines/               pdfplumber and pymupdf comparison
  results/*.json           committed measurement output, read by the notebook
tests/                     52 unit tests + 12 end-to-end (determinism, CLI, schema)
tools/                     probe.py, dump_tables.py, measure_pitch.py, vendor_deps.sh
docs/
  ARCHITECTURE.md          stage by stage, with the measurements that forced each design
  DECISIONS.md             every non-obvious choice and the alternatives rejected
  LIMITATIONS.md           every known failure, with severity
  queries.sql              worked SQL against the output schema
  renders/                 committed debug overlays used by ARCHITECTURE.md
.github/workflows/ci.yml   tests + determinism + fixture regression gate
```

---

## Reproducing the numbers

```bash
pip install -r requirements-dev.txt

python -m pytest -q                        # 64 tests

python metrics/run_all.py --only-fixtures  # 8 fixtures, no sample PDFs needed
python metrics/eval_fixtures.py --min-pass 56
```

The three sample PDFs are not redistributable and are not in this repository. To
reproduce the sample-document numbers, point `GRIDSMITH_SAMPLES` at a directory
containing them:

```bash
export GRIDSMITH_SAMPLES=/path/to/samples   # PowerShell: $env:GRIDSMITH_SAMPLES="..."
python metrics/run_all.py                   # all configurations, all documents
python metrics/baselines/run_baselines.py   # pdfplumber + pymupdf comparison
python metrics/calibration_report.py        # pooled reliability + PR curve
```

Committed results under `metrics/results/` are what the notebook reads, so the notebook
renders without re-running anything.

---

## License

[MIT](LICENSE).
