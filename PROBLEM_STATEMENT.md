# Problem statement and objectives

This document states the problem `gridsmith` was built to solve, the objectives it was
held to, and the criteria by which it should be judged. It is the "what and why";
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) is the "how",
[docs/DECISIONS.md](docs/DECISIONS.md) is the "why not otherwise", and
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) is the "where it still falls short".

---

## 1. Context

Financial and regulatory PDFs — annual reports, Form 10-Ks, rating-agency notes, audited
financial statements — are the primary record for an enormous amount of quantitative
information, and essentially none of it is machine-readable. A PDF has no notion of a
table. It has glyphs at coordinates and, sometimes, line segments at other coordinates.
Any "table" a reader perceives is an emergent property of that geometry.

Downstream consumers — analysts, data pipelines, valuation models — need that content as
structured rows and columns with correct numeric values, correct headers, and correct
units. The gap between the two is the problem.

## 2. The problem

> Given a large, messy, real-world PDF, recover every table it contains as structured
> data — with headers, units, footnotes and surrounding context attached — such that a
> consumer can trust each extracted value, and such that any value that cannot be
> trusted is *visibly marked as untrustworthy rather than silently wrong*.

The second clause is the whole difficulty. Extracting *something* from a PDF table is a
solved problem with a dozen off-the-shelf libraries. Extracting something you can stake
a number on is not, because the failure mode of every existing tool is the same: it
returns a clean-looking, confidently-formatted, wrong answer, and nothing in the output
distinguishes it from a right one.

### 2.1 Why it is hard

The difficulty is not one problem but a family of them, and the sample corpus exhibits
most of them at once.

| Difficulty | What actually happens in the documents |
|---|---|
| **No table primitive** | The PDF format has no table object. Structure must be inferred from glyph positions and drawing operators. |
| **Inconsistent grid evidence** | One document draws full rules; one draws dotted rules; one uses per-cell background fills and no rules at all. A fourth case — horizontal rules only, no verticals — defeats rule-following and whitespace-projection alike. |
| **Degenerate vector geometry** | `Shell.pdf` does not draw rules. It draws **18,066 disconnected 0.72 pt squares** that a human eye resolves into lines. Any detector that looks for long segments finds nothing. |
| **False positives that look perfect** | A table of contents with alternating row shading is, geometrically, a flawless seven-column table. So are indexes, two-column prose, and bordered callout boxes. A detector good enough to find real tables is good enough to find these. |
| **Tables that span pages** | A single logical table runs across six pages, repeats its header on some of them but not all, drifts ±2 pt in column position, and is interrupted by running headers and footers. |
| **Multi-level and implicit structure** | Headers nest two or three levels deep with spanning cells. Row hierarchy is frequently encoded *only* as indentation depth. |
| **Semantics outside the grid** | The scale ("₹ in crore") is declared in a sentence above the table. The footnote marker is in a cell on page 1 and its text is on page 2. The unit belongs to the column, not the cell. |
| **Value ambiguity** | `(1,234)` is negative. `1,23,456` is Indian grouping. An empty cell, `-`, `–`, `Nil`, `NA` and `0` mean five different things, and conflating them corrupts every downstream sum. |
| **Degraded pages** | A 45° watermark crosses the table body. A scanned page is skewed −1.6° and has no text layer at all. |
| **Genuine errors in the source** | Some subtotals in real documents do not add up. A system that "fixes" them has destroyed evidence; a system that ignores the discrepancy has missed a signal. |

### 2.2 The governing constraint

The brief's section 4 states the rule that shapes the entire design:

> **Every cell is either correct, or flagged as uncertain.**

This is deliberately not "be accurate". It is a statement about the *joint* distribution
of correctness and confidence. A system that is 99% accurate and silent about the other
1% is worse, for this purpose, than one that is 90% accurate and flags exactly the 10% it
got wrong — because the first requires re-checking everything and the second requires
re-checking a tenth of it.

The quantity that matters is therefore not accuracy. It is the **silent error**: a cell
that is wrong, shipped with high confidence, and carries no flag. The target for that
number is zero, and it is the number this project should be held to.

---

## 3. Objectives

### 3.1 Primary objectives

| # | Objective | How it is measured |
|---|---|---|
| **O1** | **Detect tables** across heterogeneous grid regimes, including the rules-only and fills-only cases | Detection precision / recall / F1 at IoU ≥ 0.7 against hand-labelled regions |
| **O2** | **Refuse non-tables** — TOC, index, prose, callouts — with a recorded, inspectable reason | Trap negatives refused vs. leaked |
| **O3** | **Recover structure**: multi-level headers with spans, indentation hierarchies, merged cells | TEDS-Struct (tree-edit-distance similarity, structure only) |
| **O4** | **Recover values faithfully**, distinguishing negatives, digit-grouping conventions, magnitude suffixes and the five distinct kinds of "blank" | Exact-match accuracy on `raw_text` and on `normalized_value` |
| **O5** | **Attach context**: column units, declared scale, section title, footnote-to-cell links, cross-page continuation | Behavioural checks on fixtures with exact ground truth |
| **O6** | **Emit calibrated confidence** such that a stated 0.9 means an empirical 0.9 | Expected calibration error; reliability diagram |
| **O7** | **Ship zero silent errors** | Count of cells that are wrong ∧ confidence ≥ 0.85 ∧ unflagged |

### 3.2 Non-functional objectives

| # | Objective | Rationale and target |
|---|---|---|
| **N1** | **Determinism** | Two runs over the same input must produce byte-identical output, timestamps excepted. Required by §3.2, and a precondition for the output being auditable at all. |
| **N2** | **Offline operation** | No network access, no API keys, no hosted inference. Enforced by a socket guard that raises on connect, not merely asserted in prose. |
| **N3** | **No language model** | §3.4 forbids an off-the-shelf extractor as the whole pipeline; the stronger choice taken here is that no LLM touches the document at any stage (see [D1](docs/DECISIONS.md)). |
| **N4** | **Resource budget** | Under 20 minutes and 4 GB for a 100-page document. |
| **N5** | **Ablatability** | §3.1 permits learned components but requires showing performance with each switched off. Every major detector has a `--no-*` flag and a measured ablation row. |
| **N6** | **Auditable output** | A SQLite database matching the §6.1 minimum schema, per-cell provenance back to page geometry, and a ranked review queue with reason codes. |

### 3.3 Explicit non-goals

Stated so that their absence reads as a decision rather than an oversight:

- **Not a general document-understanding system.** Prose, images and forms are out of
  scope except where they supply context to a table.
- **Not a chart-data recovery system.** Charts are extracted as image assets with their
  captions; the underlying series are not reconstructed.
- **Not an error-correcting system.** When a subtotal fails to reconcile, the discrepancy
  is flagged and the original value preserved. Repairing it would destroy the evidence
  that something is wrong.
- **Not a learned detector.** See N3 and [D1](docs/DECISIONS.md).

---

## 4. Success criteria

The project is successful if, on documents it was not tuned against:

1. **Zero silent errors.** The single criterion that matters most.
2. Detection F1 materially above the off-the-shelf baselines, *particularly on the
   negatives* — the baselines' characteristic failure is emitting the table of contents.
3. Structure recovery good enough that headers and units survive into the database
   without string surgery at query time.
4. Confidence that is calibrated, so the review queue can be truncated at a budget and
   the operator knows what that costs them.
5. Byte-identical reruns.

### 4.1 On the review-budget target

The brief sets a target of flagging ≤ 8% of cells while catching ≥ 95% of errors. That
target is achievable only in a regime where the error rate is well below 8%. On the
pooled evaluation set — which deliberately includes ablated runs so that it *contains*
errors to catch — the error rate is 21%, and no threshold can catch 95% of errors by
reviewing 8% of cells when a fifth of them are wrong. This is arithmetic, not a tuning
failure, and it is reported as such rather than worked around. In the regime the target
actually describes, the default configuration flags 3–11% per document and has zero
errors to catch.

---

## 5. Inputs and deliverables

### 5.1 Inputs

Three supplied documents, chosen to exercise different failure modes:

| Document | Pages | What it tests |
|---|---|---|
| Apple Form 10-K | 121 | Scale; a TOC that is geometrically a perfect table; adornment columns (`$`, `%`) that are not data |
| Shell financial statements | 3 | 18,066 disconnected fill squares in place of rules; cash-flow layout |
| ICRA rating report (MRPL) | 7 | Three-level headers; dotted rules; Indian digit grouping; tables 18 pt apart that must not merge |

Plus **eight generated adversarial fixtures** whose ground truth is exact by
construction, covering continuation across six pages, side-by-side tables, all four trap
types, watermarks, a deliberately failing subtotal, cross-page footnotes, landscape
rotation, and a skewed scan with no text layer.

> The three source PDFs are not redistributable and are therefore not in this repository.
> Point `GRIDSMITH_SAMPLES` at a directory containing them to reproduce the sample-document
> numbers; the fixture suite runs without them.

### 5.2 Deliverables

- A deterministic, offline CLI (`run`, `audit`, `debug`, `verify`, `query`).
- A SQLite database implementing the §6.1 minimum schema plus additive columns.
- One `.xlsx` per logical table, with Data / Context / Provenance / Issues sheets.
- A ranked review queue with reason codes and plain-English explanations.
- An evaluation harness computing every metric above, plus ablations and baselines.
- Documentation of the design, the rejected alternatives, and the known failures.

---

## 6. How to judge this repository

Read in this order:

1. **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)** — first, deliberately. It states what
   does not work before the numbers have a chance to impress.
2. **[README.md](README.md)** — the measured results, including the ablations and the
   baseline comparison.
3. **[notebooks/01_results_and_milestones.ipynb](notebooks/01_results_and_milestones.ipynb)** —
   the same results as reliability diagrams, PR curves and ablation charts.
4. **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** and
   **[docs/DECISIONS.md](docs/DECISIONS.md)** — the design and its rejected alternatives.

The numbers on the three sample documents include several 1.000s. Those documents were
developed against, and every threshold in `config.py` was chosen with them in view; they
are reported for completeness but they are not evidence of generalisation. The
uncontaminated evidence is the **fixture suite** (56/57 behavioural checks, ground truth
exact by construction) and the **ablation table** (which shows what each component is
actually worth). Weight them accordingly.
