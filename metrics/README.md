# metrics/

The ruler, and the results it produced. Build the ruler before the machine — this
directory is why the numbers in the top-level README mean anything.

## Reproduce everything

```bash
python metrics/fixtures/make_fixtures.py     # regenerate the adversarial PDFs + GT
python metrics/run_all.py                    # default + 3 ablations, samples + fixtures
python metrics/fit_calibration.py            # refit src/gridsmith/calibration.json
python metrics/run_all.py                    # re-measure with the new curve
python metrics/calibration_report.py         # pooled reliability + PR curve
python metrics/baselines/run_baselines.py    # pdfplumber / pymupdf comparison
```

`run_all.py` takes about three minutes (the Apple 10-K is extracted four times).
Sample documents are located via `$GRIDSMITH_SAMPLES`, defaulting to the path they were
supplied at.

## Two labelled sets, for two different jobs

### `ground_truth/` — hand-labelled, real documents

202 cells across 6 tables, structure-only labels for 4 more, 1 labelled negative, and
`value_expectations` for specific values (`#REF!`, a parenthesised negative). Scored
statistically by `eval.py`.

**How it was labelled.** By reading the rendered page text (`page.get_text` per page)
and cross-checking against the page image. Table extents were read off the document's
*own* geometry — horizontal rule coordinates and cell-fill rectangles are the author's
cell boundaries — rather than off the extractor's output. Judgement calls are recorded
in each file:

- `$ 167,045` keeps the space the PDF's token positions imply.
- The roman numeral in `I Revenue from Operations` is part of the label, because a
  seventh column for it is blocked by one row starting at the same x.
- A header cell carries no `normalized_value` (a header is a label — DECISIONS D15).
- A "Note" column is typed `reference`, and its values are not quantities.

**Precision on labelled pages only.** `pages_exhaustively_labelled` lists the pages
where I labelled *every* table. Detection precision is computed only there — scoring
precision on unlabelled pages would count every correct table found as a false positive,
which measures nothing.

**The labels were wrong twice and the harness caught both.** Recorded in the files as
`labelling_correction`:

1. Shell p2 originally carried page 1's header (`As at March 31, 2023`) onto page 2,
   which is the profit-and-loss statement headed `2022-23`. `eval.py` reported 24% cell
   accuracy — the extractor was right.
2. The same label gave the Note column numeric values (25.0, 26.0…). `eval.py` reported
   eight silent errors. Typing it `reference` with a null value is correct.

I regard that as the strongest evidence the harness works, and as a reason to treat
"1.000 accuracy" with the caution in LIMITATIONS L5.

### `fixtures/` — generated, ground truth exact by construction

The samples are all digital, single-column, and contain no multi-page table, no rotated
page, no scan, no watermark, no two-column prose and no failing subtotal — every one of
which §7 promises in the held-out set. Generating those cases means the truth is not an
opinion: `make_fixtures.py` places every glyph, so where the extractor disagrees, the
extractor is wrong.

Scored by `eval_fixtures.py` as **behavioural pass/fail**, because a fixture asserts a
specific behaviour ("these six page-level tables are one logical table"; "this subtotal
must be flagged and must not be corrected") rather than an average.

Fully deterministic — no randomness, fixed fonts and coordinates — so the PDFs can be
committed or rebuilt at will.

## What `eval.py` measures, and why each choice

| criterion | measure | note |
|---|---|---|
| detection | P / R / F1 at IoU ≥ 0.7 | labelled pages only; traps scored separately |
| structure | TEDS-Struct | implemented directly; the reference implementation needs a dependency we cannot vendor offline, and for a two-level tree the general tree-edit distance collapses to row-sequence edit distance with cell-label substitution — exact, not approximate |
| content | exact match on `raw_text`, and separately on `normalized_value` | whitespace runs collapsed, because inter-token spacing is a rendering artefact; every other difference counts |
| silent errors | wrong in *either* field, confidence ≥ 0.85, not mandatory-flagged | the headline number |
| calibration | reliability diagram + ECE, flagging PR curve | pooled across configurations by `calibration_report.py` |

A table matches ground truth at IoU ≥ 0.5 for *content* scoring — a looser bar than the
0.7 detection threshold, deliberately, so a poor detection still has its cells scored
rather than vanishing from the content metrics.

**Abstentions are excluded from calibration.** A cell that is empty in the ground truth
and empty in the output is not a prediction. Including them inflated accuracy and, since
empty cells score low confidence, poisoned the reliability diagram from the other side —
the 0.0–0.1 bin came out 58% "accurate", giving a maximum calibration error of 0.58 that
said nothing about any value a person would act on.

## Where the calibration negatives come from

With the default configuration the labelled set contains **no errors**. That is good for
accuracy and useless for calibration. The ablated runs extract the same documents against
the same ground truth with a degraded configuration, producing genuine errors with genuine
confidence scores.

This is legitimate because the confidence model is not fitted to the ablation *labels*;
it is asked, across a wide range of extraction quality, "when you said 0.7, how often
were you right?" The fit is isotonic (pool-adjacent-violators) so the curve is monotone
by construction — a confidence function that is not monotone in its own score is
indefensible.

## `results/`

| file | contents |
|---|---|
| `all_configs.json` | every configuration's aggregate metrics and per-document timings |
| `calibration_points.json` | the pooled (confidence, outcome) pairs |
| `calibration_report.json` | reliability bins, ECE/MCE, the full PR curve, operating point |
| `fixtures.json` | every fixture check with its pass/fail and detail |
| `baselines.json` | the baseline comparison |
| `summary.json` | aggregate over the sample documents |

Each output directory also gets `ground_truth_eval.json` next to its `metrics.json`, and
`gridsmith audit --out <dir>` prints both together.
