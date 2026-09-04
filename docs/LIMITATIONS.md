# LIMITATIONS

What does not work, what is untested, and where the reported numbers should not be
trusted. Written to be useful to whoever picks this up next, including me.

---

## The headline number needs a caveat before anything else

**`raw_text` accuracy 1.000 over 202 hand-labelled cells is not a generalisation
claim.** Those cells come from the three documents I developed against, and every
threshold in `config.py` was chosen with them in view. On a held-out document I would
expect materially worse — I would guess 0.90–0.97 on digital financial tables of similar
shape, and considerably worse on anything scanned (see L1).

The evidence that is *not* contaminated by tuning:

- **The fixture suite** (56/57 behavioural checks). Ground truth is exact by
  construction and the cases were written to cover what the samples lack.
- **The ablation table.** Removing rule and fill evidence drops detection F1 from 1.000
  to 0.444 and cell accuracy from 1.000 to 0.287. That is the honest measure of how much
  of the result rests on author-drawn grid evidence being present. On a document with
  neither, expect the `no_ruled` column.

I would rather state this plainly than have it discovered.

---

## L1 — The OCR path is implemented but has never read a character

**Severity: high. This is the biggest gap in the submission.**

No Tesseract binary was available on the development machine, so
`TesseractBackend.available` was `False` for the entire project. What that means
concretely:

- The raster geometry path **is** exercised end to end: page typing identifies the
  scanned fixture, skew estimation recovers 1.55° against a true 1.6°, morphological
  rule extraction finds 23 intersections, and the grid is rebuilt from that lattice
  alone (15 cells) even though the page has no text layer at all.
- The OCR path itself — cell cropping at 300 dpi, `image_to_data`, per-character
  confidence, the `psm` choice, the digit-repair rule — has been **executed but never
  validated against real OCR output**. It is the least trustworthy code in the repo.
- The `NullBackend` fallback *is* tested, and it fails closed: every raster cell is
  written `raw_text = NULL`, `source = 'none'`, confidence 0, with an
  `OCR_UNAVAILABLE` issue. The fixture asserts that no cell text is invented, and it
  passes. So a scanned page in the held-out set will produce **no values and a full set
  of flags**, not wrong values.

That is the right failure mode but it is still a failure: on a scanned page the pipeline
extracts structure and refuses content — 15 cells, all flagged, no values. The Dockerfile installs
`tesseract-ocr`, so the image would behave differently from every measurement here — but
the image was never built either (L8), so that path has been exercised nowhere. Treat
every OCR-related number and code path as unverified.

**What I would do first with more time:** install Tesseract, run the `f8` fixture,
and check the digit-repair rule against real confusions rather than the table of
substitutions I wrote from memory.

---

## L2 — Rotated *content* is detected and then ignored

`detect_content_rotation` identifies a page whose text runs vertically without a
`/Rotate` entry, and `transpose_box` exists to map coordinates for it. **Nothing calls
`transpose_box`.** A page with sideways content will be detected, recorded in
`pages.content_rotation`, and then processed as though upright — which will produce
garbage rows and columns.

Pages with a proper `/Rotate` entry are fine, because PyMuPDF applies the rotation
matrix and everything downstream sees an upright page; the landscape fixture (`f7`)
confirms that. The gap is specifically *content rotated without the page declaring it*,
which happens when a landscape exhibit is pasted into a portrait page as vector content.

I discovered this while building the fixture, initially built the fixture wrong (drawing
landscape content and then setting `/Rotate 90`, which makes a viewer see sideways text —
not what a landscape exhibit is), and corrected the fixture rather than implement the
transposition. So the *fixture* now tests the case that actually occurs and passes; the
rarer case remains unhandled and undetected-in-effect.

---

## L3 — Structure failures I know about

**Shell balance sheet (p1), 47 rows.** The enumerator column (`(a)`, `(i)`) and the Note
column are both narrow, and three long labels overflow into the Note column's x range.
The label column therefore absorbs the enumerators. Column *count* now matches ground
truth (6) and TEDS-Struct is 0.988, but the row-label text for those rows includes the
enumerator, e.g. `"(a) Property, Plant and Equipment"` rather than
`"(a)" | "Property, Plant and Equipment"`. Defensible, not what a strict reading of the
sheet would say.

**Header spans are approximate at the outer level.** On the ICRA rating-history table the
three header levels and the `FY2025 → Date/Rating` pairs come out exactly, but the
level-1 group `Current rating (FY2026)` widens one column too far left. `header_path` for
that column reads `["Current rating (FY2026)", "FY2026", "Amount rated (Rs. crore)"]`
where "FY2026" should not be in the path. The centred-extension heuristic
(`_header_row_cells`) is the weakest part of S4.

**Apple consolidated statements of operations (p32).** Ground truth says 4 columns and 2
header rows; the pipeline emits 5 columns and 2–4 header rows depending on the run. The
extra column comes from the `$` adornment band on the first data row not folding
consistently, because folding decides from body rows and that table's first body row is
the only one carrying the symbol. TEDS-Struct 0.81 on that table against 0.93 mean.

**Shell cash-flow statement (p3).** Only 2 columns are recovered against a true 3–4. Not
in the labelled set (I labelled p1 and p2), so it does not appear in the metrics — I am
listing it because it is visibly wrong in `out_shell/debug/page_003.png` and it is the
page containing four of the six `#REF!` values. Those four are consequently *not*
extracted as individual cells, which is why the `#REF!` value expectation in the Shell
ground truth reports "found 1, need 4". The failure is loud (`GRID_LOW_SUPPORT` on every
cell of that table) rather than silent, but it is a failure.

---

## L4 — The calibration set is small and the curve is coarse

698 labelled (prediction, outcome) pairs, of which 146 are errors — and the errors come
almost entirely from the deliberately-degraded ablation runs, not from the default
configuration. Consequences:

- The fitted curve has **9 monotone steps**, so the pipeline emits only about nine
  distinct confidence values. That is why the precision–recall curve is flat between 0.5
  and 0.9: there are no cells with those confidences to move across the threshold. A
  larger labelled set would give a smoother curve; I would not read the flat region as
  meaningful.
- ECE 0.029 pooled is good, but MCE 0.46 is not, and it comes from a 12-point bin. Do not
  trust any single bin with fewer than ~50 points.
- The default configuration produces **no errors on the labelled set**, so its own
  reliability diagram is uninformative (every bin is 100% accurate). The pooled diagram
  is the honest one and it necessarily includes configurations nobody would ship.

**On the 8% / 95% target.** The brief asks for ≤8% flagged catching ≥95% of errors. On
the pooled set the shipped operating point flags 24.8% and catches 91.8%. Read carefully:
the pooled set is 21% errors, and you cannot catch 95% of errors by reviewing 8% of cells
when a fifth of the cells are wrong — that is arithmetic, not a tuning failure. In the
regime the target describes (error rate well under 8%), the default configuration flags
3–11% per document and has zero errors to catch. The number to hold me to is the
**silent-error count**, and on the labelled set with the default configuration it is 0.

---

## L5 — Ground truth is small, and I labelled it wrong twice

202 cells across 6 tables, plus structure-only labels for 4 more, plus 1 labelled
negative in the samples and 4 in the fixtures. That is at the low end of the "8–10
tables" the brief suggests and it limits every statistic above.

More importantly: **the harness caught two labelling errors of mine**, both on the Shell
profit-and-loss table (page 1's header copied onto page 2; the Note column given numeric
values). Both are recorded in the ground-truth file with the reasoning. I take that as
evidence the ruler works, but it also means the labels are hand-made artefacts that were
wrong 2 times out of the ~10 assertions I checked closely, and other errors may remain.

Where the labels required a judgement call, I recorded the convention in the file rather
than pretending there was an obvious answer:

- `$ 167,045` keeps the space the PDF's token positions imply.
- The roman numeral in `I Revenue from Operations` is part of the label, because a
  seventh column for it is blocked by one row that starts at the same x.
- Header cells carry no `normalized_value` (D15).

A different labeller would reasonably disagree with at least one of those.

---

## L6 — Not implemented

- **Chart data recovery** (+3 bonus). `assets` has a `series_json` column and charts are
  detected and cropped; nothing populates it. The Apple stock-performance chart is a
  raster, so this would need axis detection and curve tracing on an image.
- **The active-learning loop** (+3 bonus). Not started. The schema supports it — a
  corrections file keyed by `cell_id` could feed back into later pages of the same
  document — but nothing reads one.
- **`row_span` detection is conservative.** Implemented only for a cell whose text box
  reaches into an empty cell below. Header row-spans and any vertical merge in the body
  that does not look like that are missed.
- **No multi-column page layout handling.** A two-column *prose* page is correctly
  refused as a trap, but a genuine table inside one column of a two-column layout would
  probably be detected with the wrong extent.
- **Watermark handling removes glyphs from the value and flags the cell**, which is
  right, but the flag is applied per cell rather than per overlapping glyph, so a
  watermark crossing twelve cells flags all twelve including the ones it did not touch.

---

## L7 — Performance and resources

Measured on this machine (Windows 11, Python 3.11, 4 cores available to the process):

| document | pages | wall | s/page | peak RSS |
|---|---|---|---|---|
| Apple Form 10-K | 121 | 15.1 s | 0.12 | 173 MB |
| Shell.pdf | 3 | 2.5 s | 0.83 | 122 MB |
| Shell MRPL | 7 | 2.2 s | 0.31 | 137 MB |

Extrapolating, a 100-page document lands near 13 s and well under 200 MB, against a
budget of 20 minutes and 4 GB. Two caveats:

- **The token model for the whole document is held in memory.** At ~1.4 MB per page that
  is 560 MB for a 400-page document — still inside budget, but it grows linearly and the
  cross-page stages (chrome detection, scale inheritance, stitching) are what require it.
  A streaming design would need a two-pass structure.
- **The timings are for a run with no OCR.** Per-cell OCR at 300 dpi is the expensive
  path and is unmeasured (L1). A scanned 100-page document with ~2,000 cells could
  plausibly take 10–20 minutes on 4 vCPU, which would put it at or over the budget. I do
  not know, and I would not claim the budget is met for scanned input.

---

## L8 — Things that are fragile rather than wrong

- **`WRAP_PITCH_RATIO = 0.90`** separates wrapped lines from separate rows and was fitted
  to four measurements from two documents. The two clusters were far apart (0.61 vs 1.19)
  so I expect it to hold, but a document with unusually tight row padding would break it,
  and the failure mode is merged rows — which breaks sums.
- **`MIN_ROWS_FOR_TOLERANCE = 10`** disables the gutter persistence tolerance on short
  tables. Chosen because a 5-row table has no outliers; the specific value is a guess.
- **The `f4` bordered-callout trap passes by never being proposed**, not by being
  refused with a reason. That is a correct outcome but weaker evidence than the other
  three traps, which each record an explicit code.
- **`no_borderless` costs nothing on any of the three samples**, because every table in
  them carries rule or fill evidence. So the borderless detector is, on this corpus,
  entirely insurance. I believe it will earn its place on the held-out set — the brief
  promises "completely borderless tables" — but I have no measurement showing it does,
  only the fixtures I wrote myself.
- **The Docker image has never been built.** Docker Desktop was not running on the
  development machine, so `docker build` was never executed. The Dockerfile is
  therefore **unverified**: the commands inside it are the ones I run locally (the same
  `pip install`, the same fixture generation, the same `gridsmith run` smoke test, all of
  which pass outside the container), but layer ordering, the `tesseract-ocr` apt install
  and the editable-install fallback are untested. Expect to fix something on first build.
  This also means the OCR path remains unvalidated *everywhere* (L1), since the image is
  the only place Tesseract would have been present.
- **Windows-only measurement.** Everything was developed and measured on Windows 11 with
  Python 3.11. Nothing here is platform-specific by design — paths go through `os.path`,
  text I/O is forced to UTF-8 with LF endings, and the determinism machinery pins the
  values that would otherwise vary — but no run on Linux or macOS has been observed.
