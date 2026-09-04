# DECISIONS

Every non-obvious choice, the alternatives, and why they were rejected. Ordered roughly
by how much they shaped the result.

Nearly all of these were *forced* by measuring the sample documents rather than reasoned
out in advance, and where that is true I say so — including the several places where the
measurement contradicted what I had already built.

---

## D1 — Pure classical: no learned model at all

**Decision.** No learned model anywhere. Not Table Transformer, not a YOLO detector, not
even a small classifier for the trap rules.

**Alternatives rejected.**

*Table Transformer / YOLO as one ensemble signal.* Permitted by §3.1 and it would very
likely raise detection recall on the held-out set. Rejected for three reasons. First,
§3.1 requires showing performance with it switched off, so the classical path has to be
strong regardless — the learned model is pure upside on top, and the upside is worth
less than the time it costs. Second, the live review asks me to explain and modify any
line; "the detector fired because these weights say so" is not an explanation I can act
on at 2 a.m. Third, weights would need vendoring, and a 100 MB checkpoint in the repo
for a component that cannot be the sole basis of any decision is poor value.

*A learned classifier for the traps.* Would probably beat my four rules on recall. But
each rule currently keys on a structural invariant and can therefore explain itself in
the `rejected_regions` row ("23 of 26 rows end in a bare integer, 100% ascending, median
1.0 numbers per row"). That explanation is what makes "we correctly excluded the table
of contents" auditable rather than asserted.

**Consequence.** This also claims the +4 bonus by construction: the best configuration
*is* the zero-learned-model configuration, so the gap is 0 points rather than 10.

---

## D2 — Detection produces evidence, not verdicts; one place resolves it

**Decision.** S3 emits rules, dots, cell fills, rule breakpoints, intersections and
gutters, tagged by source. S4 is the only place that turns evidence into boundaries.

**Why.** I started with each detector emitting a finished grid and arbitration picking a
winner. That fails on the ICRA report, where the ruled detector knows the four row
boundaries exactly and knows nothing about the header band above the first rule, while
the borderless detector knows the extent and gets the rows wrong. Neither is a winner;
each is right about a different axis. Separating "what did we observe" from "what do we
conclude" also made the weights (D4) expressible at all.

**Cost.** `Candidate` carries six parallel evidence lists and is fiddlier than a grid.
Worth it.

---

## D3 — Three detection regimes, because three is what the documents use

**Decision.** Treat cell background fills and horizontal-rule breakpoints as first-class
grid evidence alongside drawn rules.

**Why (measured, not assumed).** `tools/probe.py` on the three samples:

- `Shell.pdf` draws every "line" as a row of 0.72 pt squares — 18,066 of them per page.
- `Shell MRPL…pdf` has hairline horizontal rules segmented per cell and **no vertical
  rules at all**.
- `Apple Form 10-K.pdf` has no rules on its table pages; alternating rows carry one
  0.937-grey rectangle **per cell**.

A pipeline built on "thin rectangle ⇒ rule" finds 18,000 rules of length 0.72 on the
first, 3-point fragments on the second and nothing on the third. Ignoring Apple's fills
and falling back to whitespace throws away the best signal on the page: the fill
rectangles *are* the cell lattice.

**Alternative rejected.** Rasterise everything and use morphology uniformly. Simpler and
regime-blind, but it discards sub-point vector precision, costs a 200 dpi render per
page, and cannot see a fill lattice as anything but a grey block.

---

## D4 — Weighted boundary voting, with vetoes

**Decision.** vertical rule 3.0, rule breakpoint 2.5, fill edge 2.0, gutter 1.0; then
veto on straddling, on landing in a word gap, and on duplicating a nearby precise
boundary.

**Alternatives rejected.** *Strict source priority* ("use rules if any exist, else
fills, else gutters") loses the ICRA case where breakpoints give three boundaries and a
gutter is needed for the fourth. *Unweighted union* promotes a whitespace guess to the
authority of a drawn rule — and did exactly that: `cand.x_lines` was the union of all
members and I weighted all of it at 3.0, which put boundaries at x=224 and x=379 on
pages without a header while pages with one used the real rules at 200 and 330. The two
halves of one table stopped matching and stitching refused to join them. One
mis-weighting, three stages of damage.

The vetoes were each added in response to a specific measured failure, which is why they
look ad hoc and are not:

| veto | what it caught |
|---|---|
| straddle, header rows excluded | Shell p2's year/subtotal boundaries were straddled by exactly two lines, both header — vetoing them collapsed four value columns into two |
| word gap | a boundary landed in the 2.4 pt space between `non-fund` and `based`, splitting the ICRA label column mid-phrase; it cut no glyph, so the straddle test could not see it |
| duplicate | a breakpoint at x=249 and a refined gutter at x=254 are one boundary; kept as two they inserted a phantom empty column that shifted every cell one place right |

**The straddle slack matters more than the threshold.** At 0.6 pt a boundary could graze
the last 0.08 pt of a glyph and not be counted — which is how the `non-fund` case got
through. It is now 0.15 pt: a rounding tolerance and nothing more.

---

## D5 — Rows pick one source; columns combine several

**Decision.** Column evidence is unioned and weighted. Row evidence is not: each source
proposes a complete band set and exactly one wins.

**Why.** They fail in opposite directions. An extra *column* boundary usually costs one
spurious narrow column, which folding removes. An extra *row* boundary splits a row, and
a split row breaks every sum in the table. Unioning rule positions with cell-shading
edges turned the ICRA rating table into 8 rows instead of 4 and the Apple segment table
into 2 rows instead of 7 — in both cases because the union added boundaries neither
source would have proposed alone.

**The scoring function took three attempts**, which is worth recording because the first
two look reasonable:

1. `labelled + tall_enough`. Chose two rule-derived bands over six fill-derived ones for
   the Apple segment table, because the header row's label cell is empty and that costs
   the finer set on `labelled`. Five data rows merged into one band.
2. `+ single_row`, counting lines with text in ≥2 columns. Broke the ICRA rating table,
   which wraps four columns at once: one correct row contains five such lines,
   indistinguishable from five real rows.
3. `+ coherent`, on **line pitch**. Wrapped lines sit at 0.61 glyph heights, separate
   rows at 1.19–1.27 (measured; `tools/measure_pitch.py`). Two clusters, nothing between
   them, threshold at the midpoint. This is a property of typesetting rather than of any
   document, which is why it holds on both.

---

## D6 — Stitching: two hard gates, then a weighted score

**Decision.** Gates: adjacent pages, and same column count with ≥85% of *interior*
boundaries agreeing within 6 pt. Then a weighted sum of five signals ≥ 0.62.

**Why gates and not one score.** Adjacency and arity are the two conditions whose
violation produces an error nobody can see downstream: merging the receivables schedule
on page 12 with the payables schedule on page 15 yields a table that looks fine and is
nonsense. No accumulation of soft evidence should be able to buy past them.

**Why interior boundaries, not band IoU.** Band IoU is the obvious measure and it is
wrong: a table's outer edges are set by its widest cell, so a page carrying the repeated
header is wider than one without it. On the six-page fixture that alone dragged mean band
IoU to 0.79 against a 0.85 gate and split one logical table into six. Interior boundaries
drift by only the point or two the brief warns about. Band IoU is still recorded in the
vote.

**Why weights and not a decision tree.** Every signal is absent on some real document —
the brief says so explicitly. A tree must commit to an interrogation order and inherits
the blind spot of whichever signal it asks about first. A weighted sum degrades
gracefully: two strong signals carry the decision when three are silent.

**Header de-duplication compares against the first part, not the previous one.** On the
six-page fixture the header repeats on pages 1, 3 and 5; comparing page 3 against page 2
(whose first row is data) found no repeat and duplicated two header rows into the data.

---

## D7 — The mechanical guarantee, not just a good score

**Decision.** A set of `MANDATORY_REVIEW_CODES` hard-caps confidence below the
high-confidence floor and enters the review queue regardless of the 8% budget.

**Why.** §4 says every cell must be correct *or flagged*. A purely statistical
confidence, however well calibrated, cannot promise that: there is always a tail. But
some conditions are *known* — OCR was unavailable, the cell contains `#REF!`, the decimal
separator is ambiguous, a token straddles a boundary, the row does not add up. For those
the model should not get a vote. Capping them mechanically converts the invariant from a
statistical hope into a structural property of the output.

The corollary is that the 8% budget can only govern the discretionary tail. A budget able
to evict a known-suspect cell would make the 8% figure a lie, and
`review_queue.csv`'s `priority` column says which kind each row is.

---

## D8 — Refusing to parse is a feature

**Decision.** `1.234,56` and `03/04/2024` yield `normalized_value = NULL` plus a flag,
rather than a guess.

**Why.** These are genuinely ambiguous in one cell. Continental `1.234,56` and Western
`1.234` cannot be told apart without document-level context; `03/04/2024` is 3 April or
4 March. A pipeline that picks one is right about half the time and *confident* every
time, which is the exact failure §4 punishes. `raw_text` is preserved, so a reviewer
loses nothing.

Dates get a second chance: `DateOrderModel` observes every unambiguous date in the
document first (`31/03/2024` can only be day-first) and resolves the ambiguous ones from
that, flagged `order_from_document`. Learning it in a pre-pass rather than incrementally
is what keeps the result independent of processing order, and therefore deterministic.

**Text fidelity choices.** Unicode space separators collapse to U+0020; the seven Latin
f-ligatures are expanded (a reader sees "fi", not U+FB01) with `LIGATURE_EXPANDED`
recorded. Nothing else is touched — Unicode minus, en-dash, superscript digits and
currency glyphs reach `raw_text` unchanged and S7 interprets them additively. A blanket
NFKC would have turned superscript footnote markers into ordinary digits.

**The grave-accent rupee.** Both Indian documents render `₹` as U+0060 via a legacy font
(`(` Million)`). Reading it as INR is a *flagged* inference (`LEGACY_RUPEE_GLYPH`), never
silent.

---

## D9 — Reconciliation tolerance from printed precision, not a percentage

**Decision.** `tol = n_addends · 0.5 · 10⁻ᵈ` where `d` is the decimals actually printed,
compared strictly.

**Why.** A relative tolerance cannot distinguish the two cases that matter. At 0.5% of
the stated value, a subtotal of 359 against addends summing to 360 passes — and that is
the deliberate typo the brief plants and asks us to flag. What bounds an *innocent*
discrepancy is display rounding: each addend hides at most half of its last printed
digit. Three values printed as 33.3 summing to 99.9 against a stated 100.0 passes
(0.1 < 3 × 0.05); 1,184 + 2,421 = 3,605 against a stated 3,505 fails.

**Honest consequence.** At integer precision with two addends, a discrepancy of 1 *is*
consistent with rounding and is not flagged. Building the watermark fixture exposed that
my own planted "typo" (118 + 242 vs 359) was ambiguous by construction, and I changed the
fixture rather than the rule — the rule is right, the test was wrong.

**False-alarm guards, each from a measured false positive.** Ratio labels are excluded
("Total outside liabilities/Tangible net worth (times)" begins with "Total" and is a
ratio); percentage columns are never addends; a "total" smaller than its largest addend
is not a total; a column whose inferred type is not a plain amount is not reconciled at
all. Before these, one correct ICRA table produced two confident `DOWNFOOT_FAIL` errors
and the Apple segment table produced four.

---

## D10 — Calibration fitted offline, applied as a lookup table

**Decision.** Logistic weights fixed by hand in `config.py`; the map from that score to
a probability fitted by isotonic regression (pool-adjacent-violators) and vendored as
`calibration.json`.

**Why.** §3.2 forbids fitting anything at run time. Splitting it this way keeps the run a
pure function of the input while still making the number a probability rather than a
vibe. PAV rather than plain binning because with a few hundred points some bin always
inverts by sampling noise, and a confidence function that is not monotone in its own
score is indefensible ("0.7 means less than 0.6 here").

**Where the negatives come from.** With the default configuration the labelled set
contains no errors, which is good for accuracy and useless for calibration. The ablated
runs extract the *same* documents against the *same* ground truth with a degraded
configuration, generating genuine errors with genuine scores. This is legitimate because
the model is not being fitted to the ablation labels; it is being asked, across a wide
range of extraction quality, "when you said 0.7, how often were you right?"

**Guard rails, because the set is small.** Bins under 8 points merge into their
neighbour; fitted values shrink toward the base rate with weight `n/(n+12)`; the top bin
caps at 0.99, so no cell ever claims certainty. Absent the file, the fallback is
deliberately pessimistic — the raw sigmoid shrunk 25% toward 0.5 — because an
uncalibrated pipeline that *overstates* confidence is the failure mode being punished.

---

## D11 — Ground truth: hand-labelled, and the harness corrected it twice

**Decision.** Two labelled sets: hand-labelled cells from the three samples
(`metrics/ground_truth/`) and generated fixtures whose truth is exact by construction
(`metrics/fixtures/`).

**Why both.** The samples are real but all digital, single-column, and contain no
multi-page table, no rotated page, no scan, no watermark, no two-column prose and no
failing subtotal — every one of which §7 promises in the held-out set. Generating those
means the ground truth is not an opinion: we place every glyph, so there is no arguing
about whether `$ 167,045` should carry a space.

**The labelling was wrong twice, and the harness caught it.** Worth recording, because it
is the strongest evidence the ruler works:

1. The first Shell p2 label carried page 1's header (`As at March 31, 2023`) onto page 2,
   which is the profit-and-loss statement headed `2022-23`. `eval.py` reported 24% cell
   accuracy; the extractor was right.
2. The same label gave the Note column numeric values (25.0, 26.0…). `eval.py` reported
   eight silent errors. The extractor types that column `reference` and leaves
   `normalized_value` null — which is correct, because "25" under a header reading "Note"
   is a pointer to note 25, not a quantity.

Both were fixed in the label with the reasoning recorded in the file. This is also why
"raw_text accuracy 1.000" must be read with D12 in mind.

---

## D12 — What the headline numbers do and do not mean

**Decision.** Report accuracy on the labelled sample set *and* the ablation table *and*
the fixture pass list, and say plainly that the first is not a generalisation claim.

1.000 raw-text accuracy over 202 hand-labelled cells is a real measurement, but those
cells come from documents I iterated against, and every threshold in `config.py` was
chosen with them in view. The honest generalisation evidence is elsewhere: the fixtures
(52/53 behavioural checks, ground truth by construction, written to test cases the
samples do not contain) and the ablation table (removing rule/fill evidence drops
detection F1 from 1.000 to 0.444, which is what tells you how much of the result rests
on that evidence being present).

---

## D13 — Two-tier arbitration clustering

**Decision.** Author-drawn evidence seeds the clusters; whitespace-inferred regions
attach to at most one, and are dropped rather than allowed to bridge two.

**Why.** Single-link clustering over all votes has a specific failure: two independent
tables side by side are found separately by the ruled detector, but the borderless
detector proposes one wide region covering both. Because that region overlaps each,
single-link merges all three and two tables become one — item 5 of the adversarial set,
silently destroyed.

**And the first version of this fix was too strict.** Restricting "strong" to grid and
fills split the ICRA summary table in two, because its header shading and body shading
are separate fill groups and the horizontal rules spanning both were being treated as a
mere bridge. Strong now means *any* author-drawn evidence, and strong clusters that are
stacked (contiguous in y, sharing x) merge with each other — a shaded header block above
a shaded body block is one table, while two side-by-side tables overlap in y and are
disjoint in x. Testing both axes separates the cases without knowing in advance which is
which.

---

## D14 — Content-addressed ids, and rewriting the xlsx archive

**Decision.** Every id is a blake2b digest of its content (`sid("cell", table_id, row,
col)`), never an autoincrement. Excel files are saved to memory and the zip rebuilt with
a constant timestamp.

**Why.** Both are required for §3.2 and neither is obvious. A counter reorders whenever
anything upstream changes, so a diff of two runs after an unrelated edit is unreadable.
And `openpyxl` stamps `docProps/core.xml` plus every zip entry with `datetime.now()`, so
two runs differ in dozens of bytes even when every cell matches — not the kind of thing a
two-run diff lets you overlook.

`--fixed-time` pins the three `extraction_runs` timestamps too, so a reviewer running
twice and diffing `out/` byte-for-byte gets a clean result rather than a one-row
explanation. `gridsmith verify --a … --b …` proves it either way.

---

## D15 — Header cells carry no normalized value

**Decision.** A header cell keeps `raw_text` and a `value_type` but its
`normalized_value` is NULL.

**Why.** A header is a label. Emitting `2024.0` for the column headed "2024" is not
exactly wrong, but it invites a consumer to sum a column label, and the meaning of a
header belongs in `columns.header_path` where it can be queried properly. This surfaced
as six apparent silent errors against ground truth that had said `null` from the start —
the label reflected the intent and the code did not.

---

## D16 — Rejected alternatives worth naming

**`camelot` / `tabula` / `docling` as the pipeline.** An automatic fail per §3.4, and
they would not survive `Shell.pdf` anyway: `camelot`'s lattice mode needs drawn lines,
and 18,066 disconnected 0.72 pt squares are not lines. They are used as baselines
instead (`metrics/baselines/`).

**Reading order.** Discarded entirely; only geometry is used. In `Shell.pdf` the header
row `Particulars | Note | As at …` is the *last* thing in the content stream. Any
extractor that trusts block order emits garbage on that file, and it is a printed Excel
sheet — one of the most common shapes of real financial PDF there is.

**PyMuPDF's own `find_tables()`.** Genuinely good and it would have saved days. Rejected
as the primary path because §3.4 forbids an off-the-shelf extractor as the whole
solution, and because it gives no per-cell provenance or confidence, which is where most
of this assignment's marks live. It is included as a baseline.

**A single global confidence threshold.** Rejected in favour of mandatory codes plus a
ranked queue, because a threshold cannot express "I do not know" differently from "I am
70% sure".
