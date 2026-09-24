# Corpus survey of the geometric PDF table lane

**What ran:** the harness `carmel/tools/corpus_table_survey.py` over a stratified random
sample of **300 PDFs** drawn across all seven corpus folders (`_2 Post`, `_2021`…`_2026`;
2,087 PDFs total), seed 0. Each PDF was ingested into a scratch workspace and carried
through the exact lane `carmel report-tables` runs
(`carmel.services.general_table_report.report_document_tables`, no series agent). This is
the first time the lane has been measured over a real corpus.

## Reproduce

```bash
python -m carmel.tools.corpus_table_survey \
  --workspace <scratch>/ws \
  --out docs/i102/corpus_survey_results_n300_seed0.jsonl \
  --root "$CORPUS_ROOT/_2 Post" \
  --root "$CORPUS_ROOT/_2021" \
  --root "$CORPUS_ROOT/_2022" \
  --root "$CORPUS_ROOT/_2023" \
  --root "$CORPUS_ROOT/_2024" \
  --root "$CORPUS_ROOT/_2025" \
  --root "$CORPUS_ROOT/_2026" \
  --sample-size 300 --seed 0 --prune-unstored
```

`$CORPUS_ROOT` is the operator's own local paper library; set it to wherever that lives.

The committed `corpus_survey_results_n300_seed0.jsonl` is the raw per-document output
(one `DocumentRow` JSON per line). The sample is deterministic given the same corpus,
size, and seed. Each row's identity is an opaque `doc_id` (a content sha256, never an
absolute path or a paper's filename) — no full extracted text is stored anywhere in the
file. `caption_fragment` is a short excerpt of real extracted text, capped at 48
characters, and every `detail` string is bounded lane diagnostics (geometry and
classifier messages), not document content.

## Probe answers (established by running, not reading)

1. **Local PDF → stored artifact, no network.** Yes. The lane needs only a stored
   `raw.bin` that re-hashes to its sha; it re-extracts fragments from the bytes itself and
   never reads the stored text. The route is `extract_text` (real extractor) →
   `store_artifact(provenance=MANUAL)`. **Seam finding:** there is no public one-call door
   to ingest an *arbitrary* local PDF. The only manual door,
   `acquisition.admit_file`, is gated on a literature-request identity + full-article check
   that arbitrary corpus PDFs cannot pass, so the harness assembles a metadata-only
   `FetchedArtifact` by hand around the real extractor. The extracted text is real; only the
   fetch metadata is synthesised. This is a missing seam, not a blocker.
2. **`carmel report-tables --workspace W --sha SHA` runs.** Yes — verified on one paper: it
   authenticates the bytes, reports the candidate and a typed refusal
   (`grid_not_derived`), and exits 0.
3. **Cost.** ~0.53 s/paper mean (median 0.10 s, max 8.26 s); 160 s for all 300. Memory flat,
   no parser crashes. The full 2,087-PDF corpus would take ≈ 20 min — the 150–300 target was
   never cost-constrained.

## Yield

**0 of 300** papers produced a MEASURED grid that stored and replayed. Rate **0.0%**.
Every document completed as a clean `reported` outcome: **zero crashes, zero whole-document
refusals, zero ingest failures.** The fail-closed contract held across the entire sample.

## Refusal taxonomy, ranked (what stops a paper), N = 300

| Rank | Stage | Count | Share |
|---|---|---:|---:|
| 1 | **No candidate proposed at all** | 263 | 87.7% |
| | ↳ fragment extraction lossy / page-failed | 211 | |
| | ↳ non-lossy, no proposable `Table N` caption | 52 | |
| 2 | **Candidate proposed, refused by geometry before the classifier** | 33 docs / 144 cand. | |
| | ↳ `grid_not_derived` | 140 | |
| | ↳ `too_few_columns` | 4 | |
| 3 | **Reached the classifier** | 4 cand. | |
| | ↳ `undecided` | 2 | |
| | ↳ `not_measured` | 2 | |
| | ↳ `measured` | 0 | |

Inner breakdown of the 140 `grid_not_derived` refusals:

| inner reason | count |
|---|---:|
| `page_incomplete` | 93 |
| `straddling_fragment_at_the_box_edge` | 25 |
| `column_structure_unresolved` | 13 |
| `unattachable_affix_band` | 5 |
| `orphaned_band_below_the_box` | 2 |
| `unmapped_member` | 1 |
| `ambiguous_affix_band` | 1 |

**Root-cause rollup — where the next ticket goes.** The single dominant blocker is the
**fragment extractor failing pages** on `UnsupportedContentConstruct`: a text-show operator
a clipping path does not provably contain, or a `/Do` on a `/Form` XObject the module cannot
position. It marks pages failed → the extraction is lossy → `build_inventory` refuses the
page as `page_incomplete`. This one cause accounts for **211** of the zero-candidate
documents **plus 93** of the `grid_not_derived` candidate refusals — over 300 of the ~407
refusal events in the sample. Example: an 18-page paper whose text `extract_text` read
cleanly (53 kB of prose) yielded **0 fragments** because all 18 pages hit the clipping-path
construct. The classifier — the part everyone assumed was the bottleneck — saw only **4
grids in 300 papers**. It is not the floor; fragment extraction is, with geometric grid
derivation a distant second (~47 genuine non-lossy refusals).

## Are the refusals honest? (hand inspection)

Yes, with one deeper finding. I inspected the caption context of the top refusals via the
text extractor (a different extractor than the fragment lane, so this judges *whether a
measured table exists*, not the exact geometry). No refusal fabricated a table, and I found
**no case of a clean measured experimental grid wrongly refused or misclassified**. The
refusals decompose into three honest kinds:

- **Prose "Table N" mentions, correctly refused.** Curran 2017 and Nakamura 2017 each
  anchored on an in-text sentence — *"Table 1. Furthermore, five recent mechanisms…"* — not
  the caption; `column_structure_unresolved` over that paragraph is the correct refusal.
- **Real captioned tables refused on provable-geometry grounds.** `Table1: Recommended
  thermodynamic parameters`, `Table1: Thermodynamic and Kinetic Library`, etc. — refused for
  `straddling_fragment_at_the_box_edge` / affix bands / bridged column boundaries. These are
  coverage-limited but true to the project's refuse-rather-than-guess doctrine.
- **Tables that are present but carry the wrong *kind* of data.** This is the deeper finding.
  Gotama 2022 — a paper whose title is *Measurement of the laminar burning velocity* — has
  exactly two tables: Table 1 is a **literature survey** (Year / Author / Method / H₂ ratio;
  classified UNDECIDED, correctly) and Table 2 is **computed Arrhenius rate constants**
  (A, n, Eₐ; refused, and would classify NOT_MEASURED even if derived). Its measured
  burning-velocity-vs-φ data is in a **figure**, not a table. This pattern is pervasive:
  across the corpus, measured combustion data lives in plots, while tables carry computed
  rate parameters, thermodynamic values, and literature/method surveys.

The two NOT_MEASURED verdicts (an ML "types of distribution shifts" table; an objective-value
table) and two UNDECIDED verdicts (the Gotama survey; a validation-success-rate table) are
all defensible on their content.

**Verdict:** the 0/300 yield is *mostly honest*, not mostly a defect. A large share of papers
have no measured table to find because measured data is plotted; the lane is additionally
floored, well before the classifier, by fragment-extraction page failures and — secondarily —
by conservative grid geometry. Raising yield means (1) surviving clipping-path/`/Form`-XObject
pages in the fragment extractor, then (2) loosening grid derivation on real captioned tables —
in that order — and even then the ceiling is bounded by how little measured data is tabulated
at all. **None of that is this ticket's to change** (the non-goals forbid tuning); it is where
the evidence says the next tickets should go.

## Crashes

None. No traceback, no non-typed error, no non-zero exit outside a clean refusal, across all
300 documents. The fail-closed contract is intact on real input.

## What I could not verify

- I judged table content from the **text** extractor, not by rendering pages, so "a measured
  grid exists / does not exist" is inferred from extracted text, not from pixels. A page the
  fragment lane dropped for a clipping path *might* hide a table that even the text extractor
  also missed; I did not render to rule that out.
- I ran **300** of 2,087 PDFs (a seeded stratified sample), not the whole corpus. Cost would
  allow the full run (~20 min); the sample is what the ticket targeted.
