# I-062 — How often is a dimensionless coordinate undeclarable?

**Question.** Carmel's flagship stored dataset plots a laminar flame speed against an
**equivalence ratio** coordinate. Equivalence ratio is dimensionless; the units vocabulary
accepts exactly three unit tokens for it (`-`, `1`, `dimensionless`, all normalising to `1`)
and refuses everything else. The target paper prints none of the three — its header renders a
symbol-font φ that decodes to `/`, which is refused — so the axis fell back to the schema's
"quantity this table does not model" state and the coordinate is semantically unidentifiable.
That fallback is the honest encoding and is not in question. **The question is how often this
happens across the corpus**, because equivalence ratio is the single most common coordinate in
combustion literature.

This is a **measurement, not a code change.** No encoding, schema, axis, or unit vocabulary was
touched. No unit alias was added.

## What this report is — and is not (read this first)

🔴 **This is a TEXT-LEVEL proxy, not a table-level census.** For each document it runs the
project's own extractor (`carmel.agents.tools.extract.extract_text`) over the stored `raw.bin`
and searches the **linear extracted text** — exactly the surface the project grounds quotes
against. It does **not** enumerate table headers: this project has no automatic table discovery,
so there is no way to ask "what unit token sits in *that* column's header." Finding a usable
token *somewhere* in a paper does not prove it could ground *that column's* unit.

**Bias direction — stated explicitly:**

- The **"could declare" count is an upper bound.** A token counted as present may sit far from
  the equivalence-ratio column (a different table, a different quantity). Grounding needs the
  token *at* the column, so the true declarable count is **≤** what this proxy reports. The one
  document this proxy flags as "could declare" is shown below to be a **false positive** for
  exactly this reason.
- There is a weaker **opposite** bias: pypdf linearises a table's cells into running text and
  can garble or re-tokenise a lone header dash, so a `-` unit token that exists in a header cell
  might be mis-counted. This would *under*-count groundability. It does not rescue any document
  here, because the canonical bracketed dimensionless notation (`(-)`, `[-]`) appears **nowhere**
  in any document's extracted text (ASCII or otherwise).

Every number below is one that was actually counted. Nothing is estimated or extrapolated.

## Corpus

Eight documents, the operator's `live-syngas` literature store
(`…/evidence/literature/<sha256>/raw.bin`), read at runtime. Document bytes and extracted text
are non-redistributable and are **not** reproduced here; only short evidence fragments appear.
The target paper is DOI `10.1115/1.4007737` (sha `c2be4138…`).

## Re-derivation of the brief's claims (all confirmed)

Calling the project's own `normalize_unit` against the shipped table `TABLE_V1`:

```
known_units(EQUIVALENCE_RATIO): ['1']
aliases for EQUIVALENCE_RATIO:  [('-', '1'), ('dimensionless', '1')]
normalize_unit('1')            -> '1'
normalize_unit('-')            -> '1'
normalize_unit('dimensionless')-> '1'
normalize_unit('/')            -> UnknownUnitError: unknown unit '/' for quantity
                                  'equivalence_ratio'; known units: ['1']
```

The three accepted tokens (`1`, `-`, `dimensionless`) and the refusal of `/` are exactly as the
brief states. **None of the brief's claims was found wrong.**

## How the counts were discriminated

- **Equivalence ratio in words** — regex `equivalence[\s\-\u00AD]*ratio`, tolerant of a hyphen/line
  break split; each hit tagged with the extractor's section label.
- **`dimensionless`** — literal word, case-insensitive. Clean signal, no discrimination needed.
- **`-`** — a naive character count is meaningless (hyphenation, ranges, minus signs). Three
  tiers are reported: **naive** (every `-`), **standalone** (a `-` bounded by non-word/non-digit
  on both sides — still includes minus signs, range dashes, cell fillers), and **bracket-ASCII**
  (`(-)`/`[-]` with an ASCII hyphen inside — the canonical way a dimensionless unit is printed in
  a header or axis label). Only the last is genuinely unit-like, and only an ASCII `-` (U+002D)
  is an accepted alias — an en-dash or minus inside the brackets would still be refused, so the
  bracket dash's codepoint is checked.
- **`1`** — a bare `1` as a unit is indistinguishable from equation numbers, reference markers,
  and data values; both a naive count and a standalone-token count are reported, and neither is
  treated as evidence of a unit.
- **symbol-font φ** — the count of proper Greek φ characters (U+03C6 / U+03D5) surviving in the
  extracted text is a proxy: a paper whose φ survives as real Greek did **not** suffer the
  target's failure mode; a paper with zero φ characters that still discusses equivalence ratio
  has had its φ replaced by an impostor glyph, as the target did.

## Per-document results

| sha (short) | pp | eq-ratio in words (sections) | `-` naive / standalone / bracket-ASCII | `dimensionless` | proper φ chars | φ symbol as extracted | verdict (text-proxy) |
|---|---|---|---|---|---|---|---|
| `251a7d03` | 7 | 9 (body) | 219 / 0 / 0 | 0 | 24 | `φ` (survives) | would fall back |
| `26c137b7` | 8 | 19 (abstract+body) | 163 / 1 / 0 | 0 | 34 | `φ` (survives) | would fall back |
| `4a0b5bbd` | 7 | 12 (abstract+body) | 148 / 1 / 0 | 0 | 0 | `ER` (spelled out) | would fall back |
| `5483f9ea` | 7 | 3 (body) | 169 / 0 / 0 | **3** | 0 | `U` (impostor) | *could declare*¹ |
| `7cc54415` | 12 | 13 (abstract+body) | 183 / 0 / 0 | 0 | 0 | `F` (impostor, `¼` for `=`) | would fall back |
| `9c59f1c6` | 14 | 10 (abstract+body) | 138 / 0 / 0 | 0 | 11 | `φ` / `f` (mixed) | would fall back |
| `a08397af` | 9 | 29 (abstract+body) | 95 / 0 / 0 | 0 | 0 | `f` (impostor) | would fall back |
| `c2be4138` **(target)** | 9 | 18 (body+refs) | 226 / 11 / 0 | 0 | 0 | none (`/` impostor) | would fall back |

Every document has `1` occurring dozens to hundreds of times (naive 272–645; standalone-token
14–51) — all equation numbers, reference markers, and data values, none a unit. `bracket-ASCII`
and any bracketed dash (ASCII **or** en-dash/minus) is **0 in every document**: the canonical
dimensionless-unit notation `(-)`/`[-]` appears nowhere in the linearised text of any paper.

¹ **`5483f9ea` "could declare" is a demonstrated false positive.** Its three `dimensionless`
occurrences are all the same phrase — "**dimensionless** relative variation of laminar flame
speeds" (a normalised flame-speed quantity), not a unit label for equivalence ratio. And this
paper renders equivalence ratio as the impostor `U` (e.g. "equivalence ratio of U = 0.8"), i.e.
its φ did not survive either. So even the one apparent positive, inspected, would not ground the
equivalence-ratio column. This is precisely the over-count the text-proxy warning predicts.

## Target reproduces the known finding

The target (`c2be4138…`) reports equivalence ratio in words 18 times ("…over a wide range of
equivalence ratios…") yet prints **none** of the three accepted tokens in any unit-like form
(bracket-ASCII dash = 0, `dimensionless` = 0), and carries **zero** proper φ characters — its φ
is gone, consistent with the established `/`-impostor finding. **The measurement agrees with the
existing conclusion; it does not overturn it.**

(Aside, reported not fixed: the OS `file` tool calls the target's `raw.bin` a 71-page PDF, while
both pypdf and the project extractor see 9 pages. pypdf's 9 is authoritative for what the project
processes; the extraction is not truncated. Noted only so a later reader is not alarmed by the
discrepancy.)

## Symbol-font φ is not unique to the target — it is the corpus norm

The φ impostor that broke the target is **widespread**, not a one-paper quirk. Of eight papers,
only three keep a proper Greek φ (`251a`, `26c1`, `9c59`). The rest substitute a Latin impostor
or spell it out: `4a0b` → `ER`, `5483` → `U`, `7cc5` → `F` (with `¼` for `=`), `a083` → `f`,
target → `/`. This matches the corpus's known symbol-font Latin-impostor problem. It means the
column-header glyph that *should* carry the equivalence-ratio label is itself frequently damaged
— so even where a paper prints a dimensionless unit near it, the label the unit attaches to may
be an impostor.

## Aggregate answer

**Of the 8 corpus documents, all 8 report equivalence ratio in words. At the text level, 7 have
no accepted dimensionless unit token in any unit-like form, and the 1 that superficially does is
a demonstrated false positive (its `dimensionless` describes a different quantity). So 0–1 of 8
could plausibly declare an equivalence-ratio unit with a grounded token, and the honest figure
after inspection is effectively 0.** The target's fallback is not a quirk of one paper: it is the
systematic outcome for this corpus. Equivalence ratio is essentially never printed with one of
the three accepted unit tokens attached to it.

This measurement **informs** the decision of whether to add a unit alias (e.g. `/`→`1`); it does
not make that decision, and no alias was added. Whatever is decided, the evidence here is that
the affected coordinate is the corpus norm, not an outlier.
