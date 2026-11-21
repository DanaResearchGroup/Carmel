# Iso-pentane RCM temperature cross-check

The pinned thermo reader admits 64 points in eleven files. Of these, 49 have a
unique replicate assignment under the independent source-matching method below
(49/64 = 76.5625% coverage). Fourteen are ambiguous and one has no matching
replicate block. All 49 uniquely matched points satisfy |ΔT_C| ≤ 15 K: 100% of
matched points, exceeding the required 75%. No ambiguous assignment contributes
to that fraction. The [per-point table](iso-pentane-rcm-crosscheck.csv) includes all
64 admitted points, including exclusions. This is a temperature cross-check, not
a kinetic-model score.

## Source matching and limits

Sources are Bugler et al., *An ignition delay time and chemical kinetic modeling
study of the pentane isomers*, DOI [10.1016/j.combustflame.2015.09.014](https://doi.org/10.1016/j.combustflame.2015.09.014),
the supplement's iso-pentane RCM table on pages 29–32, and its RCM Chemkin input
ZIP. The 160 tabulated rows were checked against the rendered PDF pages.

| Source | SHA-256 |
|---|---|
| `c5_15_supplementary_material.pdf` | `8eced09c395dd7f2c17d2602c8ad4680d8c1e6c0788bc244df3dc5e42869bbdf` |
| `c5_rcm_chemkin_input_files.zip` | `f8cd48222d0d37d913fb5061bf6fee197ab5c85a5de2925eab3c0717ce4edae5` |
| `pentane_indirect_v2.5.zip` | `b34d6e78a0600bffba9bf7f04f59d9479d6d1314df9964e4071902c929a65e56` |

1. Each XML point joins uniquely to an author input by identical initial
   temperature and composition, and initial pressure rounded to the XML's five
   decimal places in atm. This identifies all 75 original RCM points; author
   directory names supply nominal compressed pressure and phi. XML initial
   pressure is never equated to the supplement's compressed pressure.
2. Within that pressure/phi group, enumerate contiguous blocks of one to three
   supplement rows spanning at most 10 K of **reported supplement temperature**.
   Compare their arithmetic mean delay, rounded half-even to the XML delay's
   printed precision, with the XML delay. This averaging correspondence is
   inferred from matching data, not an explicit replicate-ID field in either
   source. Singletons are retained; no nearest-temperature selection is used.
3. Enumerate every complete assignment for points with candidates, forbidding
   reuse of a supplement row by two XML points. Retain a point only if its
   replicate block is identical in every valid assignment. Every pressure/phi
   group has at least one valid assignment. Reversing point order and candidate
   enumeration preserves all solution counts and assignments. A synthetic
   row-reuse collision has no solution.
4. Freeze this mapping before calculating derived T_C. The candidate document
   hash is `a6615b4e77d5bf5ffa8f272918cd7c1510e384d2d95475d4a1cc06c34512267e`;
   the frozen joint-assignment document hash is
   `ae06cb3a1070a397c6654185509132f223a81a3d8a4a57624e5df197bb1995a2`.
   Compare each retained XML point once with its replicate block's unweighted
   mean reported T_C. No assignment, rounding tolerance, or grouping threshold
   is chosen or revised from the resulting deviations.

This resolves correspondence for the 49 reported comparisons; it does not
identify the excluded points. In particular, `x40011006.xml:3` reports 37,000 µs
and has no compatible block. Possible causes of that difference are not repaired
or guessed. The ambiguous points are:

- `x40011002.xml:4`
- `x40011004.xml:1`
- `x40011004.xml:4`
- `x40011004.xml:6`
- `x40011004.xml:8`
- `x40011007.xml:2`
- `x40011009.xml:3`
- `x40011009.xml:4`
- `x40011010.xml:4`
- `x40011011.xml:1`
- `x40011011.xml:2`
- `x40011011.xml:4`
- `x40011012.xml:3`
- `x40011012.xml:4`

`x40011008.xml` is excluded in full: its first initial temperature is 292 K,
below the existing 295 K mixture floor. The thermo range is not extended to
compute values for its refused record. Second-maximum records are also excluded.

## Full deviation distribution

ΔT_C means derived minus source-block mean, in kelvin. Quartiles use linear
interpolation (NumPy's `quantile`, method `linear`), including singleton files.

| Population | n | min | Q1 | median | Q3 | max | within 15 K |
|---|---:|---:|---:|---:|---:|---:|---:|
| All matched, signed | 49 | -5.608863 | -2.846424 | -1.554551 | -0.458790 | 2.552611 | 49 |
| All matched, absolute | 49 | 0.066990 | 1.103728 | 1.663242 | 2.846424 | 5.608863 | 49 |
| `x40011001.xml`, signed | 3 | -3.267313 | -2.884605 | -2.501897 | -1.472380 | -0.442863 | 3 |
| `x40011001.xml`, absolute | 3 | 0.442863 | 1.472380 | 2.501897 | 2.884605 | 3.267313 | 3 |
| `x40011002.xml`, signed | 6 | -2.507342 | -1.862329 | -0.443222 | 0.852395 | 2.552611 | 6 |
| `x40011002.xml`, absolute | 6 | 0.427653 | 0.663862 | 1.804626 | 2.463051 | 2.552611 | 6 |
| `x40011003.xml`, signed | 9 | -5.088259 | -2.846424 | -1.771289 | -0.256900 | 1.825312 | 9 |
| `x40011003.xml`, absolute | 9 | 0.066990 | 0.887617 | 1.825312 | 2.846424 | 5.088259 | 9 |
| `x40011004.xml`, signed | 5 | -3.205025 | -1.314843 | -1.239061 | 0.210493 | 0.325209 | 5 |
| `x40011004.xml`, absolute | 5 | 0.210493 | 0.325209 | 1.239061 | 1.314843 | 3.205025 | 5 |
| `x40011005.xml`, signed | 1 | -4.881585 | -4.881585 | -4.881585 | -4.881585 | -4.881585 | 1 |
| `x40011005.xml`, absolute | 1 | 4.881585 | 4.881585 | 4.881585 | 4.881585 | 4.881585 | 1 |
| `x40011006.xml`, signed | 9 | -5.608863 | -2.743589 | -2.031970 | -1.182084 | 0.824356 | 9 |
| `x40011006.xml`, absolute | 9 | 0.576738 | 1.182084 | 2.031970 | 2.743589 | 5.608863 | 9 |
| `x40011007.xml`, signed | 1 | -3.304963 | -3.304963 | -3.304963 | -3.304963 | -3.304963 | 1 |
| `x40011007.xml`, absolute | 1 | 3.304963 | 3.304963 | 3.304963 | 3.304963 | 3.304963 | 1 |
| `x40011009.xml`, signed | 3 | -4.040952 | -2.702235 | -1.363518 | -1.233623 | -1.103728 | 3 |
| `x40011009.xml`, absolute | 3 | 1.103728 | 1.233623 | 1.363518 | 2.702235 | 4.040952 | 3 |
| `x40011010.xml`, signed | 6 | -3.178541 | -2.230247 | -1.413083 | -1.289785 | 0.153251 | 6 |
| `x40011010.xml`, absolute | 6 | 0.153251 | 1.289785 | 1.413083 | 2.230247 | 3.178541 | 6 |
| `x40011011.xml`, signed | 1 | -1.663242 | -1.663242 | -1.663242 | -1.663242 | -1.663242 | 1 |
| `x40011011.xml`, absolute | 1 | 1.663242 | 1.663242 | 1.663242 | 1.663242 | 1.663242 | 1 |
| `x40011012.xml`, signed | 5 | -3.384640 | -3.064691 | -1.602886 | -1.467982 | -0.290108 | 5 |
| `x40011012.xml`, absolute | 5 | 0.290108 | 1.467982 | 1.602886 | 3.064691 | 3.384640 | 5 |

The 64 admitted points span 662.187982–953.779077 K by derived T_C; 50 lie in
600–900 K inclusive. The [A-2 acceptance revision](iso-pentane-low-t.md#acceptance-revision-a-2)
adopts these counts for case 3 only. Source coordinates, scoring fields and
holdout remain unchanged.
