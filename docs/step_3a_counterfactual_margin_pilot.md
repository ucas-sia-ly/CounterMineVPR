# Step 3A: Counterfactual Retrieval-Margin Pilot

This frozen diagnostic asks whether nuisance appearance differences help a
pretrained SALAD encoder reject structurally confusing wrong places. Each
query has a same-place positive, a SALAD RGB hard negative, and a random
negative. The hard and random triplets share exactly the same query and
positive. No training, mining score, optimization objective, or automatic
Top-K CounterMine selection is introduced.

## Frozen population and inference

The existing 8,000-image GSV mini manifest and `safe250` candidate output
provide the population. The pilot uses 500 unique queries selected in seeded
SHA256 order. Positives and random negatives use independent per-query
deterministic draws from manifest order. The hard negative has the smallest
available eligible saved RGB rank, with negative row index breaking rank
ties. Both negative types must differ in place and satisfy the saved 250 m
geographic exclusion. Random negatives receive no similarity filter. Missing
eligibility is reported explicitly; downstream failures never replace a query
or image. Fewer than 500 eligible queries would retain all eligible queries.

Construction selected 500 queries and 1,777 unique images from 8,000 eligible
queries, with zero unavailable images, query exclusions, or random/hard image
overlaps. `population.json` saves every selection, the complete configuration,
input SHA256 links, availability and exclusion accounting, seed, and original
image/source PNG digests. Every inference stage rechecks this construction.

Original RGB descriptors are frozen before generation. The existing SALAD
wrapper performs the miner's 322x322 bilinear square resize, RGB conversion,
ToTensor, ImageNet normalization, CUDA float16 autocast, and float32 L2 output
normalization. Native probe geometry does not change this SALAD policy. Pilot
descriptors are streamed into float32 disk memory maps; cosine inputs are
calculated in float64 after normalization. The original and relit passes use
the same model, preprocessing, seed, and batch size.

IC-Light preserves Step 2D2 model IDs, FC checkpoint, negative prompt,
scheduler, dtypes, full-scene conditioning, and two-stage inference. Geometry
is native full-FOV 640x480, CFG 2, 25 steps, highres scale 1, denoise 0.5, and
the prompt is `soft diffuse overcast daylight, uniform outdoor illumination,
natural lighting`. RMBG is never loaded. No historical 512-based artifact is
modified or regenerated. Local model files are resolved offline, and their
current snapshot revisions and checkpoint hashes are recorded. The historical
Step 2D2 JSON did not record resolved weight hashes, so historical asset
identity can be compared only at the level it logged.

Each unique image receives one independent deterministic diffusion seed:

```python
int.from_bytes(
    hashlib.sha256(("CounterMineVPR-Step3A|" + image_id).encode("utf-8")).digest()[:8],
    "big", signed=False,
) % 2**63
```

The shared condition comes from configuration and prompt. Noise is
independent across image IDs. Sources and relights are lossless RGB PNGs;
filenames use the full image-ID SHA256. Completed generation records are
reused, and partially completed runs resume without duplicate probes.

## Measurements and interpretation

Source-to-relight ALIKED/LightGlue fidelity contributes only R8, calculated
with the historical float64 coordinate distances and inclusive 8 px boundary. Its frozen
flag is `source_relit_R8 >= 0.40`; coverage and D95 do not reject Step 3A
records. All failed images and triplets remain saved. Every summary has an
all-triplets version and a common subset requiring all four q/p/hard/random
images to pass R8, preserving the paired comparison. A separate three-image
hard flag is retained for inspection but does not determine the paired subset.

For each negative, the margin is q-positive cosine minus q-negative cosine.
Delta margin is relit minus RGB margin. Negative delta indicates lost positive
advantage. Negative-relative gain equals the negative cosine change minus the
positive cosine change and is verified numerically to equal minus delta
margin. Strictly positive margins are correct; ties are wrong. Flip fractions
use the appropriate eligible RGB state as denominator.

Paired delta difference is hard delta minus random delta. Its sign counts,
distribution, and fraction below zero are descriptive. The random-relative
gain null supplies mean and sample standard deviation (`ddof=1`). Hard gains
are centered and divided by `sigma + 1e-12`, with no mining threshold.
Empirical percentiles use the weak ECDF: 100 times the fraction of random
gains less than or equal to the hard gain. Main per-row null fields use all
triplets; explicitly named `*_R8_pass_null` fields recalibrate within the common
subset. Undefined singleton standard deviations/correlations are JSON null,
never NaN or infinity.

ALIKED uses 2,048 maximum keypoints and detection threshold 0.2. LightGlue
uses `features=aliked`, depth/width confidence -1, filter threshold 0.1,
`mp=False`, seed 42, and extraction `resize=None`. The same lazy models serve
fidelity and cross-place matching. The separate cross-place method computes
no same-coordinate displacement, R4/R8, homography, fundamental matrix, or
registration. It records match count, count divided by the smaller keypoint
count, and occupied normalized 8x8 source/target cells divided by 64. Local
features are released after each pair; there is no full-dataset local cache.

Positive cosine changes are reported alongside both negative changes. Raw
cosine increases alone are not evidence. RGB MAE divided by 255 and absolute
stored-RGB luma mean delta are per-image appearance diagnostics. Their means
over q/p/hard are correlated descriptively with hard delta margin, relative
gain, and delta cross-match ratio using Pearson and tie-aware Spearman.
No significance test or causal inference is made.

## Runtime commands

From the repository root, use the existing `countermine-vpr` environment for
SALAD and `countermine-probe` for IC-Light/local matching/figures. These
environments contain the previously validated inference dependencies.

```bash
python tools/23_build_step3a_population.py
conda run -n countermine-vpr python tools/24_freeze_step3a_salad.py --phase rgb
conda run -n countermine-probe python tools/25_generate_step3a_probes.py --device cuda
conda run -n countermine-probe python tools/26_measure_step3a_local.py --device cuda
conda run -n countermine-vpr python tools/24_freeze_step3a_salad.py --phase relit
conda run -n countermine-probe python tools/27_export_step3a_snapshot.py
```

The defaults use seed 42, the existing safe250 mining summary, the existing
index summary, Step 2D2 snapshot, and `data/gsv-cities`. Population construction
can name alternative input files explicitly, but it inherits their saved
geographic rule and rejects changed SALAD/IC-Light policies. Generation and
local stages save atomic progress journals. They fail explicitly on inconsistent
or unlogged cached data. A fresh experiment needs a new dedicated directory
under `cache/`; protected historical cache directories and child symlinks are
rejected.

CPU validation loads no model weights and requires no CUDA:

```bash
python -m compileall countermine tools
python -m unittest discover -s tests -v
```

## Artifacts

Ignored `cache/step3a/` stores the population, original pixel copies, image-ID
probe cache with every seed, generation and fidelity records, cross-place
support, original/relit descriptor banks and cosine measurements, progress
journals, full scalar analysis, and per-triplet CSV. Synthetic pixels and
descriptors are diagnostic inference inputs only. There is no training call,
optimizer step, backward pass, or metric-learning-loss entry point in this
pipeline; CPU guards check that boundary.

The exporter consumes completed observations, rechecks descriptor hashes and
their saved scalar cosines, validates unique identities/configurations, and
writes finite portable JSON using `json.dumps(..., allow_nan=False)`. The
snapshot has no absolute paths, model blobs, or matched-coordinate arrays:

`docs/audits/step3a_counterfactual_margin_metrics.json`

The four ignored figures under `outputs/step3a/` are copied after successful
analysis to `docs/audits/` with `step3a_` prefixes:

- `top_margin_collapse_50.jpg`: the 50 smallest hard delta margins in the common
  R8 subset, with q/p/hard RGB and relit scenes, cosines, margins, margin/local
  shifts, and q/p/hard R8. Selection uses margin alone; no joint score exists.
- `hard_vs_random_margin_shift.png`: paired random (x) and hard (y) margin
  changes, with identity and zero references.
- `margin_vs_local_confusion.png`: hard delta margin against local-match-ratio
  change, with zero references.
- `rgb_vs_relit_margin.png`: RGB against relit hard margin, with identity and
  zero references.

The montage supports manual inspection of repeated facades, windows,
rooflines, columns, and scene layout versus generated artifacts, sky,
vegetation, and generic texture. Numerical pass flags alone do not establish
the scientific explanation or select a training intervention.

## Logged pilot results

The complete run produced exactly 1,777 unique probes with 1,777 distinct
image-derived seeds. Source-to-relight R8 passed for 1,767 images and failed
for 10; all observations remain saved. The common all-four-image R8 subset
contains 490 of the 500 paired queries. All 2,000 RGB/relit hard/random
cross-place observations were measured. Generation image inference totaled
3,406.431354 seconds, excluding the separately measured 3.568001-second model
load. Peak CUDA allocation was 2,920,736,256 bytes.

Mean retrieval measurements are:

| Population | Negative | RGB margin | Relit margin | Delta margin | Relative gain |
| --- | --- | ---: | ---: | ---: | ---: |
| All 500 | Hard | 0.274646 | 0.225190 | -0.049456 | 0.049456 |
| All 500 | Random | 0.450696 | 0.371411 | -0.079286 | 0.079286 |
| Common R8-pass 490 | Hard | 0.277864 | 0.228796 | -0.049068 | 0.049068 |
| Common R8-pass 490 | Random | 0.453385 | 0.374539 | -0.078846 | 0.078846 |

Hard delta-margin medians are -0.046961 (all) and -0.046612 (R8 subset).
Random medians are -0.075528 and -0.075333, respectively. The paired
hard-minus-random difference has mean 0.029830 and median 0.029483 over all
queries; only 115/500 (23.0%) have a more negative hard shift. In the common
subset, the mean is 0.029778, median 0.029771, and fraction is 110/490
(22.449%). The all-triplet random null has mu 0.079286 and sample sigma
0.063674; the common-subset null has mu 0.078846 and sigma 0.062815.

| Population | Negative | Correct-to-wrong / eligible RGB correct | Wrong-to-correct / eligible RGB wrong or tied |
| --- | --- | --- | --- |
| All 500 | Hard | 12/477 (2.516%) | 2/23 (8.696%) |
| All 500 | Random | 3/499 (0.601%) | 0/1 (0%) |
| Common R8-pass 490 | Hard | 11/470 (2.340%) | 2/20 (10%) |
| Common R8-pass 490 | Random | 3/490 (0.612%) | 0/0 (undefined) |

The higher descriptive hard flip fraction must be considered alongside the
smaller starting hard margins and the positive stability control:

| Population | Original positive cosine mean | Relit positive cosine mean | Positive change mean | Hard-negative change mean | Random-negative change mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 500 | 0.471053 | 0.396154 | -0.074899 | -0.025443 | 0.004387 |
| Common R8-pass 490 | 0.473946 | 0.399524 | -0.074423 | -0.025355 | 0.004423 |

Positive similarity falls substantially on average. Consequently, margin
collapse alone does not isolate nuisance-assisted rejection of wrong places.
The local matching control also does not show an overall increase in hard
confusion:

| Population | Negative | RGB cross-match ratio mean | Relit ratio mean | Delta mean |
| --- | --- | ---: | ---: | ---: |
| All 500 | Hard | 0.038645 | 0.034680 | -0.003965 |
| All 500 | Random | 0.027584 | 0.026131 | -0.001453 |
| Common R8-pass 490 | Hard | 0.038495 | 0.034775 | -0.003721 |
| Common R8-pass 490 | Random | 0.027692 | 0.026292 | -0.001400 |

The joint sign diagnostic occurs in 188/500 all-triplet cases (37.6%) and
185/490 R8-pass cases (37.755%). These are cases for inspection, not an
automatic score or proof of a specific mechanism. Intervention-strength
correlations have small descriptive magnitudes here: across the six requested
associations, maximum absolute Pearson/Spearman values are 0.086467/0.049551
for all queries and 0.081695/0.046809 for the common subset. These correlations
do not establish causality or exclude generator artifacts.

Manual inspection of the strongest-collapse rows found a material limitation
of the R8-only control. The first row's original positive shows a largely
blank wall, while its relit version contains a glazed/windowed facade despite
positive-image R8 of 0.5030. Other leading rows show changes to signage or
road surface appearance. Thus, R8 acceptance does not certify semantic
preservation or a pure illumination intervention. The formal 0.40 threshold
is unchanged, and these records remain in the requested summaries and montage.

This completed pilot has **not yet validated the core hypothesis**. Hard
correct-to-wrong flips are descriptively more frequent, but most paired margin
changes favor a stronger random collapse, positive recognition is destabilized,
and mean hard local confusion decreases. The saved cases can support further
scientific review; they do not justify training modifications, a mining
threshold, or a final CounterMine objective. No significance test was run.

Validation completed with `python -m compileall countermine tools` and
`python -m unittest discover -s tests -v`: **401 CPU tests passed**. The exact
workspace executable commands for the completed stages were:

```bash
python tools/23_build_step3a_population.py
/home/admin123/miniconda3/envs/countermine-vpr/bin/python tools/24_freeze_step3a_salad.py --phase rgb
/home/admin123/miniconda3/envs/countermine-probe/bin/python tools/25_generate_step3a_probes.py --device cuda
/home/admin123/miniconda3/envs/countermine-probe/bin/python tools/26_measure_step3a_local.py --device cuda
/home/admin123/miniconda3/envs/countermine-vpr/bin/python tools/24_freeze_step3a_salad.py --phase relit
/home/admin123/miniconda3/envs/countermine-probe/bin/python tools/27_export_step3a_snapshot.py
```
