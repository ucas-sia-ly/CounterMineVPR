# Step 2C: local-structure fidelity

This audit measures SOURCE-to-RELIGHT structural fidelity in the existing
canonical 512x512 physical image coordinate system. Relit images are diagnostic
probes only. It does not measure cross-place query-negative confusion, define a
CounterMine score, or train a VPR model.

Matched coordinates are compared directly: `d_i = ||p_source_i - p_relit_i||_2`
in canonical pixels. No homography, fundamental matrix, or other registration is
fitted. A systematic warp is itself a fidelity failure, so registration would
remove the effect this audit needs to measure. The center-crop and 512x512
preprocessing policy are unchanged.

## Inputs and explicit execution

Run from the repository root in the probe environment. The required local
matching dependencies are listed in `environment_probe.yml`. Step 2C reads only
the sources present in `iclight_smoke.csv`, joins their identities to
`audit_manifest.csv`, and compares both `official_rmbg` and `full_scene` for each
source. The current Step 2B smoke audit contains ten sources, so a complete
fidelity CSV contains 30 rows including identity controls.

The existing cached qualitative audit uses JPEG source and relit references.
Before quantitative measurement, explicitly rebuild the lossless canonical
sources and rerun the ten-source smoke generation:

```bash
python tools/07_build_generator_audit_set.py
python tools/08_iclight_audit_smoke.py
```

The second command starts IC-Light generation; it is a separate manual step.
It retains the audited inference algorithm and settings, including the existing
generation seed. Do not rename JPEGs to PNG or mix old JPEG probes with newly
generated PNGs. Sources must be actual RGB PNGs at
`cache/generator_audit/source_512/<row_index>.png`, and relit images must be RGB
PNGs in the corresponding `relit/official_rmbg/` and `relit/full_scene/`
directories. Row-index filenames retain their eight-digit padding. CSV paths
are interpreted relative to the invocation working directory, as in Step 2B.

After these PNG prerequisites, explicitly run fidelity measurement:

```bash
python tools/09_measure_iclight_fidelity.py \
  --audit-manifest cache/generator_audit/audit_manifest.csv \
  --smoke-csv cache/generator_audit/iclight_smoke.csv \
  --output-dir cache/generator_audit/fidelity \
  --device auto \
  --max-keypoints 2048 \
  --seed 42 \
  --plot-dir outputs/step2
```

These flags show the defaults. `--plot-dir` changes only the diagnostic figure
directory. `--device auto` selects CUDA when available and
otherwise CPU. Fidelity seed 42 controls this audit and its deterministic
visualization selection; it does not change IC-Light's generation seed. Model
constructors use the vendored pretrained ALIKED and LightGlue weights, using
PyTorch's external cache and downloading missing weights only during an explicit
measurement run. Imports, CLI help, and the CPU unit tests do not instantiate
these models or download weights.

## Fixed matching configuration

The implementation uses the read-only `third_party/LightGlue` checkout:

```python
ALIKED(max_num_keypoints=2048, detection_threshold=0.2)
LightGlue(
    features="aliked",
    depth_confidence=-1,
    width_confidence=-1,
    filter_threshold=0.1,
    mp=False,
)
```

Both models use evaluation mode; the matcher is not compiled. ALIKED normally
resizes its input to 1024. Every audit extraction explicitly calls
`extractor.extract(image, resize=None)` to disable that preprocessing. Images
are loaded as RGB float tensors in `[0,1]` and must have shape `[3,512,512]`.

Processing streams one source at a time: extract source features, independently
extract the same source again for `identity_control`, match the official RMBG
relight, match the full-scene relight, then discard local features before the
next source. No full-audit feature cache is retained in RAM. The independently
extracted identity control exercises the same extraction and matching path;
identity correspondences are not fabricated. Approximately zero displacement
and high repeatability are qualitative sanity expectations, with no hard-coded
numeric pass threshold.

## Metrics and interpretation

For each pair, let `K = min(num_keypoints_source, num_keypoints_relit)` and let
`M = num_matches` after LightGlue's configured match filtering. The CSV records
the keypoint and match counts and `match_ratio_min = M / K`.

For each inclusive tolerance `epsilon` in 2, 4, 8, and 16 pixels:

- `good_matches_eps` counts matched displacements `d_i <= epsilon`.
- `repeatability_min_eps = good_matches_eps / K` measures retained local
  correspondences relative to the smaller detected keypoint set.
- `precision_matches_eps = good_matches_eps / M` measures the fraction of
  accepted matches that preserve their physical coordinates.

The CSV suffixes use `2px`, `4px`, `8px`, and `16px`, for example
`repeatability_min_8px`. A zero denominator produces a ratio of zero. For empty
matches, displacement and match-score statistics are null (blank cells in CSV)
rather than fabricated zero values. Nonempty displacement statistics are mean, median, q75, q90, q95,
and max; LightGlue scores have mean and median. Quantiles use inclusive linear
interpolation over the observed values.

The source image is divided into an 8x8 grid of 64-pixel cells. For the good
matches at 4 and 8 pixels, `occupied_grid_cells_eps` counts distinct source-side
cells and `grid_coverage_eps = occupied_grid_cells_eps / 64`. Coverage therefore
reveals whether preserved matches are spread across the image or concentrated
in a small region. Coverage alone does not establish structural fidelity; read
it together with repeatability, precision, match counts, and displacement.

## Artifacts

The default measurement directory contains:

- `cache/generator_audit/fidelity/fidelity_10.csv`, with one row per source and
  mode (`identity_control`, `official_rmbg`, `full_scene`).
- `cache/generator_audit/fidelity/fidelity_summary.json`, with run configuration,
  seed, provenance, and per-mode median, q05, q25, q75, and q95 for `num_matches`,
  `match_ratio_min`, each repeatability tolerance, `displacement_median`,
  `displacement_q95`, and both grid-coverage tolerances.

Mode summaries give each source equal weight. Undefined pair statistics are
excluded from the corresponding quantiles, with `valid_count` and
`missing_count` recorded for every metric. Quantiles are null when all pair
statistics for a metric are undefined.

Local diagnostic plots remain gitignored:

- `outputs/step2/fidelity_repeatability.png` compares the distributions of
  repeatability at 2, 4, 8, and 16 pixels for the two relighting modes.
- `outputs/step2/fidelity_displacement_cdf.png` shows empirical distributions of
  all matched source-to-relight displacements, including displacements above
  the good-match tolerances. Each matched correspondence has equal weight in
  this CDF, so sources with more accepted matches contribute more samples.
- `outputs/step2/fidelity_worst5.jpg` shows the five lowest
  `repeatability_min_8px` examples within each relighting mode as
  SOURCE | RELIT | MATCH OVERLAY. The overlay shows only matches at most 16 pixels
  apart, distinguishing `<=4`, `4-8`, and `8-16` pixels, with a deterministic cap
  on displayed matches. The cap affects visualization only, not metric counts.

The worst-example selection is for inspection. The summary does not choose a
winning relighting mode or create an automatic fidelity rejection threshold.
No quantitative fidelity results or GPU validation are claimed by this
implementation; they require the explicit measurement command and inspection
of its saved artifacts.

## CPU checks

Synthetic tests cover zero and known displacement, inclusive tolerance
boundaries, denominator behavior, empty matches, source-grid coverage, and
deterministic quantiles. They do not download model weights or require a GPU.

```bash
python -m compileall countermine tools
python -m unittest discover -s tests -v
```
