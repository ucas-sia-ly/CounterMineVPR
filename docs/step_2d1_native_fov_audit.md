# Step 2D1: native full-FOV geometry audit

This diagnostic tests whether the 640x480 to 512x384 downsampling contributed
to the fidelity loss in Step 2D0.5. The preferred candidate `native_full_fov`
keeps the original frozen GSV-Cities image geometry: 640x480 RGB pixels, no
EXIF transposition, crop, resize, padding, or stretch. Both scales and retained
FOV fractions are exactly 1.0. Unexpected source dimensions fail explicitly.
The two historical policies and their snapshots remain unchanged.

IC-Light accepts configured dimensions of at least 256 pixels divisible by 8.
With the required `highres_scale=1.0`, its refinement uses the configured
dimensions directly. Runtime checks require both conditioning images, decoded
images, refinement images, and final outputs to remain 640x480; VAE latents are
80x60. Future highres scales other than 1.0 fail. Same-size adapter operations
copy pixels without resampling. Full-scene conditioning bypasses RMBG.

Generation preserves the frozen Step 2C model, FC checkpoint, prompt, seed
12345, CFG 2, 25 steps, scheduler, denoise settings, and two-stage inference.
Matching uses seed 42, ALIKED 2048 maximum keypoints and detection threshold
0.2, unchanged LightGlue settings, `resize=None`, and no registration. Each
source-relight pair is processed and discarded before the next pair.
Synthetic images remain diagnostic probes and never enter the VPR training
loss. No training or CounterMine pair scoring is implemented.

## Commands

Run CPU validation with the workspace Python:

```bash
python -m compileall countermine tools
python -m unittest discover -s tests -v
```

From the repository root in the existing `countermine-probe` environment:

```bash
python tools/15_build_native_fov_audit.py
python tools/16_generate_native_fov_audit.py --device cuda
python tools/17_measure_native_fov_fidelity.py --device cuda
python tools/18_export_native_fov_snapshot.py
```

Tool 15 recovers the exact ten `audit_index`/`row_index`/`image_id` identities
from `docs/audits/step2d0_metrics.json` and joins original JPEG paths from the
existing audit manifest. Its lossless source PNGs contain exactly the decoded
original RGB pixels. Tool 16 generates only ten new native full-scene relights.
Tool 17 matches only these native pairs; historical scalar results come from
the frozen snapshot. Existing square/full-512 relights are read only for the
contact sheet. Historical generation and matching are not rerun.

## Measurements and paired comparisons

Native output pixels equal original-image pixels because the scale is 1.0.
Repeatability at 2/4/8/16 pixels uses unchanged LightGlue matches and divides
qualifying counts by the smaller source/relit keypoint count. Match ratio and
displacement median/q95 preserve their established definitions. Empty matches
have zero ratios and genuinely undefined displacement statistics (`null` in
JSON, blank in CSV).

Coverage uses the existing normalized source-side 8x8 grid: occupied qualifying
cells divided by 64. At 640x480 each cell spans 80x60 pixels. Native coverage
uses native 4/8-pixel masks, while historical coverage retains its original
canonical-output-pixel masks. Coverage is a spatial-distribution diagnostic;
this experiment does not change its historical definition.

For each of the ten identical sources, native minus historical square compares
R4, R8, D95, and coverage8. Native minus historical full-FOV-512 compares R4,
R8, and D95. Historical repeatability/displacement use the frozen **original
pixel** metrics, so all R4/R8/D95 differences share original-image pixel units.
Comparison summaries report median, q25, q75, min, max, positive/zero/negative
counts, and valid/missing counts. Undefined displacement deltas remain null.
No significance test, automatic geometry winner, or new acceptance threshold
is introduced.

## Artifacts

The ignored `cache/native_fov_audit/` contains:

- `native_manifest.csv` and `native_build_summary.json` with source identity,
  JPEG/PNG dimensions, native scales, retained fractions, and pixel provenance.
- `native_full_fov/source/` and `native_full_fov/relit/` with 640x480 RGB PNGs.
- `native_generation.csv` and `native_generation_summary.json` with frozen
  configuration, elapsed seconds, and CUDA allocated/reserved memory peaks.
- `native_fidelity.csv`, `native_paired.csv`, and `native_summary.json` with
  scalar native measurements and both paired historical comparisons.

Ignored diagnostic figures are
`outputs/step2d1/native_fov_compare_10.jpg` (original, both historical relights,
native source, native relight) and `outputs/step2d1/native_fov_fidelity.png`
(all three policies' original-pixel R4/R8/D95).

The standard-library-only exporter writes
`docs/audits/step2d1_native_fov_metrics.json`. It copies logged summary values
exactly after checking identities, scalar consistency, runtime, and provenance.
It contains no absolute filesystem paths, image-hash collections, or matched
coordinate arrays. `json.dumps(..., allow_nan=False)` enforces finite JSON;
publication is atomic and protects curated historical audit files.

## Logged ten-source results

The native generation completed in 24.638851 seconds, with peak CUDA allocation
2,920,736,256 bytes and reservation 3,571,449,856 bytes. All ten generated
sources and relights were exactly 640x480. The original historical snapshots,
source images, relights, and diagnostic figures were preserved.

| Policy | Median R4 | Median R8 | Median per-image D95, original pixels |
| --- | ---: | ---: | ---: |
| Historical `square_crop_512` | 0.555246 | 0.565774 | 3.355323 |
| Historical `full_fov_512` | 0.478124 | 0.535851 | 5.416484 |
| `native_full_fov` | 0.557862 | 0.574876 | 3.483421 |

Paired differences below are native minus the named historical policy.
Positive R4/R8 means higher repeatability; negative D95 means lower displacement.
Every comparison has ten valid pairs and zero missing pairs.

| Baseline | Metric | Median | q25 | q75 | Min | Max | Positive / zero / negative |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Square | R4 | 0.018564 | -0.019512 | 0.046464 | -0.048821 | 0.062527 | 7 / 0 / 3 |
| Square | R8 | 0.024878 | -0.016975 | 0.050284 | -0.049165 | 0.065696 | 7 / 0 / 3 |
| Square | D95 | 0.264991 | -0.306458 | 0.355867 | -0.421218 | 0.936582 | 6 / 0 / 4 |
| Square | Coverage8 | 0.031250 | -0.011719 | 0.054688 | -0.046875 | 0.078125 | 6 / 1 / 3 |
| Full-FOV-512 | R4 | 0.096680 | 0.083829 | 0.111629 | 0.056361 | 0.124214 | 10 / 0 / 0 |
| Full-FOV-512 | R8 | 0.037828 | 0.029555 | 0.061338 | -0.012989 | 0.097048 | 9 / 0 / 1 |
| Full-FOV-512 | D95 | -1.608047 | -2.194547 | -1.189002 | -2.552353 | -0.923771 | 0 / 0 / 10 |

On this fixed diagnostic set, native geometry improves R4 on all ten sources,
R8 on nine, and D95 on all ten relative to downsampled full-FOV-512. The median
paired R4/R8 gaps to square become positive, while a small positive median D95
gap remains. This evidence is consistent with the downsampling hypothesis.
It does not establish a VPR performance improvement or define a geometry winner
or final acceptance gate. Exact values remain in the version-controlled snapshot.
