# Step 2D0.5: scale-fair canonical FOV geometry audit

This paired diagnostic compares the existing `square_crop_512` policy with
`full_fov_512` for exactly the ten source identities in the completed
`docs/audits/step2c_metrics.json` snapshot. It tests whether discarding horizontal
field of view during canonical preprocessing affects the fidelity diagnostic.
Step 2C remains frozen historical evidence; its source images, relights,
metrics, and curated figures are preserved.

Each source receives two independent canonical variants:

- `square_crop_512`: the existing center crop and LANCZOS resize to 512x512.
- `full_fov_512`: RGB conversion and LANCZOS resize of the complete image so its
  longest side is 512. The current 4:3 sources become 512x384, with retained area
  and retained long-axis fractions both equal to 1.0. There is no crop, padding,
  stretching, or EXIF transposition. Unsupported dimensions fail validation.

The generation stage uses IC-Light `full_scene` conditioning only. It preserves
the Step 2C prompt, checkpoint, CFG, sampler, steps, denoise settings, highres
scale, two-stage inference, and seed 12345. No RMBG run is performed. Synthetic
images remain diagnostic probes and are never inputs to a VPR optimization loss.

## Corrected measurement commands

Reuse the existing geometry manifest, generation CSV, generation summary, and
20 lossless relights. From the repository root in `countermine-probe`, run:

```bash
python tools/13_measure_geometry_fidelity.py --device cuda
python tools/14_export_step2d0_snapshot.py
```

The corrected measurement reruns ALIKED and LightGlue because the previous
measurement did not persist matched coordinates. It does not call IC-Light.
The exporter reads only JSON/CSV files and does not load any model. No training,
identity control, cross-place matching, or significance test is performed.

Tools 11 and 12 remain the original source-build and generation stages. They
are not needed when the completed inputs are available. Restoring missing
generation outputs uses the unchanged tool 12, fixed seed 12345, and the saved
Step 2C generation settings; it does not change the correction's measurement
algorithm or resample source identities.

CSV paths are relative to the invocation working directory. Run all commands
from the same repository root so the saved references resolve consistently.
The tools save deterministic seeds and configurations with their summaries.
Generation reports elapsed time and peak CUDA memory. Processing handles one
source pair at a time, without caching all local features in RAM.

## Metrics and paired analysis

ALIKED and LightGlue retain the Step 2C model settings: 2048 maximum ALIKED
keypoints, detection threshold 0.2, and LightGlue configured for ALIKED with
depth confidence -1, width confidence -1, filter threshold 0.1, and `mp=False`.
The fidelity seed is 42. ALIKED is always called with `resize=None`. Source and
relit image dimensions must match exactly; every fidelity record includes
`canonical_width` and `canonical_height`.

The initial output-pixel comparison was not scale-fair. For a 640x480 original,
the square policy scales by `512/480`, while full FOV scales by `512/640`.
Eight canonical output pixels therefore represent 7.5 original pixels in the
square policy and 10 original pixels in full FOV.

The corrected metrics require finite positive isotropic scales, with
`abs(scale_x - scale_y) <= 1e-12`. Anisotropic scaling fails before model loading.
Both source and relit matched coordinates are divided by the same recorded
scale. Crop translation cancels in their displacement and is not added.
Distances and inclusive 2/4/8/16 tolerances are measured in **original-image
pixels**. A `1e-12` pixel arithmetic guard only resolves floating-point roundoff
at inclusive boundaries; it does not define a new acceptance gate.

Each record retains the output-pixel metrics and adds
`repeatability_original_{2,4,8,16}px`, `precision_original_{2,4,8,16}px`, and
original-pixel displacement mean, median, q75, q90, q95, and max. Repeatability
still divides by the smaller keypoint count; precision divides by the number
of unchanged LightGlue matches. Zero-denominator ratios are zero. Empty-match
displacement statistics remain undefined (`null` in JSON and blank in CSV),
with no zero imputation. No registration is fitted.

Spatial coverage uses an 8x8 normalized source-side grid. A keypoint `(x, y)`
occupies cell `(floor(x * 8 / width), floor(y * 8 / height))`. Cell sizes are
64x64 for square probes and 64x48 for current full-FOV probes. Coverage remains
the number of occupied cells among qualifying matches divided by 64. The
existing canonical 4/8-pixel match masks are unchanged. Coverage is a spatial
distribution diagnostic in normalized canonical image coordinates; it is not
recomputed in original coordinates or used as a replacement for fair R4/R8.

For each `row_index`, fair paired deltas subtract square from full-FOV for
original-pixel R4, original-pixel R8, original-pixel displacement q95, and the
existing normalized grid coverage at 8 canonical pixels. Their names are
`delta_R4_original`, `delta_R8_original`, `delta_displacement_original_q95`, and
`delta_grid_coverage_8`. Per-policy summaries preserve median, q05, q25, q75,
and q95 using linear quantiles. Paired summaries preserve median, min, max,
valid_count, and missing_count.

Historical metrics remain under `output_pixel_metrics`, and their deltas
remain under `legacy_output_pixel_deltas`. Both explicitly state:
"Not directly comparable across geometry policies because canonical scale
differs." Output-pixel R4/R8 and displacement q95 must not select a geometry.
Geometry assessment uses the paired original-pixel metrics together with FOV
retention, without automatically declaring a winner.

## Historical gate

The previous gate is preserved as `legacy_output_pixel_gate`, with
`cross_policy_comparable: false`. Its unchanged conditions are:

- `repeatability_min_8px >= 0.40`
- `grid_coverage_8px >= 0.60`
- `displacement_q95 <= 5.0`

The record flag is `legacy_output_pixel_gate_pass`. Undefined displacement
statistics do not pass. Old acceptance counts are historical diagnostics and
must not be compared between policies. Every pair remains in the CSV, paired
analysis, summaries, and figures. Step 2D0.5 introduces no new acceptance
threshold. A final full-scene fidelity gate will be defined after geometry is
chosen.

## Artifacts

The build stage writes PNG source variants under
`cache/geometry_audit/square_crop_512/source/` and
`cache/geometry_audit/full_fov_512/source/`, plus
`cache/geometry_audit/geometry_manifest.csv` and
`cache/geometry_audit/geometry_build_summary.json`.

The generation stage writes lossless PNG relights and provenance in
`cache/geometry_audit/geometry_generation.csv` and
`cache/geometry_audit/geometry_generation_summary.json`.

The measurement stage writes:

- `cache/geometry_audit/geometry_fidelity.csv`: all 20 pair records, original
  and output-pixel metrics, scales, and explicitly legacy gate flags.
- `cache/geometry_audit/geometry_paired.csv`: ten rows of fair original-pixel
  full-FOV minus square deltas.
- `cache/geometry_audit/geometry_summary.json`: configuration, provenance,
  exact fair per-policy/paired summaries, preserved legacy metrics/deltas/gate,
  and coverage interpretation.
- `outputs/step2d0/geometry_compare_10.jpg`: ORIGINAL, SQUARE SOURCE,
  SQUARE RELIT, FULL-FOV SOURCE, and FULL-FOV RELIT for each source, with aspect
  ratios preserved.
- `outputs/step2d0/geometry_fidelity_compare.png`: paired R4, R8, and displacement
  q95 comparisons in original-image pixels only.
- `docs/audits/step2d0_metrics.json`: a compact version-controlled snapshot with
  frozen Step 2C SHA256, seeds, model settings, exact summaries, geometry,
  ten paired per-source records, and clearly labeled legacy metrics. The
  exporter rejects absolute paths and nonfinite numbers and stores no matched
  coordinates or large collections of image hashes.

The cache and transient figures are gitignored; the compact snapshot is tracked
under `docs/audits/`. Conclusions require the actual logged scale-fair results.

## Recorded workspace run

The previous completed generation cache could not be recovered from the available
local workspaces. Following the user's fallback authorization, the unchanged
tool 12 generated 20 new `full_scene` PNG relights with seed 12345 and the frozen
Step 2C generation settings. The existing 20 source PNGs and geometry manifest
were reused unchanged. The snapshot records this regenerated run, measured with
seed 42 and the unchanged ALIKED/LightGlue settings.

The ten-source medians are:

| Policy | Original-pixel R4 | Original-pixel R8 | Per-image D95, original-image pixels |
| --- | ---: | ---: | ---: |
| `square_crop_512` | 0.555246 | 0.565774 | 3.355323 |
| `full_fov_512` | 0.478124 | 0.535851 | 5.416484 |

Median paired full-FOV-minus-square deltas are -0.077122 for original-pixel R4,
-0.038203 for original-pixel R8, +2.119518 original-image pixels for D95, and
-0.031250 for normalized coverage8. All ten deltas are valid for each metric.
The snapshot retains exact values and quantiles. These are diagnostic fidelity
results; no geometry winner, new acceptance threshold, or VPR performance
improvement is declared.

## CPU validation

Tests cover both canonical-to-original scales, identical classification of
exactly eight original-pixel displacement, anisotropic-scale rejection,
original-pixel paired deltas, normalized coverage preservation, and model-free
snapshot validation:

```bash
python -m compileall countermine tools
python -m unittest discover -s tests -v
```
