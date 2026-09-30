# Step 2D0: canonical FOV geometry audit

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

## Explicit commands

Run these commands from the repository root in the probe environment, in order:

```bash
python tools/11_build_geometry_audit.py
python tools/12_generate_geometry_audit.py --device cuda
python tools/13_measure_geometry_fidelity.py --device cuda
```

The build command is CPU-only. It joins the identities from the frozen snapshot
to `cache/generator_audit/audit_manifest.csv` and reads originals from
`data/gsv-cities`. It does not select a new sample. The generation command
explicitly generates 20 relights, using lossless PNG inputs and outputs. The
measurement command explicitly runs ALIKED and LightGlue for the 20
source-to-relight pairs. Neither GPU stage runs as part of imports, help, or
unit tests. No training or cross-place matching is performed.

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

Matched keypoint displacements are measured directly in output pixels, with
inclusive 2, 4, 8, and 16 pixel tolerances. No registration is fitted. Match
ratios, repeatability, precision, and displacement statistics have the same
definitions as Step 2C. Empty-match displacement statistics remain undefined
(`null` in JSON and blank in CSV), with no zero imputation.

Spatial coverage uses an 8x8 normalized source-side grid. A keypoint `(x, y)`
occupies cell `(floor(x * 8 / width), floor(y * 8 / height))`. Cell sizes are
64x64 for square probes and 64x48 for current full-FOV probes. Coverage remains
the number of occupied cells among qualifying matches divided by 64.

For each `row_index`, paired deltas subtract square metrics from full-FOV
metrics for R4, R8, displacement q95, and grid coverage at 8 pixels. Per-policy
summaries report exact median, q05, q25, q75, and q95 using linear quantiles.
Paired summaries report median, min, and max; undefined values have explicit
valid and missing counts. No significance test is run on ten pairs, and the
tools do not declare a winning policy.

## Provisional gate

`pilot_gate_pass` is a diagnostic flag requiring all three conditions:

- `repeatability_min_8px >= 0.40`
- `grid_coverage_8px >= 0.60`
- `displacement_q95 <= 5.0`

Undefined displacement statistics do not pass. Acceptance is reported
separately for the two geometry policies. Every pair remains in the CSV,
paired analysis, summaries, and figures regardless of the flag. These values
are provisional diagnostics and do not define a final paper threshold.

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

- `cache/geometry_audit/geometry_fidelity.csv`: all 20 pair records and gate flags.
- `cache/geometry_audit/geometry_paired.csv`: ten rows of full-FOV minus square deltas.
- `cache/geometry_audit/geometry_summary.json`: configuration, provenance,
  exact per-policy and paired summaries, and gate acceptance.
- `outputs/step2d0/geometry_compare_10.jpg`: ORIGINAL, SQUARE SOURCE,
  SQUARE RELIT, FULL-FOV SOURCE, and FULL-FOV RELIT for each source, with aspect
  ratios preserved.
- `outputs/step2d0/geometry_fidelity_compare.png`: paired R4, R8, and displacement
  q95 comparisons.

These experiment outputs are gitignored. Numerical results require explicit
generation and measurement; this implementation does not claim an observed
performance or fidelity improvement.

## CPU validation

Tests cover canonical policies, geometry validation, rectangular coordinates,
normalized coverage, and paired deltas without loading model weights:

```bash
python -m compileall countermine tools
python -m unittest discover -s tests -v
```
