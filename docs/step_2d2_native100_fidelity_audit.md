# Step 2D2: 100-image native full-scene fidelity audit

This audit applies a **fixed shared illumination intervention** to exactly the
existing Step 2 generator-audit population: 50 deterministic random sources
and 50 distinct sources associated with high-similarity RGB candidate pairs.
It never resamples. Geometry is frozen as `native_full_fov`, 640x480, unit
scales, and 100% retained FOV. No further geometry experiment is performed.
Historical 512 policies and the Step 2C/2D0/2D1 evidence remain frozen.

The generator is a diagnostic counterfactual probe. Synthetic images never
enter the VPR optimization loss. This stage adds no CounterMine pair scoring,
training, SALAD changes, or RMBG invocation.

## Runtime commands

Run from the repository root in the existing `countermine-probe` environment:

```bash
python tools/19_build_native100_audit.py
python tools/20_generate_native100_audit.py --device cuda
python tools/21_measure_native100_fidelity.py --device cuda
python tools/22_export_native100_snapshot.py
```

CPU validation requires no model weights or CUDA:

```bash
python -m compileall countermine tools
python -m unittest discover -s tests -v
```

Tool 19 validates all 100 unique audit, row, and image identities against the
existing selection provenance and hashed historical mini manifest. Because
the original audit did not save original JPEG digests, it also checks that
each original reproduces the saved historical source PNG pixels. That is an
integrity check using the existing preprocessing, not a geometry comparison.
Only the native pixel copy is written. All historical manifest columns are
preserved; the new source is lossless 640x480 RGB PNG, with no EXIF transpose,
crop, resize, padding, or stretch. Unexpected originals fail explicitly.

Tool 20 preserves the completed Step 2D1 model, FC checkpoint, negative prompt,
scheduler, dtypes, and two-stage inference. Width/height are 640/480, seed
12345, CFG 2, 25 steps, high-resolution scale 1, and denoise 0.5. It generates
one `full_scene` probe per source. Both stages must retain exact geometry.
Model loading is timed separately from per-image inference; elapsed quantiles
and CUDA allocated/reserved peaks are logged.

Tool 21 streams one aligned source-relight pair at a time. ALIKED uses maximum
2048 keypoints and detection threshold 0.2. LightGlue uses features `aliked`,
depth/width confidence -1, filter threshold 0.1, and `mp=False`. Extraction
uses `resize=None`, seed 42, and no registration, homography, F-matrix, or
geometric fitting. Native pixel displacement equals original-image pixel
displacement. Repeatability divides qualifying matches by the smaller
keypoint count; precision divides by the match count. Coverage is occupied
source-side cells in a normalized 8x8 grid divided by 64.

## Frozen diagnostic gate and tail inspection

The inherited gate is named `inherited_pilot_gate_pass` and requires all three:

- R8 >= 0.40;
- coverage8 >= 0.60;
- displacement q95 <= 5.0 original-image pixels.

Each component has a saved flag. These thresholds precede this audit and are
never tuned to its distribution. Every record is retained, including failures.
The report includes pass/fail counts, acceptance fraction, and a 95% Wilson
interval computed directly without SciPy. Zero matches have zero ratios and
coverage, null displacement statistics, and fail the displacement gate;
undefined values have explicit missing counts.

All images receive equal weight. Quantiles use linear interpolation and cover
min, q01/q05/q10/q25, median, q75/q90/q95/q99, and max. Raw tail tables retain
the lowest 20 R8, highest 20 D95, lowest 20 coverage8, and every gate failure.
Audit index then row index break ties. A separate deduplicated manual audit set
prioritizes failures, low R8, high D95, then low coverage. The contact sheet
shows source and relit images with geometry-preserving display scaling.

RGB/luma differences are CPU-only appearance diagnostics. They can reveal
near-no-op interventions, very large changes, and unusual tails. They do not
measure exact nuisance equalization, define a pass/fail threshold, or provide
a CounterMine mining score. Luminance is `Y' = 0.2126 R + 0.7152 G + 0.0722 B`
on stored 8-bit RGB values, without gamma linearization. Luma means and delta
use 0–255 units; luma MAE is divided by 255. This is not photometric calibration.

## Saved evidence

Ignored `cache/native100_audit/` contains source/relit PNGs, the manifest and
build summary, generation CSV/summary, and fidelity CSV/summary/tail tables.
Ignored `outputs/step2d2/` contains the completed fidelity distribution,
intervention strength, and worst-20 contact sheet. Final figures are copied to
the same names prefixed `step2d2_` under `docs/audits/` as curated evidence.

Tool 22 exports `docs/audits/step2d2_native100_metrics.json` using CPU metadata
only. It records repository HEAD at export, SHA256 links to the three frozen
snapshots, source-selection provenance, configurations and seeds, all
requested summaries, frozen gate and Wilson interval, all 100 compact source
records, and bottom tails. It rejects nonfinite JSON and absolute paths and
does not load IC-Light, ALIKED, LightGlue, SALAD, or CUDA.

The next stage is not training. Only if this fidelity audit is acceptable will
the project proceed to the random-negative / SALAD-hard-negative /
same-place-positive CounterMine pair experiment, including random-pair null
statistics for shared-condition bias. Acceptance is a scientific review of
the frozen evidence, not an automatically selected training objective.

## Logged 100-source results

The completed run retained exactly 100 distinct frozen identities and all 100
measured pairs. All originals, source PNGs, and relit PNGs were 640x480; scales
and retained FOV were exactly 1.0. The inherited gate accepted 86 sources and
failed 14, for acceptance 0.86 with a Wilson 95% interval
`[0.7786281178903616, 0.9147365634006086]`. No image failed R8; 11 failed
coverage8 and 3 failed D95. These groups did not overlap. Thresholds remained
R8 >= 0.40, coverage8 >= 0.60, and D95 <= 5.0.

| Per-source metric | Median |
| --- | ---: |
| Matches | 732 |
| Match ratio | 0.572204 |
| R2 | 0.373696 |
| R4 | 0.548930 |
| R8 | 0.571461 |
| R16 | 0.571721 |
| Displacement median, original-image pixels | 1.488041 |
| Displacement q95, original-image pixels | 3.672953 |
| Coverage4 | 0.843750 |
| Coverage8 | 0.859375 |

Image inference totaled 190.553838 seconds; model loading was separately
measured at 4.032217 seconds. Peak CUDA allocation was 2,920,736,256 bytes,
and peak reservation was 3,571,449,856 bytes. Per-image elapsed quantiles are
saved alongside the runtime scope.

RGB MAE normalized had median 0.149065, min 0.082174, and max 0.328665. Luma
MAE normalized had median 0.146364; the median per-image luma mean delta was
-23.146501 stored RGB units. These are nonzero appearance differences, not
evidence of exact illumination or nuisance equalization. The distributions
and contact sheet show appearance changes that still require manual review.
Passing the structural gate alone does not establish that the intervention
is scientifically acceptable for the next pair experiment.

The three raw bottom-20 tables retain overlapping identities, and the full
manual audit union contains 42 unique sources, including all 14 gate failures.
All requested quantiles and the 100 compact records are preserved in
`docs/audits/step2d2_native100_metrics.json`, with three completed curated
figures alongside it. This audit makes no VPR performance improvement claim.
