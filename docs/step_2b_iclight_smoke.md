# Step 2B: IC-Light smoke audit

This adapter prepares diagnostic probes only. It uses the foreground-conditioned
IC-Light checkpoint and the base model, architecture, tensor normalization,
sampler, and two-stage inference from `third_party/IC-Light/gradio_demo.py`.
The Gradio module is never imported. Model loading is lazy, and the checkout
remains read-only. The upstream `briarmbg.py` architecture is read only when
the `official_rmbg` mode needs it, without writing Python bytecode there.

Run from the repository root in a CUDA environment with the IC-Light inference
dependencies available. See `third_party/IC-Light/requirements.txt` for the
upstream dependencies. Neither model loading nor GPU generation has been
validated by the static/CPU tests.

```bash
python tools/08_iclight_audit_smoke.py
```

The command explicitly starts generation. It uses the first ten audit manifest
rows and runs each source sequentially in both modes:

- `official_rmbg`: upstream RMBG matting and composition over gray 127.
- `full_scene`: the complete canonical image as foreground conditioning;
  RMBG is neither loaded nor run.

Both use 512x512, one sample, seed 12345, 25 steps, CFG 2.0, and highres scale
1.0. The positive prompt is exactly:

```text
soft diffuse overcast daylight, uniform outdoor illumination, natural lighting
```

The background initial latent is `None`. No added positive prompt is used.
The upstream negative prompt and highres denoise 0.5 are retained. Scale 1.0
retains the upstream refinement pass; it does not skip that pass.

Model weights use the external Hugging Face cache. A downloaded foreground
checkpoint can be supplied with `--checkpoint-path`; `--cache-dir` selects a
cache outside `third_party/`, and `--local-files-only` disables downloads.
`--count` allows 1 through 10 sources. Larger audits are disabled for this step.

Canonical audit sources are built separately with:

```bash
python tools/07_build_generator_audit_set.py
```

They are lossless PNGs at
`cache/generator_audit/source_512/<row_index>.png`, using the existing eight-digit
row-index filename. Preprocessing still center-crops to a square and resizes to
512x512. The audit manifest records `crop_long_axis_fraction = crop_size /
max(original_width, original_height)` and `crop_area_fraction = crop_size**2 /
(original_width * original_height)`. The audit summary has top-level
`crop_long_axis_fraction` and `crop_area_fraction` objects with min, median, q05,
q25, q75, and q95 for each fraction. These are retention diagnostics; they do not
change the crop policy.

Completed smoke runs publish:

- Lossless PNGs at `cache/generator_audit/relit/official_rmbg/<row_index>.png`
  and `cache/generator_audit/relit/full_scene/<row_index>.png`.
- `cache/generator_audit/iclight_smoke.csv`, with the requested inference fields.
- `cache/generator_audit/iclight_smoke_summary.json`, with all configuration,
  source/output hashes, timing, and peak CUDA memory statistics.
- `outputs/step2/iclight_smoke_10.jpg`, with SOURCE | OFFICIAL_RMBG | FULL_SCENE.

Source/output references use paths relative to the invocation working directory.
Every source and output is checked for RGB mode, exact resolution, finite pixels,
and successful file decoding. Publishing waits for the entire smoke run and
contact sheet to pass these checks; a failed run preserves previous artifacts.
Legacy JPEG source references, disguised JPEG content, and coexistence of JPEGs
with PNGs in `source_512` are rejected before model loading. Rebuild the canonical
source set and explicitly rerun the smoke command before quantitative fidelity
measurement. A successful rerun replaces both relit mode directories with PNGs.
The PNG change does not change the inference settings described above.

For each `official_rmbg` result, the smoke summary's `runs` entries record scalar
diagnostics from the alpha used for composition: `alpha_mean`, `alpha_q05`,
`alpha_q50`, `alpha_q95`, `alpha_fraction_lt_0_5`, and
`alpha_fraction_gt_0_9`. Each field is null for `full_scene`. RMBG behavior is
unchanged, and full alpha masks are not saved.

The existing ten-image qualitative audit figure is preserved at
[docs/audits/step2_iclight_smoke_10.jpg](audits/step2_iclight_smoke_10.jpg).
It documents the completed JPEG-era audit, not a newly generated PNG run.
New contact sheets still use the ignored `outputs/step2/iclight_smoke_10.jpg`
path and may remain JPEG because they are for visual inspection.

File validity does not establish structural fidelity or a preferred mode.
Review the ten-source contact sheet before proceeding to any larger audit.
