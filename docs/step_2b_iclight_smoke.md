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

Completed runs publish:

- `cache/generator_audit/relit/official_rmbg/` and `relit/full_scene/`.
- `cache/generator_audit/iclight_smoke.csv`, with the requested inference fields.
- `cache/generator_audit/iclight_smoke_summary.json`, with all configuration,
  source/output hashes, timing, and peak CUDA memory statistics.
- `outputs/step2/iclight_smoke_10.jpg`, with SOURCE | OFFICIAL_RMBG | FULL_SCENE.

Source/output references use paths relative to the invocation working directory.
Every source and output is checked for RGB mode, exact resolution, finite pixels,
and successful file decoding. Publishing waits for the entire smoke run and
contact sheet to pass these checks; a failed run preserves previous artifacts.
File validity does not establish structural fidelity or a preferred mode.
Review the ten-source contact sheet before proceeding to any larger audit.
