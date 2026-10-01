#!/usr/bin/env python3
"""Generate the frozen 100 native full-scene diagnostic probes on invocation.

IC-Light applies a fixed shared illumination intervention. Its two inference
stages retain exact 640x480 geometry, and full_scene never loads or uses RMBG.
Model initialization is completed and timed before per-image measurements.
No inference library or model is initialized merely by importing this tool.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path, PureWindowsPath
import sys
import tempfile
import time

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.probe.geometry_audit import (  # noqa: E402
    publish, read_csv, reference, sha256, validate_png, write_csv, write_json,
)


NATIVE_SIZE = (640, 480)
POLICY = "native_full_fov"
SOURCE_COUNT = 100
PROMPT = "soft diffuse overcast daylight, uniform outdoor illumination, natural lighting"
GENERATION_COLUMNS = (
    "audit_index", "row_index", "image_id", "policy", "mode", "source_path", "output_path",
    "canonical_width", "canonical_height", "seed", "elapsed_seconds",
    "peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes",
    "source_sha256", "output_sha256",
)
SCIENTIFIC_CONFIG_FIELDS = (
    "added_prompt", "background", "base_model", "base_revision", "cfg", "height",
    "highres_denoise", "highres_scale", "lowres_denoise", "negative_prompt", "num_samples",
    "offset_filename", "offset_model", "offset_revision", "prompt", "scheduler_algorithm_type",
    "scheduler_beta_end", "scheduler_beta_start", "scheduler_num_train_timesteps",
    "scheduler_steps_offset", "scheduler_use_karras_sigmas", "seed", "steps",
    "text_encoder_dtype", "unet_dtype", "vae_dtype", "width",
)


def _read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{Path(path).name} must contain a JSON object")
    # Reject even nested non-finite values before any model construction.
    json.dumps(value, allow_nan=False)
    return value


def read_frozen_generation_settings(step2d1_snapshot_path):
    """Read and validate the frozen Step 2D1 settings without importing models."""
    snapshot = _read_object(step2d1_snapshot_path)
    provenance = snapshot.get("provenance", {})
    config = provenance.get("iclight_config")
    if not isinstance(config, dict) or set(SCIENTIFIC_CONFIG_FIELDS).difference(config):
        raise ValueError("the frozen Step 2D1 snapshot is missing scientific IC-Light settings")
    fixed = {
        "width": 640, "height": 480, "seed": 12345, "prompt": PROMPT, "cfg": 2.0,
        "steps": 25, "highres_scale": 1.0, "highres_denoise": .5, "num_samples": 1,
        "added_prompt": "", "background": None,
    }
    if any(config.get(name) != expected for name, expected in fixed.items()):
        raise ValueError("native100 generation must preserve completed Step 2D1 inference settings")
    if (provenance.get("prompt") != PROMPT or provenance.get("iclight_seed") != 12345
            or provenance.get("rmbg_used") is not False
            or provenance.get("two_stage_inference") is not True):
        raise ValueError("frozen Step 2D1 must document the fixed prompt, seed, two stages, and no RMBG")
    geometry = snapshot.get("native_geometry", {})
    geometry_values = {
        "policy": POLICY, "original_width": 640, "original_height": 480,
        "output_width": 640, "output_height": 480, "scale_x": 1.0, "scale_y": 1.0,
        "retained_area_fraction": 1.0, "retained_long_axis_fraction": 1.0,
        "crop": None, "resize": None, "padding": None, "exif_transpose": False,
    }
    if any(name not in geometry or geometry[name] != expected
           for name, expected in geometry_values.items()):
        raise ValueError("frozen Step 2D1 geometry must be exact native 640x480 full FOV")
    for name in ("frozen_step2c_snapshot_sha256", "frozen_step2d0_snapshot_sha256"):
        value = provenance.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"frozen Step 2D1 requires {name}")
    return dict(config)


def frozen_generation_config(step2d1_snapshot_path, device="cuda"):
    """Construct the lazy adapter configuration from Step 2D1, with device override."""
    from countermine.probe.iclight_adapter import ICLightConfig

    saved = read_frozen_generation_settings(step2d1_snapshot_path)
    # Only runtime device selection may differ from the completed native pilot.
    saved["device"] = device
    return ICLightConfig(**saved)


def validate_saved_generation_config(metadata, step2d1_snapshot_path):
    frozen = read_frozen_generation_settings(step2d1_snapshot_path)
    config = metadata.get("config", {})
    actual = config.get("iclight_config", {})
    if (config.get("policy") != POLICY or config.get("mode") != "full_scene"
            or config.get("seed") != 12345 or not isinstance(actual, dict)
            or any(actual.get(name) != value for name, value in frozen.items() if name != "device")
            or actual.get("rmbg_sigma", 0.0) != 0.0):
        raise ValueError("native100 IC-Light configuration differs from frozen Step 2D1 settings")
    if metadata.get("rmbg_used") is not False or metadata.get("two_stage_inference") is not True:
        raise ValueError("native100 generation must document two inference stages without RMBG")


def _runtime(stats):
    elapsed = stats["elapsed_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("generation elapsed_seconds must be finite and nonnegative")
    values = {"elapsed_seconds": float(elapsed)}
    for name in ("peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes"):
        value = stats.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"generation {name} must be a nonnegative integer or undefined")
        values[name] = value
    return values


def summarize_runtime(runs):
    """Equal-weight image inference timing; pretrained-model loading is excluded."""
    if len(runs) != SOURCE_COUNT:
        raise ValueError("runtime summary requires exactly 100 native100 image runs")
    runtimes = [_runtime(run) for run in runs]
    times = np.asarray([run["elapsed_seconds"] for run in runtimes], dtype=np.float64)
    result = {
        "elapsed_seconds": {name: float(np.quantile(times, quantile)) for name, quantile in
                            (("median", .5), ("q05", .05), ("q25", .25), ("q75", .75), ("q95", .95))},
        "timing_scope": "per-image inference after explicit model preload; model loading excluded",
    }
    for name in ("peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes"):
        values = [run[name] for run in runtimes if run[name] is not None]
        if len(values) not in (0, SOURCE_COUNT):
            raise ValueError("CUDA memory measurements must be present for all runs or absent for all runs")
        result[name] = max(values) if values else None
    return result


def _relative_image_reference(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty relative reference")
    path, windows = Path(value), PureWindowsPath(value)
    if path.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"{label} must be a relative reference")
    return path.resolve()


def load_native100_generated_pairs(rows, generation_csv, generation_summary_path,
                                   manifest_path, step2d1_snapshot_path):
    """Validate completed run provenance and PNGs before loading local matchers."""
    metadata = _read_object(generation_summary_path)
    provenance = _read_object(step2d1_snapshot_path)["provenance"]
    if (metadata.get("number_of_sources") != SOURCE_COUNT or metadata.get("count_outputs") != SOURCE_COUNT
            or metadata.get("manifest_sha256") != sha256(manifest_path)
            or metadata.get("step2d1_snapshot_sha256") != sha256(step2d1_snapshot_path)
            or metadata.get("step2c_snapshot_sha256") != provenance["frozen_step2c_snapshot_sha256"]
            or metadata.get("step2d0_snapshot_sha256") != provenance["frozen_step2d0_snapshot_sha256"]):
        raise ValueError("native100 generation provenance must match the 100 frozen sources and Step 2D1")
    validate_saved_generation_config(metadata, step2d1_snapshot_path)
    loading = metadata.get("model_load_elapsed_seconds")
    if isinstance(loading, bool) or not isinstance(loading, (int, float)) or not math.isfinite(loading) or loading < 0:
        raise ValueError("native100 generation must label finite model loading time separately")
    runs = metadata.get("runs")
    if not isinstance(runs, list) or len(runs) != SOURCE_COUNT:
        raise ValueError("native100 generation summary requires exactly 100 run records")
    lookup = {(row["audit_index"], row["row_index"]): row for row in rows}
    if len(rows) != SOURCE_COUNT or len(lookup) != SOURCE_COUNT:
        raise ValueError("native100 manifest requires exactly 100 distinct sources")
    saved_runs = {}
    for run in runs:
        if not isinstance(run, dict) or set(GENERATION_COLUMNS).difference(run):
            raise ValueError("native100 generation is missing run provenance")
        identity = (run["audit_index"], run["row_index"])
        if identity not in lookup or identity in saved_runs:
            raise ValueError("native100 generation has duplicate or non-frozen sources")
        _runtime(run)
        saved_runs[identity] = run
    if metadata.get("generation_runtime") != summarize_runtime(runs):
        raise ValueError("native100 runtime summary disagrees with its 100 per-image runs")
    generated = read_csv(generation_csv, GENERATION_COLUMNS)
    protected_images = {Path(row[name]).resolve() for row in rows for name in ("source_path", "original_path")}
    outputs, seen_paths = {}, set()
    for run in generated:
        identity = (int(run["audit_index"]), int(run["row_index"]))
        if (identity not in lookup or identity in outputs or run["mode"] != "full_scene"
                or run["policy"] != POLICY or int(run["seed"]) != 12345):
            raise ValueError("native100 generation CSV contains duplicate or non-frozen full_scene sources")
        saved = saved_runs[identity]
        if any(run[name] != ("" if saved[name] is None else str(saved[name])) for name in GENERATION_COLUMNS):
            raise ValueError("native100 generation CSV disagrees with the saved run provenance")
        row = lookup[identity]
        source = _relative_image_reference(run["source_path"], "source_path")
        output = _relative_image_reference(run["output_path"], "output_path")
        if (run["image_id"] != row["image_id"] or source != Path(row["source_path"]).resolve()
                or output in protected_images or output in seen_paths):
            raise ValueError("native100 generated images must retain source identities and distinct paths")
        if (int(run["canonical_width"]), int(run["canonical_height"])) != NATIVE_SIZE:
            raise ValueError("native100 generated image geometry must be exactly 640x480")
        validate_png(source, NATIVE_SIZE)
        validate_png(output, NATIVE_SIZE)
        if run["source_sha256"] != sha256(source) or run["output_sha256"] != sha256(output):
            raise ValueError("native100 source or relit PNG changed after generation")
        outputs[identity] = output
        seen_paths.add(output)
    if len(generated) != SOURCE_COUNT or set(outputs) != set(lookup):
        raise ValueError("native100 generation CSV requires exactly all 100 frozen sources")
    return outputs, metadata


def _preload(adapter):
    """Explicit preload keeps lazy first-run model loading outside image timing."""
    started = time.perf_counter()
    adapter._ensure_models()
    if getattr(getattr(adapter, "device", None), "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(adapter.device)
    if getattr(adapter, "rmbg", None) is not None:
        raise ValueError("native100 full_scene must never load RMBG")
    return time.perf_counter() - started


def run_generation(manifest_path, step2d1_snapshot_path, output_dir, *, device="cuda",
                   adapter_factory=None,
                   population_manifest_path=Path("cache/generator_audit/audit_manifest.csv"),
                   population_summary_path=Path("cache/generator_audit/audit_summary.json"),
                   build_summary_path=None):
    from countermine.probe.native100_audit import load_native100_manifest, validate_native100_destinations

    manifest_path, step2d1_snapshot_path, output_dir, population_manifest_path, population_summary_path = (
        Path(path).expanduser().resolve() for path in
        (manifest_path, step2d1_snapshot_path, output_dir, population_manifest_path, population_summary_path)
    )
    if build_summary_path is None:
        build_summary_path = manifest_path.parent / "native100_build_summary.json"
    build_summary_path = Path(build_summary_path).expanduser().resolve()
    rows = load_native100_manifest(manifest_path, population_manifest_path, population_summary_path)
    config = frozen_generation_config(step2d1_snapshot_path, device)
    frozen = _read_object(step2d1_snapshot_path)["provenance"]
    build_summary = _read_object(build_summary_path)
    population = build_summary.get("source_population_provenance")
    if (build_summary.get("number_of_sources") != SOURCE_COUNT
            or build_summary.get("manifest_sha256") != sha256(manifest_path)
            or not isinstance(population, dict) or population.get("number_of_sources") != SOURCE_COUNT
            or population.get("audit_manifest_sha256") != sha256(population_manifest_path)
            or population.get("audit_summary_sha256") != sha256(population_summary_path)):
        raise ValueError("native100 build summary must preserve the frozen source-population provenance")
    destinations = [output_dir / "relit", output_dir / "native100_generation.csv",
                    output_dir / "native100_generation_summary.json"]
    inputs = [manifest_path, step2d1_snapshot_path, population_manifest_path, population_summary_path, build_summary_path]
    inputs += [row[name] for row in rows for name in ("source_path", "original_path")]
    validate_native100_destinations(destinations, inputs)
    if (output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir())
            or (output_dir / "relit").is_symlink()
            or ((output_dir / "relit").exists() and not (output_dir / "relit").is_dir())):
        raise ValueError("managed native100 relit paths must be real directories")
    if adapter_factory is None:
        from countermine.probe.iclight_adapter import ICLightAdapter

        adapter_factory = ICLightAdapter
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runs = []
    with tempfile.TemporaryDirectory(prefix=".native100-generation-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        (staging / "relit").mkdir()
        adapter = adapter_factory(config)
        model_load_elapsed_seconds = _preload(adapter)
        print(f"Model preload: {model_load_elapsed_seconds:.3f}s (excluded from image elapsed time)", flush=True)
        for row in rows:
            name = f"{row['row_index']:08d}.png"
            staged, destination = staging / "relit" / name, output_dir / "relit" / name
            validate_png(row["source_path"], NATIVE_SIZE)
            with Image.open(row["source_path"]) as source:
                source.load()
                output = adapter.relight(source, "full_scene")
                try:
                    if output.size != NATIVE_SIZE or output.mode != "RGB":
                        raise ValueError("native100 IC-Light output must remain exact 640x480 RGB")
                    if getattr(adapter, "rmbg", None) is not None:
                        raise ValueError("native100 full_scene must never load RMBG")
                    output.save(staged, format="PNG")
                finally:
                    output.close()
            validate_png(staged, NATIVE_SIZE)
            run = {name: row[name] for name in ("audit_index", "row_index", "image_id", "policy",
                                               "canonical_width", "canonical_height")}
            run.update(mode="full_scene", seed=config.seed, source_path=reference(row["source_path"]),
                       output_path=reference(destination), source_sha256=sha256(row["source_path"]),
                       output_sha256=sha256(staged), **_runtime(adapter.last_run_stats))
            runs.append(run)
            print(f"audit {row['audit_index']} | row {row['row_index']} | {POLICY} | full_scene | "
                  f"elapsed={run['elapsed_seconds']:.3f}s | "
                  f"peak CUDA allocated={run['peak_cuda_memory_allocated_bytes']} bytes | "
                  f"reserved={run['peak_cuda_memory_reserved_bytes']} bytes", flush=True)
        if len(runs) != SOURCE_COUNT:
            raise ValueError("native100 generation must complete exactly 100 sources")
        summary = {
            "number_of_sources": SOURCE_COUNT, "count_outputs": len(runs),
            "config": {"seed": 12345, "mode": "full_scene", "policy": POLICY, "iclight_config": asdict(config)},
            "source_population_provenance": population,
            "manifest_reference": reference(manifest_path), "manifest_sha256": sha256(manifest_path),
            "step2d1_snapshot_reference": reference(step2d1_snapshot_path),
            "step2d1_snapshot_sha256": sha256(step2d1_snapshot_path),
            "step2c_snapshot_sha256": frozen["frozen_step2c_snapshot_sha256"],
            "step2d0_snapshot_sha256": frozen["frozen_step2d0_snapshot_sha256"],
            "build_summary_reference": reference(build_summary_path),
            "build_summary_sha256": sha256(build_summary_path),
            "probe_image_format": "PNG", "rmbg_used": False, "two_stage_inference": True,
            "inference_geometry": "both conditioning/decoded/refinement stages exactly 640x480; latents 80x60",
            "intervention": "fixed shared illumination intervention",
            "model_load_elapsed_seconds": model_load_elapsed_seconds,
            "generation_elapsed_seconds": sum(run["elapsed_seconds"] for run in runs),
            "elapsed_seconds": time.perf_counter() - started,
            "generation_runtime": summarize_runtime(runs), "runs": runs,
        }
        write_csv(staging / "native100_generation.csv", GENERATION_COLUMNS, runs)
        write_json(staging / "native100_generation_summary.json", summary)
        publish([(staging / "relit", output_dir / "relit"),
                 (staging / "native100_generation.csv", output_dir / "native100_generation.csv"),
                 (staging / "native100_generation_summary.json", output_dir / "native100_generation_summary.json")], staging)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/native100_audit/native100_manifest.csv"))
    parser.add_argument("--step2d1-snapshot", type=Path, default=Path("docs/audits/step2d1_native_fov_metrics.json"))
    parser.add_argument("--population-manifest", type=Path, default=Path("cache/generator_audit/audit_manifest.csv"))
    parser.add_argument("--population-summary", type=Path, default=Path("cache/generator_audit/audit_summary.json"))
    parser.add_argument("--build-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("cache/native100_audit"))
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run_generation(args.manifest, args.step2d1_snapshot, args.output_dir,
                                 device=args.device, population_manifest_path=args.population_manifest,
                                 population_summary_path=args.population_summary, build_summary_path=args.build_summary)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Generated {summary['count_outputs']} native full_scene PNGs; "
          f"image inference {summary['generation_elapsed_seconds']:.3f}s; "
          f"model preload {summary['model_load_elapsed_seconds']:.3f}s")


if __name__ == "__main__":
    main()
