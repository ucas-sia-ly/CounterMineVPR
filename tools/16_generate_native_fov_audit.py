#!/usr/bin/env python3
"""Generate only ten native 640x480 full-scene diagnostic probes, on explicit invocation."""

import argparse
from dataclasses import asdict, fields, replace
import json
import math
from pathlib import Path
import sys
import tempfile
import time

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.probe.geometry_audit import publish, reference, sha256, validate_png, write_csv, write_json  # noqa: E402
from countermine.probe.iclight_adapter import ICLightAdapter, ICLightConfig  # noqa: E402
from countermine.probe.native_fov_audit import (  # noqa: E402
    NATIVE_SIZE, POLICY, load_native_manifest, validate_native_destinations,
)


GENERATION_COLUMNS = (
    "audit_index", "row_index", "image_id", "policy", "mode", "source_path", "output_path",
    "canonical_width", "canonical_height", "seed", "elapsed_seconds",
    "peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes",
    "source_sha256", "output_sha256",
)


def frozen_generation_config(step2c_snapshot_path, historical_snapshot_path, device):
    step2c = json.loads(Path(step2c_snapshot_path).read_text(encoding="utf-8"))
    historical = json.loads(Path(historical_snapshot_path).read_text(encoding="utf-8"))
    if historical["provenance"]["frozen_step2c_snapshot_sha256"] != sha256(step2c_snapshot_path):
        raise ValueError("historical Step 2D0 and supplied frozen Step 2C snapshot disagree")
    saved = step2c["provenance"]["iclight_config"]
    required = {field.name for field in fields(ICLightConfig)} - {"iclight_root", "cache_dir"}
    if not isinstance(saved, dict) or required.difference(saved):
        raise ValueError("the frozen Step 2C snapshot is missing generation settings")
    config = ICLightConfig(**{field.name: saved[field.name] for field in fields(ICLightConfig)
                             if field.name in saved})
    if any(getattr(config, key) != value for key, value in {
        "seed": 12345, "cfg": 2.0, "steps": 25, "highres_scale": 1.0, "highres_denoise": .5,
    }.items()):
        raise ValueError("native generation must retain the fixed completed Step 2C inference settings")
    for snapshot in (step2c, historical):
        if snapshot["provenance"]["iclight_seed"] != config.seed or snapshot["provenance"]["prompt"] != config.prompt:
            raise ValueError("native generation prompt or seed differs from frozen provenance")
    return replace(config, width=NATIVE_SIZE[0], height=NATIVE_SIZE[1], device=device)


def _runtime(stats):
    elapsed = stats["elapsed_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("generation elapsed_seconds must be finite and nonnegative")
    values = {"elapsed_seconds": elapsed}
    for name in ("peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes"):
        value = stats.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"generation {name} must be a nonnegative integer or undefined")
        values[name] = value
    return values


def run_generation(manifest_path, step2c_snapshot_path, historical_snapshot_path, output_dir,
                   *, device="cuda", adapter_factory=ICLightAdapter):
    paths = (manifest_path, step2c_snapshot_path, historical_snapshot_path, output_dir)
    manifest_path, step2c_snapshot_path, historical_snapshot_path, output_dir = (
        Path(path).expanduser().resolve() for path in paths
    )
    rows = load_native_manifest(manifest_path, historical_snapshot_path)
    config = frozen_generation_config(step2c_snapshot_path, historical_snapshot_path, device)
    destinations = [output_dir / POLICY / "relit", output_dir / "native_generation.csv",
                    output_dir / "native_generation_summary.json"]
    inputs = [manifest_path, step2c_snapshot_path, historical_snapshot_path]
    inputs += [row[name] for row in rows for name in ("source_path", "original_path")]
    validate_native_destinations(destinations, inputs)
    policy_dir = output_dir / POLICY
    if policy_dir.is_symlink() or (policy_dir.exists() and not policy_dir.is_dir()) or (policy_dir / "relit").is_symlink():
        raise ValueError("managed native relit paths must not be symlinks or non-directories")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runs = []
    with tempfile.TemporaryDirectory(prefix=".native-generation-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        (staging / POLICY / "relit").mkdir(parents=True)
        adapter = adapter_factory(config)
        for row in rows:
            name = f"{row['row_index']:08d}.png"
            staged = staging / POLICY / "relit" / name
            destination = output_dir / POLICY / "relit" / name
            with Image.open(row["source_path"]) as source:
                if source.size != NATIVE_SIZE or source.mode != "RGB":
                    raise ValueError("native IC-Light input must be exact 640x480 RGB")
                output = adapter.relight(source, "full_scene")
                try:
                    if output.size != source.size or output.size != NATIVE_SIZE or output.mode != "RGB":
                        raise ValueError("native IC-Light output must remain exact 640x480 RGB")
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
        summary = {
            "number_of_sources": 10, "count_outputs": len(runs),
            "config": {"seed": 12345, "mode": "full_scene", "policy": POLICY, "iclight_config": asdict(config)},
            "manifest_reference": reference(manifest_path), "manifest_sha256": sha256(manifest_path),
            "step2c_snapshot_reference": reference(step2c_snapshot_path), "step2c_snapshot_sha256": sha256(step2c_snapshot_path),
            "historical_snapshot_reference": reference(historical_snapshot_path),
            "historical_snapshot_sha256": sha256(historical_snapshot_path),
            "probe_image_format": "PNG", "elapsed_seconds": time.perf_counter() - started,
            "runs": runs,
        }
        write_csv(staging / "native_generation.csv", GENERATION_COLUMNS, runs)
        write_json(staging / "native_generation_summary.json", summary)
        publish([(staging / POLICY / "relit", output_dir / POLICY / "relit"),
                 (staging / "native_generation.csv", output_dir / "native_generation.csv"),
                 (staging / "native_generation_summary.json", output_dir / "native_generation_summary.json")], staging)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/native_fov_audit/native_manifest.csv"))
    parser.add_argument("--snapshot", type=Path, default=Path("docs/audits/step2d0_metrics.json"),
                        help="Frozen historical geometry snapshot")
    parser.add_argument("--step2c-snapshot", type=Path, default=Path("docs/audits/step2c_metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/native_fov_audit"))
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run_generation(args.manifest, args.step2c_snapshot, args.snapshot,
                                 args.output_dir, device=args.device)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Generated {summary['count_outputs']} native full_scene PNGs in {summary['elapsed_seconds']:.3f}s")


if __name__ == "__main__":
    main()
