#!/usr/bin/env python3
"""Explicitly generate 20 full-scene relights for the frozen geometry audit.

Imports and CLI help do not initialize models. Generation is a separate manual
command; neither the builder nor the measurement tool invokes it.
"""

import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import sys
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.probe.geometry_audit import (  # noqa: E402
    POLICIES, load_manifest, publish, reference, sha256,
    validate_destinations, validate_png, write_csv, write_json,
)
from countermine.probe.iclight_adapter import ICLightAdapter, ICLightConfig  # noqa: E402


GENERATION_COLUMNS = (
    "audit_index", "row_index", "image_id", "policy", "mode", "source_path", "output_path",
    "canonical_width", "canonical_height", "seed", "elapsed_seconds",
    "peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes",
    "source_sha256", "output_sha256",
)


def frozen_generation_config(snapshot_path, device):
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    provenance = snapshot["provenance"]
    saved = provenance["iclight_config"]
    names = {field.name for field in fields(ICLightConfig)}
    config = ICLightConfig(**{key: value for key, value in saved.items() if key in names})
    if config.seed != 12345 or provenance["iclight_seed"] != 12345:
        raise ValueError("Step 2D0 requires the frozen IC-Light seed 12345")
    return replace(config, device=device)


def run_generation(manifest_path, snapshot_path, output_dir, *, device="cuda",
                   adapter_factory=ICLightAdapter):
    """Stream one pair at a time through one shared model instance, full_scene only."""
    from PIL import Image

    manifest_path, snapshot_path, output_dir = (
        Path(path).expanduser().resolve() for path in (manifest_path, snapshot_path, output_dir)
    )
    rows = load_manifest(manifest_path, snapshot_path)
    base_config = frozen_generation_config(snapshot_path, device)
    destinations = [output_dir / policy / "relit" for policy in POLICIES]
    destinations += [output_dir / name for name in
                     ("geometry_generation.csv", "geometry_generation_summary.json")]
    inputs = [manifest_path, snapshot_path]
    inputs += [row[key] for row in rows for key in ("source_path", "original_path")]
    validate_destinations(destinations, inputs)
    for row in rows:
        validate_png(row["source_path"], (row["canonical_width"], row["canonical_height"]))
        with Image.open(row["original_path"]) as image:
            if image.size != (row["original_width"], row["original_height"]):
                raise ValueError("original image dimensions disagree with the geometry manifest")
            image.load()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runs, configs = [], {}
    with tempfile.TemporaryDirectory(prefix=".geometry-generation-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        for policy in POLICIES:
            (staging / policy / "relit").mkdir(parents=True)
        adapter = adapter_factory(base_config)
        for row in rows:
            config = replace(base_config, width=row["canonical_width"], height=row["canonical_height"])
            # Width/height control inference tensors, not model architecture.
            # Reusing the same adapter avoids keeping two model copies in RAM.
            adapter.config = config
            configs[f"{config.width}x{config.height}"] = asdict(config)
            name = f"{row['row_index']:08d}.png"
            staged_path = staging / row["policy"] / "relit" / name
            destination = output_dir / row["policy"] / "relit" / name
            with Image.open(row["source_path"]) as image:
                output = adapter.relight(image, "full_scene")
                try:
                    if output.mode != "RGB" or output.size != image.size:
                        raise ValueError("geometry relight must preserve source dimensions and RGB mode")
                    output.save(staged_path, format="PNG")
                finally:
                    output.close()
            validate_png(staged_path, (config.width, config.height))
            stats = adapter.last_run_stats
            run = {key: row[key] for key in ("audit_index", "row_index", "image_id", "policy",
                                            "canonical_width", "canonical_height")}
            run.update(mode="full_scene", source_path=reference(row["source_path"]),
                       output_path=reference(destination), seed=config.seed,
                       elapsed_seconds=stats["elapsed_seconds"],
                       peak_cuda_memory_allocated_bytes=stats.get("peak_cuda_memory_allocated_bytes"),
                       peak_cuda_memory_reserved_bytes=stats.get("peak_cuda_memory_reserved_bytes"),
                       source_sha256=sha256(row["source_path"]), output_sha256=sha256(staged_path))
            runs.append(run)
            print(f"audit {row['audit_index']} | row {row['row_index']} | {row['policy']} | "
                  f"full_scene | elapsed={run['elapsed_seconds']:.3f}s | "
                  f"peak CUDA allocated={run['peak_cuda_memory_allocated_bytes']} bytes | "
                  f"reserved={run['peak_cuda_memory_reserved_bytes']} bytes", flush=True)
        summary = {
            "number_of_sources": 10, "count_outputs": len(runs),
            "config": {"seed": 12345, "mode": "full_scene", "policies": list(POLICIES),
                       "configs_by_geometry": configs, "model_retention": "one shared IC-Light adapter"},
            "manifest_reference": reference(manifest_path), "manifest_sha256": sha256(manifest_path),
            "snapshot_reference": reference(snapshot_path), "snapshot_sha256": sha256(snapshot_path),
            "probe_image_format": "PNG", "elapsed_seconds": time.perf_counter() - started,
            "runs": runs,
        }
        write_csv(staging / "geometry_generation.csv", GENERATION_COLUMNS, runs)
        write_json(staging / "geometry_generation_summary.json", summary)
        artifacts = [(staging / policy / "relit", output_dir / policy / "relit") for policy in POLICIES]
        artifacts += [(staging / name, output_dir / name) for name in
                      ("geometry_generation.csv", "geometry_generation_summary.json")]
        publish(artifacts, staging)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/geometry_audit/geometry_manifest.csv"))
    parser.add_argument("--snapshot", type=Path, default=Path("docs/audits/step2c_metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/geometry_audit"))
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run_generation(args.manifest, args.snapshot, args.output_dir, device=args.device)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Generated {summary['count_outputs']} full_scene probes in {summary['elapsed_seconds']:.3f}s")


if __name__ == "__main__":
    main()
