#!/usr/bin/env python3
"""Measure ten native full-FOV pairs and compare frozen historical fidelity.

Only this explicit command initializes the unchanged ALIKED/LightGlue models.
Historical scalar metrics are read from the Step 2D0 snapshot, never remeasured.
No generation, cross-place matching, training, significance test, or gate runs.
"""

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.probe.geometry_audit import (  # noqa: E402
    publish, read_csv, reference, sha256, validate_png, write_csv, write_json,
)
from countermine.probe.iclight_adapter import ICLightConfig  # noqa: E402
from countermine.probe.local_fidelity import LocalFidelityConfig, LocalFidelityMatcher  # noqa: E402
from countermine.probe.native_fov_audit import (  # noqa: E402
    load_native_manifest, validate_native_destinations,
)
from countermine.probe.native_fov_fidelity import (  # noqa: E402
    HISTORICAL_POLICIES, NATIVE_METRICS, NATIVE_POLICY, PAIRED_COLUMNS,
    compute_native_metrics, load_historical_metrics, paired_native_comparisons,
    plot_native_contact_sheet, plot_native_fidelity, summarize_native,
)


CSV_COLUMNS = (
    "audit_index", "row_index", "image_id", "policy", "mode", "canonical_width", "canonical_height",
    "scale_x", "scale_y", *NATIVE_METRICS,
)
REPORT_NAMES = ("native_fidelity.csv", "native_paired.csv", "native_summary.json")
PLOT_NAMES = ("native_fov_compare_10.jpg", "native_fov_fidelity.png")
GENERATION_FIELDS = (
    "audit_index", "row_index", "image_id", "policy", "mode", "source_path", "output_path",
    "canonical_width", "canonical_height", "seed", "source_sha256", "output_sha256",
)


def _read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{Path(path).name} must contain a JSON object")
    return value


def validate_matcher_settings(metadata, provenance):
    models = metadata.get("models") or {}
    aliked = {**metadata.get("extractor_settings", {}), **models.get("extractor_config", {})}
    lightglue = {**metadata.get("matcher_settings", {}), **models.get("matcher_config", {})}
    lightglue["compiled"] = metadata.get("compiled")
    if "lightglue_weights_version" in models:
        lightglue["weights_version"] = models["lightglue_weights_version"]
    for label, actual, expected in (("ALIKED", aliked, provenance["aliked_config"]),
                                    ("LightGlue", lightglue, provenance["lightglue_config"])):
        if any(name not in actual or actual[name] != value for name, value in expected.items()):
            raise ValueError(f"{label} configuration differs from frozen Step 2C settings")
    if metadata.get("extract_resize") is not None or metadata.get("registration") is not None:
        raise ValueError("native fidelity requires resize=None and no registration")


def load_native_generated_pairs(rows, generation_csv, generation_summary_path,
                                manifest_path, snapshot_path, step2c_snapshot_path):
    metadata = _read_object(generation_summary_path)
    if (metadata.get("number_of_sources") != 10 or metadata.get("count_outputs") != 10
            or metadata.get("manifest_sha256") != sha256(manifest_path)
            or metadata.get("historical_snapshot_sha256") != sha256(snapshot_path)
            or metadata.get("step2c_snapshot_sha256") != sha256(step2c_snapshot_path)):
        raise ValueError("native generation provenance must match ten sources and the frozen inputs")
    config = metadata.get("config", {})
    if (config.get("policy") != NATIVE_POLICY or config.get("mode") != "full_scene"
            or config.get("seed") != 12345):
        raise ValueError("native generation must use native_full_fov, full_scene, seed 12345")
    saved = _read_object(step2c_snapshot_path)["provenance"]["iclight_config"]
    expected = asdict(ICLightConfig(**{field.name: saved[field.name] for field in fields(ICLightConfig)
                                      if field.name in saved}))
    actual = asdict(ICLightConfig(**config["iclight_config"]))
    infrastructure = {"width", "height", "device", "iclight_root", "cache_dir", "local_files_only"}
    if ((actual["width"], actual["height"]) != (640, 480)
            or any(actual[name] != value for name, value in expected.items() if name not in infrastructure)):
        raise ValueError("native IC-Light configuration differs from frozen Step 2C science settings")
    generated = read_csv(generation_csv, GENERATION_FIELDS)
    lookup = {(row["audit_index"], row["row_index"]): row for row in rows}
    runs = metadata.get("runs")
    if not isinstance(runs, list) or len(runs) != 10:
        raise ValueError("native generation summary requires ten run records")
    saved_runs = {}
    for run in runs:
        if not isinstance(run, dict) or set(GENERATION_FIELDS).difference(run):
            raise ValueError("native generation is missing run provenance")
        key = (run["audit_index"], run["row_index"])
        if key not in lookup or key in saved_runs:
            raise ValueError("native generation has duplicate or non-frozen sources")
        saved_runs[key] = run
    protected_images = {row[name] for row in rows for name in ("source_path", "original_path")}
    outputs, seen_paths = {}, set()
    for run in generated:
        key = (int(run["audit_index"]), int(run["row_index"]))
        if (key not in lookup or key in outputs or run["mode"] != "full_scene"
                or run["policy"] != NATIVE_POLICY or int(run["seed"]) != 12345):
            raise ValueError("native generation CSV contains duplicate or non-frozen full-scene pairs")
        if any(run[name] != str(saved_runs[key][name]) for name in GENERATION_FIELDS):
            raise ValueError("native generation CSV disagrees with saved run provenance")
        row = lookup[key]
        if run["image_id"] != row["image_id"]:
            raise ValueError("native generation image identity disagrees with the manifest")
        if any(not run[name] or Path(run[name]).is_absolute() for name in ("source_path", "output_path")):
            raise ValueError("native generation image references must be relative")
        source, output = Path(run["source_path"]).resolve(), Path(run["output_path"]).resolve()
        if source != row["source_path"] or output in protected_images or output in seen_paths:
            raise ValueError("native generated images must be distinct and reference the manifest source")
        if (int(run["canonical_width"]), int(run["canonical_height"])) != (640, 480):
            raise ValueError("native generated image geometry must be exactly 640x480")
        if run["source_sha256"] != sha256(source) or run["output_sha256"] != sha256(output):
            raise ValueError("native source or relit PNG changed after generation")
        outputs[key] = output
        seen_paths.add(output)
    if len(generated) != 10 or set(outputs) != set(lookup):
        raise ValueError("native generation CSV requires all ten frozen sources")
    return outputs, metadata


def load_historical_relits(path, historical):
    """Validate display-only historical relights without extracting features."""
    rows = read_csv(path, ("audit_index", "row_index", "policy", "mode", "output_path",
                           "canonical_width", "canonical_height", "output_sha256"))
    outputs, seen_paths = {}, set()
    sizes = {"square_crop_512": (512, 512), "full_fov_512": (512, 384)}
    for row in rows:
        identity = (int(row["audit_index"]), int(row["row_index"]))
        policy = row["policy"]
        key = (*identity, policy)
        if (identity not in historical or policy not in HISTORICAL_POLICIES
                or key in outputs or row["mode"] != "full_scene"):
            raise ValueError("historical display CSV must contain both policies for the frozen sources")
        if not row["output_path"] or Path(row["output_path"]).is_absolute():
            raise ValueError("historical display image paths must be relative")
        output = Path(row["output_path"]).resolve()
        if output in seen_paths or (int(row["canonical_width"]), int(row["canonical_height"])) != sizes[policy]:
            raise ValueError("historical display images must retain their recorded geometry")
        if row["output_sha256"] != sha256(output):
            raise ValueError("historical display relit changed after its generation run")
        validate_png(output, sizes[policy])
        outputs[key] = output
        seen_paths.add(output)
    if len(rows) != 20 or set(outputs) != {(*identity, policy) for identity in historical for policy in HISTORICAL_POLICIES}:
        raise ValueError("historical display CSV requires twenty frozen relights")
    return outputs


def run_native_fidelity(manifest_path, generation_csv, generation_summary_path,
                        snapshot_path, step2c_snapshot_path, historical_generation_csv,
                        output_dir, plot_dir, *, device="auto", matcher_factory=LocalFidelityMatcher):
    from PIL import Image

    (manifest_path, generation_csv, generation_summary_path, snapshot_path, step2c_snapshot_path,
     historical_generation_csv, output_dir, plot_dir) = (
        Path(path).expanduser().resolve() for path in
        (manifest_path, generation_csv, generation_summary_path, snapshot_path, step2c_snapshot_path,
         historical_generation_csv, output_dir, plot_dir)
    )
    historical, historical_snapshot = load_historical_metrics(snapshot_path)
    rows = load_native_manifest(manifest_path, snapshot_path)
    if {(row["audit_index"], row["row_index"]) for row in rows} != set(historical):
        raise ValueError("native manifest identities disagree with historical metrics")
    outputs, generation_metadata = load_native_generated_pairs(
        rows, generation_csv, generation_summary_path, manifest_path, snapshot_path, step2c_snapshot_path,
    )
    historical_outputs = load_historical_relits(historical_generation_csv, historical)
    step2c = _read_object(step2c_snapshot_path)
    provenance = step2c["provenance"]
    if (provenance["fidelity_seed"] != 42 or provenance["aliked_config"]["max_num_keypoints"] != 2048
            or historical_snapshot["provenance"]["frozen_step2c_snapshot_sha256"] != sha256(step2c_snapshot_path)
            or historical_snapshot["provenance"]["fidelity_seed"] != 42
            or any(historical_snapshot["provenance"][name] != provenance[name]
                   for name in ("aliked_config", "lightglue_config"))):
        raise ValueError("native and historical fidelity must retain the same frozen Step 2C configuration")
    config = LocalFidelityConfig(device=device, max_keypoints=2048, seed=42)
    destinations = [output_dir / name for name in REPORT_NAMES] + [plot_dir / name for name in PLOT_NAMES]
    inputs = [manifest_path, generation_csv, generation_summary_path, snapshot_path,
              step2c_snapshot_path, historical_generation_csv]
    inputs += [row[key] for row in rows for key in ("source_path", "original_path")]
    inputs += list(outputs.values()) + list(historical_outputs.values())
    validate_native_destinations(destinations, inputs)
    for row in rows:
        identity = (row["audit_index"], row["row_index"])
        if (row["canonical_width"], row["canonical_height"]) != (640, 480) or any(
            row[name] != 1.0 for name in ("scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction")
        ):
            raise ValueError("native transform must be exactly 640x480 with unit scale and retained FOV")
        validate_png(row["source_path"], (640, 480))
        validate_png(outputs[identity], (640, 480))
        if row["source_sha256"] != sha256(row["source_path"]) or row["original_sha256"] != sha256(row["original_path"]):
            raise ValueError("native source or original image changed after the audit build")
        with Image.open(row["original_path"]) as original:
            if original.size != (640, 480):
                raise ValueError("native original dimensions must remain exactly 640x480")
            original.load()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix=".native-fidelity-", dir=output_dir.parent) as temporary, \
            tempfile.TemporaryDirectory(prefix=".native-plots-", dir=plot_dir.parent) as plot_temporary:
        staging, plot_staging = Path(temporary), Path(plot_temporary)
        matcher = matcher_factory(config)
        settings_validated = False
        for row in rows:
            identity = (row["audit_index"], row["row_index"])
            source_features = matcher.extract(row["source_path"])
            try:
                if not settings_validated:
                    validate_matcher_settings(matcher.runtime_metadata(), provenance)
                    settings_validated = True
                relit_features = matcher.extract(outputs[identity])
                try:
                    matches = matcher.match(source_features, relit_features)
                finally:
                    del relit_features
                record = {key: row[key] for key in ("audit_index", "row_index", "image_id", "policy",
                                                  "canonical_width", "canonical_height", "scale_x", "scale_y")}
                record.update(mode="full_scene", **compute_native_metrics(matches))
                records.append(record)
                print(f"audit {identity[0]} | row {identity[1]} | native640x480 | "
                      f"matches={record['num_matches']} | R4={record['repeatability_4px']:.6f} | "
                      f"R8={record['repeatability_8px']:.6f}", flush=True)
                del matches
            finally:
                del source_features
        paired = paired_native_comparisons(records, historical)
        write_csv(staging / REPORT_NAMES[0], CSV_COLUMNS, records)
        write_csv(staging / REPORT_NAMES[1], PAIRED_COLUMNS, paired)
        plot_native_contact_sheet(rows, outputs, historical_outputs, plot_staging / PLOT_NAMES[0])
        plot_native_fidelity(records, historical, plot_staging / PLOT_NAMES[1])
        summary = {
            "number_of_sources": 10, "count_pairs": len(records),
            "config": {**asdict(config), "policy": NATIVE_POLICY, "mode": "full_scene",
                       "extract_resize": None, "registration": None, "grid_shape": [8, 8],
                       "grid_coordinates": "normalized canonical image coordinates",
                       "coverage_role": "spatial-distribution diagnostic only",
                       "historical_coverage": "frozen normalized coverage with unchanged historical output-pixel qualification masks",
                       "displacement_units": "original-image pixels; native output scale is exactly 1.0",
                       "epsilons_pixels": [2, 4, 8, 16], "epsilon_comparison": "inclusive <=",
                       "summary_quantiles": "numpy linear interpolation; equal weight per source",
                       "paired_delta_direction": "native_full_fov minus frozen historical policy",
                       "sign_counts": "exact positive/zero/negative deltas; undefined pairs excluded",
                       "feature_retention": "one source/relit pair; discard before next source"},
            "native_geometry": {"policy": NATIVE_POLICY, "canonical_width": 640, "canonical_height": 480,
                                "scale_x": 1.0, "scale_y": 1.0,
                                "retained_area_fraction": 1.0, "retained_long_axis_fraction": 1.0},
            "matcher": matcher.runtime_metadata(),
            "manifest_reference": reference(manifest_path), "manifest_sha256": sha256(manifest_path),
            "historical_snapshot_reference": reference(snapshot_path), "historical_snapshot_sha256": sha256(snapshot_path),
            "step2c_snapshot_reference": reference(step2c_snapshot_path), "step2c_snapshot_sha256": sha256(step2c_snapshot_path),
            "generation_csv_reference": reference(generation_csv), "generation_csv_sha256": sha256(generation_csv),
            "generation_summary_reference": reference(generation_summary_path),
            "generation_summary_sha256": sha256(generation_summary_path),
            "historical_generation_csv_reference": reference(historical_generation_csv),
            "historical_generation_csv_sha256": sha256(historical_generation_csv),
            "generation_config": generation_metadata["config"],
            "plots": {name: reference(plot_dir / name) for name in PLOT_NAMES},
            **summarize_native(records, paired),
        }
        write_json(staging / REPORT_NAMES[2], summary)
        publish([(staging / name, output_dir / name) for name in REPORT_NAMES]
                + [(plot_staging / name, plot_dir / name) for name in PLOT_NAMES], staging)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/native_fov_audit/native_manifest.csv"))
    parser.add_argument("--generation-csv", type=Path, default=Path("cache/native_fov_audit/native_generation.csv"))
    parser.add_argument("--generation-summary", type=Path, default=Path("cache/native_fov_audit/native_generation_summary.json"))
    parser.add_argument("--snapshot", type=Path, default=Path("docs/audits/step2d0_metrics.json"))
    parser.add_argument("--step2c-snapshot", type=Path, default=Path("docs/audits/step2c_metrics.json"))
    parser.add_argument("--historical-generation-csv", type=Path, default=Path("cache/geometry_audit/geometry_generation.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/native_fov_audit"))
    parser.add_argument("--plot-dir", type=Path, default=Path("outputs/step2d1"))
    parser.add_argument("--device", default="auto")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run_native_fidelity(args.manifest, args.generation_csv, args.generation_summary,
                                      args.snapshot, args.step2c_snapshot, args.historical_generation_csv,
                                      args.output_dir, args.plot_dir, device=args.device)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, TypeError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Measured {summary['count_pairs']} native pairs; paired deltas use original-image pixels.")
    for policy, deltas in summary["paired_comparisons"].items():
        print(f"native minus {policy}: " + ", ".join(f"{name} median={stats['median']}" for name, stats in deltas.items()))


if __name__ == "__main__":
    main()
