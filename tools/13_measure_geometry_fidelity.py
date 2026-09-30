#!/usr/bin/env python3
"""Explicitly measure source-to-relight fidelity for the paired Step 2D0 audit.

No identity control, cross-place comparison, generation, or training is run.
Original-image pixels provide the common displacement scale. Historical
output-pixel metrics and the old gate are retained as legacy diagnostics.
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
    DELTA_METRICS, ISOTROPIC_SCALE_TOLERANCE, ORIGINAL_METRIC_COLUMNS,
    ORIGINAL_THRESHOLD_ROUNDOFF_TOLERANCE, POLICIES, compute_original_pixel_metrics,
    load_manifest, paired_deltas, pilot_gate_pass,
    plot_contact_sheet, plot_fidelity, publish, read_csv, reference, sha256,
    summarize, validate_destinations, validate_isotropic_scale, validate_png, write_csv, write_json,
)
from countermine.probe.local_fidelity import (  # noqa: E402
    METRIC_COLUMNS, LocalFidelityConfig, LocalFidelityMatcher, compute_fidelity_metrics,
)
from countermine.probe.iclight_adapter import ICLightConfig  # noqa: E402


CSV_COLUMNS = (
    "audit_index", "row_index", "image_id", "policy", "mode", "scale_x", "scale_y",
    *METRIC_COLUMNS, *ORIGINAL_METRIC_COLUMNS, "legacy_output_pixel_gate_pass",
)
PAIRED_COLUMNS = ("audit_index", "row_index", *DELTA_METRICS)
PLOT_NAMES = ("geometry_compare_10.jpg", "geometry_fidelity_compare.png")
PROVENANCE_COLUMNS = (
    "audit_index", "row_index", "image_id", "policy", "mode", "source_path", "output_path",
    "canonical_width", "canonical_height", "seed", "source_sha256", "output_sha256",
)


def load_generated_pairs(rows, generation_csv, generation_summary_path, manifest_path, snapshot_path):
    metadata = json.loads(Path(generation_summary_path).read_text(encoding="utf-8"))
    if (metadata.get("number_of_sources") != 10 or metadata.get("count_outputs") != 20
            or metadata.get("config", {}).get("mode") != "full_scene"
            or metadata.get("config", {}).get("seed") != 12345):
        raise ValueError("generation metadata must describe 20 full_scene outputs for the frozen 10 sources")
    if metadata.get("manifest_sha256") != sha256(manifest_path) or metadata.get("snapshot_sha256") != sha256(snapshot_path):
        raise ValueError("generation metadata does not match the current manifest and frozen snapshot")
    saved = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))["provenance"]["iclight_config"]
    expected = asdict(ICLightConfig(**{field.name: saved[field.name] for field in fields(ICLightConfig)
                                      if field.name in saved}))
    configs = metadata["config"].get("configs_by_geometry", {})
    geometries = {(row["canonical_width"], row["canonical_height"]) for row in rows}
    if (set(configs) != {f"{width}x{height}" for width, height in geometries}
            or metadata["config"].get("policies") != list(POLICIES)):
        raise ValueError("generation metadata does not describe both canonical geometry policies")
    infrastructure = {"width", "height", "device", "iclight_root", "cache_dir", "local_files_only"}
    for size, settings in configs.items():
        actual = asdict(ICLightConfig(**settings))
        if size != f"{actual['width']}x{actual['height']}" or any(
            actual[name] != value for name, value in expected.items() if name not in infrastructure
        ):
            raise ValueError("generation configuration differs from the frozen Step 2C inference settings")
    generated = read_csv(generation_csv, PROVENANCE_COLUMNS)
    lookup = {(row["audit_index"], row["row_index"], row["policy"]): row for row in rows}
    saved_runs = metadata.get("runs", [])
    if not isinstance(saved_runs, list) or len(saved_runs) != 20:
        raise ValueError("generation summary must contain exactly 20 run records")
    run_lookup = {}
    for saved_run in saved_runs:
        if not isinstance(saved_run, dict) or set(PROVENANCE_COLUMNS).difference(saved_run):
            raise ValueError("generation summary is missing required run provenance")
        key = (saved_run["audit_index"], saved_run["row_index"], saved_run["policy"])
        if key not in lookup or key in run_lookup:
            raise ValueError("generation summary contains duplicate or non-frozen source/policy runs")
        run_lookup[key] = saved_run
    protected_images = {row[name] for row in rows for name in ("source_path", "original_path")}
    outputs, seen_paths = {}, set()
    for run in generated:
        key = (int(run["audit_index"]), int(run["row_index"]), run["policy"])
        if key not in lookup or key in outputs or run["mode"] != "full_scene" or int(run["seed"]) != 12345:
            raise ValueError("generation CSV contains a duplicate, non-frozen, or non-full_scene pair")
        row = lookup[key]
        if any(run[name] != str(run_lookup[key][name]) for name in PROVENANCE_COLUMNS):
            raise ValueError("generation CSV disagrees with its saved run provenance")
        if Path(run["source_path"]).is_absolute() or Path(run["output_path"]).is_absolute():
            raise ValueError("generation CSV image references must be relative")
        output = Path(run["output_path"]).resolve()
        if output in protected_images:
            raise ValueError("generated outputs must be distinct from all canonical and original sources")
        if Path(run["source_path"]).resolve() != row["source_path"] or output in seen_paths:
            raise ValueError("generation paths must match the manifest and be distinct")
        if (int(run["canonical_width"]), int(run["canonical_height"])) != (row["canonical_width"], row["canonical_height"]):
            raise ValueError("generation dimensions disagree with the geometry manifest")
        if run["source_sha256"] != sha256(row["source_path"]) or run["output_sha256"] != sha256(output):
            raise ValueError("canonical or generated PNG changed since the generation run")
        outputs[key] = output
        seen_paths.add(output)
    if len(generated) != 20 or set(outputs) != set(lookup):
        raise ValueError("generation CSV must contain exactly 20 paired outputs")
    return outputs, metadata


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
        raise ValueError("the geometry audit requires resize=None and no registration")


def run_geometry_fidelity(manifest_path, generation_csv, generation_summary_path,
                          snapshot_path, output_dir, plot_dir, *, device="auto",
                          matcher_factory=LocalFidelityMatcher):
    from PIL import Image

    manifest_path, generation_csv, generation_summary_path, snapshot_path, output_dir, plot_dir = (
        Path(path).expanduser().resolve() for path in
        (manifest_path, generation_csv, generation_summary_path, snapshot_path, output_dir, plot_dir)
    )
    rows = load_manifest(manifest_path, snapshot_path)
    for row in rows:
        validate_isotropic_scale(row["scale_x"], row["scale_y"])
    outputs, generation_metadata = load_generated_pairs(
        rows, generation_csv, generation_summary_path, manifest_path, snapshot_path,
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    provenance = snapshot["provenance"]
    config = LocalFidelityConfig(device=device, max_keypoints=provenance["aliked_config"]["max_num_keypoints"],
                                 seed=provenance["fidelity_seed"])
    report_names = ("geometry_fidelity.csv", "geometry_paired.csv", "geometry_summary.json")
    destinations = [output_dir / name for name in report_names] + [plot_dir / name for name in PLOT_NAMES]
    inputs = [manifest_path, generation_csv, generation_summary_path, snapshot_path]
    inputs += [row[key] for row in rows for key in ("source_path", "original_path")]
    inputs += list(outputs.values())
    validate_destinations(destinations, inputs)
    for row in rows:
        size = (row["canonical_width"], row["canonical_height"])
        validate_png(row["source_path"], size)
        validate_png(outputs[(row["audit_index"], row["row_index"], row["policy"])], size)
        with Image.open(row["original_path"]) as original:
            if original.size != (row["original_width"], row["original_height"]):
                raise ValueError("original image dimensions disagree with the geometry manifest")
            original.load()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix=".geometry-fidelity-", dir=output_dir.parent) as temporary, \
            tempfile.TemporaryDirectory(prefix=".geometry-plots-", dir=plot_dir.parent) as plot_temporary:
        staging, plot_staging = Path(temporary), Path(plot_temporary)
        matcher = matcher_factory(config)
        settings_validated = False
        for row in rows:
            output = outputs[(row["audit_index"], row["row_index"], row["policy"])]
            source_features = matcher.extract(row["source_path"])
            try:
                if not settings_validated:
                    validate_matcher_settings(matcher.runtime_metadata(), provenance)
                    settings_validated = True
                relit_features = matcher.extract(output)
                try:
                    matches = matcher.match(source_features, relit_features)
                finally:
                    del relit_features
                if (matches.canonical_width, matches.canonical_height) != (row["canonical_width"], row["canonical_height"]):
                    raise ValueError("local matcher dimensions disagree with the paired canonical images")
                metrics = compute_fidelity_metrics(
                    matches.points_source, matches.points_relit, matches.match_scores,
                    matches.num_keypoints_source, matches.num_keypoints_relit,
                    canonical_width=row["canonical_width"], canonical_height=row["canonical_height"],
                )
                original_metrics = compute_original_pixel_metrics(
                    matches.points_source, matches.points_relit,
                    matches.num_keypoints_source, matches.num_keypoints_relit,
                    scale_x=row["scale_x"], scale_y=row["scale_y"],
                )
                record = {key: row[key] for key in ("audit_index", "row_index", "image_id", "policy")}
                record.update(mode="full_scene", scale_x=row["scale_x"], scale_y=row["scale_y"],
                              **metrics, **original_metrics)
                record["legacy_output_pixel_gate_pass"] = pilot_gate_pass(record)
                records.append(record)
                print(f"audit {row['audit_index']} | row {row['row_index']} | {row['policy']} | "
                      f"matches={metrics['num_matches']} | "
                      f"R8(original-image pixels)={original_metrics['repeatability_original_8px']:.6f}",
                      flush=True)
                del matches
            finally:
                del source_features
        paired = paired_deltas(records)
        write_csv(staging / report_names[0], CSV_COLUMNS, records)
        write_csv(staging / report_names[1], PAIRED_COLUMNS, paired)
        plot_contact_sheet(rows, {(row_index, policy): path for (_, row_index, policy), path in outputs.items()},
                           plot_staging / PLOT_NAMES[0])
        plot_fidelity(records, plot_staging / PLOT_NAMES[1])
        summary = {
            "number_of_sources": 10, "count_pairs": len(records),
            "config": {**asdict(config), "mode": "full_scene", "extract_resize": None,
                       "registration": None, "grid_shape": [8, 8],
                       "grid_coordinates": "normalized source keypoints",
                       "displacement_units": "original-image pixels for fair comparisons",
                       "epsilons_pixels": [2, 4, 8, 16],
                       "original_pixel_conversion": "source and relit matched coordinates / scale_x; crop translation cancels",
                       "isotropic_scale_absolute_tolerance": ISOTROPIC_SCALE_TOLERANCE,
                       "original_epsilon_comparison": "inclusive <= with float64 roundoff guard",
                       "original_epsilon_roundoff_absolute_tolerance_pixels": ORIGINAL_THRESHOLD_ROUNDOFF_TOLERANCE,
                       "original_epsilon_roundoff_relative_tolerance": 0.0,
                       "historical_displacement_units": "canonical output pixels; not directly comparable across policies",
                       "summary_quantiles": "numpy linear interpolation; equal weight per source",
                       "paired_delta_direction": "full_fov_512 minus square_crop_512",
                       "geometry_acceptance_threshold": None,
                       "identity_control": "not rerun; validated historical Step 2C",
                       "feature_retention": "one source/relit pair; discard before next pair"},
            "matcher": matcher.runtime_metadata(),
            "manifest_reference": reference(manifest_path), "manifest_sha256": sha256(manifest_path),
            "snapshot_reference": reference(snapshot_path), "snapshot_sha256": sha256(snapshot_path),
            "generation_csv_reference": reference(generation_csv), "generation_csv_sha256": sha256(generation_csv),
            "generation_summary_reference": reference(generation_summary_path),
            "generation_summary_sha256": sha256(generation_summary_path),
            "generation_config": generation_metadata["config"],
            "plots": {name: reference(plot_dir / name) for name in PLOT_NAMES},
            **summarize(records, paired),
        }
        write_json(staging / report_names[2], summary)
        publish([(staging / name, output_dir / name) for name in report_names]
                + [(plot_staging / name, plot_dir / name) for name in PLOT_NAMES], staging)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/geometry_audit/geometry_manifest.csv"))
    parser.add_argument("--generation-csv", type=Path, default=Path("cache/geometry_audit/geometry_generation.csv"))
    parser.add_argument("--generation-summary", type=Path, default=Path("cache/geometry_audit/geometry_generation_summary.json"))
    parser.add_argument("--snapshot", type=Path, default=Path("docs/audits/step2c_metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/geometry_audit"))
    parser.add_argument("--plot-dir", type=Path, default=Path("outputs/step2d0"))
    parser.add_argument("--device", default="auto")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run_geometry_fidelity(args.manifest, args.generation_csv, args.generation_summary,
                                        args.snapshot, args.output_dir, args.plot_dir, device=args.device)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Measured {summary['count_pairs']} pairs in a common original-image pixel scale:")
    for policy, results in summary["fair_original_pixel_metrics"].items():
        metrics = results["metrics"]
        print(f"{policy}: median original-pixel R4={metrics['repeatability_original_4px']['median']}, "
              f"R8={metrics['repeatability_original_8px']['median']}, "
              f"displacement q95={metrics['displacement_original_q95']['median']}")
    print("Historical output-pixel metrics and gate are retained; they are not cross-policy comparable.")


if __name__ == "__main__":
    main()
