#!/usr/bin/env python3
"""Measure all 100 native full-scene probes using the frozen local matcher.

IC-Light remains a diagnostic counterfactual probe applying a fixed shared
illumination intervention. The inherited gate never filters the saved dataset.
ALIKED/LightGlue are loaded only when this explicit command starts matching.
"""

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.probe.geometry_audit import (  # noqa: E402
    publish, reference, sha256, validate_png, write_csv, write_json,
)
from countermine.probe.local_fidelity import LocalFidelityConfig, LocalFidelityMatcher  # noqa: E402
from countermine.probe.native100_audit import (  # noqa: E402
    NATIVE_SIZE, POLICY, POPULATION_MANIFEST, POPULATION_SUMMARY,
    load_native100_manifest, validate_native100_destinations,
)
from countermine.probe.native100_fidelity import (  # noqa: E402
    FIDELITY_METRICS, GATE_COLUMNS, LUMA_DEFINITION, PIXEL_METRICS,
    compute_intervention_strength, compute_native100_metrics, inherited_gate,
    plot_fidelity_distribution, plot_intervention_strength, plot_worst_contact_sheet,
    source_identity, summarize_native100,
)


REPORT_NAMES = ("native100_fidelity.csv", "native100_fidelity_summary.json")
PLOT_NAMES = (
    "native100_fidelity_distribution.png", "native100_intervention_strength.png", "native100_worst20.jpg",
)
CURATED_NAMES = tuple(f"step2d2_{name}" for name in PLOT_NAMES)
TAIL_NAMES = {
    "lowest_R8": "native100_lowest_R8.csv", "highest_D95": "native100_highest_D95.csv",
    "lowest_coverage8": "native100_lowest_coverage8.csv", "gate_failures": "native100_gate_failures.csv",
    "deduplicated_manual_audit_set": "native100_manual_audit_set.csv",
}


def _read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{Path(path).name} must contain a JSON object")
    json.dumps(value, allow_nan=False)
    return value


def _generation_loader():
    # Reuse the generation command's CPU integrity checks, never its run function.
    spec = importlib.util.spec_from_file_location("_native100_generation_validation", REPO_ROOT / "tools/20_generate_native100_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_native100_generated_pairs


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
            raise ValueError(f"{label} configuration differs from frozen Step 2D1 settings")
    if ("extract_resize" not in metadata or metadata["extract_resize"] is not None
            or "registration" not in metadata or metadata["registration"] is not None
            or metadata.get("config", {}).get("seed") != 42):
        raise ValueError("native100 fidelity requires seed 42, resize=None, registration=None")


def _validate_curated_paths(curated_dir, inputs):
    """Permit only the three new named audit figures, protecting historical files."""
    path = Path(curated_dir).expanduser().absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("curated figure directory must not use symlinks")
    root = path.resolve()
    forbidden = [REPO_ROOT / name for name in ("salad", "third_party", "data", "cache", "outputs")]
    if any(root.is_relative_to(folder.resolve()) for folder in forbidden):
        raise ValueError("curated scientific figures must not overwrite runtime or protected data")
    destinations = [root / name for name in CURATED_NAMES]
    if any(destination.is_symlink() or destination in inputs for destination in destinations):
        raise ValueError("curated scientific figures must not overwrite their inputs or symlinks")
    return destinations


def run_native100_fidelity(manifest_path, generation_csv, generation_summary_path,
                           step2c_snapshot_path, step2d0_snapshot_path, step2d1_snapshot_path,
                           output_dir, plot_dir, *, device="auto", curated_dir=Path("docs/audits"),
                           population_manifest_path=POPULATION_MANIFEST,
                           population_summary_path=POPULATION_SUMMARY,
                           matcher_factory=LocalFidelityMatcher):
    (manifest_path, generation_csv, generation_summary_path, step2c_snapshot_path,
     step2d0_snapshot_path, step2d1_snapshot_path, output_dir, plot_dir,
     population_manifest_path, population_summary_path) = [
        Path(path).expanduser().resolve() for path in (
            manifest_path, generation_csv, generation_summary_path, step2c_snapshot_path,
            step2d0_snapshot_path, step2d1_snapshot_path, output_dir, plot_dir,
            population_manifest_path, population_summary_path,
        )
    ]
    rows = load_native100_manifest(manifest_path, population_manifest_path, population_summary_path)
    outputs, generation_metadata = _generation_loader()(
        rows, generation_csv, generation_summary_path, manifest_path, step2d1_snapshot_path,
    )
    provenance = _read_object(step2d1_snapshot_path)["provenance"]
    if (provenance.get("fidelity_seed") != 42 or provenance.get("resize") is not None
            or provenance.get("registration") is not None
            or provenance.get("frozen_step2c_snapshot_sha256") != sha256(step2c_snapshot_path)
            or provenance.get("frozen_step2d0_snapshot_sha256") != sha256(step2d0_snapshot_path)
            or provenance.get("aliked_config", {}).get("max_num_keypoints") != 2048
            or provenance.get("aliked_config", {}).get("detection_threshold") != .2
            or any(provenance.get("lightglue_config", {}).get(name) != expected for name, expected in {
                "features": "aliked", "depth_confidence": -1, "width_confidence": -1,
                "filter_threshold": .1, "mp": False,
            }.items())):
        raise ValueError("native100 matching must preserve frozen Step 2D1 provenance and settings")
    inputs = [manifest_path, generation_csv, generation_summary_path,
              step2c_snapshot_path, step2d0_snapshot_path, step2d1_snapshot_path,
              population_manifest_path, population_summary_path]
    inputs += [row[key] for row in rows for key in ("original_path", "source_path")]
    inputs += list(outputs.values())
    destinations = [output_dir / name for name in (*REPORT_NAMES, *TAIL_NAMES.values())]
    destinations += [plot_dir / name for name in PLOT_NAMES]
    validate_native100_destinations(destinations, inputs)
    curated = _validate_curated_paths(curated_dir, inputs) if curated_dir is not None else []
    if set(curated).intersection(destinations):
        raise ValueError("curated and runtime figure paths must be distinct")
    for row in rows:
        validate_png(row["source_path"], NATIVE_SIZE)
        validate_png(outputs[source_identity(row)], NATIVE_SIZE)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.parent.mkdir(parents=True, exist_ok=True)
    config = LocalFidelityConfig(device=device, max_keypoints=2048, seed=42)
    records = []
    with tempfile.TemporaryDirectory(prefix=".native100-fidelity-", dir=output_dir.parent) as temporary, \
            tempfile.TemporaryDirectory(prefix=".native100-plots-", dir=plot_dir.parent) as plot_temporary:
        staging, plot_staging = Path(temporary), Path(plot_temporary)
        matcher = matcher_factory(config)
        settings_validated = False
        for row in rows:
            identity = source_identity(row)
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
                record = {key: value for key, value in row.items() if not isinstance(value, Path)}
                record.update(mode="full_scene", **compute_native100_metrics(matches),
                              **compute_intervention_strength(row["source_path"], outputs[identity]))
                record.update(inherited_gate(record))
                records.append(record)
                print(f"audit {identity[0]} | row {identity[1]} | matches={record['num_matches']} | "
                      f"R8={record['repeatability_8px']:.6f} | D95={record['displacement_q95']} | "
                      f"inherited gate={'PASS' if record['inherited_pilot_gate_pass'] else 'FAIL'}", flush=True)
                del matches
            finally:
                del source_features
        distributions = summarize_native100(records)
        columns = list(records[0])
        if any(list(record) != columns for record in records):
            raise ValueError("native100 records must retain identical provenance columns")
        write_csv(staging / REPORT_NAMES[0], columns, records)
        tail_columns = list(distributions["bottom_tail"]["lowest_R8"][0])
        for table, name in TAIL_NAMES.items():
            write_csv(staging / name, tail_columns, distributions["bottom_tail"][table])
        plot_fidelity_distribution(records, plot_staging / PLOT_NAMES[0])
        plot_intervention_strength(records, plot_staging / PLOT_NAMES[1])
        plot_worst_contact_sheet(records, rows, outputs, plot_staging / PLOT_NAMES[2])
        summary = {
            "number_of_sources": 100, "count_pairs": len(records),
            "config": {**asdict(config), "policy": POLICY, "mode": "full_scene", "resize": None,
                       "extract_resize": None, "registration": None, "grid_shape": [8, 8],
                       "grid_coordinates": "normalized canonical source-image coordinates",
                       "displacement_units": "original-image pixels; native scale exactly 1.0",
                       "epsilons_pixels": [2, 4, 8, 16], "epsilon_comparison": "inclusive <=",
                       "summary_quantiles": "numpy linear interpolation; equal weight per source",
                       "feature_retention": "one source/relit pair; discard before next source",
                       "zero_match_displacements": "null; missing counts explicit; inherited gate fails; D95 tail ranks undefined first",
                       "tail_tie_break": ["audit_index", "row_index"],
                       "manual_audit_priority": ["gate_failures", "lowest_R8", "highest_D95", "lowest_coverage8"],
                       "luminance_definition": LUMA_DEFINITION,
                       "pixel_diagnostic_role": "diagnostic only; no pass/fail threshold or CounterMine mining score",
                       "intervention": "fixed shared illumination intervention"},
            "native_geometry": {"policy": POLICY, "canonical_width": 640, "canonical_height": 480,
                                "scale_x": 1.0, "scale_y": 1.0,
                                "retained_area_fraction": 1.0, "retained_long_axis_fraction": 1.0},
            "matcher": matcher.runtime_metadata(),
            "manifest_reference": reference(manifest_path), "manifest_sha256": sha256(manifest_path),
            "generation_csv_reference": reference(generation_csv), "generation_csv_sha256": sha256(generation_csv),
            "generation_summary_reference": reference(generation_summary_path),
            "generation_summary_sha256": sha256(generation_summary_path),
            "step2c_snapshot_sha256": sha256(step2c_snapshot_path),
            "step2d0_snapshot_sha256": sha256(step2d0_snapshot_path),
            "step2d1_snapshot_sha256": sha256(step2d1_snapshot_path),
            "source_population_provenance": generation_metadata["source_population_provenance"],
            "generation_config": generation_metadata["config"],
            "plots": {name: reference(plot_dir / name) for name in PLOT_NAMES},
            "curated_plots": {name: reference(path) for name, path in zip(PLOT_NAMES, curated)},
            **distributions,
        }
        write_json(staging / REPORT_NAMES[1], summary)
        artifacts = [(staging / name, output_dir / name) for name in (*REPORT_NAMES, *TAIL_NAMES.values())]
        artifacts += [(plot_staging / name, plot_dir / name) for name in PLOT_NAMES]
        for name, destination in zip(PLOT_NAMES, curated):
            staged_copy = staging / f"curated_{name}"
            shutil.copyfile(plot_staging / name, staged_copy)
            artifacts.append((staged_copy, destination))
        publish(artifacts, staging)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/native100_audit/native100_manifest.csv"))
    parser.add_argument("--generation-csv", type=Path, default=Path("cache/native100_audit/native100_generation.csv"))
    parser.add_argument("--generation-summary", type=Path, default=Path("cache/native100_audit/native100_generation_summary.json"))
    parser.add_argument("--step2c-snapshot", type=Path, default=Path("docs/audits/step2c_metrics.json"))
    parser.add_argument("--step2d0-snapshot", type=Path, default=Path("docs/audits/step2d0_metrics.json"))
    parser.add_argument("--step2d1-snapshot", type=Path, default=Path("docs/audits/step2d1_native_fov_metrics.json"))
    parser.add_argument("--population-manifest", type=Path, default=POPULATION_MANIFEST)
    parser.add_argument("--population-summary", type=Path, default=POPULATION_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=Path("cache/native100_audit"))
    parser.add_argument("--plot-dir", type=Path, default=Path("outputs/step2d2"))
    parser.add_argument("--curated-dir", type=Path, default=Path("docs/audits"))
    parser.add_argument("--device", default="auto")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run_native100_fidelity(
            args.manifest, args.generation_csv, args.generation_summary,
            args.step2c_snapshot, args.step2d0_snapshot, args.step2d1_snapshot,
            args.output_dir, args.plot_dir, device=args.device, curated_dir=args.curated_dir,
            population_manifest_path=args.population_manifest, population_summary_path=args.population_summary,
        )
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, TypeError) as error:
        parser.exit(1, f"error: {error}\n")
    gate = summary["gate"]
    interval = gate["wilson_95"]
    print(f"Measured all {summary['count_pairs']} native pairs; no records filtered. "
          f"Inherited gate: {gate['pass_count']} pass / {gate['fail_count']} fail; "
          f"acceptance={gate['acceptance_fraction']:.3f}, Wilson95=[{interval['lower']:.4f},{interval['upper']:.4f}].")


if __name__ == "__main__":
    main()
