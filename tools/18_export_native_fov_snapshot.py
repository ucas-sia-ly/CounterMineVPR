#!/usr/bin/env python3
"""Export the native-FOV scalar audit with standard-library dependencies only."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = "native_full_fov"
BASELINES = ("square_crop_512", "full_fov_512")
COUNTS = ("num_keypoints_source", "num_keypoints_relit", "num_matches")
METRICS = (*COUNTS, "match_ratio_min",
           *(f"repeatability_{epsilon}px" for epsilon in (2, 4, 8, 16)),
           "displacement_median", "displacement_q95", "grid_coverage_4px", "grid_coverage_8px")
DELTA_FIELDS = {"delta_R4": ("repeatability_4px", "repeatability_original_4px"),
                "delta_R8": ("repeatability_8px", "repeatability_original_8px"),
                "delta_D95": ("displacement_q95", "displacement_original_q95"),
                "delta_coverage8": ("grid_coverage_8px", "grid_coverage_8px")}
NATIVE_QUANTILES = {"median": .5, "q05": .05, "q25": .25, "q75": .75,
                    "q95": .95, "min": 0, "max": 1}
PAIRED_QUANTILES = {"median": .5, "q25": .25, "q75": .75, "min": 0, "max": 1}


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be finite")
    return value


def integer(value, label):
    if isinstance(value, str) and re.fullmatch(r"\d+", value):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def csv_number(value, label, undefined=False):
    if value is None or not value.strip():
        if undefined:
            return None
        raise ValueError(f"{label} must be finite")
    try:
        result = finite(float(value), label)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if undefined:
        raise ValueError(f"{label} must be undefined with zero matches")
    return result


def read_json(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"{Path(path).name} must contain a JSON object")
    return result


def read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or set(required).difference(columns):
            raise ValueError(f"{Path(path).name} has missing or duplicate columns")
        rows = list(reader)
    if any(None in row or any(row[key] is None for key in required) for row in rows):
        raise ValueError(f"{Path(path).name} contains malformed rows")
    return rows


def identity(row):
    return tuple(integer(row[key], key) for key in ("audit_index", "row_index"))


def identities(rows, label):
    lookup = {}
    for row in rows:
        key = identity(row)
        if key in lookup:
            raise ValueError(f"{label} contains duplicate source identities")
        lookup[key] = row
    if (len(lookup) != 10 or len({key[0] for key in lookup}) != 10
            or len({key[1] for key in lookup}) != 10):
        raise ValueError(f"{label} must contain exactly ten distinct sources")
    return lookup


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same(actual, expected, label):
    if actual is None or expected is None:
        valid = actual is expected
    else:
        valid = math.isclose(finite(actual, label), expected, rel_tol=1e-12, abs_tol=1e-12)
    if not valid:
        raise ValueError(f"{label} is inconsistent with the scalar records")


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def copy_summary(saved, values, label, *, paired=False):
    available = [finite(value, label) for value in values if value is not None]
    valid, missing = len(available), len(values) - len(available)
    if (integer(saved["valid_count"], label) != valid
            or integer(saved["missing_count"], label) != missing):
        raise ValueError(f"{label} summary counts disagree with records")
    result = {}
    for name, probability in (PAIRED_QUANTILES if paired else NATIVE_QUANTILES).items():
        value = saved[name]
        if value is None and valid:
            raise ValueError(f"{label}.{name} is not genuinely undefined")
        same(value, quantile(available, probability) if valid else None, f"{label}.{name}")
        result[name] = value
    if paired:
        for name, expected in (("num_positive", sum(value > 0 for value in available)),
                               ("num_zero", sum(value == 0 for value in available)),
                               ("num_negative", sum(value < 0 for value in available))):
            if integer(saved[name], label) != expected:
                raise ValueError(f"{label} sign counts disagree with records")
            result[name] = saved[name]
    result.update(valid_count=saved["valid_count"], missing_count=saved["missing_count"])
    return result


def validate_serializable(value, label="snapshot"):
    if isinstance(value, dict):
        for key, child in value.items():
            validate_serializable(key, f"{label} key")
            validate_serializable(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_serializable(child, f"{label}[{index}]")
    elif isinstance(value, str):
        if (re.search(r"\bfile:", value, re.I)
                or re.search(r"(?:^|[^A-Za-z0-9])[A-Za-z]:[\\/]", value)
                or re.search(r"\\\\[^\\]+\\", value)):
            raise ValueError(f"{label} contains an absolute filesystem path")
        without_urls = re.sub(r"\bhttps?://[^\s<>\"']+", "", value, flags=re.I)
        if re.search(r"(?:^|[\s=:(\[{<,;'\"`])/", without_urls):
            raise ValueError(f"{label} contains an absolute filesystem path")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        finite(value, label)
    elif value is not None and not isinstance(value, bool):
        raise ValueError(f"{label} contains an unsupported value")


def load_records(manifest_path, fidelity_path, historical):
    frozen = identities(historical["per_source"], "historical snapshot")
    required = ("audit_index", "row_index", "image_id", "policy", "original_width", "original_height",
                "canonical_width", "canonical_height", "scale_x", "scale_y",
                "retained_area_fraction", "retained_long_axis_fraction")
    manifest = identities(read_csv(manifest_path, required), "manifest")
    raw = identities(read_csv(fidelity_path, ("audit_index", "row_index", "image_id", "policy", "mode",
                     "canonical_width", "canonical_height", "scale_x", "scale_y", *METRICS)), "fidelity")
    if set(frozen) != set(manifest) or set(raw) != set(frozen):
        raise ValueError("native and historical source identities must agree exactly")
    records = {}
    for key, row in raw.items():
        item = manifest[key]
        for name in ("original_width", "canonical_width"):
            if integer(item[name], name) != 640:
                raise ValueError("native geometry must be exactly 640x480")
        for name in ("original_height", "canonical_height"):
            if integer(item[name], name) != 480:
                raise ValueError("native geometry must be exactly 640x480")
        for name in ("scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction"):
            if csv_number(item[name], name) != 1.0:
                raise ValueError("native scales and retained fractions must equal 1.0")
        if (row["policy"] != POLICY or item["policy"] != POLICY or row["mode"] != "full_scene"
                or row["image_id"] != item["image_id"] or row["image_id"] != frozen[key]["image_id"]
                or integer(row["canonical_width"], "width") != 640
                or integer(row["canonical_height"], "height") != 480
                or csv_number(row["scale_x"], "scale_x") != 1.0
                or csv_number(row["scale_y"], "scale_y") != 1.0):
            raise ValueError("native fidelity identity, mode or geometry mismatch")
        result = {"audit_index": key[0], "row_index": key[1], "image_id": row["image_id"]}
        for name in COUNTS:
            result[name] = integer(row[name], name)
        denominator = min(result[name] for name in COUNTS[:2])
        if result["num_matches"] > denominator:
            raise ValueError("matches cannot exceed either keypoint count")
        for name in METRICS[3:]:
            undefined = name.startswith("displacement_") and not result["num_matches"]
            value = csv_number(row[name], name, undefined)
            if value is not None and (value < 0 or (not name.startswith("displacement_") and value > 1)):
                raise ValueError(f"invalid native metric {name}")
            result[name] = value
        same(result["match_ratio_min"], result["num_matches"] / denominator if denominator else 0, "match_ratio_min")
        for epsilon in (2, 4, 8, 16):
            if result[f"repeatability_{epsilon}px"] > result["match_ratio_min"] + 1e-12:
                raise ValueError("repeatability cannot exceed match ratio")
        for policy in BASELINES:
            if policy not in frozen[key]["policies"]:
                raise ValueError("historical snapshot is missing a required geometry policy")
        records[key] = result
    return records, frozen


def paired_comparisons(records, frozen, paired_path, summary):
    csv_rows = identities(read_csv(paired_path, ("audit_index", "row_index",
        *(f"{name}_vs_{policy}" for policy in BASELINES for name in DELTA_FIELDS
          if policy == BASELINES[0] or name != "delta_coverage8"))), "paired CSV")
    if set(csv_rows) != set(records):
        raise ValueError("paired source identities disagree with native records")
    comparisons = {}
    for policy in BASELINES:
        fields = {name: pair for name, pair in DELTA_FIELDS.items()
                  if policy == BASELINES[0] or name != "delta_coverage8"}
        rows = []
        for key in sorted(records):
            row = {"audit_index": key[0], "row_index": key[1]}
            baseline = frozen[key]["policies"][policy]
            for name, (native_name, original_name) in fields.items():
                native, original = records[key][native_name], baseline[original_name]
                if original is not None:
                    finite(original, f"historical {original_name}")
                elif not original_name.startswith("displacement_") or baseline["num_matches"]:
                    raise ValueError("historical metric is not genuinely undefined")
                expected = None if native is None or original is None else native - original
                value = csv_number(csv_rows[key][f"{name}_vs_{policy}"], name, expected is None)
                same(value, expected, name)
                row[name] = value
            rows.append(row)
        comparisons[policy] = {
            "delta_direction": f"native_full_fov minus {policy}",
            "displacement_units": "original-image pixels",
            "summary": {name: copy_summary(summary["paired_comparisons"][policy][name],
                                           [row[name] for row in rows], f"{policy}.{name}", paired=True)
                        for name in fields}, "per_source": rows,
        }
    return comparisons


def provenance_and_runtime(generation, summary, historical, step2c, records, manifest_path,
                           generation_path, historical_path, step2c_path, git_commit, repo_root):
    hashes = {"manifest_sha256": sha256(manifest_path), "historical_snapshot_sha256": sha256(historical_path),
              "step2c_snapshot_sha256": sha256(step2c_path)}
    if historical["provenance"]["frozen_step2c_snapshot_sha256"] != hashes["step2c_snapshot_sha256"]:
        raise ValueError("historical geometry snapshot does not match frozen Step 2C")
    for name in ("aliked_config", "lightglue_config"):
        if historical["provenance"][name] != step2c["provenance"][name]:
            raise ValueError(f"historical {name} differs from frozen Step 2C")
    for label, data, count_name in (("generation", generation, "count_outputs"), ("fidelity", summary, "count_pairs")):
        if integer(data["number_of_sources"], label) != 10 or integer(data[count_name], label) != 10:
            raise ValueError(f"{label} must describe exactly ten native pairs")
        if any(data[name] != value for name, value in hashes.items()):
            raise ValueError(f"{label} provenance does not match the supplied artifacts")
    if summary["generation_summary_sha256"] != sha256(generation_path):
        raise ValueError("fidelity generation-summary hash is inconsistent")
    generation_config = generation["config"]
    config = generation_config["iclight_config"]
    if (generation_config["mode"] != "full_scene" or generation_config["policy"] != POLICY
            or generation_config["seed"] != 12345 or (config["width"], config["height"]) != (640, 480)):
        raise ValueError("native generation mode, seed or dimensions changed")
    for name, value in step2c["provenance"]["iclight_config"].items():
        if name not in ("width", "height", "device") and config.get(name) != value:
            raise ValueError(f"IC-Light setting differs from frozen Step 2C: {name}")
    for name, value in {"seed": 12345, "cfg": 2, "steps": 25, "highres_scale": 1.0, "highres_denoise": .5}.items():
        if config.get(name) != value:
            raise ValueError(f"native generation must retain {name}={value}")
    fidelity_config = summary["config"]
    if (fidelity_config["seed"] != 42 or fidelity_config["mode"] != "full_scene"
            or fidelity_config["policy"] != POLICY or fidelity_config["grid_shape"] != [8, 8]
            or fidelity_config["extract_resize"] is not None
            or fidelity_config["registration"] is not None):
        raise ValueError("fidelity seed, resize or registration changed")
    matcher = summary["matcher"]
    models = matcher["models"]
    aliked = {**matcher["extractor_settings"], **models["extractor_config"]}
    lightglue = {**matcher["matcher_settings"], **models["matcher_config"], "compiled": matcher["compiled"]}
    if "lightglue_weights_version" in models:
        lightglue["weights_version"] = models["lightglue_weights_version"]
    for label, actual in (("aliked_config", aliked), ("lightglue_config", lightglue)):
        if actual != historical["provenance"][label]:
            raise ValueError(f"{label} differs from frozen historical geometry audit")
    runs = identities(generation["runs"], "generation runs")
    if set(runs) != set(records):
        raise ValueError("generation-run identities differ from native sources")
    runtime_rows = []
    for key in sorted(runs):
        run = runs[key]
        if (run["policy"] != POLICY or run["mode"] != "full_scene" or run["seed"] != 12345
                or run["image_id"] != records[key]["image_id"]
                or (run["canonical_width"], run["canonical_height"]) != (640, 480)):
            raise ValueError("generation run identity, mode or dimensions changed")
        elapsed = finite(run["elapsed_seconds"], "generation elapsed_seconds")
        if elapsed < 0:
            raise ValueError("generation runtime must be nonnegative")
        runtime = {"audit_index": key[0], "row_index": key[1], "elapsed_seconds": elapsed}
        for name in ("peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes"):
            value = run[name]
            runtime[name] = None if value is None else integer(value, name)
            if value is None and str(config.get("device", "")).startswith("cuda"):
                raise ValueError("CUDA run must record peak memory")
        runtime_rows.append(runtime)
    elapsed = finite(generation["elapsed_seconds"], "generation total elapsed_seconds")
    if elapsed < 0:
        raise ValueError("generation runtime must be nonnegative")
    if git_commit is None:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
                                    capture_output=True, text=True).stdout.strip()
    if not isinstance(git_commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", git_commit):
        raise ValueError("git_commit must identify the repository commit")
    scientific_config = {name: value for name, value in config.items()
                         if name not in ("device", "checkpoint_path", "local_files_only", "iclight_root", "cache_dir")
                         and not name.startswith("rmbg_")}
    provenance = {"git_commit": git_commit, "git_commit_scope": "repository HEAD at snapshot export",
                  "frozen_step2c_snapshot_sha256": hashes["step2c_snapshot_sha256"],
                  "frozen_step2d0_snapshot_sha256": hashes["historical_snapshot_sha256"],
                  "number_of_sources": 10, "iclight_seed": 12345, "fidelity_seed": 42,
                  "prompt": config["prompt"], "iclight_config": scientific_config,
                  "aliked_config": aliked, "lightglue_config": lightglue,
                  "rmbg_used": False, "resize": None, "extract_resize": None, "registration": None,
                  "displacement_units": "original-image pixels (native scale is 1.0)",
                  "coverage_coordinate_system": "normalized canonical source-image coordinates",
                  "coverage_grid_shape": [8, 8], "coverage_role": "spatial-distribution diagnostic only",
                  "historical_coverage_qualification": "unchanged historical canonical-output-pixel masks",
                  "two_stage_inference": True,
                  "inference_geometry": "both conditioning/decoded/refinement stages exactly 640x480; latents 80x60",
                  "policy_selection": None, "significance_test": None}
    runtime = {"elapsed_seconds": elapsed, "per_source": runtime_rows}
    for name in ("peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes"):
        values = [row[name] for row in runtime_rows if row[name] is not None]
        runtime[name] = max(values) if values else None
    return provenance, runtime


def export_snapshot(manifest_path, generation_summary_path, fidelity_csv_path, paired_csv_path,
                    summary_path, historical_snapshot_path, step2c_snapshot_path, output_path,
                    *, git_commit=None, repo_root=REPO_ROOT):
    """Validate exact scalar provenance and publish a compact portable snapshot."""
    historical, step2c = read_json(historical_snapshot_path), read_json(step2c_snapshot_path)
    generation, summary = read_json(generation_summary_path), read_json(summary_path)
    records, frozen = load_records(manifest_path, fidelity_csv_path, historical)
    comparisons = paired_comparisons(records, frozen, paired_csv_path, summary)
    provenance, runtime = provenance_and_runtime(generation, summary, historical, step2c, records,
        manifest_path, generation_summary_path, historical_snapshot_path, step2c_snapshot_path, git_commit, repo_root)
    output = {
        "provenance": provenance,
        "native_geometry": {"policy": POLICY, "original_width": 640, "original_height": 480,
                            "output_width": 640, "output_height": 480, "scale_x": 1.0, "scale_y": 1.0,
                            "retained_area_fraction": 1.0, "retained_long_axis_fraction": 1.0,
                            "crop": None, "resize": None, "padding": None, "exif_transpose": False},
        "generation_runtime": runtime,
        "per_source_native_metrics": [records[key] for key in sorted(records)],
        "summary_native_metrics": {name: copy_summary(summary["native_metrics"][name],
            [row[name] for row in records.values()], name) for name in METRICS},
        "paired_comparison_vs_square_crop_512": comparisons[BASELINES[0]],
        "paired_comparison_vs_full_fov_512": comparisons[BASELINES[1]],
    }
    validate_serializable(output)
    serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
    destination, root = Path(output_path).resolve(), Path(repo_root).resolve()
    inputs = (manifest_path, generation_summary_path, fidelity_csv_path, paired_csv_path,
              summary_path, historical_snapshot_path, step2c_snapshot_path)
    if destination.suffix.lower() != ".json" or any(destination == Path(path).resolve() for path in inputs):
        raise ValueError("output must be a JSON file separate from all inputs")
    for name in ("salad", "third_party", "cache/generator_audit", "cache/geometry_audit", "outputs/step2", "outputs/step2d0"):
        if destination.is_relative_to((root / name).resolve()):
            raise ValueError("output cannot modify frozen or protected artifacts")
    audits = (root / "docs/audits").resolve()
    if destination.is_relative_to(audits) and destination.name != "step2d1_native_fov_metrics.json":
        raise ValueError("output cannot overwrite curated historical audits")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in (("manifest", "cache/native_fov_audit/native_manifest.csv"),
                          ("generation-summary", "cache/native_fov_audit/native_generation_summary.json"),
                          ("fidelity-csv", "cache/native_fov_audit/native_fidelity.csv"),
                          ("paired-csv", "cache/native_fov_audit/native_paired.csv"),
                          ("summary", "cache/native_fov_audit/native_summary.json"),
                          ("snapshot", "docs/audits/step2d0_metrics.json"),
                          ("step2c-snapshot", "docs/audits/step2c_metrics.json"),
                          ("output", "docs/audits/step2d1_native_fov_metrics.json")):
        parser.add_argument(f"--{flag}", type=Path, default=Path(default))
    args = parser.parse_args(argv)
    try:
        result = export_snapshot(args.manifest, args.generation_summary, args.fidelity_csv, args.paired_csv,
                                 args.summary, args.snapshot, args.step2c_snapshot, args.output)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Exported {len(result['per_source_native_metrics'])} native sources to {args.output}")


if __name__ == "__main__":
    main()
