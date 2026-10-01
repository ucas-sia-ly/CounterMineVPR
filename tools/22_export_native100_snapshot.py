#!/usr/bin/env python3
"""Export a portable Step 2D2 scalar audit using the standard library only.

The exporter reads completed artifacts. It never loads images, ML libraries,
model weights, or CUDA, and never creates observations for an unexecuted run.
"""

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
PROMPT = "soft diffuse overcast daylight, uniform outdoor illumination, natural lighting"
LUMA_DEFINITION = (
    "Y'=0.2126 R+0.7152 G+0.0722 B on stored 8-bit RGB values, without gamma "
    "linearization; luma means/delta in [0,255] units, luma MAE divided by 255"
)
QUANTILES = {"min": 0.0, "q01": .01, "q05": .05, "q10": .10, "q25": .25,
             "median": .5, "q75": .75, "q90": .90, "q95": .95, "q99": .99, "max": 1.0}
SUMMARY_METRICS = {
    "num_matches": "num_matches", "match_ratio_min": "match_ratio_min",
    **{f"R{epsilon}": f"repeatability_{epsilon}px" for epsilon in (2, 4, 8, 16)},
    "displacement_median": "displacement_median", "displacement_q95": "displacement_q95",
    "coverage4": "grid_coverage_4px", "coverage8": "grid_coverage_8px",
}
PIXEL_METRICS = ("rgb_mae_normalized", "luma_source_mean", "luma_relit_mean",
                 "luma_mean_delta", "luma_mae_normalized")
COUNTS = ("num_keypoints_source", "num_keypoints_relit", "num_matches")
DISPLACEMENTS = tuple(f"displacement_{name}" for name in ("mean", "median", "q75", "q90", "q95", "max"))
FIDELITY_METRICS = (*COUNTS, "match_ratio_min",
                    *(f"repeatability_{epsilon}px" for epsilon in (2, 4, 8, 16)),
                    *(f"precision_{epsilon}px" for epsilon in (2, 4, 8, 16)),
                    *DISPLACEMENTS, "grid_coverage_4px", "grid_coverage_8px")
GATE_COLUMNS = ("pass_R8", "pass_coverage8", "pass_D95", "inherited_pilot_gate_pass")
FROZEN_GATE = {"repeatability_8px_min": .40, "grid_coverage_8px_min": .60,
               "displacement_q95_max": 5.0, "diagnostic_only": True, "filters_data": False}
GEOMETRY_COLUMNS = ("original_width", "original_height", "canonical_width", "canonical_height",
                    "scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction")


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


def csv_number(value, label, *, undefined=False):
    if value is None or value == "":
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


def csv_bool(value, label):
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise ValueError(f"{label} must contain a boolean")


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"{Path(path).name} contains nonfinite {value}")
    result = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError(f"{Path(path).name} must contain a JSON object")
    validate_finite_tree(result)
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
    lookup, audits, sources, images = {}, set(), set(), set()
    for row in rows:
        key = identity(row)
        image_id = row.get("image_id")
        if (key in lookup or key[0] in audits or key[1] in sources
                or not isinstance(image_id, str) or not image_id or image_id in images):
            raise ValueError(f"{label} must contain exactly 100 distinct source identities")
        lookup[key] = row
        audits.add(key[0])
        sources.add(key[1])
        images.add(image_id)
    if len(lookup) != 100:
        raise ValueError(f"{label} must contain exactly 100 distinct source identities")
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
        raise ValueError(f"{label} is inconsistent with scalar records")


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    valid = [finite(value, "distribution value") for value in values if value is not None]
    return {**{name: quantile(valid, probability) if valid else None for name, probability in QUANTILES.items()},
            "valid_count": len(valid), "missing_count": len(values) - len(valid), "total_count": len(values)}


def validate_distribution(saved, values, label):
    expected = distribution(values)
    for name, value in expected.items():
        if name.endswith("_count"):
            if integer(saved[name], f"{label}.{name}") != value:
                raise ValueError(f"{label} summary counts disagree with records")
        else:
            same(saved[name], value, f"{label}.{name}")
    return expected


def validate_finite_tree(value, label="input"):
    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        finite(value, label)


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


def selection_provenance(row):
    excluded = {"audit_index", "row_index", "image_id", "relative_path", "policy", "mode",
                "source_format", "original_format", "crop", "resize", "padding", "exif_transpose",
                *GEOMETRY_COLUMNS, *FIDELITY_METRICS, *PIXEL_METRICS, *GATE_COLUMNS}
    return {key: value for key, value in row.items() if key not in excluded
            and not key.endswith(("_path", "_sha256")) and not key.startswith("crop_")}


def compact_record(record):
    result = {name: record[name] for name in ("audit_index", "row_index", "image_id")}
    result.update(selection_provenance(record))
    for name in ("num_matches", "match_ratio_min", "displacement_median", "displacement_q95",
                 "rgb_mae_normalized", "luma_mean_delta", "luma_mae_normalized", *GATE_COLUMNS):
        result[name] = record[name]
    for alias in ("R4", "R8", "R16", "coverage4", "coverage8"):
        result[alias] = record[SUMMARY_METRICS[alias]]
    return result


def load_records(manifest_path, fidelity_path):
    manifest = identities(read_csv(manifest_path, ("audit_index", "row_index", "image_id", "policy",
                                                  *GEOMETRY_COLUMNS)), "manifest")
    raw = identities(read_csv(fidelity_path, ("audit_index", "row_index", "image_id", "policy", "mode",
                    "canonical_width", "canonical_height", "scale_x", "scale_y",
                    *FIDELITY_METRICS, *PIXEL_METRICS, *GATE_COLUMNS)), "fidelity")
    if set(raw) != set(manifest):
        raise ValueError("manifest and fidelity source identities must agree exactly")
    records = {}
    for key in sorted(raw):
        source, row = manifest[key], raw[key]
        for name in ("original_width", "canonical_width"):
            if integer(source[name], name) != 640:
                raise ValueError("native geometry must be exactly 640x480")
        for name in ("original_height", "canonical_height"):
            if integer(source[name], name) != 480:
                raise ValueError("native geometry must be exactly 640x480")
        for name in ("scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction"):
            if csv_number(source[name], name) != 1.0:
                raise ValueError("native scales and retained fractions must equal 1.0")
        if (source["policy"] != POLICY or row["policy"] != POLICY or row["mode"] != "full_scene"
                or row["image_id"] != source["image_id"]
                or integer(row["canonical_width"], "width") != 640
                or integer(row["canonical_height"], "height") != 480
                or csv_number(row["scale_x"], "scale_x") != 1.0
                or csv_number(row["scale_y"], "scale_y") != 1.0):
            raise ValueError("native fidelity identity, mode or geometry mismatch")
        if selection_provenance(row) != selection_provenance(source):
            raise ValueError("fidelity selection provenance differs from the frozen native manifest")
        record = {"audit_index": key[0], "row_index": key[1], "image_id": row["image_id"],
                  **selection_provenance(source)}
        for name in COUNTS:
            record[name] = integer(row[name], name)
        denominator = min(record[name] for name in COUNTS[:2])
        if record["num_matches"] > denominator:
            raise ValueError("matches cannot exceed either keypoint count")
        for name in FIDELITY_METRICS[3:]:
            value = csv_number(row[name], name, undefined=name in DISPLACEMENTS and not record["num_matches"])
            if value is not None and (value < 0 or (name not in DISPLACEMENTS and value > 1)):
                raise ValueError(f"invalid native metric {name}")
            record[name] = value
        same(record["match_ratio_min"], record["num_matches"] / denominator if denominator else 0, "match_ratio_min")
        for epsilon in (2, 4, 8, 16):
            if record[f"repeatability_{epsilon}px"] > record["match_ratio_min"] + 1e-12:
                raise ValueError("repeatability cannot exceed the match ratio")
            expected_precision = (record[f"repeatability_{epsilon}px"] * denominator /
                                  record["num_matches"] if record["num_matches"] else 0.0)
            same(record[f"precision_{epsilon}px"], expected_precision, f"precision_{epsilon}px")
        if any(record[f"repeatability_{first}px"] > record[f"repeatability_{second}px"]
               for first, second in ((2, 4), (4, 8), (8, 16))):
            raise ValueError("repeatability must be monotone with tolerance")
        for name in PIXEL_METRICS:
            record[name] = csv_number(row[name], name)
            lower = -255.0 if name == "luma_mean_delta" else 0.0
            upper = 1.0 if name.endswith("normalized") else 255.0
            if not lower <= record[name] <= upper:
                raise ValueError(f"{name} is outside its valid range")
        same(record["luma_mean_delta"], record["luma_relit_mean"] - record["luma_source_mean"], "luma_mean_delta")
        for name in GATE_COLUMNS:
            record[name] = csv_bool(row[name], name)
        expected = {"pass_R8": record["repeatability_8px"] >= .40,
                    "pass_coverage8": record["grid_coverage_8px"] >= .60,
                    "pass_D95": record["displacement_q95"] is not None and record["displacement_q95"] <= 5.0}
        expected["inherited_pilot_gate_pass"] = all(expected.values())
        if any(record[name] is not value for name, value in expected.items()):
            raise ValueError("inherited frozen gate component flags disagree with records")
        records[key] = record
    return records


def wilson_interval(pass_count, total):
    z = 1.959963984540054
    fraction, squared = pass_count / total, z * z
    denominator = 1 + squared / total
    center = (fraction + squared / (2 * total)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / total + squared / (4 * total * total)) / denominator
    return {"lower": 0.0 if pass_count == 0 else max(0.0, center - radius),
            "upper": 1.0 if pass_count == total else min(1.0, center + radius),
            "confidence_level": .95, "z": z}


def gate_summary(records):
    passed = sum(row["inherited_pilot_gate_pass"] for row in records)
    return {"name": "inherited_pilot_gate_pass", "thresholds": dict(FROZEN_GATE),
            "pass_count": passed, "fail_count": len(records) - passed,
            "acceptance_fraction": passed / len(records), "wilson_95": wilson_interval(passed, len(records))}


def bottom_tail_tables(records):
    identity_key = lambda row: (row["audit_index"], row["row_index"])
    low_r8 = sorted(records, key=lambda row: (row["repeatability_8px"], *identity_key(row)))[:20]
    high_d95 = sorted(records, key=lambda row: (row["displacement_q95"] is not None,
                      -row["displacement_q95"] if row["displacement_q95"] is not None else 0,
                      *identity_key(row)))[:20]
    low_coverage = sorted(records, key=lambda row: (row["grid_coverage_8px"], *identity_key(row)))[:20]
    failures = sorted((row for row in records if not row["inherited_pilot_gate_pass"]), key=identity_key)
    union, seen = [], set()
    for row in (*failures, *low_r8, *high_d95, *low_coverage):
        key = identity_key(row)
        if key not in seen:
            seen.add(key)
            union.append(row)
    return {name: [compact_record(row) for row in rows] for name, rows in (
        ("lowest_R8", low_r8), ("highest_D95", high_d95), ("lowest_coverage8", low_coverage),
        ("gate_failures", failures), ("deduplicated_manual_audit_set", union))}


def compare_gate_and_tails(summary, records):
    gate = gate_summary(records)
    thresholds = summary["gate"]["thresholds"]
    if (thresholds != FROZEN_GATE or thresholds["diagnostic_only"] is not True
            or thresholds["filters_data"] is not False):
        raise ValueError("inherited gate thresholds must remain frozen")
    if summary["gate"]["name"] != gate["name"]:
        raise ValueError("inherited gate summary disagrees with records")
    for name in ("pass_count", "fail_count"):
        if integer(summary["gate"][name], name) != gate[name]:
            raise ValueError("inherited gate summary disagrees with records")
    same(summary["gate"]["acceptance_fraction"], gate["acceptance_fraction"], "gate.acceptance_fraction")
    for name, value in gate["wilson_95"].items():
        same(summary["gate"]["wilson_95"][name], value, f"gate.wilson_95.{name}")
    tails = bottom_tail_tables(records)
    for name, rows in tails.items():
        saved = summary["bottom_tail"][name]
        if ([identity(row) for row in saved] != [identity(row) for row in rows]
                or any(row["image_id"] != expected["image_id"] for row, expected in zip(saved, rows))):
            raise ValueError(f"bottom_tail.{name} identities or deterministic order disagree with records")
    return gate, tails


def scientific_config(config):
    return {name: value for name, value in config.items()
            if name not in ("device", "checkpoint_path", "local_files_only", "iclight_root", "cache_dir")
            and not name.startswith("rmbg_")}


def validate_provenance(generation, summary, snapshots, snapshot_paths, records, manifest_path,
                        generation_path, git_commit, repo_root):
    step2c, step2d0, step2d1 = snapshots
    hashes = {f"step2{name}_snapshot_sha256": sha256(path)
              for name, path in zip(("c", "d0", "d1"), snapshot_paths)}
    if (step2d0["provenance"]["frozen_step2c_snapshot_sha256"] != hashes["step2c_snapshot_sha256"]
            or step2d1["provenance"]["frozen_step2c_snapshot_sha256"] != hashes["step2c_snapshot_sha256"]
            or step2d1["provenance"]["frozen_step2d0_snapshot_sha256"] != hashes["step2d0_snapshot_sha256"]):
        raise ValueError("frozen historical snapshot chain is inconsistent")
    hashes["manifest_sha256"] = sha256(manifest_path)
    for label, data, count_name in (("generation", generation, "count_outputs"), ("fidelity", summary, "count_pairs")):
        if integer(data["number_of_sources"], label) != 100 or integer(data[count_name], label) != 100:
            raise ValueError(f"{label} must describe exactly 100 native pairs")
        if any(data[name] != value for name, value in hashes.items()):
            raise ValueError(f"{label} provenance does not match supplied artifacts")
    if summary["generation_summary_sha256"] != sha256(generation_path):
        raise ValueError("fidelity generation-summary hash is inconsistent")
    config = generation["config"]
    if config["seed"] != 12345 or config["mode"] != "full_scene" or config["policy"] != POLICY:
        raise ValueError("native generation mode, seed or geometry changed")
    iclight = scientific_config(config["iclight_config"])
    if iclight != step2d1["provenance"]["iclight_config"]:
        raise ValueError("IC-Light configuration differs from frozen Step 2D1")
    if generation["rmbg_used"] is not False or generation["two_stage_inference"] is not True:
        raise ValueError("generation must preserve two-stage inference without RMBG")
    required = {"width": 640, "height": 480, "seed": 12345, "prompt": PROMPT,
                "cfg": 2.0, "steps": 25, "highres_scale": 1.0, "highres_denoise": .5}
    if any(iclight.get(name) != value for name, value in required.items()):
        raise ValueError("native full-scene generation settings must remain frozen")
    frozen_provenance = step2d1["provenance"]
    if not frozen_provenance["two_stage_inference"] or frozen_provenance["rmbg_used"]:
        raise ValueError("native pilot must preserve two-stage inference without RMBG")
    fidelity = summary["config"]
    if (fidelity["seed"] != 42 or fidelity["mode"] != "full_scene" or fidelity["policy"] != POLICY
            or fidelity["grid_shape"] != [8, 8] or fidelity["extract_resize"] is not None
            or fidelity["registration"] is not None):
        raise ValueError("fidelity seed, matcher geometry, resize or registration changed")
    if fidelity["luminance_definition"] != LUMA_DEFINITION:
        raise ValueError("pixel diagnostic luminance definition differs from measured stored-RGB luma")
    matcher = summary["matcher"]
    models = matcher["models"]
    aliked = {**matcher["extractor_settings"], **models["extractor_config"]}
    lightglue = {**matcher["matcher_settings"], **models["matcher_config"], "compiled": matcher["compiled"]}
    if "lightglue_weights_version" in models:
        lightglue["weights_version"] = models["lightglue_weights_version"]
    for name, actual in (("aliked_config", aliked), ("lightglue_config", lightglue)):
        if any(snapshot["provenance"][name] != actual for snapshot in snapshots):
            raise ValueError(f"{name} differs from frozen historical audits")
    if (aliked.get("max_num_keypoints") != 2048 or aliked.get("detection_threshold") != .2
            or any(lightglue.get(name) != value for name, value in
                   {"features": "aliked", "depth_confidence": -1, "width_confidence": -1,
                    "filter_threshold": .1, "mp": False}.items())):
        raise ValueError("local matcher settings must remain frozen")
    population = generation["source_population_provenance"]
    if not isinstance(population, dict) or integer(population.get("number_of_sources"), "population count") != 100:
        raise ValueError("source-population provenance is required")
    if summary["source_population_provenance"] != population:
        raise ValueError("fidelity source-population provenance differs from generation")
    group_counts = {name: sum(row.get("group") == name for row in records.values())
                    for name in ("random", "hard_candidate")}
    if any("group" in row for row in records.values()) and group_counts != {"random": 50, "hard_candidate": 50}:
        raise ValueError("frozen population must preserve 50 random and 50 hard-candidate sources")
    if "group_counts" in population and population["group_counts"] != {"random": 50, "hard_candidate": 50}:
        raise ValueError("frozen population provenance must preserve 50/50 group counts")
    for name in ("count_random", "count_hard_candidate"):
        if name in population and integer(population[name], name) != 50:
            raise ValueError("frozen population provenance must preserve 50/50 group counts")
    if git_commit is None:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
                                    capture_output=True, text=True).stdout.strip()
    if not isinstance(git_commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", git_commit):
        raise ValueError("git_commit must identify the repository commit")
    return {"git_commit": git_commit, "git_commit_scope": "repository HEAD at snapshot export",
            **{f"frozen_{name}": value for name, value in hashes.items() if name != "manifest_sha256"},
            "number_of_sources": 100, "source_population_provenance": population,
            "iclight_seed": 12345, "fidelity_seed": 42, "prompt": PROMPT,
            "iclight_config": iclight, "aliked_config": aliked, "lightglue_config": lightglue,
            "rmbg_used": False, "two_stage_inference": True, "resize": None, "extract_resize": None,
            "registration": None, "coverage_grid_shape": [8, 8],
            "luminance_definition": fidelity["luminance_definition"],
            "displacement_units": "original-image pixels (native scale is 1.0)",
            "intervention": "fixed shared illumination intervention",
            "probe_role": "diagnostic counterfactual probe only; never VPR optimization input"}


def generation_runtime(generation, records):
    runs = identities(generation["runs"], "generation runs")
    if set(runs) != set(records):
        raise ValueError("generation-run identities differ from native sources")
    elapsed, allocated, reserved = [], [], []
    for key in sorted(runs):
        row = runs[key]
        if (row["image_id"] != records[key]["image_id"] or row["policy"] != POLICY
                or row["mode"] != "full_scene" or row["seed"] != 12345
                or (row["canonical_width"], row["canonical_height"]) != (640, 480)):
            raise ValueError("generation-run identity, mode, seed or geometry changed")
        duration = finite(row["elapsed_seconds"], "elapsed_seconds")
        if duration < 0:
            raise ValueError("generation runtime must be nonnegative")
        elapsed.append(duration)
        for name, values in (("peak_cuda_memory_allocated_bytes", allocated),
                             ("peak_cuda_memory_reserved_bytes", reserved)):
            value = row[name]
            if value is None and str(generation["config"]["iclight_config"].get("device", "")).startswith("cuda"):
                raise ValueError("CUDA generation must record peak memory")
            values.append(None if value is None else integer(value, name))
    runtime = generation["generation_runtime"]
    for values in (allocated, reserved):
        if sum(value is None for value in values) not in (0, len(values)):
            raise ValueError("generation memory must be recorded for all sources or absent for all")
    result = {"elapsed_seconds": {name: quantile(elapsed, probability) for name, probability in
                                  {"median": .5, "q05": .05, "q25": .25, "q75": .75, "q95": .95}.items()},
              "peak_cuda_memory_allocated_bytes": max(allocated) if allocated[0] is not None else None,
              "peak_cuda_memory_reserved_bytes": max(reserved) if reserved[0] is not None else None}
    for name, value in result["elapsed_seconds"].items():
        same(runtime["elapsed_seconds"][name], value, f"generation_runtime.elapsed_seconds.{name}")
    for name in ("peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes"):
        saved_value = None if runtime[name] is None else integer(runtime[name], name)
        if saved_value != result[name]:
            raise ValueError("generation memory summary disagrees with run records")
    for name in ("model_load_elapsed_seconds", "generation_elapsed_seconds", "elapsed_seconds"):
        value = finite(generation[name], name)
        if value < 0:
            raise ValueError("generation runtime must be nonnegative")
        result["total_elapsed_seconds" if name == "elapsed_seconds" else name] = value
    same(result["generation_elapsed_seconds"], sum(elapsed), "generation_elapsed_seconds")
    result["timing_scope"] = "per-image generation excludes separately measured first-run model loading"
    return result


def publish_json(output, output_path, inputs, repo_root):
    validate_serializable(output)
    serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
    destination, root = Path(output_path).resolve(), Path(repo_root).resolve()
    if destination.suffix.lower() != ".json" or any(destination == Path(path).resolve() for path in inputs):
        raise ValueError("output must be a JSON file separate from all inputs")
    for name in ("salad", "third_party", "cache/generator_audit", "cache/geometry_audit",
                 "cache/native_fov_audit", "outputs/step2", "outputs/step2d0", "outputs/step2d1"):
        if destination.is_relative_to((root / name).resolve()):
            raise ValueError("output cannot modify frozen or protected artifacts")
    if destination.is_relative_to((root / "docs/audits").resolve()) and destination.name != "step2d2_native100_metrics.json":
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


def export_snapshot(manifest_path, generation_summary_path, fidelity_csv_path, summary_path,
                    step2c_snapshot_path, step2d0_snapshot_path, step2d1_snapshot_path, output_path,
                    *, git_commit=None, repo_root=REPO_ROOT):
    """Validate all 100 completed scalar records and atomically export them."""
    snapshots = tuple(read_json(path) for path in (step2c_snapshot_path, step2d0_snapshot_path, step2d1_snapshot_path))
    generation, summary = read_json(generation_summary_path), read_json(summary_path)
    records = load_records(manifest_path, fidelity_csv_path)
    rows = list(records.values())
    provenance = validate_provenance(generation, summary, snapshots,
        (step2c_snapshot_path, step2d0_snapshot_path, step2d1_snapshot_path), records,
        manifest_path, generation_summary_path, git_commit, repo_root)
    gate, tails = compare_gate_and_tails(summary, rows)
    geometry = {"policy": POLICY, "original_width": 640, "original_height": 480,
                "canonical_width": 640, "canonical_height": 480, "scale_x": 1.0, "scale_y": 1.0,
                "retained_area_fraction": 1.0, "retained_long_axis_fraction": 1.0,
                "crop": None, "resize": None, "padding": None, "exif_transpose": False}
    provenance["native_geometry"] = geometry
    output = {"provenance": provenance, "native_geometry": geometry,
              "generation_runtime": generation_runtime(generation, records),
              "fidelity_summary": {alias: validate_distribution(summary["fidelity_summary"][alias],
                  [row[name] for row in rows], alias) for alias, name in SUMMARY_METRICS.items()},
              "gate": gate,
              "intervention_strength": {name: validate_distribution(summary["intervention_strength"][name],
                  [row[name] for row in rows], name) for name in PIXEL_METRICS},
              "intervention_strength_definition": summary["config"]["luminance_definition"],
              "per_source_compact": [compact_record(row) for row in rows], "bottom_tail": tails,
              "undefined_displacement_policy": "null only for zero matches; retained as gate failures and ranked first in highest_D95"}
    publish_json(output, output_path, (manifest_path, generation_summary_path, fidelity_csv_path, summary_path,
                 step2c_snapshot_path, step2d0_snapshot_path, step2d1_snapshot_path), repo_root)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in (("manifest", "cache/native100_audit/native100_manifest.csv"),
                          ("generation-summary", "cache/native100_audit/native100_generation_summary.json"),
                          ("fidelity-csv", "cache/native100_audit/native100_fidelity.csv"),
                          ("summary", "cache/native100_audit/native100_fidelity_summary.json"),
                          ("step2c-snapshot", "docs/audits/step2c_metrics.json"),
                          ("step2d0-snapshot", "docs/audits/step2d0_metrics.json"),
                          ("step2d1-snapshot", "docs/audits/step2d1_native_fov_metrics.json"),
                          ("output", "docs/audits/step2d2_native100_metrics.json")):
        parser.add_argument(f"--{flag}", type=Path, default=Path(default))
    args = parser.parse_args(argv)
    try:
        result = export_snapshot(args.manifest, args.generation_summary, args.fidelity_csv, args.summary,
                                 args.step2c_snapshot, args.step2d0_snapshot, args.step2d1_snapshot, args.output)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Exported {len(result['per_source_compact'])} native audit sources to {args.output}")


if __name__ == "__main__":
    main()
