#!/usr/bin/env python3
"""Export validated Step 2D0.5 scalar results without loading images or models.

Saved statistics are copied exactly. CSV identities, counts, paired deltas and
summary statistics are checked before atomic publication. Only the frozen Step
2C snapshot SHA256 is retained; image hashes and matched coordinates are omitted.
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
POLICIES = ("square_crop_512", "full_fov_512")
REASON = "Not directly comparable across geometry policies because canonical scale differs."
QUANTILES = {"median": .5, "q05": .05, "q25": .25, "q75": .75, "q95": .95}
COUNTS = ("num_keypoints_source", "num_keypoints_relit", "num_matches")
FAIR_METRICS = (
    "num_matches", "match_ratio_min",
    *(f"repeatability_original_{epsilon}px" for epsilon in (2, 4, 8, 16)),
    "displacement_original_median", "displacement_original_q95",
    "grid_coverage_4px", "grid_coverage_8px",
)
ORIGINAL_METRICS = (
    *(f"repeatability_original_{epsilon}px" for epsilon in (2, 4, 8, 16)),
    *(f"precision_original_{epsilon}px" for epsilon in (2, 4, 8, 16)),
    *(f"displacement_original_{stat}" for stat in ("mean", "median", "q75", "q90", "q95", "max")),
)
LEGACY_METRICS = (
    *COUNTS, "match_ratio_min",
    *(f"repeatability_min_{epsilon}px" for epsilon in (2, 4, 8, 16)),
    *(f"precision_matches_{epsilon}px" for epsilon in (2, 4, 8, 16)),
    "displacement_median", "displacement_q95", "grid_coverage_4px", "grid_coverage_8px",
)
DELTAS = {
    "delta_R4_original": "repeatability_original_4px",
    "delta_R8_original": "repeatability_original_8px",
    "delta_displacement_original_q95": "displacement_original_q95",
    "delta_grid_coverage_8": "grid_coverage_8px",
}
LEGACY_DELTAS = {
    "delta_R4": "repeatability_min_4px", "delta_R8": "repeatability_min_8px",
    "delta_displacement_q95": "displacement_q95", "delta_grid_coverage_8": "grid_coverage_8px",
}
GEOMETRY_FIELDS = (
    "canonical_width", "canonical_height", "retained_area_fraction",
    "retained_long_axis_fraction", "scale_x", "scale_y",
)


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _required(mapping, key, label):
    if key not in mapping:
        raise ValueError(f"{label} is missing required field {key}")
    return mapping[key]


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be finite")
    return value


def _csv_integer(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"\d+", value):
        raise ValueError(f"{label} must be a nonnegative integer")
    return _integer(int(value), label)


def _csv_number(value, label, *, undefined=False):
    if value is None or not value.strip():
        if undefined:
            return None
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be finite") from error
    _finite(result, label)
    if undefined:
        raise ValueError(f"{label} must be undefined when there are no matches")
    return result


def _read_json(path):
    return _object(json.loads(Path(path).read_text(encoding="utf-8")), Path(path).name)


def _read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or set(required).difference(columns):
            raise ValueError(f"{Path(path).name}: missing or duplicate CSV columns")
        rows = list(reader)
    if any(None in row or any(row[key] is None for key in required) for row in rows):
        raise ValueError(f"{Path(path).name}: malformed CSV rows")
    return rows


def _identity(row):
    return row["audit_index"], row["row_index"]


def _sources(identities, label):
    identities = set(identities)
    if (len(identities) != 10 or len({value[0] for value in identities}) != 10
            or len({value[1] for value in identities}) != 10):
        raise ValueError(f"{label} must contain exactly 10 distinct sources")
    return identities


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{label} is inconsistent with the CSV records")
    elif not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} is inconsistent with the CSV records")


def _quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _manifest(path):
    required = ("audit_index", "row_index", "image_id", "policy", "source_path", "original_path",
                "original_width", "original_height", *GEOMETRY_FIELDS)
    rows = _read_csv(path, required)
    lookup = {}
    for row in rows:
        for key in ("audit_index", "row_index", "original_width", "original_height",
                    "canonical_width", "canonical_height"):
            row[key] = _csv_integer(row[key], f"manifest.{key}")
        if row["policy"] not in POLICIES or not row["image_id"]:
            raise ValueError("manifest requires known policies and image IDs")
        for name in ("image_id", "source_path", "original_path"):
            _validate_serializable(row[name], f"manifest.{name}")
            if not row[name]:
                raise ValueError(f"manifest.{name} is required")
        for name in GEOMETRY_FIELDS[2:]:
            row[name] = _csv_number(row[name], f"manifest.{name}")
        key = (*_identity(row), row["policy"])
        if key in lookup:
            raise ValueError("manifest duplicates a source/policy pair")
        lookup[key] = row
        ow, oh = row["original_width"], row["original_height"]
        width, height = row["canonical_width"], row["canonical_height"]
        if min(ow, oh) < 1 or min(width, height) < 256 or width % 64 or height % 64 or max(width, height) != 512:
            raise ValueError("manifest contains invalid image geometry")
        if abs(row["scale_x"] - row["scale_y"]) > 1e-12:
            raise ValueError("anisotropic scale is not supported")
        short = min(ow, oh)
        if row["policy"] == POLICIES[0]:
            if (width, height) != (512, 512):
                raise ValueError("square policy must use 512x512")
            expected = (short ** 2 / (ow * oh), short / max(ow, oh), 512 / short, 512 / short)
        else:
            if width * oh != height * ow:
                raise ValueError("full-FOV policy must preserve aspect ratio")
            expected = (1.0, 1.0, width / ow, height / oh)
        for name, value in zip(GEOMETRY_FIELDS[2:], expected):
            _same(row[name], value, f"manifest.{name}")
    identities = _sources((_identity(row) for row in rows), "manifest")
    if len(rows) != 20 or set(lookup) != {(*identity, policy) for identity in identities for policy in POLICIES}:
        raise ValueError("manifest must contain exactly 20 pairs with both policies")
    for identity in identities:
        pair = [lookup[(*identity, policy)] for policy in POLICIES]
        if any(pair[0][name] != pair[1][name] for name in
               ("image_id", "original_path", "original_width", "original_height")):
            raise ValueError("paired policies must identify the same original image")
    originals = [lookup[(*identity, POLICIES[0])] for identity in identities]
    if len({row["image_id"] for row in originals}) != 10 or len({row["original_path"] for row in originals}) != 10:
        raise ValueError("manifest must identify ten distinct original images")
    geometry = {}
    for policy in POLICIES:
        candidates = [{name: row[name] for name in GEOMETRY_FIELDS} for row in rows if row["policy"] == policy]
        if any(candidate != candidates[0] for candidate in candidates[1:]):
            raise ValueError("per-policy geometry varies; a representative geometry would be misleading")
        geometry[policy] = candidates[0]
    return lookup, identities, geometry


def _fidelity(path, manifest, identities):
    metrics = tuple(dict.fromkeys((*LEGACY_METRICS, *ORIGINAL_METRICS)))
    required = ("audit_index", "row_index", "image_id", "policy", "mode",
                "canonical_width", "canonical_height", "scale_x", "scale_y", *metrics)
    raw_rows = _read_csv(path, required)
    lookup = {}
    for raw in raw_rows:
        row = dict(raw)
        for name in ("audit_index", "row_index", "canonical_width", "canonical_height", *COUNTS):
            row[name] = _csv_integer(raw[name], f"fidelity.{name}")
        key = (*_identity(row), row["policy"])
        if key not in manifest or key in lookup or row["mode"] != "full_scene":
            raise ValueError("fidelity has duplicate, non-manifest, or non-full_scene pairs")
        for name in ("image_id", "canonical_width", "canonical_height"):
            if row[name] != manifest[key][name]:
                raise ValueError(f"fidelity.{name} disagrees with the manifest")
        for name in ("scale_x", "scale_y"):
            row[name] = _csv_number(raw[name], f"fidelity.{name}")
            _same(row[name], manifest[key][name], f"fidelity.{name}")
        for name in metrics:
            if name not in COUNTS:
                row[name] = _csv_number(raw[name], f"fidelity.{name}", undefined=(
                    name.startswith("displacement_") and row["num_matches"] == 0
                ))
            value = row[name]
            if value is not None and value < 0:
                raise ValueError(f"fidelity.{name} must be nonnegative")
            if value is not None and name.startswith(("repeatability_", "precision_", "grid_coverage_", "match_ratio_")) and value > 1:
                raise ValueError(f"fidelity.{name} must lie in [0, 1]")
        denominator = min(row["num_keypoints_source"], row["num_keypoints_relit"])
        if row["num_matches"] > denominator:
            raise ValueError("matches cannot exceed extracted keypoints")
        _same(row["match_ratio_min"], row["num_matches"] / denominator if denominator else 0.0, "match_ratio_min")
        for repeatability, precision in (("repeatability_min_", "precision_matches_"),
                                         ("repeatability_original_", "precision_original_")):
            for epsilon in (2, 4, 8, 16):
                r, p = row[f"{repeatability}{epsilon}px"], row[f"{precision}{epsilon}px"]
                if r > row["match_ratio_min"] + 1e-12:
                    raise ValueError("repeatability cannot exceed the match ratio")
                if row["num_matches"] == 0:
                    if r != 0 or p != 0:
                        raise ValueError("repeatability and precision must be zero when there are no matches")
                else:
                    _same(r * denominator, p * row["num_matches"], "repeatability/precision counts")
        lookup[key] = row
    if len(raw_rows) != 20 or set(lookup) != set(manifest):
        raise ValueError("fidelity must contain exactly 20 pairs for the manifest's ten sources")
    _sources((_identity(row) for row in lookup.values()), "fidelity")
    return lookup


def _paired_values(fidelity, identities, metrics):
    result = {}
    for identity in identities:
        square, full = (fidelity[(*identity, policy)] for policy in POLICIES)
        result[identity] = {
            delta: None if square[metric] is None or full[metric] is None else full[metric] - square[metric]
            for delta, metric in metrics.items()
        }
    return result


def _paired(path, expected):
    rows = _read_csv(path, ("audit_index", "row_index", *DELTAS))
    seen = set()
    for row in rows:
        identity = tuple(_csv_integer(row[name], f"paired.{name}") for name in ("audit_index", "row_index"))
        if identity not in expected or identity in seen:
            raise ValueError("paired CSV contains duplicate or non-manifest source identities")
        seen.add(identity)
        for name, value in expected[identity].items():
            parsed = _csv_number(row[name], f"paired.{name}", undefined=value is None)
            _same(parsed, value, f"paired.{name}")
    if len(rows) != 10 or seen != set(expected):
        raise ValueError("paired CSV must contain exactly ten manifest source identities")


def _metric_summary(values, data, label, *, displacement=False, delta=False):
    data = _object(data, label)
    valid = _integer(_required(data, "valid_count", label), f"{label}.valid_count")
    missing = _integer(_required(data, "missing_count", label), f"{label}.missing_count")
    finite_values = [value for value in values if value is not None]
    if valid != len(finite_values) or missing != len(values) - valid:
        raise ValueError(f"{label} has inconsistent summary counts")
    statistics = {"median": .5, "min": 0.0, "max": 1.0} if delta else QUANTILES
    result = {}
    for name, probability in statistics.items():
        value = _required(data, name, label)
        if value is None:
            if valid or not (displacement or delta):
                raise ValueError(f"{label}.{name} is not genuinely undefined")
        else:
            _finite(value, f"{label}.{name}")
            if not valid:
                raise ValueError(f"{label}.{name} must be null with no valid values")
        expected = _quantile(finite_values, probability) if valid else None
        _same(value, expected, f"{label}.{name}")
        result[name] = value
    result.update(valid_count=valid, missing_count=missing)
    return result


def _policy_summaries(source, fidelity, names, label):
    source = _object(source, label)
    result = {}
    for policy in POLICIES:
        data = _object(_required(source, policy, label), f"{label}.{policy}")
        if _integer(_required(data, "count_pairs", policy), f"{policy}.count_pairs") != 10:
            raise ValueError("every policy summary must contain ten pairs")
        metrics = _object(_required(data, "metrics", policy), f"{policy}.metrics")
        rows = [row for row in fidelity.values() if row["policy"] == policy]
        result[policy] = {"count_pairs": 10, "metrics": {
            name: _metric_summary([row[name] for row in rows], _required(metrics, name, label),
                                  f"{label}.{policy}.{name}", displacement=name.startswith("displacement_"))
            for name in names
        }}
    return result


def _delta_summary(source, values, names, label):
    source = _object(source, label)
    return {name: _metric_summary([row[name] for row in values.values()], _required(source, name, label),
                                  f"{label}.{name}", delta=True) for name in names}


def _legacy(summary, fidelity, legacy_values):
    metrics = _object(_required(summary, "output_pixel_metrics", "summary"), "output_pixel_metrics")
    deltas = _object(_required(summary, "legacy_output_pixel_deltas", "summary"), "legacy_output_pixel_deltas")
    gate = _object(_required(summary, "legacy_output_pixel_gate", "summary"), "legacy_output_pixel_gate")
    for label, wrapper in (("output_pixel_metrics", metrics), ("legacy_output_pixel_deltas", deltas),
                           ("legacy_output_pixel_gate", gate)):
        if wrapper.get("cross_policy_comparable") is not False or wrapper.get("reason") != REASON:
            raise ValueError(f"{label} must explicitly mark output-pixel comparisons as incomparable")
    threshold_values = {"repeatability_min_8px_min": .40, "grid_coverage_8px_min": .60,
                        "displacement_q95_max": 5.0, "diagnostic_only": True, "filters_data": False}
    if _object(_required(gate, "thresholds", "legacy gate"), "legacy gate thresholds") != threshold_values:
        raise ValueError("legacy gate thresholds must retain the previous diagnostic gate exactly")
    gate_policies = _object(_required(gate, "policies", "legacy gate"), "legacy gate policies")
    copied_gate = {}
    for policy in POLICIES:
        rows = [row for row in fidelity.values() if row["policy"] == policy]
        passed = sum(row["repeatability_min_8px"] >= .40 and row["grid_coverage_8px"] >= .60
                     and row["displacement_q95"] is not None and row["displacement_q95"] <= 5 for row in rows)
        data = _object(_required(gate_policies, policy, "legacy gate policies"), policy)
        if (_integer(_required(data, "pass_count", policy), "pass_count") != passed
                or _integer(_required(data, "total_count", policy), "total_count") != 10):
            raise ValueError("legacy gate counts disagree with the fidelity records")
        fraction = _finite(_required(data, "acceptance_fraction", policy), "acceptance_fraction")
        _same(fraction, passed / 10, "legacy gate acceptance_fraction")
        copied_gate[policy] = {"pass_count": data["pass_count"], "total_count": 10,
                               "acceptance_fraction": fraction}
    return {
        "cross_policy_comparable": False, "reason": REASON,
        "coverage": {
            "coordinate_system": "normalized canonical image coordinates", "grid_shape": [8, 8],
            "role": "spatial-distribution diagnostic only",
            "qualifying_displacement_units": "canonical output pixels",
            "qualifying_tolerances_pixels": [4, 8], "qualification_mask_changed": False,
        },
        "policies": _policy_summaries(_required(metrics, "policies", "output metrics"), fidelity,
                                      LEGACY_METRICS, "output_pixel_metrics"),
        "legacy_output_pixel_deltas": {
            "cross_policy_comparable": False, "reason": REASON,
            "metrics": _delta_summary(_required(deltas, "metrics", "legacy deltas"), legacy_values,
                                      LEGACY_DELTAS, "legacy_output_pixel_deltas"),
        },
        "legacy_output_pixel_gate": {
            "cross_policy_comparable": False, "reason": REASON,
            "thresholds": threshold_values, "policies": copied_gate,
        },
    }


def _generation(generation, summary, manifest, identities, manifest_path):
    for label, data, count_name in (("generation", generation, "count_outputs"), ("summary", summary, "count_pairs")):
        if (_integer(_required(data, "number_of_sources", label), f"{label}.number_of_sources") != 10
                or _integer(_required(data, count_name, label), f"{label}.{count_name}") != 20):
            raise ValueError(f"{label} must describe ten sources and twenty pairs")
        if _required(data, "manifest_sha256", label) != _sha256(manifest_path):
            raise ValueError(f"{label} does not match the current geometry manifest")
    frozen_sha = _required(generation, "snapshot_sha256", "generation")
    if (not isinstance(frozen_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", frozen_sha)
            or _required(summary, "snapshot_sha256", "summary") != frozen_sha):
        raise ValueError("generation and fidelity must share a valid frozen Step 2C SHA256")
    config = _object(_required(generation, "config", "generation"), "generation.config")
    if config.get("mode") != "full_scene" or config.get("policies") != list(POLICIES) or config.get("seed") != 12345:
        raise ValueError("generation must use both policies, full_scene only, and seed 12345")
    settings = _object(_required(config, "configs_by_geometry", "generation config"), "configs_by_geometry")
    expected_sizes = {f"{row['canonical_width']}x{row['canonical_height']}" for row in manifest.values()}
    if set(settings) != expected_sizes:
        raise ValueError("generation is missing a required canonical geometry configuration")
    prompts = []
    for size, values in settings.items():
        values = _object(values, size)
        if (values.get("seed") != 12345
                or size != f"{values.get('width')}x{values.get('height')}"):
            raise ValueError("generation geometry or seed differs from the frozen audit")
        prompt = _required(values, "prompt", size)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("generation prompt must be nonempty")
        prompts.append(prompt)
    if len(set(prompts)) != 1:
        raise ValueError("both geometry policies must use the same prompt")
    runs = _required(generation, "runs", "generation")
    if not isinstance(runs, list) or len(runs) != 20:
        raise ValueError("generation must contain exactly twenty run records")
    seen = set()
    for raw in runs:
        run = _object(raw, "generation run")
        identity = tuple(_integer(_required(run, name, "generation run"), name) for name in ("audit_index", "row_index"))
        key = (*identity, _required(run, "policy", "generation run"))
        if key not in manifest or key in seen or run.get("mode") != "full_scene" or run.get("seed") != 12345:
            raise ValueError("generation runs contain duplicate or non-manifest full-scene pairs")
        seen.add(key)
        for name in ("image_id", "canonical_width", "canonical_height"):
            if _required(run, name, "generation run") != manifest[key][name]:
                raise ValueError(f"generation run {name} disagrees with the manifest")
    if seen != set(manifest):
        raise ValueError("generation runs must cover both policies for every source")
    return frozen_sha, prompts[0]


def _provenance(summary, generation, frozen_sha, prompt, git_commit, repo_root):
    config = _object(_required(summary, "config", "summary"), "fidelity config")
    if config.get("seed") != 42 or config.get("mode") != "full_scene":
        raise ValueError("fidelity must retain seed 42 and full_scene mode")
    for name in ("extract_resize", "registration"):
        if _required(config, name, "fidelity config") is not None:
            raise ValueError("fidelity must use resize=None and no registration")
    matcher = _object(_required(summary, "matcher", "summary"), "matcher")
    models = _object(_required(matcher, "models", "matcher"), "matcher.models")
    aliked = {**_object(matcher.get("extractor_settings", {}), "extractor_settings"),
              **_object(_required(models, "extractor_config", "models"), "extractor_config")}
    lightglue = {**_object(matcher.get("matcher_settings", {}), "matcher_settings"),
                 **_object(_required(models, "matcher_config", "models"), "matcher_config")}
    if aliked.get("max_num_keypoints") != 2048 or aliked.get("detection_threshold") != .2:
        raise ValueError("ALIKED must retain max keypoints 2048 and detection threshold 0.2")
    for name, value in {"features": "aliked", "depth_confidence": -1, "width_confidence": -1,
                        "filter_threshold": .1, "mp": False}.items():
        if lightglue.get(name) != value:
            raise ValueError("LightGlue must retain the completed Step 2C audit settings")
    lightglue["compiled"] = _required(matcher, "compiled", "matcher")
    if lightglue["compiled"] is not False:
        raise ValueError("LightGlue compiled setting must remain false")
    if "lightglue_weights_version" in models:
        lightglue["weights_version"] = models["lightglue_weights_version"]
    if git_commit is None:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                                    check=True, capture_output=True, text=True).stdout.strip()
    if not isinstance(git_commit, str) or not git_commit.strip():
        raise ValueError("git commit must be a nonempty string")
    return {
        "git_commit": git_commit, "git_commit_scope": "repository HEAD at snapshot export",
        "frozen_step2c_snapshot_sha256": frozen_sha, "number_of_sources": 10,
        "iclight_seed": 12345, "fidelity_seed": 42, "prompt": prompt,
        "aliked_config": aliked, "lightglue_config": lightglue,
        "registration": None, "resize": None, "extract_resize": None,
        "coverage_coordinate_system": "normalized canonical image coordinates",
        "coverage_role": "spatial-distribution diagnostic only",
        "coverage_grid_shape": [8, 8],
        "coverage_qualifying_tolerances": "unchanged canonical-output-pixel displacement masks at 4 and 8 pixels",
        "paired_delta_direction": "full_fov_512 minus square_crop_512",
    }


def _validate_serializable(value, label="snapshot"):
    if isinstance(value, dict):
        for name, child in value.items():
            _validate_serializable(name, f"{label} key")
            _validate_serializable(child, f"{label}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_serializable(child, f"{label}[{index}]")
    elif isinstance(value, str):
        if (re.search(r"\bfile:", value, flags=re.I)
                or re.search(r"(?:^|[^A-Za-z0-9])[A-Za-z]:[\\/]", value)
                or re.search(r"\\\\[^\\]+\\", value)):
            raise ValueError(f"{label} contains an absolute filesystem path")
        without_urls = re.sub(r"\bhttps?://[^\s<>\"']+", "", value, flags=re.I)
        if re.search(r"(?:^|[\s=:(\[{<,;'\"`])/", without_urls):
            raise ValueError(f"{label} contains an absolute filesystem path")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _finite(value, label)
    elif value is not None and not isinstance(value, bool):
        raise ValueError(f"{label} contains an unsupported value")


def _validate_output(output, inputs, repo_root):
    resolved = Path(output).expanduser().resolve()
    if resolved.suffix.lower() != ".json":
        raise ValueError("snapshot output must be a JSON file")
    root = Path(repo_root).resolve()
    for name in ("salad", "third_party", "cache/generator_audit", "outputs/step2"):
        protected = (root / name).resolve()
        if resolved.is_relative_to(protected) or protected.is_relative_to(resolved):
            raise ValueError("snapshot output must not modify protected or frozen artifacts")
    if resolved.is_relative_to(root / "docs/audits") and resolved.name.startswith("step2c_"):
        raise ValueError("snapshot output must not overwrite frozen Step 2C evidence")
    if any(resolved == Path(path).resolve() or Path(path).resolve().is_relative_to(resolved) for path in inputs):
        raise ValueError("snapshot output must not overwrite input artifacts")
    return resolved


def export_snapshot(
    manifest_path, generation_summary_path, fidelity_csv_path, paired_csv_path,
    summary_path, output_path, *, git_commit=None, repo_root=REPO_ROOT,
):
    """Validate the completed scale-fair audit and atomically export six sections."""
    manifest, identities, geometry = _manifest(manifest_path)
    generation, summary = _read_json(generation_summary_path), _read_json(summary_path)
    fidelity = _fidelity(fidelity_csv_path, manifest, identities)
    paired = _paired_values(fidelity, identities, DELTAS)
    _paired(paired_csv_path, paired)
    frozen_sha, prompt = _generation(generation, summary, manifest, identities, manifest_path)
    if _required(summary, "generation_summary_sha256", "summary") != _sha256(generation_summary_path):
        raise ValueError("fidelity summary does not match the supplied generation summary")
    snapshot = {
        "provenance": _provenance(summary, generation, frozen_sha, prompt, git_commit, repo_root),
        "geometry": geometry,
        "fair_original_pixel_metrics": _policy_summaries(
            _required(summary, "fair_original_pixel_metrics", "summary"), fidelity,
            FAIR_METRICS, "fair_original_pixel_metrics",
        ),
        "paired_original_pixel_deltas": _delta_summary(
            _required(summary, "paired_original_pixel_deltas", "summary"), paired,
            DELTAS, "paired_original_pixel_deltas",
        ),
        "per_source": [
            {"audit_index": identity[0], "row_index": identity[1],
             "image_id": manifest[(*identity, POLICIES[0])]["image_id"],
             "policies": {policy: {
                 name: fidelity[(*identity, policy)][name]
                 for name in ("canonical_width", "canonical_height", *COUNTS[:2], *FAIR_METRICS)
             } for policy in POLICIES},
             "paired_original_pixel_deltas": paired[identity]}
            for identity in sorted(identities)
        ],
        "legacy_output_pixel_metrics": _legacy(
            summary, fidelity, _paired_values(fidelity, identities, LEGACY_DELTAS),
        ),
    }
    _validate_serializable(snapshot)
    serialized = json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
    output = _validate_output(output_path, (manifest_path, generation_summary_path, fidelity_csv_path,
                                          paired_csv_path, summary_path), repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return snapshot


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/geometry_audit/geometry_manifest.csv"))
    parser.add_argument("--generation-summary", type=Path, default=Path("cache/geometry_audit/geometry_generation_summary.json"))
    parser.add_argument("--fidelity-csv", type=Path, default=Path("cache/geometry_audit/geometry_fidelity.csv"))
    parser.add_argument("--paired-csv", type=Path, default=Path("cache/geometry_audit/geometry_paired.csv"))
    parser.add_argument("--summary", type=Path, default=Path("cache/geometry_audit/geometry_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/audits/step2d0_metrics.json"))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        snapshot = export_snapshot(args.manifest, args.generation_summary, args.fidelity_csv,
                                   args.paired_csv, args.summary, args.output)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(f"Exported {len(snapshot['per_source'])} paired sources to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
