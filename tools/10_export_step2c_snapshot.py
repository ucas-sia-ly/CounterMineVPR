#!/usr/bin/env python3
"""Export completed Step 2C results without loading models or measuring images.

Fidelity and crop summaries are copied exactly from the completed audit.
Only the across-image RMBG alpha summaries are calculated here, using inclusive
linear interpolation at position (n - 1) * q, with equal weight per image.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
RELIGHT_MODES = ("official_rmbg", "full_scene")
ALL_MODES = ("identity_control", *RELIGHT_MODES)
SUMMARY_KEYS = ("median", "q05", "q25", "q75", "q95")
CROP_KEYS = ("min", *SUMMARY_KEYS)
SUMMARY_METRICS = (
    "num_matches", "match_ratio_min",
    "repeatability_min_2px", "repeatability_min_4px",
    "repeatability_min_8px", "repeatability_min_16px",
    "displacement_median", "displacement_q95",
    "grid_coverage_4px", "grid_coverage_8px",
)
COMPACT_METRICS = tuple(
    metric for metric in SUMMARY_METRICS
    if metric not in ("match_ratio_min", "repeatability_min_2px")
)
ALPHA_METRICS = (
    "alpha_mean", "alpha_q05", "alpha_q50", "alpha_q95",
    "alpha_fraction_lt_0_5", "alpha_fraction_gt_0_9",
)
ICLIGHT_CONFIG_KEYS = (
    "added_prompt", "background", "base_model", "base_revision",
    "cfg", "checkpoint_path", "device", "height", "highres_denoise",
    "highres_scale", "local_files_only", "lowres_denoise", "negative_prompt",
    "num_samples", "offset_filename", "offset_model", "offset_revision",
    "prompt", "rmbg_dtype", "rmbg_model", "rmbg_revision", "rmbg_sigma",
    "scheduler_algorithm_type", "scheduler_beta_end", "scheduler_beta_start",
    "scheduler_num_train_timesteps", "scheduler_steps_offset",
    "scheduler_use_karras_sigmas", "seed", "steps", "text_encoder_dtype",
    "unet_dtype", "vae_dtype", "width",
)


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _required(mapping, key, label):
    if key not in mapping:
        raise ValueError(f"{label} is missing required field {key}")
    return mapping[key]


def _integer(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return value


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{label} must be a finite number")
    return value


def _source_identity(row, label):
    return tuple(
        _integer(_required(row, key, label), f"{label}.{key}")
        for key in ("audit_index", "row_index")
    )


def _validate_sources(identities, label):
    sources = set(identities)
    if (len(sources) != 10 or len({item[0] for item in sources}) != 10
            or len({item[1] for item in sources}) != 10):
        raise ValueError(f"{label} must contain exactly 10 distinct sources")
    return sources


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return _mapping(json.load(handle), Path(path).name)


def _read_fidelity_csv(path):
    required = ("audit_index", "row_index", "mode", *SUMMARY_METRICS)
    rows = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError("fidelity CSV contains duplicate columns")
        if set(required).difference(columns):
            raise ValueError("fidelity CSV is missing required columns")
        for line, raw in enumerate(reader, start=2):
            label = f"fidelity CSV line {line}"
            row = {}
            for key in ("audit_index", "row_index", "num_matches"):
                try:
                    parsed = int(raw[key])
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{label}.{key} must be an integer") from error
                row[key] = _integer(parsed, f"{label}.{key}")
            row["mode"] = raw["mode"]
            if row["mode"] not in ALL_MODES:
                raise ValueError(f"{label} contains an unexpected mode")
            identity = (*_source_identity(row, label), row["mode"])
            if identity in seen:
                raise ValueError(f"{label} duplicates a source/mode row")
            seen.add(identity)
            for metric in SUMMARY_METRICS[1:]:
                value = raw[metric]
                if value is None or not value.strip():
                    if metric.startswith("displacement_") and row["num_matches"] == 0:
                        row[metric] = None
                        continue
                    raise ValueError(f"{label}.{metric} must be finite")
                try:
                    number = float(value)
                except ValueError as error:
                    raise ValueError(f"{label}.{metric} must be finite") from error
                row[metric] = _finite(number, f"{label}.{metric}")
                if metric.startswith("displacement_") and row["num_matches"] == 0:
                    raise ValueError(f"{label}.{metric} must be undefined with zero matches")
            rows.append(row)
    if len(rows) != 30:
        raise ValueError("fidelity CSV must contain exactly 30 source/mode rows")
    sources = _validate_sources(
        (_source_identity(row, "fidelity CSV") for row in rows), "fidelity CSV"
    )
    for mode in ALL_MODES:
        mode_sources = {
            _source_identity(row, "fidelity CSV") for row in rows if row["mode"] == mode
        }
        if mode_sources != sources:
            raise ValueError(f"fidelity CSV is missing sources for {mode}")
    return rows, sources


def _copy_fidelity_summaries(summary, rows):
    if _integer(_required(summary, "number_of_sources", "fidelity summary"), "fidelity.number_of_sources") != 10:
        raise ValueError("fidelity summary must contain exactly 10 sources")
    if _integer(_required(summary, "count_pairs", "fidelity summary"), "fidelity.count_pairs") != 30:
        raise ValueError("fidelity summary must contain exactly 30 pairs")
    modes = _mapping(_required(summary, "modes", "fidelity summary"), "fidelity modes")
    result = {}
    for mode in ALL_MODES:
        mode_summary = _mapping(_required(modes, mode, "fidelity modes"), mode)
        count = _integer(_required(mode_summary, "count_pairs", mode), f"{mode}.count_pairs")
        if count != 10:
            raise ValueError(f"{mode} must contain exactly 10 pairs")
        metrics = _mapping(_required(mode_summary, "metrics", mode), f"{mode}.metrics")
        copied = {}
        for metric in SUMMARY_METRICS:
            label = f"{mode}.{metric}"
            values = _mapping(_required(metrics, metric, f"{mode}.metrics"), label)
            valid = _integer(_required(values, "valid_count", label), f"{label}.valid_count")
            missing = _integer(_required(values, "missing_count", label), f"{label}.missing_count")
            expected_missing = sum(
                row[metric] is None for row in rows if row["mode"] == mode
            )
            if valid + missing != count or missing != expected_missing:
                raise ValueError(f"{label} has inconsistent valid_count/missing_count")
            copied[metric] = {}
            for key in SUMMARY_KEYS:
                value = _required(values, key, label)
                if value is None:
                    if not metric.startswith("displacement_") or valid != 0:
                        raise ValueError(f"{label}.{key} is unexpectedly undefined")
                else:
                    _finite(value, f"{label}.{key}")
                    if valid == 0:
                        raise ValueError(f"{label}.{key} must be null with no valid values")
                copied[metric][key] = value
            copied[metric].update(valid_count=valid, missing_count=missing)
        result[mode] = {"count_pairs": count, "metrics": copied}
    return result


def _quantile(sorted_values, q):
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def _rmbg_alpha(smoke, fidelity_sources):
    if _integer(_required(smoke, "number_of_sources", "smoke summary"), "smoke.number_of_sources") != 10:
        raise ValueError("smoke summary must contain exactly 10 sources")
    if _integer(_required(smoke, "count_outputs", "smoke summary"), "smoke.count_outputs") != 20:
        raise ValueError("smoke summary must contain exactly 20 outputs")
    modes = _required(smoke, "modes", "smoke summary")
    if (not isinstance(modes, list) or not all(isinstance(mode, str) for mode in modes)
            or set(modes) != set(RELIGHT_MODES)):
        raise ValueError("smoke summary is missing a relighting mode")
    runs = _required(smoke, "runs", "smoke summary")
    if not isinstance(runs, list) or len(runs) != 20:
        raise ValueError("smoke summary must contain exactly 20 runs")
    seen = {mode: set() for mode in RELIGHT_MODES}
    alpha_values = {metric: [] for metric in ALPHA_METRICS}
    for run in runs:
        run = _mapping(run, "smoke run")
        mode = _required(run, "mode", "smoke run")
        if mode not in RELIGHT_MODES:
            raise ValueError("smoke run contains an unexpected mode")
        source = _source_identity(run, "smoke run")
        if source in seen[mode]:
            raise ValueError("smoke summary duplicates a source/mode run")
        seen[mode].add(source)
        if mode == "official_rmbg":
            stats = _mapping(_required(run, "adapter_stats", "smoke run"), "adapter_stats")
            for metric in ALPHA_METRICS:
                value = _required(stats, metric, "adapter_stats")
                alpha_values[metric].append(_finite(value, f"adapter_stats.{metric}"))
    for mode, sources in seen.items():
        if sources != fidelity_sources:
            raise ValueError(f"smoke {mode} sources differ from fidelity sources")
    result = {"quantile_method": "inclusive linear interpolation; equal weight per image"}
    for metric, values in alpha_values.items():
        values.sort()
        result[metric] = {
            "min": values[0], "median": _quantile(values, 0.5),
            "q05": _quantile(values, 0.05), "q25": _quantile(values, 0.25),
            "q75": _quantile(values, 0.75), "q95": _quantile(values, 0.95),
            "max": values[-1],
        }
    return result


def _provenance(audit, smoke, fidelity, git_commit, repo_root):
    count = _integer(
        _required(audit, "number_of_images", "audit summary"),
        "audit summary.number_of_images", positive=True,
    )
    resolution = _required(audit, "canonical_resolution", "audit summary")
    if resolution != [512, 512]:
        raise ValueError("canonical resolution must be [512, 512]")
    generation = _mapping(_required(smoke, "config", "smoke summary"), "IC-Light config")
    config = _mapping(_required(fidelity, "config", "fidelity summary"), "fidelity config")
    if config.get("canonical_resolution", resolution) != resolution:
        raise ValueError("fidelity canonical resolution differs from audit resolution")
    for key, value in (("width", 512), ("height", 512)):
        if _required(generation, key, "IC-Light config") != value:
            raise ValueError("IC-Light resolution differs from canonical resolution")
    prompt = _required(generation, "prompt", "IC-Light config")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("IC-Light prompt must be a nonempty string")
    matcher = _mapping(_required(fidelity, "matcher", "fidelity summary"), "matcher")
    models = _mapping(matcher.get("models", {}), "matcher.models")
    aliked = dict(_mapping(matcher.get("extractor_settings", {}), "extractor_settings"))
    aliked.update(_mapping(models.get("extractor_config", {}), "extractor_config"))
    lightglue = dict(_mapping(matcher.get("matcher_settings", {}), "matcher_settings"))
    lightglue.update(_mapping(models.get("matcher_config", {}), "matcher_config"))
    if not aliked or not lightglue:
        raise ValueError("ALIKED and LightGlue configurations are required")
    if "compiled" in matcher:
        lightglue["compiled"] = matcher["compiled"]
    if "lightglue_weights_version" in models:
        lightglue["weights_version"] = models["lightglue_weights_version"]
    if git_commit is None:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    if not isinstance(git_commit, str) or not git_commit.strip():
        raise ValueError("git_commit must be a nonempty string")
    return {
        "git_commit": git_commit,
        "git_commit_scope": "repository HEAD at snapshot export",
        "number_of_audit_sources": 10,
        "number_of_canonical_audit_images": count,
        "number_of_fidelity_sources": 10,
        "canonical_resolution": resolution,
        "iclight_seed": _integer(_required(generation, "seed", "IC-Light config"), "iclight_seed"),
        "fidelity_seed": _integer(_required(config, "seed", "fidelity config"), "fidelity_seed"),
        "prompt": prompt,
        "iclight_config": {key: generation[key] for key in ICLIGHT_CONFIG_KEYS if key in generation},
        "aliked_config": aliked,
        "lightglue_config": lightglue,
        "extract_resize": _required(config, "extract_resize", "fidelity config"),
        "registration": _required(config, "registration", "fidelity config"),
    }


def _validate_serializable(value, label="snapshot"):
    """Reject path leakage and non-finite numbers throughout the retained data."""
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_serializable(key, f"{label} key")
            _validate_serializable(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_serializable(child, f"{label}[{index}]")
    elif isinstance(value, str):
        # URLs and repository model IDs are provenance, not filesystem paths.
        if (re.search(r"\bfile:", value, flags=re.I)
                or re.search(r"(?:^|[^A-Za-z0-9])[A-Za-z]:[\\/]", value)
                or re.search(r"\\\\[^\\]+\\", value)):
            raise ValueError(f"{label} contains an absolute filesystem path")
        without_urls = re.sub(r"\bhttps?://[^\s<>\"']+", "", value, flags=re.I)
        if re.search(r"(?:^|[\s=:(\[{<,;'\"`])/", without_urls):
            raise ValueError(f"{label} contains an absolute filesystem path")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _finite(value, label)


def export_snapshot(
    audit_summary_path, smoke_summary_path, fidelity_summary_path,
    fidelity_csv_path, output_path, *, git_commit=None, repo_root=REPO_ROOT,
):
    """Validate completed artifacts, write a compact snapshot, and return it.

    ``git_commit`` can be supplied by CPU-only fixtures; normal invocation records
    the current repository HEAD. No input model, image, or feature data is loaded.
    Validation and serialization finish before an existing output is replaced.
    """
    audit = _read_json(audit_summary_path)
    smoke = _read_json(smoke_summary_path)
    fidelity = _read_json(fidelity_summary_path)
    rows, sources = _read_fidelity_csv(fidelity_csv_path)
    crop = {}
    for metric in ("crop_long_axis_fraction", "crop_area_fraction"):
        values = _mapping(_required(audit, metric, "audit summary"), metric)
        crop[metric] = {
            key: _finite(_required(values, key, metric), f"{metric}.{key}")
            for key in CROP_KEYS
        }
    compact = [
        {key: row[key] for key in ("audit_index", "row_index", "mode", *COMPACT_METRICS)}
        for row in sorted(rows, key=lambda row: (row["audit_index"], row["row_index"], row["mode"]))
        if row["mode"] in RELIGHT_MODES
    ]
    snapshot = {
        "provenance": _provenance(audit, smoke, fidelity, git_commit, repo_root),
        "canonical_crop": crop,
        "rmbg_alpha": _rmbg_alpha(smoke, sources),
        "fidelity": _copy_fidelity_summaries(fidelity, rows),
        "per_source_compact": compact,
        "worst_by_r8": {
            mode: sorted(
                (row for row in compact if row["mode"] == mode),
                key=lambda row: (row["repeatability_min_8px"], row["audit_index"], row["row_index"]),
            )[:5] for mode in RELIGHT_MODES
        },
    }
    _validate_serializable(snapshot)
    serialized = json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
    output_path = Path(output_path)
    resolved_output = output_path.resolve()
    for directory in ("salad", "third_party"):
        if resolved_output.is_relative_to((Path(repo_root) / directory).resolve()):
            raise ValueError(f"snapshot output must not modify {directory}/")
    if resolved_output in {
        Path(path).resolve() for path in (
            audit_summary_path, smoke_summary_path, fidelity_summary_path, fidelity_csv_path,
        )
    }:
        raise ValueError("snapshot output must not overwrite an input artifact")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return snapshot


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-summary", type=Path, default=REPO_ROOT / "cache/generator_audit/audit_summary.json")
    parser.add_argument("--smoke-summary", type=Path, default=REPO_ROOT / "cache/generator_audit/iclight_smoke_summary.json")
    parser.add_argument("--fidelity-summary", type=Path, default=REPO_ROOT / "cache/generator_audit/fidelity/fidelity_summary.json")
    parser.add_argument("--fidelity-csv", type=Path, default=REPO_ROOT / "cache/generator_audit/fidelity/fidelity_10.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/audits/step2c_metrics.json")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        snapshot = export_snapshot(
            args.audit_summary, args.smoke_summary, args.fidelity_summary,
            args.fidelity_csv, args.output,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(f"Exported {len(snapshot['per_source_compact'])} relighting records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
