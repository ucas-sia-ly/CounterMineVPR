"""CPU-only calibration of the frozen Step 2A real-RGB evidence.

The calibrated quantities describe graph structure. They do not specify a
training objective, loss weight, sampling probability, or final mining score.
Only compact metadata and previously measured scalar metrics enter this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np
import pandas as pd

from .structural_analysis import (
    analyze_tables, load_completed_metrics, pair_candidate_random,
    read_metadata_csv, sha256_file, validate_metrics, validate_snapshot, weak_ecdf,
)


EXPECTED_CANDIDATE_COUNT = 5000
EXPECTED_RANDOM_COUNT = 4999
MIN_RELATION_NULL_COUNT = 100
METRIC_NAMES = {
    "ratio": "local_match_ratio",
    "match_count": "num_matches",
    "coverage": "symmetric_match_coverage",
    "entropy": "symmetric_match_entropy",
}
CALIBRATION_DEFINITIONS = {
    "weak_ecdf": "F_null(x) = count(random_value <= x) / number_of_random_values",
    "primary_null": "relation-specific when at least 100 matched controls exist; otherwise global",
    "structural_bottleneck": "min(ratio_null_percentile, match_count_null_percentile)",
    "structural_geomean": "sqrt(ratio_null_percentile * match_count_null_percentile)",
    "spatial_support_percentile": "coverage_null_percentile",
    "salad_similarity_percentile": "count(candidate_max_salad_similarity <= x) / candidate_count",
    "interpretation": "graph-analysis weights only; future training mapping remains unselected",
}


def stable_pair_uid(image_id_a: str, image_id_b: str) -> str:
    """Use the existing pair identity algorithm without importing model code."""
    payload = json.dumps(sorted((image_id_a, image_id_b)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compare_frozen(actual, expected, location: str) -> None:
    """Compare complete frozen report schemas, ordered identities and numbers."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"frozen Step 2A snapshot schema differs at {location}")
        for key in expected:
            _compare_frozen(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"frozen Step 2A snapshot ordering/count differs at {location}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_frozen(left, right, f"{location}[{index}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"frozen Step 2A snapshot identity/value differs at {location}")
    elif isinstance(expected, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise ValueError(f"frozen Step 2A snapshot numeric value differs at {location}")
        if isinstance(expected, int):
            equal = isinstance(actual, int) and actual == expected
        else:
            equal = math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
        if not equal:
            raise ValueError(f"frozen Step 2A snapshot numeric value differs at {location}")
    else:
        raise ValueError(f"unsupported frozen snapshot value at {location}")


def _integer_column(frame: pd.DataFrame, name: str) -> pd.Series:
    values = pd.to_numeric(frame[name], errors="raise")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all() or (array < 0).any() or (array != np.floor(array)).any():
        raise ValueError(f"{name} must contain nonnegative integer values")
    return values.astype(np.int64)


def _validate_endpoint_identities(candidates: pd.DataFrame, random: pd.DataFrame, manifest_path: Path) -> None:
    manifest = read_metadata_csv(manifest_path)
    required = {"image_id", "row_index", "place_uid", "city_id"}
    if not required.issubset(manifest) or manifest["image_id"].duplicated().any():
        raise ValueError("manifest must contain unique image identities and endpoint metadata")
    rows = _integer_column(manifest, "row_index")
    if not np.array_equal(rows, np.arange(len(manifest))):
        raise ValueError("manifest row_index must preserve range(N) in existing row order")
    if any(not manifest[name].map(lambda value: bool(value.strip())).all() for name in required - {"row_index"}):
        raise ValueError("manifest endpoint identities must be nonempty")
    index = manifest.set_index("image_id")
    for kind, frame in (("candidate", candidates), ("random", random)):
        if frame["pair_uid"].duplicated().any():
            raise ValueError(f"{kind} pair_uid must be unique")
        if (frame["image_id_a"] == frame["image_id_b"]).any():
            raise ValueError(f"{kind} endpoints must be different images")
        for endpoint in ("a", "b"):
            ids = frame[f"image_id_{endpoint}"]
            if not ids.isin(index.index).all():
                raise ValueError(f"{kind} endpoint is absent from the frozen manifest")
            metadata = index.loc[ids]
            for name in ("place_uid", "city_id"):
                if metadata[name].tolist() != frame[f"{name}_{endpoint}"].tolist():
                    raise ValueError(f"{kind} endpoint {name} differs from frozen manifest")
            row_name = f"row_index_{endpoint}" if kind == "candidate" else (
                "anchor_query_row_index" if endpoint == "a" else "random_negative_row_index"
            )
            if row_name not in frame or not np.array_equal(
                _integer_column(frame, row_name), _integer_column(metadata, "row_index")
            ):
                raise ValueError(f"{kind} endpoint row index differs from frozen manifest")
        actual_uids = [stable_pair_uid(a, b) for a, b in zip(frame["image_id_a"], frame["image_id_b"])]
        uid_column = "pair_uid" if kind == "candidate" else "random_pair_uid"
        if uid_column not in frame or frame[uid_column].tolist() != actual_uids:
            raise ValueError(f"{kind} {uid_column} differs from canonical endpoint identity")
        if kind == "candidate" and not (frame["image_id_a"] < frame["image_id_b"]).all():
            raise ValueError("candidate endpoints must preserve canonical lexical order")
    if random["pair_uid"].tolist() != random["candidate_pair_uid"].tolist():
        raise ValueError("random pair_uid must preserve its matched candidate identity")
    pair_candidate_random(candidates, random)


def load_step2a_evidence(
    runtime_dir: str | Path, manifest_path: str | Path, candidate_csv_path: str | Path,
    candidate_summary_path: str | Path, snapshot_path: str | Path,
    *, expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
    expected_random_count: int = EXPECTED_RANDOM_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict, dict]:
    """Bind completed measurements to the frozen historical scientific snapshot.

    The historical snapshot did not record measurement CSV hashes. Completion
    marker checksums are therefore checked together with all snapshot input and
    configuration hashes, then every historical descriptive report field and
    ordered Top-50 record is reproduced on CPU. No local matching is performed.
    """
    root, snapshot_file = Path(runtime_dir), Path(snapshot_path)
    snapshot_hash = sha256_file(snapshot_file)
    with snapshot_file.open(encoding="utf-8") as stream:
        snapshot = json.load(stream)
    validate_snapshot(snapshot)
    candidates, random, population, config = load_completed_metrics(
        root, manifest_path, candidate_csv_path, candidate_summary_path,
    )
    if len(candidates) != expected_candidate_count or len(random) != expected_random_count:
        raise ValueError(f"Step 2B requires exactly {expected_candidate_count} candidates and {expected_random_count} matched controls")
    saved = snapshot.get("provenance", {})
    if saved.get("input_hashes") != config.get("input_hashes"):
        raise ValueError("frozen Step 2A snapshot population/input hashes differ")
    for key in ("manifest_sha256", "candidate_csv_sha256", "candidate_summary_sha256"):
        if saved.get(key) != config["input_hashes"].get(key):
            raise ValueError(f"frozen Step 2A snapshot hash differs: {key}")
    checks = {
        "measurement_config_sha256": sha256_file(root / "measurement_config.json"),
        "input_image_index_sha256": config.get("input_image_index_sha256"),
        "seed": config.get("seed"), "aliked_config": config.get("aliked_config"),
        "lightglue_config": config.get("lightglue_config"),
        "measurement_source_sha256": config.get("source_sha256", {}),
        "vendor_provenance": config.get("vendor_provenance", {}),
    }
    for key, value in checks.items():
        if saved.get(key) != value:
            raise ValueError(f"frozen Step 2A snapshot configuration/provenance differs: {key}")
    for key, value in saved.get("matcher_provenance", {}).items():
        if config.get("matcher_provenance", {}).get(key) != value:
            raise ValueError(f"frozen Step 2A snapshot matcher provenance differs: {key}")
    for key in snapshot.get("population", {}):
        if population.get(key) != snapshot["population"][key]:
            raise ValueError(f"frozen Step 2A snapshot population differs: {key}")
    if snapshot.get("candidate_summary", {}).get("count") != expected_candidate_count or snapshot.get("random_summary", {}).get("count") != expected_random_count:
        raise ValueError("frozen Step 2A snapshot counts differ from required pilot")
    _validate_endpoint_identities(candidates, random, Path(manifest_path))
    _, historical_report = analyze_tables(candidates, random)
    report_keys = set(snapshot) - {"schema_version", "provenance", "population"}
    if report_keys != set(historical_report):
        raise ValueError("frozen Step 2A scientific report schema differs")
    for key in historical_report:
        _compare_frozen(historical_report[key], snapshot[key], key)
    if sha256_file(snapshot_file) != snapshot_hash:
        raise ValueError("frozen Step 2A snapshot changed during validation")
    provenance = {
        "step2a_snapshot_sha256": snapshot_hash,
        "step2a_candidate_metrics_sha256": sha256_file(root / "candidate_structural_metrics.csv"),
        "step2a_random_metrics_sha256": sha256_file(root / "random_structural_metrics.csv"),
        "step2a_measurement_summary_sha256": sha256_file(root / "measurement_summary.json"),
        "step2a_measurement_config_sha256": checks["measurement_config_sha256"],
        "step2a_image_fingerprints_sha256": checks["input_image_index_sha256"],
        "step2a_input_hashes": config["input_hashes"],
        "step2a_analysis_source_sha256": saved.get("analysis_source_sha256", {}),
        "validation": {
            "completion_marker_metrics_checksums": True,
            "snapshot_input_and_configuration_checksums": True,
            "all_snapshot_report_fields_reproduced": True,
            "report_numeric_absolute_tolerance": 1e-12,
            "measurement_csv_hashes_present_in_historical_snapshot": False,
            "canonical_endpoint_and_manifest_identities": True,
        },
        "seed": config["seed"], "real_rgb_only": True,
        "synthetic_images_used": False, "no_new_local_matching": True,
    }
    return candidates, random, population, config, snapshot, provenance


def calibrate_structural_evidence(
    candidates: pd.DataFrame, random: pd.DataFrame,
    *, min_relation_null_count: int = MIN_RELATION_NULL_COUNT,
) -> tuple[pd.DataFrame, dict]:
    """Use measured matched-random weak ECDFs for four independent metrics."""
    if not isinstance(min_relation_null_count, int) or min_relation_null_count < 1:
        raise ValueError("minimum relation null count must be a positive integer")
    result = validate_metrics(candidates, candidate=True).reset_index(drop=True)
    controls = validate_metrics(random, candidate=False).reset_index(drop=True)
    if result.empty or controls.empty:
        raise ValueError("calibration requires candidates and at least one measured control")
    if controls["pair_uid"].duplicated().any():
        raise ValueError("random pair_uid must be unique")
    pair_candidate_random(result, controls)
    counts = {
        name: int((controls["same_city"] == flag).sum())
        for name, flag in (("same_city", True), ("cross_city", False))
    }
    for short, metric in METRIC_NAMES.items():
        global_name, relation_name = f"global_{short}_null_percentile", f"relation_{short}_null_percentile"
        primary_name, source_name = f"{short}_null_percentile", f"{short}_null_source"
        result[global_name] = weak_ecdf(controls[metric], result[metric])
        result[relation_name] = np.nan
        result[primary_name] = result[global_name]
        result[source_name] = "global"
        for name, flag in (("same_city", True), ("cross_city", False)):
            if counts[name] >= min_relation_null_count:
                selected = result["same_city"] == flag
                percentiles = weak_ecdf(controls.loc[controls["same_city"] == flag, metric], result.loc[selected, metric])
                result.loc[selected, relation_name] = percentiles
                result.loc[selected, primary_name] = percentiles
                result.loc[selected, source_name] = name
    result["structural_bottleneck"] = np.minimum(result["ratio_null_percentile"], result["match_count_null_percentile"])
    result["structural_geomean"] = np.sqrt(result["ratio_null_percentile"] * result["match_count_null_percentile"])
    result["spatial_support_percentile"] = result["coverage_null_percentile"]
    result["salad_similarity_percentile"] = weak_ecdf(result["max_salad_similarity"], result["max_salad_similarity"])
    result["near_geo_500m"] = result["geo_distance_m"] < 500
    result["min_num_keypoints"] = np.minimum(result["num_keypoints_a"], result["num_keypoints_b"])
    definitions = dict(CALIBRATION_DEFINITIONS)
    definitions["primary_null"] = f"relation-specific when at least {min_relation_null_count} matched controls exist; otherwise global"
    metadata = {
        "definitions": definitions, "metric_columns": dict(METRIC_NAMES),
        "global_null_count": len(controls), "relation_null_counts": counts,
        "minimum_relation_null_count": min_relation_null_count,
        "relation_null_available": {name: count >= min_relation_null_count for name, count in counts.items()},
        "primary_null_source_counts": {name: int((result["ratio_null_source"] == name).sum()) for name in ("same_city", "cross_city", "global")},
        "relation_percentile_unavailable_count": int(result["relation_ratio_null_percentile"].isna().sum()),
        "unavailable_relation_percentiles": "null in JSON; empty CSV cell",
        "coverage_in_primary_bottleneck": False,
        "salad_similarity_percentile_attribute_only": True,
    }
    return validate_calibrated_candidates(result), metadata


def validate_calibrated_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse calibrated CSV data while enforcing formulas and finite evidence."""
    result = validate_metrics(frame, candidate=True)
    for short in METRIC_NAMES:
        for prefix in ("", "global_", "relation_"):
            name = f"{prefix}{short}_null_percentile"
            if name not in result:
                raise ValueError(f"calibrated candidates lack {name}")
            values = pd.to_numeric(result[name].replace("", np.nan), errors="raise").to_numpy(dtype=float)
            missing = np.isnan(values)
            if prefix != "relation_" and missing.any():
                raise ValueError(f"{name} must contain finite percentile values")
            if not np.isfinite(values[~missing]).all() or ((values[~missing] < 0) | (values[~missing] > 1)).any():
                raise ValueError(f"{name} must contain percentiles in [0,1]")
            result[name] = values
        source_name = f"{short}_null_source"
        if source_name not in result or not result[source_name].isin(("global", "same_city", "cross_city")).all():
            raise ValueError(f"{source_name} must identify an explicit null source")
        global_selected = result[source_name] == "global"
        expected = np.where(global_selected, result[f"global_{short}_null_percentile"], result[f"relation_{short}_null_percentile"])
        if not np.isfinite(expected).all() or not np.allclose(result[f"{short}_null_percentile"], expected, rtol=0, atol=1e-12):
            raise ValueError(f"{short} primary percentile disagrees with recorded null source")
        for relation, flag in (("same_city", True), ("cross_city", False)):
            if ((result[source_name] == relation) & (result["same_city"] != flag)).any():
                raise ValueError(f"{source_name} disagrees with candidate relation")
        if not result.loc[global_selected, f"relation_{short}_null_percentile"].isna().all():
            raise ValueError("unavailable relation percentiles must be null")
    formulas = {
        "structural_bottleneck": np.minimum(result["ratio_null_percentile"], result["match_count_null_percentile"]),
        "structural_geomean": np.sqrt(result["ratio_null_percentile"] * result["match_count_null_percentile"]),
        "spatial_support_percentile": result["coverage_null_percentile"],
        "salad_similarity_percentile": weak_ecdf(result["max_salad_similarity"], result["max_salad_similarity"]),
    }
    for name, expected in formulas.items():
        if name not in result:
            raise ValueError(f"calibrated candidates lack {name}")
        values = pd.to_numeric(result[name], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.allclose(values, expected, rtol=0, atol=1e-12):
            raise ValueError(f"{name} disagrees with calibrated evidence formula")
        result[name] = values
    result["min_num_keypoints"] = np.minimum(result["num_keypoints_a"], result["num_keypoints_b"])
    result["near_geo_500m"] = result["geo_distance_m"] < 500
    for name in ("row_index_a", "row_index_b", "num_candidate_directions"):
        if name in result:
            result[name] = _integer_column(result, name)
    return result


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source_paths(repo_root: Path) -> dict[str, str]:
    names = (
        "countermine/mining/structural_calibration.py", "countermine/mining/structural_analysis.py",
        "tools/11_calibrate_structural_evidence.py",
    )
    return {name: sha256_file(repo_root / name) for name in names}


def _relative_source(path: str | Path, repo_root: Path) -> str:
    try:
        relative = Path(path).resolve().relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError("Step 2B source metadata paths must be inside the repository root") from error
    return relative.as_posix()


def _source_file_inputs(runtime_dir, manifest_path, candidate_csv_path, candidate_summary_path, snapshot_path, repo_root: Path) -> dict:
    paths = {
        "runtime_dir": runtime_dir, "manifest": manifest_path, "candidates": candidate_csv_path,
        "candidate_summary": candidate_summary_path, "snapshot": snapshot_path,
    }
    return {name: _relative_source(path, repo_root) for name, path in paths.items()}


def _resolve_sources(sources: dict, repo_root: Path) -> dict[str, Path]:
    required = {"runtime_dir", "manifest", "candidates", "candidate_summary", "snapshot"}
    if not isinstance(sources, dict) or set(sources) != required:
        raise ValueError("calibration source input paths are incomplete")
    result = {}
    for name, relative in sources.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("calibration source input paths must be relative and stay in the repository")
        path = (repo_root / relative).resolve()
        if not path.is_relative_to(repo_root.resolve()):
            raise ValueError("calibration source input path escapes repository")
        result[name] = path
    return result


def _validate_calibration_destinations(
    output_dir: str | Path, runtime_dir: str | Path, input_paths, repo_root: Path,
) -> None:
    """Protect original evidence even when a destination uses a symlink alias."""
    protected_roots = (
        Path(runtime_dir).resolve(),
        (repo_root / "cache/countermine_rgb/step2a").resolve(),
        (repo_root / "data/gsv-cities").resolve(),
        (repo_root / "third_party").resolve(),
    )
    protected_files = {Path(path).resolve() for path in input_paths}
    protected_files.add((repo_root / "docs/audits/step2a_rgb_structural_metrics.json").resolve())
    protected_files.update(path.resolve() for path in (repo_root / "docs/audits").glob("step2a_*"))
    output = Path(output_dir)
    targets = (output, output / "calibrated_candidates.csv", output / "calibration_summary.json")
    for target in targets:
        resolved = target.resolve()
        if resolved in protected_files or any(resolved == root or resolved.is_relative_to(root) for root in protected_roots):
            raise ValueError("Step 2B calibration output must not modify Step 2A evidence, original inputs, RGB images, or third_party")


def run_calibration(
    runtime_dir: str | Path, manifest_path: str | Path, candidate_csv_path: str | Path,
    candidate_summary_path: str | Path, snapshot_path: str | Path, output_dir: str | Path,
    *, repo_root: str | Path | None = None,
) -> dict:
    """Validate the fixed pilot and atomically publish deterministic artifacts."""
    repository = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    output = Path(output_dir)
    _validate_calibration_destinations(
        output, runtime_dir, (manifest_path, candidate_csv_path, candidate_summary_path, snapshot_path), repository,
    )
    candidates, controls, _, _, _, provenance = load_step2a_evidence(
        runtime_dir, manifest_path, candidate_csv_path, candidate_summary_path, snapshot_path,
    )
    historical_sources = provenance["step2a_analysis_source_sha256"]
    for name, digest in historical_sources.items():
        if sha256_file(repository / name) != digest:
            raise ValueError(f"frozen Step 2A analysis source checksum differs: {name}")
    calibrated, metadata = calibrate_structural_evidence(candidates, controls)
    payload = _csv_bytes(calibrated)
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    provenance.update({
        "calibration_source_sha256": _source_paths(repository),
        "current_git_commit": git_commit,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    })
    summary = {
        "schema_version": 1, "stage": "step2b_structural_calibration", "complete": True,
        "candidate_count": len(candidates), "random_count": len(controls),
        "configuration": {
            "seed": provenance["seed"], "minimum_relation_null_count": MIN_RELATION_NULL_COUNT,
            "expected_candidate_count": EXPECTED_CANDIDATE_COUNT, "expected_random_count": EXPECTED_RANDOM_COUNT,
        },
        "input_files": _source_file_inputs(runtime_dir, manifest_path, candidate_csv_path, candidate_summary_path, snapshot_path, repository),
        "provenance": provenance, "calibration": metadata,
        "calibrated_candidates_sha256": hashlib.sha256(payload).hexdigest(),
    }
    validate_snapshot(summary)
    csv_path, summary_path = output / "calibrated_candidates.csv", output / "calibration_summary.json"
    if csv_path.exists() or summary_path.exists():
        if not csv_path.is_file() or not summary_path.is_file():
            raise ValueError("incomplete Step 2B calibration artifacts exist; refusing to mix runs")
        with summary_path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != summary or sha256_file(csv_path) != summary["calibrated_candidates_sha256"]:
            raise ValueError("Step 2B calibration artifacts differ from current inputs/configuration; refusing to mix runs")
        return existing
    # Recheck every source byte after calculation and before publication. The
    # summary is written last and serves as the complete-stage marker.
    _, _, _, _, _, rechecked = load_step2a_evidence(
        runtime_dir, manifest_path, candidate_csv_path, candidate_summary_path, snapshot_path,
    )
    if any(provenance[key] != value for key, value in rechecked.items()) or _source_paths(repository) != provenance["calibration_source_sha256"]:
        raise ValueError("Step 2B source inputs changed during calibration")
    _validate_calibration_destinations(
        output, runtime_dir, (manifest_path, candidate_csv_path, candidate_summary_path, snapshot_path), repository,
    )
    _atomic_bytes(csv_path, payload)
    _atomic_bytes(summary_path, (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    return summary


def load_calibrated_artifacts(
    step2b_dir: str | Path, *, source_validation: bool = True, repo_root: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Validate completion, hashes, formulas, and (by default) source evidence."""
    directory = Path(step2b_dir)
    with (directory / "calibration_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    validate_snapshot(summary)
    if summary.get("schema_version") != 1 or summary.get("stage") != "step2b_structural_calibration" or summary.get("complete") is not True:
        raise ValueError("calibration artifacts require a complete supported summary")
    csv_path = directory / "calibrated_candidates.csv"
    if sha256_file(csv_path) != summary.get("calibrated_candidates_sha256"):
        raise ValueError("calibrated candidates checksum differs from summary")
    frame = validate_calibrated_candidates(read_metadata_csv(csv_path))
    if len(frame) != summary.get("candidate_count"):
        raise ValueError("calibrated candidate count differs from summary")
    if source_validation:
        repository = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        sources = _resolve_sources(summary.get("input_files"), repository)
        config = summary.get("configuration", {})
        if config != {
            "seed": 42, "minimum_relation_null_count": MIN_RELATION_NULL_COUNT,
            "expected_candidate_count": EXPECTED_CANDIDATE_COUNT, "expected_random_count": EXPECTED_RANDOM_COUNT,
        }:
            raise ValueError("calibration configuration differs from the fixed Step 2B pilot")
        candidates, controls, _, _, _, provenance = load_step2a_evidence(
            sources["runtime_dir"], sources["manifest"], sources["candidates"], sources["candidate_summary"], sources["snapshot"],
        )
        if any(summary["provenance"].get(key) != value for key, value in provenance.items()):
            raise ValueError("calibration source provenance differs from frozen Step 2A inputs")
        if _source_paths(repository) != summary["provenance"].get("calibration_source_sha256"):
            raise ValueError("calibration source code checksums differ")
        fresh, metadata = calibrate_structural_evidence(candidates, controls)
        if metadata != summary.get("calibration") or hashlib.sha256(_csv_bytes(fresh)).hexdigest() != summary["calibrated_candidates_sha256"]:
            raise ValueError("calibration artifacts differ from reproducible source evidence")
    return frame, summary
