"""CPU-only descriptive analysis and export for the real-RGB Step 2A pilot.

Only compact pair metadata is held in memory. This module never imports a
retrieval model, local feature model, or tensor runtime.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


RANK_BINS = ("rank_1", "rank_2_5", "rank_6_10", "rank_11_20", "rank_21_50")
MIN_RELATION_NULL_COUNT = 100
ALIKED_CONFIG = {"max_num_keypoints": 2048, "detection_threshold": 0.2}
LIGHTGLUE_CONFIG = {
    "features": "aliked", "depth_confidence": -1, "width_confidence": -1,
    "filter_threshold": 0.1, "mp": False,
}
METRIC_COLUMNS = (
    "num_keypoints_a", "num_keypoints_b", "num_matches", "local_match_ratio",
    "matched_source_cell_coverage", "matched_target_cell_coverage", "symmetric_match_coverage",
    "source_max_cell_match_fraction", "target_max_cell_match_fraction",
    "source_match_entropy", "target_match_entropy", "symmetric_match_entropy",
    "exact_pixel_duplicate",
)
TOP_METADATA_COLUMNS = (
    "pair_uid", "image_id_a", "image_id_b", "place_uid_a", "place_uid_b", "city_id_a", "city_id_b",
    "geo_distance_m", "same_city", "near_geo_500m", "best_rgb_rank", "rank_bin",
    "max_salad_similarity", "num_keypoints_a", "num_keypoints_b", "num_matches",
    "local_match_ratio", "matched_source_cell_coverage", "matched_target_cell_coverage",
    "global_local_null_percentile", "relation_local_null_percentile",
)
CURATED_ARTIFACTS = {
    "candidate_vs_random_local.png": "step2a_candidate_vs_random_local.png",
    "salad_vs_local.png": "step2a_salad_vs_local.png",
    "local_by_rank.png": "step2a_local_by_rank.png",
    "top50_structural_candidates.jpg": "step2a_top50_structural_candidates.jpg",
    "top20_match_overlays.jpg": "step2a_top20_match_overlays.jpg",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boolean(value, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{name} must contain booleans")


def read_metadata_csv(path: str | Path) -> pd.DataFrame:
    """Preserve string identities, including city/place IDs and empty directions."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        columns = next(csv.reader(stream), [])
    if not columns or len(columns) != len(set(columns)):
        raise ValueError(f"{path} has missing or duplicate CSV column names")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _finite_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"metrics lack {column}")
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} must contain finite numeric values") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{column} must contain finite numeric values")
    return values


def validate_metrics(table: pd.DataFrame, *, candidate: bool) -> pd.DataFrame:
    """Check measured values before descriptive statistics or evidence selection."""
    required = set(METRIC_COLUMNS) | {
        "pair_uid", "image_id_a", "image_id_b", "place_uid_a", "place_uid_b", "city_id_a",
        "city_id_b", "geo_distance_m", "same_city", "rank_bin",
    }
    if candidate:
        required |= {"best_rgb_rank", "max_salad_similarity", "mean_salad_similarity"}
    else:
        required |= {"candidate_pair_uid", "anchor_query_image_id", "random_negative_image_id"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"metrics lack required columns: {', '.join(sorted(missing))}")
    frame = table.copy()
    key = "pair_uid" if candidate else "candidate_pair_uid"
    if frame[key].duplicated().any() or not frame[key].map(lambda v: isinstance(v, str) and bool(v)).all():
        raise ValueError(f"metrics {key} must contain unique nonempty identities")
    for column in ("same_city", "exact_pixel_duplicate"):
        frame[column] = frame[column].map(lambda value: _boolean(value, column))
    if (frame["place_uid_a"] == frame["place_uid_b"]).any():
        raise ValueError("same-place pairs are not eligible negative measurements")
    if not (frame["same_city"] == (frame["city_id_a"] == frame["city_id_b"])).all():
        raise ValueError("same_city disagrees with endpoint city identities")
    frame["geo_distance_m"] = _finite_numeric(frame, "geo_distance_m")
    if (frame["geo_distance_m"] < 250).any():
        raise ValueError("measured negatives must satisfy the frozen >=250m exclusion")
    if not frame["rank_bin"].isin(RANK_BINS).all():
        raise ValueError("metrics contain an unknown frozen rank bin")
    for column in METRIC_COLUMNS[:-1]:
        values = _finite_numeric(frame, column)
        if column.startswith("num_"):
            if (values < 0).any() or (values != np.floor(values)).any() or (values > 2048).any():
                raise ValueError(f"{column} must contain keypoint/match counts in [0,2048]")
            frame[column] = values.astype(np.int64)
        else:
            if (values < 0).any() or (values > 1).any():
                raise ValueError(f"{column} must be in [0,1]")
            frame[column] = values
    denominator = np.minimum(frame["num_keypoints_a"], frame["num_keypoints_b"]).to_numpy()
    matches = frame["num_matches"].to_numpy()
    if (matches > denominator).any():
        raise ValueError("num_matches exceeds the smaller keypoint count")
    expected = np.divide(matches, denominator, out=np.zeros(len(frame)), where=denominator != 0)
    if not np.allclose(frame["local_match_ratio"], expected, rtol=0, atol=1e-12):
        raise ValueError("local_match_ratio disagrees with num_matches/min(keypoints)")
    for combined, left, right in (
        ("symmetric_match_coverage", "matched_source_cell_coverage", "matched_target_cell_coverage"),
        ("symmetric_match_entropy", "source_match_entropy", "target_match_entropy"),
    ):
        if not np.allclose(frame[combined], np.minimum(frame[left], frame[right]), rtol=0, atol=1e-12):
            raise ValueError(f"{combined} disagrees with its endpoint diagnostics")
    if candidate:
        frame["best_rgb_rank"] = _finite_numeric(frame, "best_rgb_rank")
        rank = frame["best_rgb_rank"].to_numpy()
        if (rank < 1).any() or (rank > 50).any() or (rank != np.floor(rank)).any():
            raise ValueError("best_rgb_rank must be an integer in [1,50]")
        frame["best_rgb_rank"] = rank.astype(np.int64)
        expected_bins = [RANK_BINS[0 if r == 1 else 1 if r <= 5 else 2 if r <= 10 else 3 if r <= 20 else 4] for r in rank]
        if frame["rank_bin"].tolist() != expected_bins:
            raise ValueError("rank_bin disagrees with frozen best_rgb_rank boundaries")
        for column in ("max_salad_similarity", "mean_salad_similarity"):
            frame[column] = _finite_numeric(frame, column)
    return frame


def _verify_source_columns(measured: pd.DataFrame, expected: pd.DataFrame, key: str, label: str) -> None:
    if expected[key].duplicated().any() or measured[key].duplicated().any():
        raise ValueError(f"{label} contains duplicated pair identities")
    if set(expected[key]) != set(measured[key]):
        raise ValueError(f"{label} identities do not match the frozen population/control records")
    missing = set(expected.columns).difference(measured.columns)
    if missing:
        raise ValueError(f"{label} does not preserve input metadata: {sorted(missing)}")
    actual = measured.set_index(key).loc[expected[key]].reset_index()
    for column in expected:
        left, right = expected[column].astype(str), actual[column].astype(str)
        if left.equals(right):
            continue
        numeric_metadata = (
            column.startswith("row_index_") or column.endswith("_row_index")
            or column.endswith(("_rank", "_similarity", "_distance_m"))
            or column == "num_candidate_directions"
        )
        if not numeric_metadata:
            raise ValueError(f"{label} changed input metadata {column}")
        # CSV round trips may write integral numeric metadata as floats. Empty
        # optional directed occurrences must still agree exactly.
        if not left.eq("").equals(right.eq("")):
            raise ValueError(f"{label} changed input metadata {column}")
        mask = ~left.eq("")
        try:
            lnum = pd.to_numeric(left[mask], errors="raise").to_numpy(dtype=float)
            rnum = pd.to_numeric(right[mask], errors="raise").to_numpy(dtype=float)
            numeric = np.isfinite(lnum).all() and np.isfinite(rnum).all()
        except (TypeError, ValueError):
            numeric = False
        if not numeric or not np.allclose(lnum, rnum, rtol=1e-14, atol=1e-12):
            raise ValueError(f"{label} changed input metadata {column}")


def load_completed_metrics(
    runtime_dir: str | Path,
    manifest_path: str | Path,
    candidate_csv_path: str | Path,
    candidate_summary_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Require complete inference and matching current input/output provenance."""
    root = Path(runtime_dir)
    with (root / "measurement_summary.json").open(encoding="utf-8") as stream:
        completion = json.load(stream)
    if completion.get("schema_version") != 1 or completion.get("complete") is not True:
        raise ValueError("structural measurement is incomplete; analysis requires all pairs")
    config_path = root / "measurement_config.json"
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    with (root / "population_summary.json").open(encoding="utf-8") as stream:
        population_summary = json.load(stream)
    inputs = {
        "manifest_sha256": Path(manifest_path),
        "candidate_csv_sha256": Path(candidate_csv_path),
        "candidate_summary_sha256": Path(candidate_summary_path),
        "population_sha256": root / "population.csv",
        "random_controls_sha256": root / "random_controls.csv",
        "population_summary_sha256": root / "population_summary.json",
    }
    hashes = {key: sha256_file(path) for key, path in inputs.items()}
    for source in (completion, config):
        if source.get("input_hashes") != hashes:
            raise ValueError("measurement input provenance differs from current frozen inputs")
    for key in hashes:
        if key != "population_summary_sha256" and population_summary.get(key) != hashes[key]:
            raise ValueError(f"population summary provenance differs: {key}")
    if completion.get("config_sha256") != sha256_file(config_path):
        raise ValueError("measurement configuration checksum differs from completion marker")
    policy = config.get("image_policy", {})
    if (
        config.get("schema_version") != 1 or config.get("seed") != population_summary.get("seed")
        or policy.get("real_rgb_only") is not True or policy.get("synthetic_images_used") is not False
        or policy.get("geometry") != [640, 480] or policy.get("local_resize", "missing") is not None
        or policy.get("registration", "missing") is not None
        or config.get("aliked_config") != ALIKED_CONFIG or config.get("lightglue_config") != LIGHTGLUE_CONFIG
        or population_summary.get("min_geo_distance_m") != 250
    ):
        raise ValueError("measurement does not use the frozen real-RGB Step 2A configuration")
    fingerprint_path = root / "image_fingerprints.json"
    if config.get("input_image_index_sha256") != sha256_file(fingerprint_path):
        raise ValueError("measured original-image fingerprint index checksum differs")
    candidate_path = root / "candidate_structural_metrics.csv"
    random_path = root / "random_structural_metrics.csv"
    if completion.get("candidate_metrics_sha256") != sha256_file(candidate_path):
        raise ValueError("candidate measurement checksum differs from completion marker")
    if completion.get("random_metrics_sha256") != sha256_file(random_path):
        raise ValueError("random measurement checksum differs from completion marker")
    population = read_metadata_csv(root / "population.csv")
    controls = read_metadata_csv(root / "random_controls.csv")
    controls = controls.loc[controls["random_control_available"].map(lambda v: _boolean(v, "random_control_available"))].reset_index(drop=True)
    candidate_raw, random_raw = read_metadata_csv(candidate_path), read_metadata_csv(random_path)
    _verify_source_columns(candidate_raw, population, "pair_uid", "candidate measurements")
    _verify_source_columns(random_raw, controls, "pair_uid", "random measurements")
    for kind, frame, expected in (("candidate", candidate_raw, len(population)), ("random", random_raw, len(controls))):
        if completion.get(f"expected_{kind}_count") != expected or completion.get(f"{kind}_count") != len(frame):
            raise ValueError(f"{kind} measurement completion counts disagree")
    if population_summary.get("sampled_pair_count") != len(population) or population_summary.get("matched_random_count") != len(controls):
        raise ValueError("population summary counts disagree with frozen CSV records")
    with fingerprint_path.open(encoding="utf-8") as stream:
        fingerprints = json.load(stream)
    for table in (candidate_raw, random_raw):
        for _, row in table.iterrows():
            for endpoint in ("a", "b"):
                fingerprint = fingerprints.get(row[f"image_id_{endpoint}"])
                if not isinstance(fingerprint, dict):
                    raise ValueError("measured image is absent from the original-image fingerprint index")
                for field, saved in ((f"rgb_pixel_sha_{endpoint}", "rgb_pixel_sha256"), (f"image_file_sha_{endpoint}", "image_file_sha256")):
                    if row.get(field) != fingerprint.get(saved):
                        raise ValueError("measured image hash disagrees with the original-image fingerprint index")
            duplicate = fingerprints[row["image_id_a"]]["rgb_pixel_sha256"] == fingerprints[row["image_id_b"]]["rgb_pixel_sha256"]
            if _boolean(row["exact_pixel_duplicate"], "exact_pixel_duplicate") != duplicate:
                raise ValueError("exact duplicate flag disagrees with decoded RGB pixel hashes")
    return validate_metrics(candidate_raw, candidate=True), validate_metrics(random_raw, candidate=False), population_summary, config


def validate_original_image_fingerprints(
    runtime_dir: str | Path, candidates: pd.DataFrame, random: pd.DataFrame,
    manifest_path: str | Path, dataset_root: str | Path,
) -> None:
    """Stream encoded original-image hashes so exported evidence uses measured bytes."""
    with (Path(runtime_dir) / "image_fingerprints.json").open(encoding="utf-8") as stream:
        fingerprints = json.load(stream)
    manifest = read_metadata_csv(manifest_path)
    if manifest["image_id"].duplicated().any():
        raise ValueError("manifest image identities must be unique")
    index = manifest.set_index("image_id")
    used = set(candidates["image_id_a"]) | set(candidates["image_id_b"]) | set(random["image_id_a"]) | set(random["image_id_b"])
    if set(fingerprints) != used:
        raise ValueError("original-image fingerprint identities differ from measured images")
    root = Path(dataset_root).expanduser().resolve()
    for image_id in sorted(used):
        if image_id not in index.index:
            raise ValueError("measured image is absent from the current manifest")
        relative = Path(index.loc[image_id, "relative_path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("original image path must remain inside the dataset root")
        if sha256_file(path) != fingerprints[image_id].get("image_file_sha256"):
            raise ValueError(f"original image bytes changed after inference: {image_id}")


def distribution(values, *, delta: bool = False) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("distribution inputs must be a finite one-dimensional array")
    quantiles = (("q05", .05), ("q25", .25), ("median", .5), ("q75", .75), ("q95", .95))
    if not delta:
        quantiles = (("q01", .01),) + quantiles + (("q99", .99),)
    result = {"count": int(array.size), "mean": None, "std": None, "min": None}
    result.update({name: None for name, _ in quantiles})
    result["max"] = None
    if array.size:
        result.update(mean=float(array.mean()), min=float(array.min()), max=float(array.max()))
        if array.size >= 2:
            result["std"] = float(array.std(ddof=1))
        result.update({name: float(np.quantile(array, q)) for name, q in quantiles})
    return result


def weak_ecdf(null_values, values) -> np.ndarray:
    """Empirical F(x)=count(null<=x)/N; ties are included, without smoothing."""
    null = np.asarray(null_values, dtype=float)
    queries = np.asarray(values, dtype=float)
    if null.ndim != 1 or not np.isfinite(null).all() or not np.isfinite(queries).all():
        raise ValueError("ECDF inputs must be finite")
    if not len(null):
        raise ValueError("ECDF requires at least one null control")
    return np.searchsorted(np.sort(null), queries, side="right") / len(null)


def add_null_percentiles(candidates: pd.DataFrame, random: pd.DataFrame) -> pd.DataFrame:
    result = candidates.copy()
    result["global_local_null_percentile"] = (
        weak_ecdf(random["local_match_ratio"], candidates["local_match_ratio"]) if len(random) else np.nan
    )
    result["relation_local_null_percentile"] = np.nan
    for relation in (True, False):
        null = random.loc[random["same_city"] == relation, "local_match_ratio"]
        selected = result["same_city"] == relation
        if len(null) >= MIN_RELATION_NULL_COUNT:
            result.loc[selected, "relation_local_null_percentile"] = weak_ecdf(null, result.loc[selected, "local_match_ratio"])
    return result


def pair_candidate_random(candidates: pd.DataFrame, random: pd.DataFrame) -> pd.DataFrame:
    if candidates["pair_uid"].duplicated().any() or random["candidate_pair_uid"].duplicated().any():
        raise ValueError("paired comparison requires unique candidate/control identities")
    if not set(random["candidate_pair_uid"]).issubset(set(candidates["pair_uid"])):
        raise ValueError("random measurements reference unknown candidate pairs")
    joined = candidates.merge(
        random[["candidate_pair_uid", "local_match_ratio", "same_city", "rank_bin"]],
        left_on="pair_uid", right_on="candidate_pair_uid", how="inner", suffixes=("", "_random"),
        validate="one_to_one",
    )
    if not joined["same_city"].equals(joined["same_city_random"]) or not joined["rank_bin"].equals(joined["rank_bin_random"]):
        raise ValueError("matched controls disagree with candidate relation/rank metadata")
    joined["delta_local_match_ratio"] = joined["local_match_ratio"] - joined["local_match_ratio_random"]
    return joined


def paired_summary(paired: pd.DataFrame) -> dict:
    delta = paired["delta_local_match_ratio"].to_numpy(dtype=float)
    return {
        "delta_local_match_ratio": distribution(delta, delta=True),
        "candidate_gt_random": int((delta > 0).sum()),
        "candidate_eq_random": int((delta == 0).sum()),
        "candidate_lt_random": int((delta < 0).sum()),
        "fraction_candidate_gt_random": float((delta > 0).mean()) if len(delta) else None,
    }


def _quantiles(values, names=("q25", "median", "q75")) -> dict:
    source = distribution(values)
    return {name: source[name] for name in names}


def _correlation(x, y) -> dict:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"count": len(a), "pearson": None, "spearman": None}
    return {
        "count": len(a), "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": float(np.corrcoef(pd.Series(a).rank(method="average"), pd.Series(b).rank(method="average"))[0, 1]),
    }


def select_top_structural_candidates(candidates: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    duplicate = candidates["exact_pixel_duplicate"].map(lambda value: _boolean(value, "exact_pixel_duplicate"))
    return candidates.loc[~duplicate].sort_values(
        ["local_match_ratio", "num_matches", "max_salad_similarity", "pair_uid"],
        ascending=[False, False, False, True], kind="stable",
    ).head(top_n).copy()


def _integrity_group(frame: pd.DataFrame) -> dict:
    count = len(frame)
    duplicates = int(frame["exact_pixel_duplicate"].sum())
    near = int((frame["geo_distance_m"] < 500).sum())
    return {
        "count": count, "exact_pixel_duplicate_count": duplicates, "near_geo_500m_count": near,
        "exact_pixel_duplicate_fraction": duplicates / count if count else None,
        "near_geo_500m_fraction": near / count if count else None,
    }


def analyze_tables(candidates: pd.DataFrame, random: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute descriptive evidence with frozen bins and no decision rule."""
    candidates = validate_metrics(candidates, candidate=True)
    random = validate_metrics(random, candidate=False)
    candidates = add_null_percentiles(candidates, random)
    candidates["near_geo_500m"] = candidates["geo_distance_m"] < 500
    paired = pair_candidate_random(candidates, random)
    rank_conditioned = {}
    for rank_bin in RANK_BINS:
        selected = candidates.loc[candidates["rank_bin"] == rank_bin]
        controls = random.loc[random["rank_bin"] == rank_bin]
        pair_subset = paired.loc[paired["rank_bin"] == rank_bin]
        paired_report = paired_summary(pair_subset)
        rank_conditioned[rank_bin] = {
            "pair_count": len(selected), "matched_random_count": len(controls),
            "max_salad_similarity": _quantiles(selected["max_salad_similarity"]),
            "local_match_ratio": _quantiles(selected["local_match_ratio"], ("q05", "q25", "median", "q75", "q95")),
            "random_local_match_ratio": _quantiles(controls["local_match_ratio"], ("q05", "q25", "median", "q75", "q95")),
            "paired_delta": _quantiles(pair_subset["delta_local_match_ratio"]),
            "fraction_candidate_gt_random": paired_report["fraction_candidate_gt_random"],
            "global_local_null_percentile": _quantiles(selected["global_local_null_percentile"].dropna(), ("q25", "median", "q75", "q95")),
        }
    relations = {}
    for relation, flag in (("same_city", True), ("cross_city", False)):
        c = candidates.loc[candidates["same_city"] == flag]
        r = random.loc[random["same_city"] == flag]
        relations[relation] = {
            "candidate_summary": distribution(c["local_match_ratio"]),
            "random_summary": distribution(r["local_match_ratio"]),
            "paired_candidate_vs_random": paired_summary(paired.loc[paired["same_city"] == flag]),
        }
    strongest = candidates.sort_values(
        ["local_match_ratio", "num_matches", "max_salad_similarity", "pair_uid"],
        ascending=[False, False, False, True], kind="stable",
    ).head(50)
    top = select_top_structural_candidates(candidates)
    result = {
        "candidate_summary": distribution(candidates["local_match_ratio"]),
        "random_summary": distribution(random["local_match_ratio"]),
        "paired_candidate_vs_random": paired_summary(paired),
        "null_calibration": {
            "definition": "weak ECDF: count(random_local_match_ratio <= candidate_local_match_ratio) / random_count",
            "global_null_count": len(random), "minimum_relation_null_count": MIN_RELATION_NULL_COUNT,
            "relation_null_counts": {name: int((random["same_city"] == flag).sum()) for name, flag in (("same_city", True), ("cross_city", False))},
            "global_local_null_percentile": distribution(candidates["global_local_null_percentile"].dropna()),
            "relation_percentile_unavailable_count": int(candidates["relation_local_null_percentile"].isna().sum()),
        },
        "rank_conditioned": rank_conditioned, "city_relation_conditioned": relations,
        "similarity_local_correlations": {
            "max_salad_similarity_vs_local_match_ratio": _correlation(candidates["max_salad_similarity"], candidates["local_match_ratio"]),
            "best_rgb_rank_vs_local_match_ratio": _correlation(candidates["best_rgb_rank"], candidates["local_match_ratio"]),
        },
        "integrity_audit": {
            "candidate": _integrity_group(candidates), "random": _integrity_group(random),
            "strongest50_including_duplicates": _integrity_group(strongest),
            "top50_nonduplicate": _integrity_group(top),
            "exact_pixel_duplicate_count": int(candidates["exact_pixel_duplicate"].sum()),
            "near_geo_500m_count": int(candidates["near_geo_500m"].sum()),
        },
        "top50_structural_candidates": compact_top_metadata(top),
    }
    return candidates, result


def compact_top_metadata(top: pd.DataFrame) -> list[dict]:
    """Represent missing subgroup percentiles as JSON null, never numeric NaN."""
    result = []
    for _, row in top.iterrows():
        record = {}
        for column in TOP_METADATA_COLUMNS:
            value = row[column]
            if pd.isna(value):
                if column not in ("global_local_null_percentile", "relation_local_null_percentile"):
                    raise ValueError(f"top metadata contains missing {column}")
                value = None
            elif isinstance(value, np.generic):
                value = value.item()
            record[column] = value
        result.append(record)
    return result


def validate_snapshot(snapshot: dict) -> None:
    def check(value):
        if isinstance(value, dict):
            for key, child in value.items():
                check(key)
                check(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("snapshot must contain finite JSON numbers")
        elif isinstance(value, str) and (
            value.startswith(("/", "~", "file://", "\\\\")) or (len(value) > 2 and value[1] == ":" and value[2] in ("/", "\\"))
        ):
            raise ValueError("snapshot must not contain absolute paths")
    check(snapshot)
    provenance = snapshot.get("provenance", {})
    if provenance.get("real_rgb_only") is not True or provenance.get("synthetic_images_used") is not False:
        raise ValueError("snapshot requires original real RGB provenance")
    json.dumps(snapshot, allow_nan=False)


def build_snapshot(report: dict, population: dict, config: dict, git_commit: str) -> dict:
    """Allowlist compact, path-free provenance; no runtime paths are copied."""
    snapshot = {
        "schema_version": 1,
        "provenance": {
            "current_git_commit": git_commit,
            **{key: population[key] for key in ("manifest_sha256", "candidate_csv_sha256", "candidate_summary_sha256")},
            "min_geo_distance_m": population["min_geo_distance_m"], "pilot_max_pairs": population["max_pairs"],
            "seed": population["seed"], "aliked_config": config["aliked_config"], "lightglue_config": config["lightglue_config"],
            "real_rgb_only": True, "synthetic_images_used": False, "local_resize": None, "registration": None,
            "input_hashes": config.get("input_hashes", {}),
            "input_image_index_sha256": config.get("input_image_index_sha256"),
            "measurement_source_sha256": config.get("source_sha256", {}),
            "vendor_provenance": config.get("vendor_provenance", {}),
            "matcher_provenance": {key: value for key, value in config.get("matcher_provenance", {}).items() if key in (
                "aliked_weights_sha256", "lightglue_weights_sha256", "torch_version", "cuda_version",
                "cudnn_version", "device", "device_name", "deterministic_algorithms", "cublas_workspace_config",
            )},
        },
        "population": {key: population[key] for key in (
            "raw_directed_candidate_rows", "eligible_geo_directed_rows", "eligible_unique_canonical_pairs",
            "sampled_pair_count", "matched_random_count", "reverse_candidate_count", "same_city_count",
            "cross_city_count", "rank_bin_counts",
        )},
        **report,
    }
    if "sampled_reverse_candidate_count" in population:
        snapshot["population"]["sampled_reverse_candidate_count"] = population["sampled_reverse_candidate_count"]
    validate_snapshot(snapshot)
    return snapshot


def _atomic_bytes(path: str | Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_snapshot(path: str | Path, snapshot: dict) -> None:
    validate_snapshot(snapshot)
    _atomic_bytes(path, (json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def create_figures(candidates: pd.DataFrame, random: pd.DataFrame, output_dir: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paired = pair_candidate_random(candidates, random)
    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    axis.scatter(paired["local_match_ratio_random"], paired["local_match_ratio"], s=9, alpha=.4)
    maximum = max(float(candidates["local_match_ratio"].max()), float(random["local_match_ratio"].max()) if len(random) else 0, .01)
    axis.plot([0, maximum], [0, maximum], color="black", linestyle="--", linewidth=1, label="y = x")
    axis.set(xlabel="Matched random local match ratio", ylabel="SALAD candidate local match ratio")
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "candidate_vs_random_local.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    for name, flag, color in (("Same city", True, "#276fbf"), ("Cross city", False, "#d97925")):
        group = candidates.loc[candidates["same_city"] == flag]
        axis.scatter(group["max_salad_similarity"], group["local_match_ratio"], s=9, alpha=.45, c=color, label=name)
    axis.set(xlabel="Maximum SALAD similarity", ylabel="Local match ratio")
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "salad_vs_local.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5.6))
    groups = [candidates.loc[candidates["rank_bin"] == name, "local_match_ratio"].to_numpy() for name in RANK_BINS]
    # Empty frozen bins retain their positions, without invented observations.
    populated = [(index + 1, values) for index, values in enumerate(groups) if len(values)]
    if populated:
        axis.boxplot([values for _, values in populated], positions=[index for index, _ in populated], whis=(5, 95), showfliers=True)
    axis.set_xticks(range(1, 6), RANK_BINS)
    axis.set(xlabel="Frozen best SALAD rank bin", ylabel="Candidate local match ratio")
    figure.tight_layout()
    figure.savefig(root / "local_by_rank.png", dpi=160)
    plt.close(figure)


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def create_top_montage(
    top: pd.DataFrame, manifest_path: str | Path, dataset_root: str | Path,
    output_path: str | Path, *, fingerprints: dict | None = None,
) -> None:
    """Render full original 640x480 RGB frames; no crop/resize/EXIF transform."""
    root = Path(dataset_root).expanduser().resolve()
    manifest = read_metadata_csv(manifest_path)
    if manifest["image_id"].duplicated().any():
        raise ValueError("manifest image identities must be unique")
    index = manifest.set_index("image_id")
    if len(top) == 0:
        sheet = Image.new("RGB", (1312, 100), "white")
        ImageDraw.Draw(sheet).text((16, 30), "No non-exact-duplicate structural candidates", font=_font(20), fill="black")
    else:
        margin, gap, row_height = 12, 12, 598
        sheet = Image.new("RGB", (2 * 640 + 2 * margin + gap, 52 + row_height * len(top)), "#f8f8f8")
        draw = ImageDraw.Draw(sheet)
        draw.text((margin, 12), "Real RGB structural candidates: non-exact-duplicates, descriptive ranking", font=_font(20), fill="black")
        for order, (_, row) in enumerate(top.iterrows(), start=1):
            y = 52 + (order - 1) * row_height
            for endpoint, x in (("a", margin), ("b", margin + 640 + gap)):
                image_id = row[f"image_id_{endpoint}"]
                if image_id not in index.index:
                    raise ValueError("montage image identity is absent from the manifest")
                metadata = index.loc[image_id]
                for field in ("place_uid", "city_id"):
                    if str(metadata[field]) != str(row[f"{field}_{endpoint}"]):
                        raise ValueError("montage endpoint identity differs from manifest")
                relative = Path(metadata["relative_path"])
                path = (root / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(root):
                    raise ValueError("image path must remain inside the original dataset root")
                encoded = path.read_bytes()
                if fingerprints is not None and hashlib.sha256(encoded).hexdigest() != fingerprints[image_id]["image_file_sha256"]:
                    raise ValueError("montage original image bytes changed after inference")
                with Image.open(io.BytesIO(encoded)) as original:
                    visual = original.convert("RGB")
                    if visual.size != (640, 480):
                        visual.close()
                        raise ValueError("montage source must decode to RGB 640x480")
                if fingerprints is not None and hashlib.sha256(visual.tobytes()).hexdigest() != fingerprints[image_id]["rgb_pixel_sha256"]:
                    visual.close()
                    raise ValueError("montage decoded RGB pixels differ from inference")
                sheet.paste(visual, (x, y))
                visual.close()
                draw.text((x, y + 486), f"{endpoint.upper()} place: {row[f'place_uid_{endpoint}']} | city: {row[f'city_id_{endpoint}']}", font=_font(16), fill="black")
            percentile = row["global_local_null_percentile"]
            percentile_text = "null" if pd.isna(percentile) else f"{percentile:.4f}"
            captions = (
                f"#{order}  geographic distance {row['geo_distance_m']:.1f} m | best SALAD rank {int(row['best_rgb_rank'])} | similarity {row['max_salad_similarity']:.6f}",
                f"Keypoints A/B {int(row['num_keypoints_a'])}/{int(row['num_keypoints_b'])} | matches {int(row['num_matches'])} | local match ratio {row['local_match_ratio']:.6f}",
                f"Source coverage {row['matched_source_cell_coverage']:.4f} | target coverage {row['matched_target_cell_coverage']:.4f} | global local-null percentile {percentile_text}",
            )
            for line, caption in enumerate(captions):
                draw.text((margin, y + 512 + 24 * line), caption, font=_font(17), fill="#222222")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".jpg", delete=False) as stream:
            temporary = Path(stream.name)
        sheet.save(temporary, "JPEG", quality=92, subsampling=0)
        os.replace(temporary, target)
    finally:
        sheet.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def curate_audit_artifacts(output_dir: str | Path, audit_dir: str | Path, *, require_overlays: bool = False) -> list[Path]:
    """Copy completed figures; the separate model-based tool adds its overlay."""
    root, destination = Path(output_dir), Path(audit_dir)
    required = [name for name in CURATED_ARTIFACTS if require_overlays or name != "top20_match_overlays.jpg"]
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(f"completed audit artifact missing: {name}")
    copied = []
    for name in required:
        target = destination / CURATED_ARTIFACTS[name]
        _atomic_bytes(target, (root / name).read_bytes())
        copied.append(target)
    return copied


def export_audit(
    runtime_dir: str | Path, manifest_path: str | Path, candidate_csv_path: str | Path,
    candidate_summary_path: str | Path, dataset_root: str | Path, output_dir: str | Path,
    snapshot_path: str | Path, *, repo_root: str | Path,
) -> dict:
    candidates, random, population, config = load_completed_metrics(runtime_dir, manifest_path, candidate_csv_path, candidate_summary_path)
    if candidates.empty:
        raise ValueError("completed population contains no candidate pairs")
    validate_original_image_fingerprints(runtime_dir, candidates, random, manifest_path, dataset_root)
    candidates, report = analyze_tables(candidates, random)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    snapshot = build_snapshot(report, population, config, git_commit)
    snapshot["provenance"]["measurement_config_sha256"] = sha256_file(Path(runtime_dir) / "measurement_config.json")
    snapshot["provenance"]["analysis_source_sha256"] = {
        name: sha256_file(Path(repo_root) / name) for name in (
            "countermine/mining/structural_analysis.py", "tools/09_analyze_structural_audit.py",
        )
    }
    validate_snapshot(snapshot)
    top = select_top_structural_candidates(candidates)
    create_figures(candidates, random, output_dir)
    with (Path(runtime_dir) / "image_fingerprints.json").open(encoding="utf-8") as stream:
        fingerprints = json.load(stream)
    create_top_montage(top, manifest_path, dataset_root, Path(output_dir) / "top50_structural_candidates.jpg", fingerprints=fingerprints)
    _atomic_bytes(Path(runtime_dir) / "candidate_structural_analysis.csv", candidates.to_csv(index=False, float_format="%.17g", na_rep="").encode("utf-8"))
    _atomic_bytes(Path(runtime_dir) / "top50_structural_candidates.csv", top.to_csv(index=False, float_format="%.17g", na_rep="").encode("utf-8"))
    write_snapshot(Path(runtime_dir) / "structural_audit_summary.json", snapshot)
    curate_audit_artifacts(output_dir, Path(snapshot_path).parent)
    write_snapshot(snapshot_path, snapshot)
    return snapshot
