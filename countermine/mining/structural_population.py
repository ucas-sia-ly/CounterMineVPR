"""Frozen real-RGB Step 2A population and matched, retrieval-external controls.

Only manifest and retrieval metadata enter selection. Distances are checked
against the manifest with absolute tolerance 1e-6 m (zero relative tolerance).
Local matching results never enter this module. Metadata is O(N*K); no images,
descriptors, local feature banks, or full pairwise distance matrix are loaded.
"""

import hashlib
import json
import math
import operator
from pathlib import Path
import random
import subprocess
import tempfile

import numpy as np
import pandas as pd

from countermine.mining.candidate_miner import (
    CANDIDATE_COLUMNS,
    REQUIRED_COLUMNS,
    haversine_distance_m,
    stable_pair_uid,
)


MIN_GEO_DISTANCE_M = 250.0
GEO_DISTANCE_ATOL_M = 1e-6
RANK_BINS = ("rank_1", "rank_2_5", "rank_6_10", "rank_11_20", "rank_21_50")
GEO_DISTANCE_BINS = (
    (250.0, 500.0, "[250,500)"),
    (500.0, 1000.0, "[500,1000)"),
    (1000.0, 2000.0, "[1000,2000)"),
    (2000.0, 5000.0, "[2000,5000)"),
    (5000.0, math.inf, "[5000,+inf)"),
)
POPULATION_COLUMNS = (
    "pair_uid", "image_id_a", "image_id_b", "row_index_a", "row_index_b",
    "place_uid_a", "place_uid_b", "city_id_a", "city_id_b", "geo_distance_m",
    "same_city", "num_candidate_directions", "best_rgb_rank",
    "max_salad_similarity", "mean_salad_similarity", "a_to_b_rank",
    "a_to_b_similarity", "b_to_a_rank", "b_to_a_similarity",
    "anchor_query_image_id", "anchor_query_row_index", "anchor_negative_image_id",
    "anchor_negative_row_index", "anchor_rank", "anchor_similarity", "rank_bin",
)
CONTROL_COLUMNS = (
    "pair_uid", "candidate_pair_uid", "random_pair_uid", "random_control_available",
    "anchor_query_image_id", "anchor_query_row_index", "anchor_query_place_uid",
    "anchor_query_city_id", "candidate_negative_image_id", "candidate_negative_row_index",
    "candidate_negative_place_uid", "candidate_negative_city_id", "random_negative_image_id",
    "random_negative_row_index", "random_negative_place_uid", "random_negative_city_id",
    "candidate_geo_distance_m", "random_geo_distance_m", "candidate_same_city",
    "random_same_city", "same_city", "geo_distance_bin", "rank_bin",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value, name: str, minimum: int = 0) -> int:
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if isinstance(value, (bool, np.bool_)) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def rank_bin(rank: int) -> str:
    rank = _integer(rank, "rank", 1)
    if rank == 1:
        return RANK_BINS[0]
    if rank <= 5:
        return RANK_BINS[1]
    if rank <= 10:
        return RANK_BINS[2]
    if rank <= 20:
        return RANK_BINS[3]
    if rank <= 50:
        return RANK_BINS[4]
    raise ValueError("Step 2A rank must be in [1, 50]")


def geo_distance_bin(distance_m: float) -> str:
    distance_m = float(distance_m)
    if not math.isfinite(distance_m) or distance_m < MIN_GEO_DISTANCE_M:
        raise ValueError("distance must be finite and >=250 m")
    return next(label for lower, upper, label in GEO_DISTANCE_BINS if lower <= distance_m < upper)


def validate_manifest(manifest: pd.DataFrame) -> None:
    if not manifest.columns.is_unique:
        raise ValueError("manifest column names must be unique")
    missing = set(REQUIRED_COLUMNS).difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest lacks required columns: {', '.join(sorted(missing))}")
    if manifest.empty or manifest.loc[:, REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("manifest is empty or contains missing required metadata")
    if (
        not pd.api.types.is_integer_dtype(manifest["row_index"])
        or not np.array_equal(manifest["row_index"].to_numpy(), np.arange(len(manifest)))
    ):
        raise ValueError("manifest row_index must equal range(N) in existing row order")
    for name in ("image_id", "place_uid", "city_id"):
        if not manifest[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"manifest {name} must contain nonempty strings")
    if not manifest["image_id"].is_unique:
        raise ValueError("manifest image_id must be unique")
    try:
        coordinates = manifest[["lat", "lon"]].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("manifest coordinates must be numeric") from error
    if (
        not np.isfinite(coordinates).all()
        or (np.abs(coordinates[:, 0]) > 90).any()
        or (np.abs(coordinates[:, 1]) > 180).any()
    ):
        raise ValueError("manifest coordinates must be finite and within geographic ranges")


def raw_candidate_statistics(candidates: pd.DataFrame) -> dict:
    """Compute the Step 1C summary fields independently from the saved CSV."""
    similarities = candidates["similarity"].to_numpy(dtype=np.float64)
    distances = candidates["geo_distance_m"].to_numpy(dtype=np.float64)
    def stats(values, include_mean=False):
        result = {"min": float(values.min()), "max": float(values.max())}
        result.update(zip(("q05", "q25", "median", "q75", "q95"),
                          map(float, np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95]))))
        if include_mean:
            result["mean"] = float(values.mean())
        return result
    counts = candidates.groupby("pair_uid", sort=False).size()
    reverse = int(counts.eq(2).sum())
    return {
        "number_of_queries": int(candidates["query_row_index"].nunique()),
        "total_candidate_pairs": len(candidates),
        "similarity_statistics": stats(similarities, True),
        "geo_distance_statistics": stats(distances),
        "fraction_same_city": float(candidates["same_city"].mean()),
        "geo_distance_counts_below_m": {
            str(cutoff): int(np.count_nonzero(distances < cutoff))
            for cutoff in (150, 200, 250, 500, 1000)
        },
        "rank_distribution": {str(rank): int(count) for rank, count in
                              candidates["rank"].value_counts().sort_index().items()},
        "unique_pair_uids": len(counts),
        "reverse_duplicate_pairs": reverse,
        "reverse_duplicated_candidate_rows": 2 * reverse,
        "reverse_duplicate_fraction": 2 * reverse / len(candidates),
        "top_candidates": candidates.sort_values(
            ["similarity", "query_row_index", "negative_row_index"],
            ascending=[False, True, True], kind="stable",
        ).head(20).to_dict(orient="records"),
    }


def _compare_summary(actual, expected, location="summary") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{location} must be an object")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{location} missing {key}")
            _compare_summary(actual[key], value, f"{location}.{key}")
        # Stats/rank/count subobjects must not silently contain contradictory bins.
        if location != "summary" and set(actual) != set(expected):
            raise ValueError(f"{location} has unexpected keys")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{location} does not agree with raw CSV")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_summary(left, right, f"{location}[{index}]")
    elif isinstance(expected, bool):
        if not isinstance(actual, bool) or actual != expected:
            raise ValueError(f"{location} does not agree with raw CSV")
    elif isinstance(expected, int):
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise ValueError(f"{location} does not agree with raw CSV")
    elif isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{location} does not agree with raw CSV")
    elif actual != expected:
        raise ValueError(f"{location} does not agree with raw CSV")


def validate_raw_candidates(
    manifest: pd.DataFrame,
    candidates: pd.DataFrame,
    summary: dict,
    *,
    manifest_sha256: str | None = None,
    expected_top_k: int = 50,
) -> None:
    """Fail closed on inconsistent Step 1C metadata, order, distances or summary.

    ``expected_top_k`` exists for small CPU fixtures. The experiment loader
    always uses the frozen value 50 and requires raw geographic exclusion 0.
    """
    validate_manifest(manifest)
    expected_top_k = _integer(expected_top_k, "expected_top_k", 1)
    if not isinstance(summary, dict):
        raise ValueError("candidate summary must be an object")
    if summary.get("top_k") != expected_top_k or isinstance(summary.get("top_k"), bool):
        raise ValueError(f"raw summary must declare top_k={expected_top_k}")
    raw_geo = summary.get("min_geo_distance_m")
    if isinstance(raw_geo, bool) or not isinstance(raw_geo, (int, float)) or raw_geo != 0.0:
        raise ValueError("raw summary must declare min_geo_distance_m=0.0")
    if summary.get("stage") != "1C raw RGB diagnostic candidates":
        raise ValueError("candidate summary must identify the Step 1C raw RGB stage")
    if manifest_sha256 is not None and summary.get("manifest_sha256") != manifest_sha256:
        raise ValueError("raw summary manifest_sha256 does not match current manifest")
    if not candidates.columns.is_unique or tuple(candidates.columns) != CANDIDATE_COLUMNS:
        raise ValueError("raw candidate schema must match CANDIDATE_COLUMNS exactly")
    if candidates.empty or candidates.isna().any().any():
        raise ValueError("raw candidates are empty or contain missing values")
    for name in ("query_row_index", "negative_row_index", "rank"):
        if not pd.api.types.is_integer_dtype(candidates[name]):
            raise ValueError(f"candidate {name} must contain integers")
    queries = candidates["query_row_index"].to_numpy(dtype=np.int64)
    negatives = candidates["negative_row_index"].to_numpy(dtype=np.int64)
    ranks = candidates["rank"].to_numpy(dtype=np.int64)
    if not np.array_equal(queries, np.repeat(np.arange(len(manifest)), expected_top_k)):
        raise ValueError("raw CSV must contain consecutive complete Top-K for every manifest query")
    if not np.array_equal(ranks, np.tile(np.arange(1, expected_top_k + 1), len(manifest))):
        raise ValueError("raw candidate rank must equal 1..K exactly for each query")
    if (negatives < 0).any() or (negatives >= len(manifest)).any():
        raise ValueError("negative_row_index outside manifest range")
    if candidates.duplicated(["query_row_index", "negative_row_index"]).any():
        raise ValueError("duplicated directed candidate")
    for name in ("image_id", "place_uid", "city_id"):
        values = manifest[name].to_numpy()
        for prefix, indices in (("query", queries), ("negative", negatives)):
            if not np.array_equal(candidates[f"{prefix}_{name}"].to_numpy(), values[indices]):
                raise ValueError(f"candidate {prefix}_{name} does not agree with manifest")
    if candidates["query_place_uid"].eq(candidates["negative_place_uid"]).any():
        raise ValueError("same-place candidates are forbidden")
    if not pd.api.types.is_bool_dtype(candidates["same_city"]) or not np.array_equal(
        candidates["same_city"].to_numpy(),
        candidates["query_city_id"].to_numpy() == candidates["negative_city_id"].to_numpy(),
    ):
        raise ValueError("same_city must be boolean and agree with manifest cities")
    try:
        similarities = candidates["similarity"].to_numpy(dtype=np.float64)
        distances = candidates["geo_distance_m"].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("similarity and geo_distance_m must be numeric") from error
    if not np.isfinite(similarities).all() or (np.abs(similarities) > 1.0 + 1e-6).any():
        raise ValueError("candidate cosine similarities must be finite and in [-1,1]")
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("candidate distances must be finite and nonnegative")
    scores = similarities.reshape(len(manifest), expected_top_k)
    targets = negatives.reshape(len(manifest), expected_top_k)
    if (np.diff(scores, axis=1) > 0).any() or (
        (np.diff(scores, axis=1) == 0) & (np.diff(targets, axis=1) <= 0)
    ).any():
        raise ValueError("raw Top-K must descend by similarity and break exact ties by row_index")
    coords = manifest[["lat", "lon"]].to_numpy(dtype=np.float64)
    recomputed = haversine_distance_m(
        coords[queries, 0], coords[queries, 1], coords[negatives, 0], coords[negatives, 1],
    )
    bad = ~np.isclose(distances, recomputed, atol=GEO_DISTANCE_ATOL_M, rtol=0)
    if bad.any():
        index = int(np.flatnonzero(bad)[0])
        raise ValueError(f"saved geo_distance_m disagrees with manifest haversine at CSV row {index}; "
                         f"absolute tolerance={GEO_DISTANCE_ATOL_M} m, rtol=0")
    expected_uids = [stable_pair_uid(a, b) for a, b in zip(
        candidates["query_image_id"], candidates["negative_image_id"])]
    if not np.array_equal(candidates["pair_uid"].to_numpy(), expected_uids):
        raise ValueError("candidate pair_uid does not equal existing stable_pair_uid")
    _compare_summary(summary, raw_candidate_statistics(candidates))
    if "descriptor_shape" in summary and (
        not isinstance(summary["descriptor_shape"], list)
        or len(summary["descriptor_shape"]) != 2
        or summary["descriptor_shape"][0] != len(manifest)
        or not isinstance(summary["descriptor_shape"][1], int)
        or summary["descriptor_shape"][1] <= 0
    ):
        raise ValueError("raw summary descriptor_shape disagrees with manifest")


def load_population_inputs(manifest_path, candidate_path, summary_path):
    """Read metadata only and validate the frozen default Step 1C cache."""
    manifest = pd.read_csv(manifest_path, float_precision="round_trip", dtype={
        "image_id": str, "place_uid": str, "city_id": str,
    })
    candidates = pd.read_csv(candidate_path, float_precision="round_trip", dtype={
        name: str for name in CANDIDATE_COLUMNS if name.endswith(("_id", "_uid"))
    })
    with Path(summary_path).open(encoding="utf-8") as source:
        summary = json.load(source, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value}")))
    validate_raw_candidates(manifest, candidates, summary,
                            manifest_sha256=sha256_file(manifest_path), expected_top_k=50)
    return manifest, candidates, summary


def canonicalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Keep >=250 m different-place pairs, aggregate directions, choose anchors."""
    missing = set(CANDIDATE_COLUMNS).difference(candidates.columns)
    if missing:
        raise ValueError(f"raw candidate schema missing columns: {sorted(missing)}")
    if candidates["query_place_uid"].eq(candidates["negative_place_uid"]).any():
        raise ValueError("same-place candidates are forbidden")
    if candidates.duplicated(["query_row_index", "negative_row_index"]).any():
        raise ValueError("duplicated directed candidate")
    expected = [stable_pair_uid(a, b) for a, b in zip(
        candidates["query_image_id"], candidates["negative_image_id"])]
    if not np.array_equal(candidates["pair_uid"].to_numpy(), expected):
        raise ValueError("candidate pair_uid does not equal existing stable_pair_uid")
    eligible = candidates.loc[candidates["geo_distance_m"] >= MIN_GEO_DISTANCE_M,
                              CANDIDATE_COLUMNS].copy()
    if eligible.empty:
        return pd.DataFrame(columns=POPULATION_COLUMNS)
    forward = eligible["query_image_id"].to_numpy() < eligible["negative_image_id"].to_numpy()
    for suffix, is_forward in (("a", forward), ("b", ~forward)):
        for name in ("image_id", "row_index", "place_uid", "city_id"):
            eligible[f"{name}_{suffix}"] = np.where(
                is_forward, eligible[f"query_{name}"], eligible[f"negative_{name}"])
    grouped = eligible.groupby("pair_uid", sort=True)
    canonical = grouped.agg(
        image_id_a=("image_id_a", "first"), image_id_b=("image_id_b", "first"),
        row_index_a=("row_index_a", "first"), row_index_b=("row_index_b", "first"),
        place_uid_a=("place_uid_a", "first"), place_uid_b=("place_uid_b", "first"),
        city_id_a=("city_id_a", "first"), city_id_b=("city_id_b", "first"),
        geo_distance_m=("geo_distance_m", "first"), same_city=("same_city", "first"),
        num_candidate_directions=("rank", "size"), best_rgb_rank=("rank", "min"),
        max_salad_similarity=("similarity", "max"), mean_salad_similarity=("similarity", "mean"),
    )
    if canonical["num_candidate_directions"].gt(2).any():
        raise ValueError("unordered image pair has more than two directed occurrences")
    for prefix, mask in (("a_to_b", forward), ("b_to_a", ~forward)):
        direction = eligible.loc[mask].set_index("pair_uid")
        canonical[f"{prefix}_rank"] = direction["rank"]
        canonical[f"{prefix}_similarity"] = direction["similarity"]
    representatives = eligible.sort_values(
        ["rank", "similarity", "query_row_index", "negative_row_index"],
        ascending=[True, False, True, True], kind="stable",
    ).drop_duplicates("pair_uid").set_index("pair_uid")
    for target, source in (
        ("anchor_query_image_id", "query_image_id"), ("anchor_query_row_index", "query_row_index"),
        ("anchor_negative_image_id", "negative_image_id"),
        ("anchor_negative_row_index", "negative_row_index"), ("anchor_rank", "rank"),
        ("anchor_similarity", "similarity"),
    ):
        canonical[target] = representatives[source]
    canonical["rank_bin"] = canonical["best_rgb_rank"].map(rank_bin)
    return canonical.reset_index().loc[:, POPULATION_COLUMNS]


def sample_candidate_population(canonical: pd.DataFrame, max_pairs: int = 5000, seed: int = 42):
    max_pairs = _integer(max_pairs, "max_pairs")
    seed = _integer(seed, "seed")
    if seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if not canonical["pair_uid"].is_unique:
        raise ValueError("canonical pair_uid must be unique")
    sampled = canonical.copy()
    sampled["sampling_key"] = sampled["pair_uid"].map(lambda uid: hashlib.sha256(
        f"CounterMineVPR-Step2A|{seed}|{uid}".encode("utf-8")).hexdigest())
    sampled = sampled.sort_values(["sampling_key", "pair_uid"], kind="stable")
    if max_pairs:
        sampled = sampled.head(max_pairs)
    return sampled.reset_index(drop=True)


def build_random_controls(
    population: pd.DataFrame,
    manifest: pd.DataFrame,
    candidates: pd.DataFrame,
    seed: int = 42,
    min_available_fraction: float = 0.95,
) -> pd.DataFrame:
    """One matched target per pair, drawn outside the anchor's entire Top-50.

    Sorted eligible row indices feed Python Random(full SHA256 integer).randrange.
    A missing stratum has an explicit unavailable row; no fallback is permitted.
    The optional availability argument supports isolated CPU fixtures; production
    construction always enforces >=95% without exposing a CLI override.
    """
    validate_manifest(manifest)
    seed = _integer(seed, "seed")
    if seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if not math.isfinite(min_available_fraction) or not 0 <= min_available_fraction <= 1:
        raise ValueError("min_available_fraction must be in [0,1]")
    if not population["pair_uid"].is_unique:
        raise ValueError("population pair_uid must be unique")
    retrieved = {int(q): set(map(int, group["negative_row_index"])) for q, group in
                 candidates.groupby("query_row_index", sort=False)}
    cities = manifest["city_id"].to_numpy()
    places = manifest["place_uid"].to_numpy()
    image_ids = manifest["image_id"].to_numpy()
    coords = manifest[["lat", "lon"]].to_numpy(dtype=np.float64)
    city_rows = {city: np.flatnonzero(cities == city) for city in manifest["city_id"].unique()}
    records = []
    # Only one anchor's O(N) distances/eligible rows are kept at a time.
    ordered = population.sort_values(["anchor_query_row_index", "pair_uid"], kind="stable")
    for query, group in ordered.groupby("anchor_query_row_index", sort=True):
        query = int(query)
        if query not in retrieved:
            raise ValueError("anchor query has no saved Top-K list")
        distances = haversine_distance_m(coords[query, 0], coords[query, 1],
                                         coords[:, 0], coords[:, 1])
        forbidden = np.zeros(len(manifest), dtype=bool)
        forbidden[list(retrieved[query])] = True
        forbidden[query] = True
        eligible_base = (~forbidden) & (places != places[query]) & (distances >= MIN_GEO_DISTANCE_M)
        for pair in group.itertuples(index=False):
            negative = int(pair.anchor_negative_row_index)
            if image_ids[query] != pair.anchor_query_image_id or image_ids[negative] != pair.anchor_negative_image_id:
                raise ValueError("population anchor identities do not agree with manifest")
            if negative not in retrieved[query]:
                raise ValueError("population candidate is outside anchor saved Top-K list")
            if places[query] == places[negative] or float(pair.geo_distance_m) < MIN_GEO_DISTANCE_M:
                raise ValueError("population anchor must be a >=250m different-place pair")
            relation = bool(cities[query] == cities[negative])
            if relation != pair.same_city or not math.isclose(
                float(pair.geo_distance_m), float(distances[negative]),
                abs_tol=GEO_DISTANCE_ATOL_M, rel_tol=0,
            ):
                raise ValueError("population city relation/distance disagrees with manifest")
            label = geo_distance_bin(pair.geo_distance_m) if relation else None
            targets = city_rows[cities[negative]]
            valid = eligible_base[targets].copy()
            if relation:
                lower, upper, _ = next(item for item in GEO_DISTANCE_BINS if item[2] == label)
                valid &= (distances[targets] >= lower) & (distances[targets] < upper)
            targets = targets[valid]  # flatnonzero and filtering preserve ascending row_index.
            draw_seed = int(hashlib.sha256(
                f"CounterMineVPR-Step2A-random|{seed}|{pair.pair_uid}".encode("utf-8")
            ).hexdigest(), 16)
            target = int(targets[random.Random(draw_seed).randrange(len(targets))]) if len(targets) else None
            record = {
                "pair_uid": pair.pair_uid, "candidate_pair_uid": pair.pair_uid,
                "random_pair_uid": stable_pair_uid(image_ids[query], image_ids[target]) if target is not None else None,
                "random_control_available": target is not None,
                "anchor_query_image_id": image_ids[query], "anchor_query_row_index": query,
                "anchor_query_place_uid": places[query], "anchor_query_city_id": cities[query],
                "candidate_negative_image_id": image_ids[negative], "candidate_negative_row_index": negative,
                "candidate_negative_place_uid": places[negative], "candidate_negative_city_id": cities[negative],
                "random_negative_image_id": image_ids[target] if target is not None else None,
                "random_negative_row_index": target,
                "random_negative_place_uid": places[target] if target is not None else None,
                "random_negative_city_id": cities[target] if target is not None else None,
                "candidate_geo_distance_m": float(pair.geo_distance_m),
                "random_geo_distance_m": float(distances[target]) if target is not None else None,
                "candidate_same_city": relation,
                "random_same_city": bool(cities[query] == cities[target]) if target is not None else None,
                "same_city": relation, "geo_distance_bin": label, "rank_bin": pair.rank_bin,
            }
            records.append(record)
    controls = pd.DataFrame(records, columns=CONTROL_COLUMNS)
    if not controls.empty:
        controls = controls.set_index("pair_uid").loc[population["pair_uid"]].reset_index()
    available = int(controls["random_control_available"].sum())
    if len(population) and available / len(population) < min_available_fraction:
        raise ValueError(f"matched random controls available for {available}/{len(population)} "
                         f"({available / len(population):.3%}); require >=95% for the pilot")
    return controls


def _distribution(values: pd.Series) -> dict:
    data = values.to_numpy(dtype=np.float64)
    if not len(data):
        return {"count": 0}
    result = {"count": len(data), "min": float(data.min()), "max": float(data.max()),
              "mean": float(data.mean())}
    result.update(zip(("q05", "q25", "median", "q75", "q95"),
                      map(float, np.quantile(data, [0.05, 0.25, 0.5, 0.75, 0.95]))))
    return result


def build_structural_population(manifest, candidates, *, max_pairs=5000, seed=42):
    canonical = canonicalize_candidates(candidates)
    population = sample_candidate_population(canonical, max_pairs, seed)
    if population.empty:
        raise ValueError("no eligible >=250m different-place canonical candidates")
    controls = build_random_controls(population, manifest, candidates, seed)
    same_city_count = int(population["same_city"].sum())
    summary = {
        "stage": "2A real-RGB structural-confusion population",
        "raw_directed_candidate_rows": len(candidates),
        "eligible_geo_directed_rows": int(candidates["geo_distance_m"].ge(MIN_GEO_DISTANCE_M).sum()),
        "eligible_unique_canonical_pairs": len(canonical),
        "reverse_candidate_count": int(canonical["num_candidate_directions"].eq(2).sum()),
        "sampled_reverse_candidate_count": int(population["num_candidate_directions"].eq(2).sum()),
        "sampled_pair_count": len(population),
        "matched_random_count": int(controls["random_control_available"].sum()),
        "matched_random_fraction": float(controls["random_control_available"].mean()),
        "same_city_count": same_city_count,
        "cross_city_count": len(population) - same_city_count,
        "best_rank_distribution": {str(rank): int(count) for rank, count in
                                   population["best_rgb_rank"].value_counts().sort_index().items()},
        "rank_bin_counts": {name: int(population["rank_bin"].eq(name).sum()) for name in RANK_BINS},
        "similarity_distribution": _distribution(population["max_salad_similarity"]),
        "seed": int(seed), "max_pairs": int(max_pairs), "top_k": 50,
        "raw_min_geo_distance_m": 0.0, "min_geo_distance_m": MIN_GEO_DISTANCE_M,
        "geo_distance_validation": {"absolute_tolerance_m": GEO_DISTANCE_ATOL_M, "relative_tolerance": 0.0},
        "sampling_definition": f'SHA256("CounterMineVPR-Step2A|{seed}|" + pair_uid), ascending sampling_key then pair_uid; first max_pairs (0 means all)',
        "random_sampling_definition": f'Sorted eligible row_index; Python Random(int(SHA256("CounterMineVPR-Step2A-random|{seed}|" + pair_uid),16)).randrange(count)',
        "random_control_min_available_fraction": 0.95,
        "same_city_geo_distance_bins": [label for _, _, label in GEO_DISTANCE_BINS],
        "real_rgb_only": True, "synthetic_images_used": False,
        "rank_bins": list(RANK_BINS),
    }
    return population, controls, summary


def write_population_artifacts(population, controls, summary, output_dir, *,
                               manifest_path, candidate_path, candidate_summary_path):
    """Publish CSVs atomically, then the summary as the completion marker.

    A reader must verify summary output hashes. It cannot reuse an interrupted
    mixed-generation publication. Changed inputs/configuration fail before any
    existing completed pilot is overwritten.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"population": output_dir / "population.csv",
             "random_controls": output_dir / "random_controls.csv",
             "summary": output_dir / "population_summary.json"}
    inputs = [Path(path).resolve() for path in (manifest_path, candidate_path, candidate_summary_path)]
    if any(path.resolve() in inputs for path in paths.values()):
        raise ValueError("population output paths must be distinct from input files")
    provenance = dict(summary)
    current_hashes = {"manifest_sha256": sha256_file(manifest_path),
                      "candidate_csv_sha256": sha256_file(candidate_path),
                      "candidate_summary_sha256": sha256_file(candidate_summary_path)}
    for key, value in current_hashes.items():
        if key in provenance and provenance[key] != value:
            raise ValueError(f"input changed during construction: {key}")
    provenance.update(current_hashes)
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]),
                                 "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=5, check=False)
        provenance["counterminevpr_git_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        provenance["counterminevpr_git_commit"] = None
    provenance["population_builder_sha256"] = sha256_file(__file__)
    provenance["python_random_selection"] = "CPython random.Random integer-seeded randrange"
    if not paths["summary"].exists() and any(output_dir.glob("*structural_metrics.csv")):
        raise ValueError("existing structural measurements have no population provenance; use a separate output directory")
    if paths["summary"].exists():
        previous = json.loads(paths["summary"].read_text(encoding="utf-8"))
        for key in ("manifest_sha256", "candidate_csv_sha256", "candidate_summary_sha256",
                    "seed", "max_pairs", "min_geo_distance_m", "sampling_definition",
                    "random_sampling_definition", "population_builder_sha256"):
            if previous.get(key) != provenance.get(key):
                raise ValueError(f"existing population provenance differs: {key}; use a separate output directory")
    temporary = []
    try:
        for name, table in (("population", population), ("random_controls", controls)):
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                             dir=output_dir, prefix=paths[name].name + ".",
                                             suffix=".partial", delete=False) as destination:
                temp_path = Path(destination.name)
                temporary.append((temp_path, paths[name]))
                table.to_csv(destination, index=False, float_format="%.17g")
            provenance[f"{name}_sha256"] = sha256_file(temp_path)
        if paths["summary"].exists():
            for key in ("population_sha256", "random_controls_sha256"):
                if previous.get(key) != provenance[key]:
                    raise ValueError(f"same provenance produced a different {key}; refuse to overwrite")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir,
                                         prefix=paths["summary"].name + ".", suffix=".partial",
                                         delete=False) as destination:
            temp_path = Path(destination.name)
            temporary.append((temp_path, paths["summary"]))
            destination.write(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n")
        for temporary_path, destination_path in temporary:
            temporary_path.replace(destination_path)
    finally:
        for temporary_path, _ in temporary:
            temporary_path.unlink(missing_ok=True)
    return provenance
