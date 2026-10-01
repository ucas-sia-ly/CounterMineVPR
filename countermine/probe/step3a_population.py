"""Deterministic real-RGB triplet population for the diagnostic Step 3A pilot.

Metadata is O(N); candidate CSV rows are streamed and only the best eligible
hard candidate per query is retained. No encoder, generator, or training API
is called here. The geographic threshold must come from the RGB miner's saved
summary rather than an independent policy introduced by this pilot.
"""

from collections import Counter, defaultdict
import csv
import hashlib
import math
from pathlib import Path, PureWindowsPath
import random

import numpy as np


SEED_DERIVATION_RULE = (
    'int.from_bytes(SHA256(("CounterMineVPR-Step3A|" + image_id).encode("utf-8"))'
    '.digest()[:8], "big", signed=False) % 2**63'
)
MANIFEST_COLUMNS = ("row_index", "image_id", "relative_path", "place_uid", "city_id", "lat", "lon")
CANDIDATE_COLUMNS = ("query_row_index", "negative_row_index", "query_image_id", "negative_image_id", "rank", "similarity")
ROLES = ("q", "p", "n_hard", "n_random")


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Frozen RGB miner's great-circle formula without its pandas/Torch imports.

    The mean-earth radius, float64 conversion, clipping, and broadcasting match
    ``countermine.mining.candidate_miner.haversine_distance_m`` exactly. Probe
    inference environments only need NumPy for this metadata-only operation.
    """
    lat1, lon1, lat2, lon2 = (
        np.deg2rad(np.asarray(value, dtype=np.float64))
        for value in (lat1, lon1, lat2, lon2)
    )
    haversine = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * 6_371_008.8 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if str(value).strip() != str(result) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _relative(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty relative identifier/path")
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute() or ".." in path.parts or ".." in PureWindowsPath(value).parts:
        raise ValueError(f"{name} must not contain an absolute path or parent traversal")
    return value


def derive_image_seed(image_id):
    """Independent reproducible per-image diffusion noise seed, in [0, 2**63)."""
    _relative(image_id, "image_id")
    digest = hashlib.sha256(("CounterMineVPR-Step3A|" + image_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**63)


def _selection_key(selection_seed, kind, image_id):
    # Distinct namespaces avoid coupling query selection to positive/control
    # choices, and each query's choices survive unrelated metadata reordering.
    return hashlib.sha256(f"CounterMineVPR-Step3A-selection|{selection_seed}|{kind}|{image_id}".encode("utf-8")).digest()


def _csv_rows(path, required):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError(f"{path.name} has duplicate CSV columns")
        missing = set(required).difference(columns)
        if missing:
            raise ValueError(f"{path.name} lacks required columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"{path.name}, CSV line {line}: unexpected extra fields")
            yield row


def _validate_manifest(manifest):
    if hasattr(manifest, "to_dict"):
        manifest = manifest.to_dict("records")
    rows, indices, identifiers = [], set(), set()
    for original in manifest:
        row = dict(original)
        missing = set(MANIFEST_COLUMNS).difference(row)
        if missing:
            raise ValueError(f"manifest lacks required columns: {', '.join(sorted(missing))}")
        row["row_index"] = _integer(row["row_index"], "row_index")
        for key in ("image_id", "relative_path"):
            _relative(row[key], key)
        for key in ("place_uid", "city_id"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"manifest {key} must be a nonempty string")
        for name, limit in (("lat", 90), ("lon", 180)):
            row[name] = float(row[name])
            if not math.isfinite(row[name]) or abs(row[name]) > limit:
                raise ValueError(f"manifest {name} must be a finite coordinate within range")
        if row["row_index"] in indices or row["image_id"] in identifiers:
            raise ValueError("manifest has duplicate row_index or image_id")
        indices.add(row["row_index"])
        identifiers.add(row["image_id"])
        rows.append(row)
    if not rows:
        raise ValueError("manifest contains no images")
    return sorted(rows, key=lambda row: (row["row_index"], row["image_id"]))


def load_manifest(path):
    """Load real-image metadata without filtering or substituting any source."""
    return _validate_manifest(_csv_rows(path, MANIFEST_COLUMNS))


def iter_candidates(path):
    """Stream existing RGB candidates; no descriptor bank is materialized."""
    yield from _csv_rows(path, CANDIDATE_COLUMNS)


def _threshold(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("min_geo_distance_m must be finite and nonnegative")
    return value


def _distance(a, b):
    return float(haversine_distance_m(a["lat"], a["lon"], b["lat"], b["lon"]))


def _role_record(role, row):
    return {f"{role}_{key}": row[key] for key in ("row_index", "image_id", "place_uid", "relative_path")}


def _validate_triplet_with_index(triplet, by_id, threshold):
    images = {}
    for role in ROLES:
        identifier = triplet[f"{role}_image_id"]
        if identifier not in by_id:
            raise ValueError(f"{role}_image_id is outside the original RGB manifest")
        source = by_id[identifier]
        for key in ("row_index", "place_uid", "relative_path"):
            value = triplet[f"{role}_{key}"]
            if key == "row_index":
                value = _integer(value, f"{role}_{key}")
            if value != source[key]:
                raise ValueError(f"{role}_{key} disagrees with the original RGB manifest")
        images[role] = source
    q, p = images["q"], images["p"]
    if q["image_id"] == p["image_id"] or q["place_uid"] != p["place_uid"]:
        raise ValueError("q and p must be distinct images from the same place_uid")
    for role in ("n_hard", "n_random"):
        negative = images[role]
        if negative["image_id"] in (q["image_id"], p["image_id"]) or negative["place_uid"] == q["place_uid"]:
            raise ValueError(f"{role} must be a different image and place_uid from q/p")
        if _distance(q, negative) < threshold:
            raise ValueError(f"{role} violates the saved RGB geographic exclusion")
    return True


def validate_triplet(triplet, manifest, *, min_geo_distance_m):
    """Reject identity, pairing, place, or geography errors in a matched unit."""
    threshold = _threshold(min_geo_distance_m)
    by_id = {row["image_id"]: row for row in _validate_manifest(manifest)}
    return _validate_triplet_with_index(triplet, by_id, threshold)


def validate_triplets(triplets, manifest, *, min_geo_distance_m):
    """Validate a pilot in O(N + T), without revalidating metadata T times."""
    threshold = _threshold(min_geo_distance_m)
    by_id = {row["image_id"]: row for row in _validate_manifest(manifest)}
    seen = set()
    for triplet in triplets:
        _validate_triplet_with_index(triplet, by_id, threshold)
        query_id = triplet["q_image_id"]
        if query_id in seen:
            raise ValueError("Pilot contains duplicate queries")
        seen.add(query_id)
    return True


def select_population(manifest, candidates, *, min_geo_distance_m,
                      query_count=500, selection_seed=42, available_image_ids=None):
    """Return matched triplets and complete eligibility/selection accounting.

    Query selection uses seeded SHA256 ordering, positives and random negatives
    use independent per-query seeded draws over row-index sorted eligible real
    images. Random negatives use precisely the same place/geographic rules as
    hard negatives; they are not filtered by RGB similarity. A chance overlap
    with the selected hard negative is retained and explicitly counted.

    Optional source availability must be supplied explicitly by the caller.
    Every missing source and failed query eligibility is accounted for. Once
    this population is frozen, downstream generation failures never cause this
    function to replace a triplet or image.
    """
    query_count = _integer(query_count, "query_count", minimum=1)
    selection_seed = _integer(selection_seed, "selection_seed")
    threshold = _threshold(min_geo_distance_m)
    ordered = _validate_manifest(manifest)
    by_index = {row["row_index"]: row for row in ordered}
    identifiers = {row["image_id"] for row in ordered}
    available = identifiers if available_image_ids is None else set(available_image_ids)
    if available.difference(identifiers):
        raise ValueError("available_image_ids contains identities outside the RGB manifest")
    candidate_counts = Counter({"candidate_rows": 0, "candidate_rejected_same_place": 0,
                                "candidate_rejected_geography": 0, "candidate_rejected_unavailable": 0,
                                "candidate_eligible_rows": 0})
    best_hard = {}
    if isinstance(candidates, (str, Path)):
        candidates = iter_candidates(candidates)
    elif hasattr(candidates, "to_dict"):
        candidates = candidates.to_dict("records")
    for candidate in candidates:
        candidate_counts["candidate_rows"] += 1
        missing = set(CANDIDATE_COLUMNS).difference(candidate)
        if missing:
            raise ValueError(f"RGB candidate lacks required columns: {', '.join(sorted(missing))}")
        q_index = _integer(candidate["query_row_index"], "query_row_index")
        n_index = _integer(candidate["negative_row_index"], "negative_row_index")
        if q_index not in by_index or n_index not in by_index:
            raise ValueError("RGB candidate index is outside the original manifest")
        q, n = by_index[q_index], by_index[n_index]
        for side, source in (("query", q), ("negative", n)):
            for key in ("image_id", "place_uid", "city_id"):
                name = f"{side}_{key}"
                if name in candidate and candidate[name] != source[key]:
                    raise ValueError(f"RGB candidate {name} disagrees with manifest")
        rank = _integer(candidate["rank"], "rank", minimum=1)
        similarity = float(candidate["similarity"])
        if not math.isfinite(similarity):
            raise ValueError("RGB candidate similarity must be finite")
        distance = _distance(q, n)
        if "geo_distance_m" in candidate:
            recorded = float(candidate["geo_distance_m"])
            if not math.isfinite(recorded) or not math.isclose(recorded, distance, rel_tol=1e-8, abs_tol=1e-5):
                raise ValueError("RGB candidate geo_distance_m disagrees with manifest")
        if q["place_uid"] == n["place_uid"]:
            candidate_counts["candidate_rejected_same_place"] += 1
            continue
        if distance < threshold:
            candidate_counts["candidate_rejected_geography"] += 1
            continue
        if q["image_id"] not in available or n["image_id"] not in available:
            candidate_counts["candidate_rejected_unavailable"] += 1
            continue
        candidate_counts["candidate_eligible_rows"] += 1
        priority = (rank, n_index)
        previous = best_hard.get(q_index)
        if previous is None or priority < previous["priority"]:
            best_hard[q_index] = {"priority": priority, "negative": n,
                                  "rank": rank, "similarity": similarity, "distance": distance}

    available_rows = [row for row in ordered if row["image_id"] in available]
    by_place = defaultdict(list)
    for row in available_rows:
        by_place[row["place_uid"]].append(row)
    places = np.asarray([row["place_uid"] for row in available_rows], dtype=object)
    latitudes = np.asarray([row["lat"] for row in available_rows], dtype=np.float64)
    longitudes = np.asarray([row["lon"] for row in available_rows], dtype=np.float64)
    excluded_counts = Counter({"unavailable_query_source": 0, "missing_same_place_positive": 0,
                               "missing_eligible_rgb_hard_negative": 0, "missing_eligible_random_negative": 0})
    excluded_queries, eligible = [], []
    for q in ordered:
        reason = None
        positives = [row for row in by_place[q["place_uid"]] if row["image_id"] != q["image_id"]]
        if q["image_id"] not in available:
            reason = "unavailable_query_source"
        elif not positives:
            reason = "missing_same_place_positive"
        elif q["row_index"] not in best_hard:
            reason = "missing_eligible_rgb_hard_negative"
        if reason is not None:
            excluded_counts[reason] += 1
            excluded_queries.append({"q_image_id": q["image_id"], "reason": reason})
            continue
        distances = haversine_distance_m(q["lat"], q["lon"], latitudes, longitudes)
        random_indices = np.flatnonzero((places != q["place_uid"]) & (distances >= threshold))
        if not len(random_indices):
            reason = "missing_eligible_random_negative"
            excluded_counts[reason] += 1
            excluded_queries.append({"q_image_id": q["image_id"], "reason": reason})
            continue
        positive_rng = random.Random(int.from_bytes(_selection_key(selection_seed, "positive", q["image_id"]), "big"))
        random_rng = random.Random(int.from_bytes(_selection_key(selection_seed, "random-negative", q["image_id"]), "big"))
        p = positives[positive_rng.randrange(len(positives))]
        random_index = int(random_indices[random_rng.randrange(len(random_indices))])
        n_random = available_rows[random_index]
        hard = best_hard[q["row_index"]]
        triplet = {**_role_record("q", q), **_role_record("p", p),
                   **_role_record("n_hard", hard["negative"]), **_role_record("n_random", n_random),
                   "hard_rgb_rank": hard["rank"], "hard_candidate_similarity": hard["similarity"],
                   "hard_geo_distance_m": hard["distance"], "random_geo_distance_m": float(distances[random_index])}
        eligible.append(triplet)
    eligible.sort(key=lambda row: (_selection_key(selection_seed, "query", row["q_image_id"]), row["q_image_id"]))
    selected = [dict(row, pilot_index=index) for index, row in enumerate(eligible[:query_count])]
    unique_images = {row[f"{role}_image_id"] for row in selected for role in ROLES}
    accounting = {
        "requested_query_count": query_count, "query_count": len(selected),
        "manifest_image_count": len(ordered), "available_image_count": len(available),
        "unavailable_image_count": len(identifiers - available),
        "unavailable_image_ids": sorted(identifiers - available),
        "eligible_query_count": len(eligible), "excluded_query_count": len(excluded_queries),
        "eligible_queries_not_selected": max(0, len(eligible) - query_count),
        "fewer_than_requested": len(selected) < query_count,
        "unique_image_count": len(unique_images),
        "random_hard_image_overlap_count": sum(row["n_random_image_id"] == row["n_hard_image_id"] for row in selected),
        "selection_seed": selection_seed, "min_geo_distance_m": threshold,
        "query_selection": "first requested eligible queries in seeded SHA256 query order",
        "positive_selection": "per-query seeded random draw among available distinct same-place images in row_index order",
        "hard_selection": "smallest eligible existing RGB rank; negative row_index breaks rank ties",
        "random_selection": "per-query seeded random draw among available different-place images satisfying saved geographic rule in row_index order; no similarity filter",
        "selection_hash_rule": 'SHA256(UTF8("CounterMineVPR-Step3A-selection|{selection_seed}|{kind}|{image_id}")); kinds=query,positive,random-negative; full digest big-endian for random.Random seed',
        "query_exclusion_counts": dict(excluded_counts), "query_exclusions": excluded_queries,
        "candidate_accounting": dict(candidate_counts),
        "missing_examples_policy": "explicit eligibility accounting; frozen downstream identities are never replaced",
    }
    assert len(eligible) + len(excluded_queries) == len(ordered)
    return selected, accounting
