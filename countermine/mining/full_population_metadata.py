"""CPU metadata adapter of the frozen Step 2A canonicalization logic.

The selected constants and functions are retained from the immutable pilot
modules. Their retrieval module imports Torch eagerly; this project-owned
adapter keeps population building and bank metadata independent of Torch.
Original module hashes and adapter hashes are bound in Step 2D provenance.
No candidate retrieval, model inference or random resampling runs here.
"""
import hashlib
import json
import math
import operator
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ('row_index', 'image_id', 'place_uid', 'city_id', 'lat', 'lon')

CANDIDATE_COLUMNS = ('query_row_index', 'negative_row_index', 'query_image_id', 'negative_image_id', 'query_place_uid', 'negative_place_uid', 'query_city_id', 'negative_city_id', 'rank', 'similarity', 'geo_distance_m', 'same_city', 'pair_uid')

EARTH_RADIUS_M = 6371008.8

def stable_pair_uid(image_id_a: str, image_id_b: str) -> str:
    """SHA256 of an unambiguous, lexicographically sorted pair of image IDs."""
    payload = json.dumps(sorted((image_id_a, image_id_b)), ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def haversine_distance_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distances in meters, with NumPy broadcasting.

    Coordinates are in degrees. Pass ``[:, None]`` query coordinates and
    ``[None, :]`` reference coordinates for a block of pairwise distances, or
    equally shaped vectors to measure only the selected candidate pairs.
    """
    lat1, lon1, lat2, lon2 = (np.deg2rad(np.asarray(value, dtype=np.float64)) for value in (lat1, lon1, lat2, lon2))
    haversine = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))

MIN_GEO_DISTANCE_M = 250.0

GEO_DISTANCE_ATOL_M = 1e-06

RANK_BINS = ('rank_1', 'rank_2_5', 'rank_6_10', 'rank_11_20', 'rank_21_50')

GEO_DISTANCE_BINS = ((250.0, 500.0, '[250,500)'), (500.0, 1000.0, '[500,1000)'), (1000.0, 2000.0, '[1000,2000)'), (2000.0, 5000.0, '[2000,5000)'), (5000.0, math.inf, '[5000,+inf)'))

POPULATION_COLUMNS = ('pair_uid', 'image_id_a', 'image_id_b', 'row_index_a', 'row_index_b', 'place_uid_a', 'place_uid_b', 'city_id_a', 'city_id_b', 'geo_distance_m', 'same_city', 'num_candidate_directions', 'best_rgb_rank', 'max_salad_similarity', 'mean_salad_similarity', 'a_to_b_rank', 'a_to_b_similarity', 'b_to_a_rank', 'b_to_a_similarity', 'anchor_query_image_id', 'anchor_query_row_index', 'anchor_negative_image_id', 'anchor_negative_row_index', 'anchor_rank', 'anchor_similarity', 'rank_bin')

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _integer(value, name: str, minimum: int=0) -> int:
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f'{name} must be an integer >= {minimum}') from error
    if isinstance(value, (bool, np.bool_)) or result < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return result

def rank_bin(rank: int) -> str:
    rank = _integer(rank, 'rank', 1)
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
    raise ValueError('Step 2A rank must be in [1, 50]')

def validate_manifest(manifest: pd.DataFrame) -> None:
    if not manifest.columns.is_unique:
        raise ValueError('manifest column names must be unique')
    missing = set(REQUIRED_COLUMNS).difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest lacks required columns: {', '.join(sorted(missing))}")
    if manifest.empty or manifest.loc[:, REQUIRED_COLUMNS].isna().any().any():
        raise ValueError('manifest is empty or contains missing required metadata')
    if not pd.api.types.is_integer_dtype(manifest['row_index']) or not np.array_equal(manifest['row_index'].to_numpy(), np.arange(len(manifest))):
        raise ValueError('manifest row_index must equal range(N) in existing row order')
    for name in ('image_id', 'place_uid', 'city_id'):
        if not manifest[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f'manifest {name} must contain nonempty strings')
    if not manifest['image_id'].is_unique:
        raise ValueError('manifest image_id must be unique')
    try:
        coordinates = manifest[['lat', 'lon']].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError('manifest coordinates must be numeric') from error
    if not np.isfinite(coordinates).all() or (np.abs(coordinates[:, 0]) > 90).any() or (np.abs(coordinates[:, 1]) > 180).any():
        raise ValueError('manifest coordinates must be finite and within geographic ranges')

def raw_candidate_statistics(candidates: pd.DataFrame) -> dict:
    """Compute the Step 1C summary fields independently from the saved CSV."""
    similarities = candidates['similarity'].to_numpy(dtype=np.float64)
    distances = candidates['geo_distance_m'].to_numpy(dtype=np.float64)

    def stats(values, include_mean=False):
        result = {'min': float(values.min()), 'max': float(values.max())}
        result.update(zip(('q05', 'q25', 'median', 'q75', 'q95'), map(float, np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95]))))
        if include_mean:
            result['mean'] = float(values.mean())
        return result
    counts = candidates.groupby('pair_uid', sort=False).size()
    reverse = int(counts.eq(2).sum())
    return {'number_of_queries': int(candidates['query_row_index'].nunique()), 'total_candidate_pairs': len(candidates), 'similarity_statistics': stats(similarities, True), 'geo_distance_statistics': stats(distances), 'fraction_same_city': float(candidates['same_city'].mean()), 'geo_distance_counts_below_m': {str(cutoff): int(np.count_nonzero(distances < cutoff)) for cutoff in (150, 200, 250, 500, 1000)}, 'rank_distribution': {str(rank): int(count) for rank, count in candidates['rank'].value_counts().sort_index().items()}, 'unique_pair_uids': len(counts), 'reverse_duplicate_pairs': reverse, 'reverse_duplicated_candidate_rows': 2 * reverse, 'reverse_duplicate_fraction': 2 * reverse / len(candidates), 'top_candidates': candidates.sort_values(['similarity', 'query_row_index', 'negative_row_index'], ascending=[False, True, True], kind='stable').head(20).to_dict(orient='records')}

def _compare_summary(actual, expected, location='summary') -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f'{location} must be an object')
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f'{location} missing {key}')
            _compare_summary(actual[key], value, f'{location}.{key}')
        if location != 'summary' and set(actual) != set(expected):
            raise ValueError(f'{location} has unexpected keys')
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f'{location} does not agree with raw CSV')
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_summary(left, right, f'{location}[{index}]')
    elif isinstance(expected, bool):
        if not isinstance(actual, bool) or actual != expected:
            raise ValueError(f'{location} does not agree with raw CSV')
    elif isinstance(expected, int):
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise ValueError(f'{location} does not agree with raw CSV')
    elif isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or (not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f'{location} does not agree with raw CSV')
    elif actual != expected:
        raise ValueError(f'{location} does not agree with raw CSV')

def validate_raw_candidates(manifest: pd.DataFrame, candidates: pd.DataFrame, summary: dict, *, manifest_sha256: str | None=None, expected_top_k: int=50) -> None:
    """Fail closed on inconsistent Step 1C metadata, order, distances or summary.

    ``expected_top_k`` exists for small CPU fixtures. The experiment loader
    always uses the frozen value 50 and requires raw geographic exclusion 0.
    """
    validate_manifest(manifest)
    expected_top_k = _integer(expected_top_k, 'expected_top_k', 1)
    if not isinstance(summary, dict):
        raise ValueError('candidate summary must be an object')
    if summary.get('top_k') != expected_top_k or isinstance(summary.get('top_k'), bool):
        raise ValueError(f'raw summary must declare top_k={expected_top_k}')
    raw_geo = summary.get('min_geo_distance_m')
    if isinstance(raw_geo, bool) or not isinstance(raw_geo, (int, float)) or raw_geo != 0.0:
        raise ValueError('raw summary must declare min_geo_distance_m=0.0')
    if summary.get('stage') != '1C raw RGB diagnostic candidates':
        raise ValueError('candidate summary must identify the Step 1C raw RGB stage')
    if manifest_sha256 is not None and summary.get('manifest_sha256') != manifest_sha256:
        raise ValueError('raw summary manifest_sha256 does not match current manifest')
    if not candidates.columns.is_unique or tuple(candidates.columns) != CANDIDATE_COLUMNS:
        raise ValueError('raw candidate schema must match CANDIDATE_COLUMNS exactly')
    if candidates.empty or candidates.isna().any().any():
        raise ValueError('raw candidates are empty or contain missing values')
    for name in ('query_row_index', 'negative_row_index', 'rank'):
        if not pd.api.types.is_integer_dtype(candidates[name]):
            raise ValueError(f'candidate {name} must contain integers')
    queries = candidates['query_row_index'].to_numpy(dtype=np.int64)
    negatives = candidates['negative_row_index'].to_numpy(dtype=np.int64)
    ranks = candidates['rank'].to_numpy(dtype=np.int64)
    if not np.array_equal(queries, np.repeat(np.arange(len(manifest)), expected_top_k)):
        raise ValueError('raw CSV must contain consecutive complete Top-K for every manifest query')
    if not np.array_equal(ranks, np.tile(np.arange(1, expected_top_k + 1), len(manifest))):
        raise ValueError('raw candidate rank must equal 1..K exactly for each query')
    if (negatives < 0).any() or (negatives >= len(manifest)).any():
        raise ValueError('negative_row_index outside manifest range')
    if candidates.duplicated(['query_row_index', 'negative_row_index']).any():
        raise ValueError('duplicated directed candidate')
    for name in ('image_id', 'place_uid', 'city_id'):
        values = manifest[name].to_numpy()
        for prefix, indices in (('query', queries), ('negative', negatives)):
            if not np.array_equal(candidates[f'{prefix}_{name}'].to_numpy(), values[indices]):
                raise ValueError(f'candidate {prefix}_{name} does not agree with manifest')
    if candidates['query_place_uid'].eq(candidates['negative_place_uid']).any():
        raise ValueError('same-place candidates are forbidden')
    if not pd.api.types.is_bool_dtype(candidates['same_city']) or not np.array_equal(candidates['same_city'].to_numpy(), candidates['query_city_id'].to_numpy() == candidates['negative_city_id'].to_numpy()):
        raise ValueError('same_city must be boolean and agree with manifest cities')
    try:
        similarities = candidates['similarity'].to_numpy(dtype=np.float64)
        distances = candidates['geo_distance_m'].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError('similarity and geo_distance_m must be numeric') from error
    if not np.isfinite(similarities).all() or (np.abs(similarities) > 1.0 + 1e-06).any():
        raise ValueError('candidate cosine similarities must be finite and in [-1,1]')
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError('candidate distances must be finite and nonnegative')
    scores = similarities.reshape(len(manifest), expected_top_k)
    targets = negatives.reshape(len(manifest), expected_top_k)
    if (np.diff(scores, axis=1) > 0).any() or ((np.diff(scores, axis=1) == 0) & (np.diff(targets, axis=1) <= 0)).any():
        raise ValueError('raw Top-K must descend by similarity and break exact ties by row_index')
    coords = manifest[['lat', 'lon']].to_numpy(dtype=np.float64)
    recomputed = haversine_distance_m(coords[queries, 0], coords[queries, 1], coords[negatives, 0], coords[negatives, 1])
    bad = ~np.isclose(distances, recomputed, atol=GEO_DISTANCE_ATOL_M, rtol=0)
    if bad.any():
        index = int(np.flatnonzero(bad)[0])
        raise ValueError(f'saved geo_distance_m disagrees with manifest haversine at CSV row {index}; absolute tolerance={GEO_DISTANCE_ATOL_M} m, rtol=0')
    expected_uids = [stable_pair_uid(a, b) for a, b in zip(candidates['query_image_id'], candidates['negative_image_id'])]
    if not np.array_equal(candidates['pair_uid'].to_numpy(), expected_uids):
        raise ValueError('candidate pair_uid does not equal existing stable_pair_uid')
    _compare_summary(summary, raw_candidate_statistics(candidates))
    if 'descriptor_shape' in summary and (not isinstance(summary['descriptor_shape'], list) or len(summary['descriptor_shape']) != 2 or summary['descriptor_shape'][0] != len(manifest) or (not isinstance(summary['descriptor_shape'][1], int)) or (summary['descriptor_shape'][1] <= 0)):
        raise ValueError('raw summary descriptor_shape disagrees with manifest')

def load_population_inputs(manifest_path, candidate_path, summary_path):
    """Read metadata only and validate the frozen default Step 1C cache."""
    manifest = pd.read_csv(manifest_path, float_precision='round_trip', dtype={'image_id': str, 'place_uid': str, 'city_id': str})
    candidates = pd.read_csv(candidate_path, float_precision='round_trip', dtype={name: str for name in CANDIDATE_COLUMNS if name.endswith(('_id', '_uid'))})
    with Path(summary_path).open(encoding='utf-8') as source:
        summary = json.load(source, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'non-finite JSON constant {value}')))
    validate_raw_candidates(manifest, candidates, summary, manifest_sha256=sha256_file(manifest_path), expected_top_k=50)
    return (manifest, candidates, summary)

def canonicalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Keep >=250 m different-place pairs, aggregate directions, choose anchors."""
    missing = set(CANDIDATE_COLUMNS).difference(candidates.columns)
    if missing:
        raise ValueError(f'raw candidate schema missing columns: {sorted(missing)}')
    if candidates['query_place_uid'].eq(candidates['negative_place_uid']).any():
        raise ValueError('same-place candidates are forbidden')
    if candidates.duplicated(['query_row_index', 'negative_row_index']).any():
        raise ValueError('duplicated directed candidate')
    expected = [stable_pair_uid(a, b) for a, b in zip(candidates['query_image_id'], candidates['negative_image_id'])]
    if not np.array_equal(candidates['pair_uid'].to_numpy(), expected):
        raise ValueError('candidate pair_uid does not equal existing stable_pair_uid')
    eligible = candidates.loc[candidates['geo_distance_m'] >= MIN_GEO_DISTANCE_M, CANDIDATE_COLUMNS].copy()
    if eligible.empty:
        return pd.DataFrame(columns=POPULATION_COLUMNS)
    forward = eligible['query_image_id'].to_numpy() < eligible['negative_image_id'].to_numpy()
    for suffix, is_forward in (('a', forward), ('b', ~forward)):
        for name in ('image_id', 'row_index', 'place_uid', 'city_id'):
            eligible[f'{name}_{suffix}'] = np.where(is_forward, eligible[f'query_{name}'], eligible[f'negative_{name}'])
    grouped = eligible.groupby('pair_uid', sort=True)
    canonical = grouped.agg(image_id_a=('image_id_a', 'first'), image_id_b=('image_id_b', 'first'), row_index_a=('row_index_a', 'first'), row_index_b=('row_index_b', 'first'), place_uid_a=('place_uid_a', 'first'), place_uid_b=('place_uid_b', 'first'), city_id_a=('city_id_a', 'first'), city_id_b=('city_id_b', 'first'), geo_distance_m=('geo_distance_m', 'first'), same_city=('same_city', 'first'), num_candidate_directions=('rank', 'size'), best_rgb_rank=('rank', 'min'), max_salad_similarity=('similarity', 'max'), mean_salad_similarity=('similarity', 'mean'))
    if canonical['num_candidate_directions'].gt(2).any():
        raise ValueError('unordered image pair has more than two directed occurrences')
    for prefix, mask in (('a_to_b', forward), ('b_to_a', ~forward)):
        direction = eligible.loc[mask].set_index('pair_uid')
        canonical[f'{prefix}_rank'] = direction['rank']
        canonical[f'{prefix}_similarity'] = direction['similarity']
    representatives = eligible.sort_values(['rank', 'similarity', 'query_row_index', 'negative_row_index'], ascending=[True, False, True, True], kind='stable').drop_duplicates('pair_uid').set_index('pair_uid')
    for target, source in (('anchor_query_image_id', 'query_image_id'), ('anchor_query_row_index', 'query_row_index'), ('anchor_negative_image_id', 'negative_image_id'), ('anchor_negative_row_index', 'negative_row_index'), ('anchor_rank', 'rank'), ('anchor_similarity', 'similarity')):
        canonical[target] = representatives[source]
    canonical['rank_bin'] = canonical['best_rgb_rank'].map(rank_bin)
    return canonical.reset_index().loc[:, POPULATION_COLUMNS]
