"""Complete canonical frozen SALAD population; no sampling or local evidence."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .full_population_metadata import stable_pair_uid
from .full_structural_io import (
    ROOT, DEFAULT_DIR, code_hashes, frozen_provenance, guard_step2d, read_json,
    relative_path, verify_hashes, write_csv, write_json,
)
from .structural_analysis import read_metadata_csv, sha256_file
from .full_population_metadata import (
    POPULATION_COLUMNS, RANK_BINS, canonicalize_candidates, load_population_inputs, rank_bin,
)

SOURCE_FILES = ('countermine/mining/full_structural_io.py',
                'countermine/mining/full_population_metadata.py',
                'countermine/mining/full_structural_population.py',
                'tools/17_build_full_structural_population.py')


def build_full_population(manifest, candidates, frozen_summary):
    """Reproduce all eligibility/aggregation counts, retaining every edge."""
    result = canonicalize_candidates(candidates).sort_values('pair_uid', kind='stable').reset_index(drop=True)
    counts = {'raw_directed_candidate_rows': len(candidates),
              'eligible_geo_directed_rows': int((candidates['geo_distance_m'] >= 250).sum()),
              'eligible_unique_canonical_pairs': len(result),
              'reverse_candidate_count': int((result['num_candidate_directions'] == 2).sum())}
    for key, value in counts.items():
        if value != frozen_summary.get(key):
            raise ValueError(f'full population disagrees with frozen Step 2A provenance: {key}')
    if not len(result):
        raise ValueError('full population cannot be empty')
    counts.update({'manifest_image_count': len(manifest), 'full_canonical_edge_count': len(result),
                   'same_city_count': int(result['same_city'].sum()),
                   'cross_city_count': int((~result['same_city']).sum()),
                   'unique_images': len(set(result['image_id_a']) | set(result['image_id_b'])),
                   'unique_places': len(set(result['place_uid_a']) | set(result['place_uid_b'])),
                   'rank_bin_counts': {name: int((result['rank_bin'] == name).sum()) for name in RANK_BINS}})
    return result, counts


def validate_full_population(frame):
    if tuple(frame.columns) != POPULATION_COLUMNS:
        raise ValueError('full population schema differs from frozen Step 2A population schema')
    result = frame.copy()
    if result.empty or result['pair_uid'].duplicated().any() or result['pair_uid'].tolist() != sorted(result['pair_uid']):
        raise ValueError('full population must contain unique sorted pair_uids')
    if (result['place_uid_a'] == result['place_uid_b']).any() or not (result['image_id_a'] < result['image_id_b']).all():
        raise ValueError('full population must contain canonical different-place image pairs')
    expected = [stable_pair_uid(a, b) for a, b in zip(result['image_id_a'], result['image_id_b'])]
    if result['pair_uid'].tolist() != expected:
        raise ValueError('full population pair_uid differs from canonical endpoint identities')
    for column in ('row_index_a', 'row_index_b', 'num_candidate_directions', 'best_rgb_rank',
                   'anchor_query_row_index', 'anchor_negative_row_index', 'anchor_rank'):
        values = pd.to_numeric(result[column], errors='raise').to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f'full population {column} must contain nonnegative integers')
        result[column] = values.astype(np.int64)
    for column in ('geo_distance_m', 'max_salad_similarity', 'mean_salad_similarity', 'anchor_similarity'):
        values = pd.to_numeric(result[column], errors='raise').to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f'full population {column} must be finite')
        result[column] = values
    if (result['geo_distance_m'] < 250).any() or not result['num_candidate_directions'].isin((1, 2)).all():
        raise ValueError('full population violates geographic exclusion or reverse aggregation')
    from .graph_analysis import boolean_values
    result['same_city'] = boolean_values(result['same_city'])
    if not np.array_equal(result['same_city'], result['city_id_a'] == result['city_id_b']):
        raise ValueError('full population city relation differs from endpoints')
    if result['rank_bin'].tolist() != result['best_rgb_rank'].map(rank_bin).tolist():
        raise ValueError('full population rank bin differs from best rank')
    for prefix in ('a_to_b', 'b_to_a'):
        for suffix in ('rank', 'similarity'):
            name = prefix + '_' + suffix
            result[name] = pd.to_numeric(result[name].replace('', np.nan), errors='raise')
        if not np.array_equal(result[prefix + '_rank'].notna(), result[prefix + '_similarity'].notna()):
            raise ValueError('directional rank/similarity presence differs')
        values = result.loc[result[prefix + '_rank'].notna(), prefix + '_rank'].to_numpy()
        if not np.isfinite(values).all() or (values < 1).any() or (values > 50).any() or (values != np.floor(values)).any():
            raise ValueError('directional ranks must stay within frozen Top-50')
        similarities = result.loc[result[prefix + '_similarity'].notna(), prefix + '_similarity'].to_numpy()
        if not np.isfinite(similarities).all():
            raise ValueError('present directional similarities must be finite')
    directions = result['a_to_b_rank'].notna().astype(int) + result['b_to_a_rank'].notna().astype(int)
    if not np.array_equal(directions, result['num_candidate_directions']):
        raise ValueError('full population reverse aggregation disagrees with direction metadata')
    rank_values = result[['a_to_b_rank', 'b_to_a_rank']]
    similarity_values = result[['a_to_b_similarity', 'b_to_a_similarity']]
    if not np.array_equal(result['best_rgb_rank'], rank_values.min(axis=1)):
        raise ValueError('full population best rank differs from directed occurrences')
    for name, values in (('max_salad_similarity', similarity_values.max(axis=1)),
                          ('mean_salad_similarity', similarity_values.mean(axis=1))):
        if not np.allclose(result[name], values, rtol=0, atol=1e-12):
            raise ValueError('full population SALAD aggregation differs from directed occurrences')
    return result


def run_build_full_population(manifest_path=ROOT / 'cache/gsv_mini/manifest.csv',
                              candidate_path=ROOT / 'cache/gsv_mini/rgb_candidates_raw.csv',
                              summary_path=ROOT / 'cache/gsv_mini/rgb_candidates_raw_summary.json',
                              output_dir=DEFAULT_DIR, *, repo_root=ROOT):
    root, directory = Path(repo_root).resolve(), Path(output_dir)
    targets = [directory / 'full_population.csv', directory / 'full_population_summary.json']
    guard_step2d(targets, root)
    provenance = frozen_provenance(root)
    frozen = read_json(root / 'cache/countermine_rgb/step2a/population_summary.json')
    hashes = {key: sha256_file(path) for key, path in (
        ('manifest_sha256', manifest_path), ('candidate_csv_sha256', candidate_path),
        ('candidate_summary_sha256', summary_path))}
    if any(hashes[key] != frozen[key] for key in hashes):
        raise ValueError('full population inputs differ from frozen Step 2A source hashes')
    manifest, candidates, _ = load_population_inputs(manifest_path, candidate_path, summary_path)
    population, counts = build_full_population(manifest, candidates, frozen)
    pilot = read_metadata_csv(root / 'cache/countermine_rgb/step2a/population.csv')
    if not set(pilot['pair_uid']).issubset(set(population['pair_uid'])):
        raise ValueError('full population does not contain every frozen pilot pair')
    provenance.update(hashes)
    provenance['code_sha256'] = code_hashes(SOURCE_FILES, root)
    config = {'seed': 42, 'top_k': 50, 'min_geo_distance_m': 250.0,
              'sampling': False, 'rank_filter': False, 'ordering': 'pair_uid ascending'}
    summary = {'schema_version': 1, 'stage': '2D complete canonical candidate population',
               'complete': True, 'configuration': config, 'population': counts, 'provenance': provenance,
               'input_files': {'manifest': relative_path(manifest_path, root),
                               'raw_candidates': relative_path(candidate_path, root),
                               'raw_candidate_summary': relative_path(summary_path, root)},
               'columns': list(population.columns)}
    if any(path.exists() for path in targets):
        existing, saved = load_full_population(directory, repo_root=root)
        if saved['provenance'] != provenance or saved['population'] != counts:
            raise ValueError('existing full population provenance differs; use a fresh Step 2D directory')
        # Reproduce bytes, rather than relying on tolerant scalar comparisons.
        import hashlib
        expected = hashlib.sha256(population.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
        if expected != saved['full_population_sha256']:
            raise ValueError('existing full population differs from deterministic reconstruction')
        return saved
    verify_hashes(provenance['input_file_sha256'], root)
    write_csv(targets[0], population, repo_root=root)
    summary['full_population_sha256'] = sha256_file(targets[0])
    write_json(targets[1], summary, repo_root=root)
    return summary


def load_full_population(output_dir=DEFAULT_DIR, *, repo_root=ROOT):
    directory = Path(output_dir)
    summary = read_json(directory / 'full_population_summary.json')
    if summary.get('complete') is not True or summary.get('schema_version') != 1:
        raise ValueError('full population needs a complete supported summary')
    path = directory / 'full_population.csv'
    if sha256_file(path) != summary['full_population_sha256']:
        raise ValueError('full population CSV checksum differs')
    provenance = summary['provenance']
    verify_hashes(provenance['input_file_sha256'], repo_root)
    if code_hashes(SOURCE_FILES, repo_root) != provenance['code_sha256']:
        raise ValueError('full population implementation changed')
    frame = validate_full_population(read_metadata_csv(path))
    if len(frame) != summary['population']['full_canonical_edge_count'] or list(frame.columns) != summary['columns']:
        raise ValueError('full population count/schema differs')
    return frame, summary
