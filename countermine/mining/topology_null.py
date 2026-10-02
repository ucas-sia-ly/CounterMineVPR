"""Stratified evidence-bundle permutations on a fixed real-RGB candidate graph."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from countermine.mining.graph_analysis import boolean_values
from countermine.mining.joint_null import (
    ROOT, RANK_BINS, guard_destinations, load_frozen_inputs, publish_stage,
)
from pathlib import Path
from countermine.mining.structural_analysis import distribution

BUNDLE_COLUMNS = ('ratio_null_percentile', 'match_count_null_percentile', 'structural_bottleneck',
                  'structural_geomean', 'core_q95', 'core_q99')
SLICES = ('core_q95', 'core_q99', 'core_q95_geo500')
STATISTICS = ('edge_count', 'repeated_support_2', 'repeated_support_3', 'core_independent_support_2',
              'core_independent_support_3', 'active_places', 'non_singleton_place_components',
              'largest_place_component', 'max_place_degree', 'active_images',
              'largest_image_component', 'max_image_degree')


def positive_integer(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if allow_zero else "positive"} integer')
    return int(value)


def build_strata(edges):
    relations = boolean_values(edges['same_city'])
    ranks = edges['rank_bin'].to_numpy()
    geography = pd.to_numeric(edges['geo_distance_m'], errors='raise').to_numpy(float)
    if not np.isfinite(geography).all() or (geography < 250).any() or not edges['rank_bin'].isin(RANK_BINS).all():
        raise ValueError('strata require frozen rank bins and finite >=250m distances')
    core_flags = {name: boolean_values(edges[name]) for name in ('core_q95', 'core_q99') if name in edges}
    strata, records = [], []
    for relation, same in (('same_city', True), ('cross_city', False)):
        for rank in RANK_BINS:
            for geo, near in (('geo_250_500', True), ('geo_ge500', False)):
                positions = np.flatnonzero((relations == same) & (ranks == rank) & ((geography < 500) == near))
                strata.append(positions)
                records.append({'city_relation': relation, 'rank_bin': rank, 'geography_bin': geo,
                                'edge_count': len(positions), 'unchanged': len(positions) < 2,
                                **{name + '_edge_count': int(flags[positions].sum()) for name, flags in core_flags.items()}})
    return strata, records


def permutation_indices(strata, num_edges, *, seed=42, permutation_index=0):
    seed = positive_integer(seed, 'seed', allow_zero=True)
    index = positive_integer(permutation_index, 'permutation_index', allow_zero=True)
    num_edges = positive_integer(num_edges, 'num_edges', allow_zero=True)
    flattened = np.concatenate(strata) if strata else np.array([], dtype=int)
    if len(flattened) != num_edges or sorted(flattened.tolist()) != list(range(num_edges)):
        raise ValueError('strata must partition every candidate position exactly once')
    # A permutation is reproducible directly from its index, without advancing
    # any previous permutation's generator. No shared mutable RNG state exists.
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, index])))
    order = np.arange(num_edges)
    for positions in strata:
        if len(positions) >= 2:
            order[positions] = rng.permutation(positions)
    return order


def permute_evidence_bundles(edges, *, seed=42, permutation_index=0):
    """Reference table implementation: non-bundle edge attributes remain fixed."""
    strata, _ = build_strata(edges)
    order = permutation_indices(strata, len(edges), seed=seed, permutation_index=permutation_index)
    result = edges.copy()
    for column in BUNDLE_COLUMNS:
        result[column] = edges[column].to_numpy()[order]
    for name in ('core_q95', 'core_q99'):
        if name + '_geo500' in result:
            result[name + '_geo500'] = boolean_values(result[name]) & (result['geo_distance_m'].to_numpy() >= 500)
    return result


def _simple_topology(node_count, a, b):
    """Sparse union-find on active nodes; fixed-universe isolates remain implicit."""
    if len(a) != len(b) or (a == b).any():
        raise ValueError('topology requires unique different-node endpoint pairs')
    if len(a) and (min(a.min(), b.min()) < 0 or max(a.max(), b.max()) >= node_count):
        raise ValueError('topology endpoints absent from fixed node universe')
    degree = np.bincount(np.concatenate((a, b)), minlength=node_count)
    active = np.flatnonzero(degree)
    parent = {int(node): int(node) for node in active}
    size = {int(node): 1 for node in active}
    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for left, right in zip(a.tolist(), b.tolist()):
        left, right = find(left), find(right)
        if left == right:
            continue
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]
    roots = {find(int(node)) for node in active}
    return {'active': len(active), 'non_singleton_components': len(roots),
            'largest_component': max([size[root] for root in roots] + ([1] if node_count else [0])),
            'max_degree': int(degree.max()) if node_count else 0}


class FixedGraph:
    """Integer endpoints and one small bundle matrix; no per-permutation features."""
    def __init__(self, edges):
        if edges['pair_uid'].duplicated().any():
            raise ValueError('candidate pair_uid values must be unique')
        self.images = sorted(set(edges['image_id_a']) | set(edges['image_id_b']))
        self.places = sorted(set(edges['place_uid_a']) | set(edges['place_uid_b']))
        image_index, place_index = ({value: index for index, value in enumerate(values)} for values in (self.images, self.places))
        self.ia = edges['image_id_a'].map(image_index).to_numpy(int)
        self.ib = edges['image_id_b'].map(image_index).to_numpy(int)
        if (self.ia == self.ib).any() or len(set(zip(np.minimum(self.ia, self.ib), np.maximum(self.ia, self.ib)))) != len(edges):
            raise ValueError('candidate graph must contain unique undirected non-self image edges')
        self.pa = edges['place_uid_a'].map(place_index).to_numpy(int)
        self.pb = edges['place_uid_b'].map(place_index).to_numpy(int)
        if (self.pa == self.pb).any():
            raise ValueError('same-place edges cannot enter topology null')
        reverse = self.pa > self.pb
        self.view_a, self.view_b = np.where(reverse, self.ib, self.ia), np.where(reverse, self.ia, self.ib)
        self.pa, self.pb = np.minimum(self.pa, self.pb), np.maximum(self.pa, self.pb)
        pair_ids = {pair: index for index, pair in enumerate(sorted(set(zip(self.pa, self.pb))))}
        self.pair = np.array([pair_ids[pair] for pair in zip(self.pa, self.pb)])
        self.geo500 = edges['geo_distance_m'].to_numpy(float) >= 500
        self.strata, self.stratum_records = build_strata(edges)
        ratio, count, bottleneck, geomean = (pd.to_numeric(edges[column], errors='raise').to_numpy(float) for column in BUNDLE_COLUMNS[:4])
        if not np.isfinite(np.column_stack((ratio, count, bottleneck, geomean))).all() or (ratio < 0).any() or (ratio > 1).any() or (count < 0).any() or (count > 1).any():
            raise ValueError('bundle evidence must be finite percentiles')
        if not np.allclose(bottleneck, np.minimum(ratio, count), rtol=0, atol=1e-12) or not np.allclose(geomean, np.sqrt(ratio * count), rtol=0, atol=1e-12):
            raise ValueError('evidence bundle formulas disagree')
        q95, q99 = boolean_values(edges['core_q95']), boolean_values(edges['core_q99'])
        duplicates = boolean_values(edges['exact_pixel_duplicate'])
        if not np.array_equal(q95, (bottleneck >= .95) & ~duplicates) or not np.array_equal(q99, (bottleneck >= .99) & ~duplicates):
            raise ValueError('frozen bundle core flags disagree')
        self.bundle = np.column_stack((ratio, count, bottleneck, geomean, q95, q99))
        self.num_edges = len(edges)

    def statistics(self, bundle=None):
        bundle = self.bundle if bundle is None else bundle
        if bundle.shape != self.bundle.shape:
            raise ValueError('permutation bundle shape differs')
        q95, q99 = bundle[:, 4].astype(bool), bundle[:, 5].astype(bool)
        result = {}
        for name, selected in zip(SLICES, (q95, q99, q95 & self.geo500)):
            positions = np.flatnonzero(selected)
            counts = Counter(self.pair[positions].tolist())
            views_a, views_b = defaultdict(set), defaultdict(set)
            representatives = {}
            for index in positions:
                pair = int(self.pair[index])
                views_a[pair].add(int(self.view_a[index]))
                views_b[pair].add(int(self.view_b[index]))
                representatives.setdefault(pair, index)
            place_positions = np.array(list(representatives.values()), dtype=int)
            places = _simple_topology(len(self.places), self.pa[place_positions], self.pb[place_positions])
            images = _simple_topology(len(self.images), self.ia[positions], self.ib[positions])
            row = {'edge_count': len(positions),
                   'active_places': places['active'], 'non_singleton_place_components': places['non_singleton_components'],
                   'largest_place_component': places['largest_component'], 'max_place_degree': places['max_degree'],
                   'active_images': images['active'], 'largest_image_component': images['largest_component'],
                   'max_image_degree': images['max_degree']}
            for minimum in (2, 3):
                row[f'repeated_support_{minimum}'] = sum(count >= minimum for count in counts.values())
                row[f'core_independent_support_{minimum}'] = sum(count >= minimum and len(views_a[pair]) >= 2 and len(views_b[pair]) >= 2 for pair, count in counts.items())
            result[name] = row
        return result


def validate_observed_statistics(observed, frozen):
    """Bind null-observed values to the frozen Step 2B science, not literals."""
    for name in SLICES:
        image, place = frozen['image_graph'][name], frozen['place_graph'][name]
        expected = {'edge_count': image['edge_count'], 'active_images': image['active_node_count'],
                    'largest_image_component': image['component_sizes']['max'], 'max_image_degree': image['degree']['max'],
                    'active_places': place['active_node_count'], 'non_singleton_place_components': place['non_singleton_component_count'],
                    'largest_place_component': place['component_sizes']['max'], 'max_place_degree': place['degree']['max']}
        if any(observed[name][key] != value for key, value in expected.items()):
            raise ValueError('observed topology differs from frozen Step 2B')
    repeated = frozen['repeated_support']
    for key in ('repeated_support_2', 'repeated_support_3'):
        if observed['core_q95'][key] != repeated[key]:
            raise ValueError('observed repeated support differs from Step 2B')
    for key in ('core_independent_support_2', 'core_independent_support_3'):
        if observed['core_q95'][key] != repeated['strict_core_only'][key]:
            raise ValueError('observed core-only view support differs from Step 2B')
    geo = frozen['geo_sensitivity']['repeated_support_geo500']
    for key in ('repeated_support_2', 'repeated_support_3', 'core_independent_support_2', 'core_independent_support_3'):
        if observed['core_q95_geo500'][key] != geo[key + '_geo500']:
            raise ValueError('observed geographic repeated support differs from Step 2B')


def summarize_permutations(observed, permutations):
    result = {}
    for name in SLICES:
        result[name] = {}
        for statistic in STATISTICS:
            values = permutations[name + '__' + statistic].to_numpy()
            target = observed[name][statistic]
            result[name][statistic] = {'observed': target, 'permutation': distribution(values),
                'empirical_exceedance_fraction': (1 + int((values >= target).sum())) / (1 + len(values))}
    return result


def analyze_topology_null(edges, *, permutations=1000, seed=42, frozen_snapshot=None, progress=None):
    permutations = positive_integer(permutations, 'permutations')
    seed = positive_integer(seed, 'seed', allow_zero=True)
    graph = FixedGraph(edges)
    observed = graph.statistics()
    if frozen_snapshot is not None:
        validate_observed_statistics(observed, frozen_snapshot)
    rows = []
    expected = {name: observed[name]['edge_count'] for name in SLICES}
    for index in range(permutations):
        order = permutation_indices(graph.strata, graph.num_edges, seed=seed, permutation_index=index)
        # All six fields move together; endpoints and all other metadata remain
        # in the FixedGraph. Only this bounded edge-bundle copy is materialized.
        statistics = graph.statistics(graph.bundle[order])
        if any(statistics[name]['edge_count'] != expected[name] for name in SLICES):
            raise ValueError('stratified permutation changed frozen core totals')
        rows.append({'permutation_index': index, **{name + '__' + key: statistics[name][key] for name in SLICES for key in STATISTICS}})
        if progress and ((index + 1) % 100 == 0 or index + 1 == permutations):
            progress(index + 1, permutations)
    table = pd.DataFrame(rows)
    report = {'topology_null': {
        'stratum_definitions': {'city_relation': ['same_city', 'cross_city'], 'rank_bin': list(RANK_BINS),
                               'geography_bin': ['geo_250_500', 'geo_ge500']},
        'stratum_counts': graph.stratum_records,
        'strata_lt2_count': sum(record['edge_count'] < 2 for record in graph.stratum_records),
        'empty_strata_count': sum(record['edge_count'] == 0 for record in graph.stratum_records),
        'singleton_strata_count': sum(record['edge_count'] == 1 for record in graph.stratum_records),
        'unchanged_edges_in_lt2_strata': sum(record['edge_count'] for record in graph.stratum_records if record['edge_count'] < 2),
        'bundle_columns': list(BUNDLE_COLUMNS), 'graph_endpoints_fixed': True,
        'random_generator': 'NumPy PCG64(SeedSequence([seed, zero_based_permutation_index]))',
        'permutations': permutations, 'seed': seed,
        'observed_statistics': observed, 'comparisons': summarize_permutations(observed, table),
        'full_distribution_file': 'topology_permutations.csv',
        'definitions': {'place_graph': 'simple unordered place-pair edges with >=1 selected image support',
                        'independent_support': '>=2 or >=3 selected image edges and >=2 distinct selected views on BOTH canonical place sides',
                        'isolates': 'fixed full-pilot node universe retained; largest component is 1 for an empty nonempty-universe slice',
                        'geo500': 'selected bundle core_q95 AND fixed edge geographic distance >=500',
                        'duplicates': 'core flags move as part of the frozen bundle; no new duplicate gate after permutation',
                        'exceedance': '(1 + count(permutation >= observed))/(1 + num_permutations); diagnostic, not a definitive hypothesis-test p-value',
                        'std': 'sample standard deviation (ddof=1); null when fewer than two permutations'}},
    }
    return table, report


def run_topology_null(step2b_dir, step2b_snapshot, output_dir, *, permutations=1000, seed=42, repo_root=ROOT, progress=None):
    permutations = positive_integer(permutations, 'permutations')
    seed = positive_integer(seed, 'seed', allow_zero=True)
    guard_destinations([Path(output_dir) / name for name in (
        'topology_null_summary.json', 'topology_permutations.csv')], repo_root)
    inputs = load_frozen_inputs(step2b_dir, step2b_snapshot, repo_root=repo_root)
    table, report = analyze_topology_null(inputs['image_edges'], permutations=permutations, seed=seed,
                                         frozen_snapshot=inputs['step2b_snapshot'], progress=progress)
    return publish_stage(output_dir, 'topology_null', {'topology_permutations.csv': table}, report,
                         inputs['provenance'], {'seed': seed, 'permutations': permutations, 'candidate_count': 5000}, repo_root=repo_root)
