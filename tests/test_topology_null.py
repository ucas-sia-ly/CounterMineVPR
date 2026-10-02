"""Small fixed graphs exercise bundle permutation and independent view topology."""
import copy
import unittest

import numpy as np
import pandas as pd

from countermine.mining.topology_null import (
    BUNDLE_COLUMNS, SLICES, FixedGraph, analyze_topology_null, build_strata,
    permutation_indices, permute_evidence_bundles, validate_observed_statistics,
)


def topology_fixture():
    frame = pd.DataFrame({
        'pair_uid': [f'e{i}' for i in range(8)],
        'image_id_a': ['a1', 'b2', 'a1', 'a1', 'c1', 'a2', 'b1', 'e1'],
        'image_id_b': ['b1', 'a2', 'b2', 'c1', 'd1', 'c1', 'd1', 'f1'],
        'place_uid_a': ['P', 'Q', 'P', 'P', 'R', 'P', 'Q', 'T'],
        'place_uid_b': ['Q', 'P', 'Q', 'R', 'S', 'R', 'S', 'U'],
        'same_city': [True] * 7 + [False], 'rank_bin': ['rank_6_10'] * 8,
        'geo_distance_m': [800, 900, 300, 800, 800, 800, 800, 1000000],
        'ratio_null_percentile': [.995, .97, .96, .995, .5, .2, .96, .94],
        'match_count_null_percentile': [.997, .98, .97, .999, .8, .6, .999, .96],
        'exact_pixel_duplicate': [False] * 8,
    })
    frame['structural_bottleneck'] = np.minimum(frame['ratio_null_percentile'], frame['match_count_null_percentile'])
    frame['structural_geomean'] = np.sqrt(frame['ratio_null_percentile'] * frame['match_count_null_percentile'])
    for name, q in (('q95', .95), ('q99', .99)):
        frame['core_' + name] = frame['structural_bottleneck'] >= q
        frame['core_' + name + '_geo500'] = frame['core_' + name] & (frame['geo_distance_m'] >= 500)
    return frame


def frozen_fixture(observed):
    frozen = {'image_graph': {}, 'place_graph': {}, 'repeated_support': {'strict_core_only': {}},
              'geo_sensitivity': {'repeated_support_geo500': {}}}
    for name in SLICES:
        row = observed[name]
        frozen['image_graph'][name] = {'edge_count': row['edge_count'], 'active_node_count': row['active_images'],
            'component_sizes': {'max': row['largest_image_component']}, 'degree': {'max': row['max_image_degree']}}
        frozen['place_graph'][name] = {'active_node_count': row['active_places'],
            'non_singleton_component_count': row['non_singleton_place_components'],
            'component_sizes': {'max': row['largest_place_component']}, 'degree': {'max': row['max_place_degree']}}
    for key in ('repeated_support_2', 'repeated_support_3', 'core_independent_support_2', 'core_independent_support_3'):
        if key.startswith('core_'):
            frozen['repeated_support']['strict_core_only'][key] = observed['core_q95'][key]
        else:
            frozen['repeated_support'][key] = observed['core_q95'][key]
        frozen['geo_sensitivity']['repeated_support_geo500'][key + '_geo500'] = observed['core_q95_geo500'][key]
    return frozen


class TopologyNullTests(unittest.TestCase):
    def test_support_components_degrees_and_geo_sensitivity(self):
        observed = FixedGraph(topology_fixture()).statistics()
        q95 = observed['core_q95']
        self.assertEqual(q95['edge_count'], 5)
        self.assertEqual(q95['repeated_support_2'], 1)
        self.assertEqual(q95['repeated_support_3'], 1)
        self.assertEqual(q95['core_independent_support_2'], 1)
        self.assertEqual(q95['core_independent_support_3'], 1)
        self.assertEqual(q95['active_places'], 4)
        self.assertEqual(q95['largest_place_component'], 4)
        self.assertEqual(q95['non_singleton_place_components'], 1)
        self.assertEqual(q95['max_place_degree'], 2)  # place supports collapse into simple edges
        self.assertEqual(q95['active_images'], 6)
        self.assertEqual(q95['largest_image_component'], 6)
        self.assertEqual(q95['max_image_degree'], 3)
        geo = observed['core_q95_geo500']
        self.assertEqual(geo['edge_count'], 4)
        self.assertEqual(geo['core_independent_support_2'], 1)
        self.assertEqual(geo['core_independent_support_3'], 0)
        self.assertEqual(observed['core_q99']['edge_count'], 2)
        self.assertEqual(observed['core_q99']['largest_place_component'], 3)

    def test_core_only_views_do_not_use_weak_edges(self):
        frame = topology_fixture()
        for index in (1, 6):
            frame.loc[index, list(BUNDLE_COLUMNS)] = [.5, .5, .5, .5, False, False]
        row = FixedGraph(frame).statistics()['core_q95']
        self.assertEqual(row['repeated_support_2'], 1)  # a1-b1 and a1-b2
        self.assertEqual(row['core_independent_support_2'], 0)  # a2-b2 is weak

    def test_bundle_dependence_and_all_fixed_attributes(self):
        original = topology_fixture()
        permuted = permute_evidence_bundles(original, seed=42, permutation_index=3)
        fixed = [column for column in original if column not in BUNDLE_COLUMNS and not column.endswith('_geo500')]
        pd.testing.assert_frame_equal(original[fixed], permuted[fixed])
        bundles = lambda frame: sorted(frame[list(BUNDLE_COLUMNS)].itertuples(index=False, name=None))
        self.assertEqual(bundles(original), bundles(permuted))
        np.testing.assert_allclose(permuted['structural_bottleneck'], np.minimum(permuted['ratio_null_percentile'], permuted['match_count_null_percentile']))
        np.testing.assert_allclose(permuted['structural_geomean'], np.sqrt(permuted['ratio_null_percentile'] * permuted['match_count_null_percentile']))
        self.assertEqual(original['core_q95'].sum(), permuted['core_q95'].sum())
        np.testing.assert_array_equal(permuted['core_q95_geo500'], permuted['core_q95'] & (original['geo_distance_m'] >= 500))

    def test_stratification_singletons_and_per_stratum_totals(self):
        frame = topology_fixture()
        strata, records = build_strata(frame)
        self.assertEqual(len(records), 20)
        self.assertEqual(sum(record['edge_count'] for record in records), len(frame))
        self.assertEqual(sum(record['core_q95_edge_count'] for record in records), int(frame['core_q95'].sum()))
        self.assertEqual(sum(record['core_q99_edge_count'] for record in records), int(frame['core_q99'].sum()))
        self.assertEqual(sum(record['edge_count'] == 1 for record in records), 2)
        order = permutation_indices(strata, len(frame), seed=42, permutation_index=8)
        for positions in strata:
            self.assertEqual(set(order[positions]), set(positions))
            if len(positions) == 1:
                self.assertEqual(order[positions[0]], positions[0])
            for flag in ('core_q95', 'core_q99'):
                self.assertEqual(frame[flag].to_numpy()[order[positions]].sum(), frame[flag].to_numpy()[positions].sum())

    def test_independent_index_determinism(self):
        strata, _ = build_strata(topology_fixture())
        first = permutation_indices(strata, 8, seed=42, permutation_index=0)
        other = permutation_indices(strata, 8, seed=42, permutation_index=1)
        again = permutation_indices(strata, 8, seed=42, permutation_index=0)
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, other))
        with self.assertRaises(ValueError):
            permutation_indices([np.array([0, 0])], 2)
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                permutation_indices(strata, 8, permutation_index=invalid)

    def test_full_distribution_and_plus_one_exceedance(self):
        frame = topology_fixture()
        observed = FixedGraph(frame).statistics()
        frozen = frozen_fixture(observed)
        table, report = analyze_topology_null(frame, permutations=17, seed=42, frozen_snapshot=frozen)
        again, _ = analyze_topology_null(frame, permutations=17, seed=42)
        pd.testing.assert_frame_equal(table, again)
        self.assertEqual(table['permutation_index'].tolist(), list(range(17)))
        for name in SLICES:
            self.assertTrue((table[name + '__edge_count'] == observed[name]['edge_count']).all())
        statistic = report['topology_null']['comparisons']['core_q95']['repeated_support_2']
        values = table['core_q95__repeated_support_2'].to_numpy()
        expected = (1 + (values >= observed['core_q95']['repeated_support_2']).sum()) / 18
        self.assertEqual(statistic['empirical_exceedance_fraction'], expected)
        self.assertEqual(statistic['permutation']['q99'], np.quantile(values, .99))
        self.assertEqual(statistic['permutation']['std'], np.std(values, ddof=1))
        broken = copy.deepcopy(frozen)
        broken['place_graph']['core_q95']['component_sizes']['max'] += 1
        with self.assertRaises(ValueError):
            validate_observed_statistics(observed, broken)

    def test_empty_core_keeps_fixed_singleton_universe(self):
        frame = topology_fixture()
        frame.loc[:, 'ratio_null_percentile'] = .4
        frame.loc[:, 'match_count_null_percentile'] = .4
        frame.loc[:, 'structural_bottleneck'] = .4
        frame.loc[:, 'structural_geomean'] = .4
        frame.loc[:, 'core_q95'] = False
        frame.loc[:, 'core_q99'] = False
        observed = FixedGraph(frame).statistics()
        self.assertEqual(observed['core_q95']['active_places'], 0)
        self.assertEqual(observed['core_q95']['largest_place_component'], 1)
        self.assertEqual(observed['core_q95']['max_place_degree'], 0)
        self.assertEqual(observed['core_q95']['largest_image_component'], 1)

    def test_rejects_invalid_graph_bundles_and_config(self):
        frame = topology_fixture()
        bad = frame.copy()
        bad.loc[0, 'structural_bottleneck'] = .3
        with self.assertRaises(ValueError):
            FixedGraph(bad)
        with self.assertRaises(ValueError):
            FixedGraph(pd.concat([frame, frame.iloc[:1]]))
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                analyze_topology_null(frame, permutations=count)


if __name__ == '__main__':
    unittest.main()
