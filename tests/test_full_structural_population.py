"""CPU regression against the frozen canonicalization and complete counts."""
import copy
import unittest
import pandas as pd

from countermine.mining.full_structural_population import build_full_population, validate_full_population
from countermine.mining.full_population_metadata import canonicalize_candidates as cpu_canonicalize
from countermine.mining.structural_population import canonicalize_candidates
from test_structural_population import raw_fixture


class FullPopulationTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.raw, _ = raw_fixture()
        self.expected = canonicalize_candidates(self.raw).sort_values('pair_uid').reset_index(drop=True)
        self.summary = {'raw_directed_candidate_rows': len(self.raw),
                        'eligible_geo_directed_rows': int((self.raw['geo_distance_m'] >= 250).sum()),
                        'eligible_unique_canonical_pairs': len(self.expected),
                        'reverse_candidate_count': int((self.expected['num_candidate_directions'] == 2).sum())}

    def test_all_canonical_edges_and_order_match_frozen_logic(self):
        actual, summary = build_full_population(self.manifest, self.raw.sample(frac=1, random_state=42), self.summary)
        pd.testing.assert_frame_equal(actual, self.expected)
        pd.testing.assert_frame_equal(cpu_canonicalize(self.raw), canonicalize_candidates(self.raw))
        self.assertEqual(summary['full_canonical_edge_count'], len(self.expected))
        self.assertEqual(summary['same_city_count'] + summary['cross_city_count'], len(actual))
        self.assertEqual(summary['manifest_image_count'], len(self.manifest))
        parsed = validate_full_population(actual)
        self.assertEqual(parsed['pair_uid'].tolist(), actual['pair_uid'].tolist())
        self.assertTrue((actual.geo_distance_m >= 250).all())

    def test_every_frozen_population_count_must_match(self):
        for key in self.summary:
            changed = copy.deepcopy(self.summary)
            changed[key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'frozen Step 2A'):
                build_full_population(self.manifest, self.raw, changed)

    def test_unknown_identity_order_geo_direction_or_rank_rejected(self):
        changes = [self.expected.assign(pair_uid='unknown'), self.expected.iloc[::-1],
                   self.expected.assign(geo_distance_m=249.999), self.expected.assign(num_candidate_directions=3),
                   self.expected.assign(same_city=False), self.expected.assign(rank_bin='rank_21_50'),
                   self.expected.assign(a_to_b_similarity=float('inf')),
                   self.expected.assign(max_salad_similarity=.123),
                   self.expected.assign(place_uid_b=self.expected.place_uid_a)]
        for changed in changes:
            with self.assertRaises(ValueError):
                validate_full_population(changed)
