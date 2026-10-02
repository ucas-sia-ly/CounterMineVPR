"""Synthetic CPU fixtures for self-inclusive joint nulls and paired evidence."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from countermine.mining.joint_null import (
    SOURCE_FILES, analyze_joint_null, align_paired_joint, bind_joint_measurements, code_hashes, compare_rates,
    guard_destinations, load_stage, publish_stage, random_joint_evidence,
    validate_step2c_snapshot, weak_ecdf_threshold,
)
from countermine.mining.structural_analysis import sha256_file, weak_ecdf


def joint_fixture():
    controls = pd.DataFrame({
        'pair_uid': [f'r{i}' for i in range(8)], 'candidate_pair_uid': [f'c{i}' for i in range(8)],
        'same_city': [True] * 4 + [False] * 4,
        'rank_bin': ['rank_1', 'rank_2_5', 'rank_6_10', 'rank_11_20'] * 2,
        'local_match_ratio': [.1, .2, .3, .4] * 2, 'num_matches': [10, 20, 30, 40] * 2,
    })
    candidates = pd.DataFrame({
        'pair_uid': [f'c{i}' for i in range(9)], 'same_city': [True] * 4 + [False] * 4 + [True],
        'rank_bin': controls['rank_bin'].tolist() + ['rank_21_50'],
        'local_match_ratio': [.4, .2, .05, .4, .5, .1, .3, .4, .5],
        'num_matches': [40, 10, 5, 30, 50, 10, 30, 40, 50], 'exact_pixel_duplicate': [False] * 9,
    })
    for short, column in (('ratio', 'local_match_ratio'), ('match_count', 'num_matches')):
        candidates[short + '_null_percentile'] = 0.
        for flag in (True, False):
            mask = candidates['same_city'] == flag
            candidates.loc[mask, short + '_null_percentile'] = weak_ecdf(controls.loc[controls['same_city'] == flag, column], candidates.loc[mask, column])
    candidates['structural_bottleneck'] = np.minimum(candidates['ratio_null_percentile'], candidates['match_count_null_percentile'])
    candidates['structural_geomean'] = np.sqrt(candidates['ratio_null_percentile'] * candidates['match_count_null_percentile'])
    for name, q in (('q95', .95), ('q99', .99)):
        candidates['core_' + name] = candidates['structural_bottleneck'] >= q
    return candidates, controls


class JointNullTests(unittest.TestCase):
    def test_exact_threshold_ties_and_realized_tail(self):
        result = weak_ecdf_threshold([0, 1, 1, 1, 2], .6)
        self.assertEqual(result['threshold'], 1)
        self.assertEqual(result['realized_tail_count'], 4)
        self.assertEqual(result['realized_tail_fraction'], .8)
        self.assertEqual(result['weak_ecdf_at_threshold'], .8)
        self.assertEqual(weak_ecdf_threshold(range(100), .95)['threshold'], 94)
        self.assertEqual(weak_ecdf_threshold(range(100), .99)['realized_tail_count'], 2)
        self.assertEqual(weak_ecdf_threshold([3] * 100, .99)['realized_tail_fraction'], 1.)
        for values, q in (([], .95), ([1, np.nan], .95), ([1], 0), ([1], 1.1)):
            with self.assertRaises(ValueError):
                weak_ecdf_threshold(values, q)

    def test_self_inclusive_and_loo_conventions(self):
        random = pd.DataFrame({'same_city': [True] * 100, 'local_match_ratio': np.arange(100) / 100,
                               'num_matches': np.arange(100)})
        result, thresholds = random_joint_evidence(random)
        self.assertEqual(result['random_core_q95'].sum(), 6)
        self.assertEqual(result['loo_core_q95'].sum(), 5)
        self.assertEqual(result['random_core_q99'].sum(), 2)
        self.assertEqual(result['loo_core_q99'].sum(), 1)
        self.assertEqual(result.loc[94, 'random_ratio_null_percentile'], .95)
        self.assertEqual(result.loc[94, 'loo_ratio_null_percentile'], 94 / 99)
        self.assertEqual(thresholds['same_city']['count_q95_threshold'], 94)
        # A tied maximum is not removed wholesale in leave-one-out.
        tied = pd.DataFrame({'same_city': [False] * 3, 'local_match_ratio': [.1, .4, .4], 'num_matches': [1, 4, 4]})
        result, _ = random_joint_evidence(tied)
        self.assertEqual(result['loo_ratio_null_percentile'].tolist(), [0., 1., 1.])

    def test_loo_singleton_is_undefined(self):
        result, _ = random_joint_evidence(pd.DataFrame({'same_city': [False], 'local_match_ratio': [.2], 'num_matches': [2]}))
        self.assertEqual(result.loc[0, 'random_structural_bottleneck'], 1.)
        self.assertTrue(pd.isna(result.loc[0, 'loo_core_q95']))
        self.assertTrue(pd.isna(result.loc[0, 'loo_structural_bottleneck']))

    def test_joint_requires_both_metrics_and_same_relation_null(self):
        c, r = joint_fixture()
        controls, paired, report = analyze_joint_null(c, r)
        self.assertEqual(controls['random_core_q95'].sum(), 2)
        self.assertEqual(report['joint_null']['rates']['same_city']['random']['q95_count'], 1)
        self.assertEqual(report['joint_null']['rates']['cross_city']['random']['q95_count'], 1)
        self.assertEqual(report['joint_null']['rates']['all']['candidate']['count'], 9)
        self.assertFalse(c.loc[3, 'core_q95'])  # perfect ratio percentile, only .75 count percentile
        self.assertEqual(len(paired), 8)  # the unpaired candidate remains in rates
        c.loc[0, 'ratio_null_percentile'] = .4
        with self.assertRaisesRegex(ValueError, 'relation-specific'):
            analyze_joint_null(c, r)

    def test_exact_paired_alignment_deltas_and_groups(self):
        c, r = joint_fixture()
        controls, paired, report = analyze_joint_null(c.sample(frac=1, random_state=7), r.sample(frac=1, random_state=3))
        lookup = paired.set_index('candidate_pair_uid')
        self.assertEqual(lookup.loc['c0', 'random_pair_uid'], 'r0')
        self.assertEqual(lookup.loc['c0', 'delta_bottleneck'], .75)
        self.assertEqual(lookup.loc['c1', 'delta_bottleneck'], -.25)
        self.assertEqual(report['paired_joint_evidence']['all']['count'], 8)
        self.assertEqual(report['paired_joint_evidence']['same_city']['count'], 4)
        self.assertEqual(report['paired_joint_evidence']['rank_bins']['rank_1']['count'], 2)
        self.assertEqual(report['paired_joint_evidence']['rank_bins']['rank_21_50']['count'], 0)
        self.assertEqual(report['paired_joint_evidence']['all']['candidate_eq_random'], 2)
        # CSV roundtrip noise must not turn a true tie into an apparent loss.
        c.loc[c['pair_uid'] == 'c7', 'structural_bottleneck'] -= 1e-16
        _, _, report2 = analyze_joint_null(c, r)
        self.assertEqual(report2['paired_joint_evidence']['all']['candidate_eq_random'], 2)

    def test_alignment_rejects_duplicates_unknown_or_mismatched_metadata(self):
        c, r = joint_fixture()
        controls, _ = random_joint_evidence(r)
        for changed in (pd.concat([controls, controls.iloc[:1]]), controls.assign(candidate_pair_uid=['unknown'] + controls['candidate_pair_uid'].tolist()[1:]),
                        controls.assign(rank_bin=['rank_21_50'] + controls['rank_bin'].tolist()[1:])):
            with self.assertRaises(ValueError):
                align_paired_joint(c, changed)

    def test_zero_rate_denominator_is_null(self):
        candidate = {'q95_fraction': .07, 'q99_fraction': .015}
        random = {'q95_fraction': 0., 'q99_fraction': .005}
        report = compare_rates(candidate, random)
        self.assertIsNone(report['q95_rate_ratio'])
        self.assertEqual(report['q95_absolute_difference'], .07)
        self.assertEqual(report['q99_rate_ratio'], 3.)

    def test_graph_csv_roundtrip_preserves_original_raw_ecdf_ties(self):
        candidates, controls = joint_fixture()
        ratio = .0291130670277589
        candidates.loc[0, 'local_match_ratio'] = ratio
        controls.loc[0, 'local_match_ratio'] = ratio
        for flag in (True, False):
            selected = candidates['same_city'] == flag
            candidates.loc[selected, 'ratio_null_percentile'] = weak_ecdf(
                controls.loc[controls['same_city'] == flag, 'local_match_ratio'],
                candidates.loc[selected, 'local_match_ratio'])
        candidates['structural_bottleneck'] = np.minimum(candidates['ratio_null_percentile'], candidates['match_count_null_percentile'])
        candidates['structural_geomean'] = np.sqrt(candidates['ratio_null_percentile'] * candidates['match_count_null_percentile'])
        for name, q in (('q95', .95), ('q99', .99)):
            candidates['core_' + name] = candidates['structural_bottleneck'] >= q
        expected_controls, expected_pairs, expected_report = analyze_joint_null(candidates, controls)
        graph = candidates.copy()
        graph['local_match_ratio'] = pd.to_numeric(graph['local_match_ratio'].map(lambda value: format(value, '.17g')))
        self.assertLess(graph.loc[0, 'local_match_ratio'], ratio)
        with self.assertRaisesRegex(ValueError, 'relation-specific'):
            analyze_joint_null(graph, controls)
        before = graph.copy(deep=True)
        restored = bind_joint_measurements(graph.sample(frac=1, random_state=7), candidates.sample(frac=1, random_state=3))
        actual_controls, actual_pairs, actual_report = analyze_joint_null(restored, controls)
        pd.testing.assert_frame_equal(graph, before)
        pd.testing.assert_frame_equal(actual_controls, expected_controls)
        pd.testing.assert_frame_equal(actual_pairs, expected_pairs)
        self.assertEqual(actual_report, expected_report)
        self.assertEqual(actual_report['paired_joint_evidence']['all']['candidate_eq_random'], 3)

    def test_raw_measurement_binding_rejects_identity_or_evidence_changes(self):
        candidates, _ = joint_fixture()
        for changed in (candidates.iloc[:-1], pd.concat([candidates, candidates.iloc[:1]]),
                        candidates.assign(pair_uid=['unknown'] + candidates['pair_uid'].tolist()[1:]),
                        candidates.assign(local_match_ratio=candidates['local_match_ratio'] + .01),
                        candidates.assign(num_matches=candidates['num_matches'] + 1),
                        candidates.assign(same_city=~candidates['same_city']),
                        candidates.assign(exact_pixel_duplicate=True),
                        candidates.assign(rank_bin='rank_21_50')):
            with self.assertRaises(ValueError):
                bind_joint_measurements(candidates, changed)


class ArtifactIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in SOURCE_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('pass\n')
        original = self.root / 'input.csv'
        original.write_text('frozen\n')
        self.provenance = {'real_rgb_only': True, 'synthetic_images_used': False, 'no_new_local_matching': True,
                           'input_file_sha256': {'input.csv': sha256_file(original)}, 'step2c_code_sha256': code_hashes(self.root)}
        self.output = self.root / 'cache/countermine_rgb/step2c'

    def publish(self, table=None):
        return publish_stage(self.output, 'fixture', {'fixture.csv': pd.DataFrame({'number': [1, 2]}) if table is None else table},
                             {'statistic': 2}, self.provenance, {'seed': 42}, repo_root=self.root)

    def test_completion_hashes_idempotence_and_tamper_rejection(self):
        summary = self.publish()
        self.assertEqual(summary, self.publish())
        loaded, tables = load_stage(self.output, 'fixture', self.provenance, repo_root=self.root)
        self.assertEqual(loaded, summary)
        self.assertEqual(tables['fixture.csv']['number'].tolist(), [1, 2])
        (self.output / 'fixture.csv').write_text('number\n3\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_stage(self.output, 'fixture', self.provenance, repo_root=self.root)
        with self.assertRaises(ValueError):
            self.publish()

    def test_source_and_code_changes_rejected(self):
        self.publish()
        (self.root / 'input.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'frozen source changed'):
            self.publish()
        (self.root / 'input.csv').write_text('frozen\n')
        (self.root / SOURCE_FILES[0]).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'implementation changed'):
            self.publish()

    def test_destination_guards_and_symlink_aliases(self):
        for target in (self.root / 'cache/countermine_rgb/step2a/new.csv', self.root / 'cache/countermine_rgb/step2b/new.csv',
                       self.root / 'docs/audits/step2b_snapshot.json', self.root / 'third_party/new.csv', self.root / 'data/new.csv'):
            with self.assertRaises(ValueError):
                guard_destinations([target], self.root)
            self.assertFalse(target.exists())
        guard_destinations([self.output / 'file.csv', self.root / 'outputs/step2c/plot.png', self.root / 'docs/audits/step2c_snapshot.json'], self.root)
        self.output.mkdir(parents=True)
        alias = self.output / 'alias.csv'
        alias.symlink_to(self.root / 'input.csv')
        with self.assertRaises(ValueError):
            guard_destinations([alias], self.root)
        self.assertEqual((self.root / 'input.csv').read_text(), 'frozen\n')

    def test_snapshot_finite_path_free_exact_flags(self):
        good = {'provenance': self.provenance, 'missing': None, 'value': 1.}
        validate_step2c_snapshot(good)
        for key, value in (('value', float('nan')), ('value', float('inf')), ('path', '/tmp/private'),
                           ('path', 'C:\\private'), ('coordinates', [1, 2])):
            bad = copy.deepcopy(good)
            bad[key] = value
            with self.assertRaises(ValueError):
                validate_step2c_snapshot(bad)
        for key, value in (('real_rgb_only', False), ('synthetic_images_used', True), ('no_new_local_matching', False)):
            bad = copy.deepcopy(good)
            bad['provenance'][key] = value
            with self.assertRaises(ValueError):
                validate_step2c_snapshot(bad)


if __name__ == '__main__':
    unittest.main()
