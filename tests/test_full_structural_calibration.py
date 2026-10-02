"""Frozen relation calibration, diagnostic slices and pilot failure gates."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from countermine.mining.full_structural_calibration import (
    FLOAT_METRICS, INTEGER_METRICS, QUANTILES, SOURCE_FILES, calibrate_full_evidence,
    compare_calibration_to_measurements, compare_pilot_calibration, compare_pilot_metrics, validate_full_calibrated,
    load_full_calibrated, run_calibration,
)
from countermine.mining.full_structural_io import write_csv, write_json
from countermine.mining.structural_analysis import sha256_file, weak_ecdf
from test_structural_calibration import measured_tables


class FullCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.candidates, self.controls, _ = measured_tables()

    def test_exact_relation_ecdf_bottleneck_and_spatial_diagnostic(self):
        actual = calibrate_full_evidence(self.candidates, self.controls)
        for flag in (True, False):
            selected = self.candidates.same_city == flag
            controls = self.controls.same_city == flag
            for short, name in (('ratio', 'local_match_ratio'), ('match_count', 'num_matches'),
                                ('coverage', 'symmetric_match_coverage'), ('entropy', 'symmetric_match_entropy')):
                np.testing.assert_array_equal(actual.loc[selected, short + '_null_percentile'],
                                              weak_ecdf(self.controls.loc[controls, name], self.candidates.loc[selected, name]))
                self.assertEqual(set(actual.loc[selected, short + '_null_source']), {'same_city' if flag else 'cross_city'})
        np.testing.assert_array_equal(actual.structural_bottleneck, np.minimum(actual.ratio_null_percentile, actual.match_count_null_percentile))
        np.testing.assert_array_equal(actual.spatial_support_percentile, actual.coverage_null_percentile)
        changed = self.candidates.copy()
        changed.loc[:, ['matched_source_cell_coverage','matched_target_cell_coverage','symmetric_match_coverage']] = 0.
        alternate = calibrate_full_evidence(changed, self.controls)
        np.testing.assert_array_equal(actual.structural_bottleneck, alternate.structural_bottleneck)
        validate_full_calibrated(actual)

    def test_missing_relation_never_falls_back_or_invents_controls(self):
        with self.assertRaisesRegex(ValueError, 'no fallback'):
            calibrate_full_evidence(self.candidates, self.controls.loc[self.controls.same_city])

    def test_four_core_slices_duplicate_exclusion_and_geo500(self):
        self.assertEqual(QUANTILES, {'q95': .95, 'q975': .975, 'q99': .99, 'q995': .995})
        candidate = self.candidates.copy()
        candidate.loc[0,'exact_pixel_duplicate'] = True
        actual = calibrate_full_evidence(candidate, self.controls)
        for name, q in QUANTILES.items():
            expected = (actual.structural_bottleneck >= q) & ~actual.exact_pixel_duplicate
            np.testing.assert_array_equal(actual['core_' + name], expected)
            np.testing.assert_array_equal(actual['core_' + name + '_geo500'], expected & (actual.geo_distance_m >= 500))
        tampered = actual.copy()
        tampered.loc[0,'structural_bottleneck'] = .123
        with self.assertRaisesRegex(ValueError, 'formula'):
            validate_full_calibrated(tampered)

    def test_full_subset_comparison_is_uid_aligned_and_fails_closed_on_all_metrics(self):
        report = compare_pilot_metrics(self.candidates.sample(frac=1, random_state=7), self.candidates)
        self.assertTrue(report['passed'])
        for column in (*INTEGER_METRICS, *FLOAT_METRICS, 'exact_pixel_duplicate'):
            changed = self.candidates.copy()
            changed.loc[0, column] = True if column == 'exact_pixel_duplicate' else changed.loc[0,column] + (1 if column in INTEGER_METRICS else 1e-9)
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'pilot reproduction failed'):
                compare_pilot_metrics(changed, self.candidates)
        for changed in (self.candidates.iloc[1:], self.candidates.assign(pair_uid='duplicate')):
            with self.assertRaises(ValueError):
                compare_pilot_metrics(changed, self.candidates)

    def test_frozen_labels_counts_and_calibration_drift_fail_closed(self):
        actual = calibrate_full_evidence(self.candidates, self.controls)
        snapshot = {'joint_null': {'rates': {'all': {'candidate': {
            name+'_count': int(actual['core_'+name].sum()) for name in ('q95','q99')}}}}}
        self.assertTrue(compare_pilot_calibration(actual,actual,snapshot)['passed'])
        changed = actual.copy()
        changed.loc[0,'ratio_null_percentile'] += .01
        with self.assertRaisesRegex(ValueError, 'Step 2B'):
            compare_pilot_calibration(changed,actual,snapshot)
        snapshot['joint_null']['rates']['all']['candidate']['q95_count'] += 1
        with self.assertRaisesRegex(ValueError, 'Step 2C'):
            compare_pilot_calibration(actual,actual,snapshot)

    def test_every_full_row_binds_to_original_measurements_and_frozen_ecdf(self):
        actual = calibrate_full_evidence(self.candidates, self.controls)
        compare_calibration_to_measurements(actual, self.candidates, self.controls)
        changed = actual.copy()
        changed.loc[0,'coverage_null_percentile'] = .123
        changed.loc[0,'spatial_support_percentile'] = .123
        # The primary formulas are still valid; the frozen-null binding is not.
        validate_full_calibrated(changed)
        with self.assertRaisesRegex(ValueError, 'frozen null'):
            compare_calibration_to_measurements(changed, self.candidates, self.controls)


class FullCalibrationPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = self.root / 'cache/countermine_rgb/step2d'
        for name in SOURCE_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('pass\n')
        candidates, controls, _ = measured_tables()
        self.candidates = candidates
        self.controls = controls
        path = self.root / 'cache/countermine_rgb/step2a/random_structural_metrics.csv'
        path.parent.mkdir(parents=True)
        controls.to_csv(path,index=False)
        frozen_source = self.root / 'frozen.txt'
        frozen_source.write_text('unchanged')
        self.provenance = {'input_file_sha256': {'frozen.txt': sha256_file(frozen_source),
            path.relative_to(self.root).as_posix(): sha256_file(path)},
            'real_rgb_only': True, 'synthetic_images_used': False}
        write_csv(self.runtime / 'full_candidate_structural_metrics.csv', candidates, repo_root=self.root)
        self.measurement = {'full_metrics_sha256': sha256_file(self.runtime / 'full_candidate_structural_metrics.csv')}
        write_json(self.runtime / 'full_measurement_summary.json', self.measurement, repo_root=self.root)
        self.calibrated = calibrate_full_evidence(candidates,controls)
        snapshot = {'joint_null': {'rates': {'all': {'candidate': {
            name+'_count': int(self.calibrated['core_'+name].sum()) for name in ('q95','q99')},
            'random': {'count': len(controls)}}}, 'thresholds': {}}}
        self.addCleanup(patch.stopall)
        patch('countermine.mining.full_structural_measurement.load_full_metrics',
              return_value=(candidates,self.measurement)).start()
        patch('countermine.mining.full_structural_calibration.load_frozen_null',
              return_value=(controls,candidates,self.calibrated,snapshot,self.provenance,{'passed':True})).start()

    def test_publish_reload_idempotence_and_source_reproduction(self):
        summary = run_calibration(self.runtime,repo_root=self.root)
        frame, loaded = load_full_calibrated(self.runtime,repo_root=self.root)
        self.assertEqual(summary,loaded)
        self.assertEqual(frame['pair_uid'].tolist(),self.candidates['pair_uid'].tolist())
        self.assertTrue(summary['pilot_reproduction']['passed'])
        self.assertEqual(summary,run_calibration(self.runtime,repo_root=self.root))

    def test_frozen_null_reconstruction_detects_self_consistent_csv_drift(self):
        summary = run_calibration(self.runtime,repo_root=self.root)
        changed = self.calibrated.copy()
        changed.loc[0,['coverage_null_percentile','spatial_support_percentile']] = .123
        write_csv(self.runtime / 'full_calibrated_candidates.csv',changed,repo_root=self.root)
        summary['full_calibrated_candidates_sha256'] = sha256_file(self.runtime / 'full_calibrated_candidates.csv')
        write_json(self.runtime / 'full_calibration_summary.json',summary,repo_root=self.root)
        with self.assertRaisesRegex(ValueError,'frozen null'):
            load_full_calibrated(self.runtime,repo_root=self.root)
