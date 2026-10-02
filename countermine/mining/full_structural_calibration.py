"""Frozen relation-null calibration and fail-closed pilot-to-full audit."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .full_structural_io import (
    ROOT, DEFAULT_DIR, code_hashes, frozen_provenance, guard_step2d, read_json,
    verify_hashes, write_csv, write_json,
)
from .graph_analysis import boolean_values
from .joint_null import random_joint_evidence
from .structural_analysis import METRIC_COLUMNS, read_metadata_csv, sha256_file, validate_metrics, weak_ecdf
from .structural_calibration import METRIC_NAMES, _compare_frozen, load_calibrated_artifacts

QUANTILES = {'q95': .95, 'q975': .975, 'q99': .99, 'q995': .995}
SOURCE_FILES = ('countermine/mining/full_structural_io.py',
                'countermine/mining/full_structural_calibration.py',
                'tools/22_calibrate_full_structural_evidence.py')
INTEGER_METRICS = ('num_keypoints_a', 'num_keypoints_b', 'num_matches')
FLOAT_METRICS = tuple(name for name in METRIC_COLUMNS if name not in (*INTEGER_METRICS, 'exact_pixel_duplicate'))
PERCENTILES = tuple(name + '_null_percentile' for name in METRIC_NAMES)


def calibrate_full_evidence(candidates, controls):
    """Apply only the frozen same/cross-city weak ECDF, without fallback."""
    result = validate_metrics(candidates, candidate=True)
    null = validate_metrics(controls, candidate=False)
    for short, metric in METRIC_NAMES.items():
        values = np.empty(len(result), dtype=np.float64)
        sources = np.empty(len(result), dtype=object)
        for relation, flag in (('same_city', True), ('cross_city', False)):
            selected = (result['same_city'] == flag).to_numpy()
            population = null.loc[null['same_city'] == flag, metric]
            if not len(population):
                raise ValueError('frozen relation-specific null is missing; no fallback allowed')
            values[selected] = weak_ecdf(population, result.loc[selected, metric])
            sources[selected] = relation
        result[short + '_null_percentile'] = values
        result[short + '_null_source'] = sources
    result['structural_bottleneck'] = np.minimum(result['ratio_null_percentile'], result['match_count_null_percentile'])
    result['structural_geomean'] = np.sqrt(result['ratio_null_percentile'] * result['match_count_null_percentile'])
    result['spatial_support_percentile'] = result['coverage_null_percentile']
    result['min_num_keypoints'] = np.minimum(result['num_keypoints_a'], result['num_keypoints_b'])
    for name, quantile in QUANTILES.items():
        result['core_' + name] = (result['structural_bottleneck'] >= quantile) & ~result['exact_pixel_duplicate']
        result['core_' + name + '_geo500'] = result['core_' + name] & (result['geo_distance_m'] >= 500)
    return result


def compare_pilot_metrics(measured, historical):
    """Require every reference UID once; exact integer/boolean and 1e-12 floats."""
    if measured['pair_uid'].duplicated().any() or historical['pair_uid'].duplicated().any():
        raise ValueError('pilot reproduction requires unique measured/reference pair_uids')
    reference = historical.set_index('pair_uid')
    if not set(reference.index).issubset(set(measured['pair_uid'])):
        raise ValueError('full metrics are missing frozen pilot pairs')
    actual = measured.set_index('pair_uid').loc[reference.index]
    maximum = {}
    for name in INTEGER_METRICS:
        left, right = pd.to_numeric(actual[name]).to_numpy(float), pd.to_numeric(reference[name]).to_numpy(float)
        if not np.isfinite(left).all() or not np.array_equal(left, right):
            raise ValueError(f'pilot reproduction failed: {name} must match exactly')
        maximum[name] = 0
    if not np.array_equal(boolean_values(actual['exact_pixel_duplicate']), boolean_values(reference['exact_pixel_duplicate'])):
        raise ValueError('pilot reproduction failed: exact_pixel_duplicate must match exactly')
    maximum['exact_pixel_duplicate'] = 0
    for name in FLOAT_METRICS:
        left, right = pd.to_numeric(actual[name]).to_numpy(float), pd.to_numeric(reference[name]).to_numpy(float)
        delta = np.abs(left - right)
        if not np.isfinite(delta).all() or (delta > 1e-12).any():
            raise ValueError(f'pilot reproduction failed: {name}, maximum absolute difference {delta.max()}')
        maximum[name] = float(delta.max()) if len(delta) else 0.0
    for column in ('image_id_a', 'image_id_b', 'place_uid_a', 'place_uid_b', 'city_id_a', 'city_id_b', 'rank_bin'):
        if not np.array_equal(actual[column], reference[column]):
            raise ValueError(f'pilot reproduction failed: endpoint/metadata {column}')
    return {'passed': True, 'pair_count': len(reference), 'exact_integer_and_duplicate_metrics': True,
            'absolute_tolerance': 1e-12, 'maximum_metric_differences': maximum}


def compare_pilot_calibration(calibrated, frozen, snapshot):
    reference = frozen.set_index('pair_uid')
    if calibrated['pair_uid'].duplicated().any() or not set(reference.index).issubset(set(calibrated['pair_uid'])):
        raise ValueError('pilot calibration pairs are missing or duplicated')
    actual = calibrated.set_index('pair_uid').loc[reference.index]
    maximum = {}
    for name in (*PERCENTILES, 'structural_bottleneck', 'structural_geomean'):
        left, right = actual[name].to_numpy(float), reference[name].to_numpy(float)
        differences = np.abs(left - right)
        if not np.isfinite(differences).all() or (differences > 1e-12).any():
            raise ValueError(f'pilot calibration differs from frozen Step 2B: {name}')
        maximum[name] = float(differences.max()) if len(differences) else 0.0
    for name in ('q95', 'q99'):
        # Step 2B calibration CSV precedes graph labels, so reproduce its rule
        # from frozen bottleneck and duplicate flag if the labels are absent.
        expected = (reference['structural_bottleneck'].to_numpy(float) >= QUANTILES[name]) & ~boolean_values(reference['exact_pixel_duplicate'])
        actual_flags = boolean_values(actual['core_' + name])
        if not np.array_equal(actual_flags, expected):
            raise ValueError(f'pilot calibration differs from frozen Step 2B labels: {name}')
        if int(actual_flags.sum()) != snapshot['joint_null']['rates']['all']['candidate'][name + '_count']:
            raise ValueError(f'pilot calibration differs from frozen Step 2C count: {name}')
    return {'passed': True, 'maximum_calibration_differences': maximum,
            'q95_count': int(boolean_values(actual['core_q95']).sum()),
            'q99_count': int(boolean_values(actual['core_q99']).sum())}


def load_frozen_null(*, repo_root=ROOT):
    root = Path(repo_root)
    provenance = frozen_provenance(root)
    historical = validate_metrics(read_metadata_csv(root / 'cache/countermine_rgb/step2a/candidate_structural_metrics.csv'), candidate=True)
    controls = validate_metrics(read_metadata_csv(root / 'cache/countermine_rgb/step2a/random_structural_metrics.csv'), candidate=False)
    snapshot = read_json(root / 'docs/audits/step2c_null_topology_metrics.json')
    expected = snapshot['joint_null']['rates']['all']
    if len(controls) != expected['random']['count'] or len(historical) != expected['candidate']['count']:
        raise ValueError('frozen pilot/null population count differs from Step 2C')
    random, thresholds = random_joint_evidence(controls)
    _compare_frozen(thresholds, snapshot['joint_null']['thresholds'], 'frozen_null_thresholds')
    for relation, mask in (('all', np.ones(len(random), bool)),
                           ('same_city', random['same_city']), ('cross_city', ~random['same_city'])):
        for name in ('q95', 'q99'):
            flags = random.loc[mask, 'random_core_' + name]
            saved = snapshot['joint_null']['rates'][relation]['random']
            if int(flags.sum()) != saved[name + '_count'] or not np.isclose(float(flags.mean()), saved[name + '_fraction'], rtol=0, atol=1e-12):
                raise ValueError('random joint rates disagree with frozen Step 2C')
    frozen, _ = load_calibrated_artifacts(root / 'cache/countermine_rgb/step2b', repo_root=root)
    pilot = calibrate_full_evidence(historical, controls)
    audit = compare_pilot_calibration(pilot, frozen, snapshot)
    return controls, historical, frozen, snapshot, provenance, audit


def validate_pilot_reproduction(full_metrics, *, repo_root=ROOT):
    controls, historical, frozen, snapshot, _, _ = load_frozen_null(repo_root=repo_root)
    report = compare_pilot_metrics(full_metrics, historical)
    report.update(compare_pilot_calibration(calibrate_full_evidence(
        full_metrics.loc[full_metrics['pair_uid'].isin(historical['pair_uid'])], controls), frozen, snapshot))
    return report


def validate_full_calibrated(frame):
    result = validate_metrics(frame, candidate=True)
    for name in (*PERCENTILES, 'structural_bottleneck', 'structural_geomean', 'spatial_support_percentile'):
        values = pd.to_numeric(result[name], errors='raise').to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError('calibrated evidence must be finite percentiles in [0,1]')
        result[name] = values
    expected = {'structural_bottleneck': np.minimum(result['ratio_null_percentile'], result['match_count_null_percentile']),
                'structural_geomean': np.sqrt(result['ratio_null_percentile'] * result['match_count_null_percentile']),
                'spatial_support_percentile': result['coverage_null_percentile']}
    for name, values in expected.items():
        if not np.allclose(result[name], values, atol=1e-12, rtol=0):
            raise ValueError(f'calibrated formula differs: {name}')
    for short in METRIC_NAMES:
        source = np.where(result['same_city'], 'same_city', 'cross_city')
        if not np.array_equal(result[short + '_null_source'], source):
            raise ValueError('calibrated relation null source differs from endpoint relation')
    for name, quantile in QUANTILES.items():
        flags = (result['structural_bottleneck'] >= quantile) & ~result['exact_pixel_duplicate']
        for column, values in (('core_' + name, flags), ('core_' + name + '_geo500', flags & (result['geo_distance_m'] >= 500))):
            parsed = boolean_values(result[column])
            if not np.array_equal(parsed, values):
                raise ValueError(f'calibrated core flag differs: {column}')
            result[column] = parsed
    result['min_num_keypoints'] = np.minimum(result['num_keypoints_a'], result['num_keypoints_b'])
    return result


def compare_calibration_to_measurements(calibrated, measurements, controls):
    """Bind every calibrated row to raw measurements and the unchanged null.

    Recompute from the original measured CSV parse, not the calibrated CSV's
    reserialized raw ratios: the latter can again lose an exact null tie.
    """
    expected = calibrate_full_evidence(measurements, controls)
    if calibrated['pair_uid'].tolist() != expected['pair_uid'].tolist():
        raise ValueError('full calibration identity/order differs from raw measurements')
    for name in expected:
        if name not in calibrated:
            raise ValueError(f'full calibration lacks source/evidence column: {name}')
        left, right = calibrated[name], expected[name]
        if pd.api.types.is_bool_dtype(right):
            equal = np.array_equal(boolean_values(left), right)
        elif pd.api.types.is_numeric_dtype(right):
            # Frozen geographic metadata uses 1e-6 m, including large
            # cross-city distances whose CSV parse can move by one float ULP.
            # Structural measurement/calibration metrics remain at 1e-12.
            tolerance = 1e-6 if name == 'geo_distance_m' else 1e-12
            equal = np.allclose(pd.to_numeric(left), right, atol=tolerance, rtol=0, equal_nan=True)
        else:
            equal = np.array_equal(left, right)
        if not equal:
            raise ValueError(f'full calibration differs from raw measurement/frozen null: {name}')


def run_calibration(runtime_dir=DEFAULT_DIR, *, repo_root=ROOT):
    from .full_structural_measurement import load_full_metrics
    directory = Path(runtime_dir)
    targets = [directory / 'full_calibrated_candidates.csv', directory / 'full_calibration_summary.json']
    guard_step2d(targets, repo_root)
    metrics, measurement = load_full_metrics(directory, repo_root=repo_root)
    original_hashes = {'full_metrics_sha256': sha256_file(directory / 'full_candidate_structural_metrics.csv'),
                       'full_measurement_summary_sha256': sha256_file(directory / 'full_measurement_summary.json')}
    if original_hashes['full_metrics_sha256'] != measurement['full_metrics_sha256']:
        raise ValueError('full measurements changed after validation')
    controls, historical, frozen, snapshot, provenance, preflight = load_frozen_null(repo_root=repo_root)
    audit = compare_pilot_metrics(metrics, historical)
    result = calibrate_full_evidence(metrics, controls)
    audit.update(compare_pilot_calibration(result, frozen, snapshot))
    provenance.update({**original_hashes, 'code_sha256': code_hashes(SOURCE_FILES, repo_root)})
    config = {'seed': 42, 'null': 'same/cross-city weak ECDF count(null <= x)/N_relation',
              'null_fallback': False, 'quantiles': QUANTILES, 'coverage_in_primary_bottleneck': False,
              'diagnostic_slices_only': True}
    summary = {'schema_version': 1, 'complete': True, 'stage': '2D frozen matched-random calibration',
               'configuration': config, 'provenance': provenance, 'candidate_count': len(result),
               'random_count': len(controls), 'relation_null_counts': {
                   'same_city': int(controls['same_city'].sum()), 'cross_city': int((~controls['same_city']).sum())},
               'frozen_q95_q99_thresholds': snapshot['joint_null']['thresholds'],
               'frozen_random_joint_rates': {name: rates['random'] for name, rates in snapshot['joint_null']['rates'].items()},
               'pilot_reproduction': audit, 'frozen_null_preflight': preflight,
               'columns': list(result.columns)}
    import hashlib
    digest = hashlib.sha256(result.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
    summary['full_calibrated_candidates_sha256'] = digest
    if any(path.exists() for path in targets):
        _, saved = load_full_calibrated(directory, repo_root=repo_root)
        if saved != summary:
            raise ValueError('existing full calibration differs; use a fresh Step 2D directory')
        return saved
    verify_hashes(provenance['input_file_sha256'], repo_root)
    for filename, key in (('full_candidate_structural_metrics.csv', 'full_metrics_sha256'),
                          ('full_measurement_summary.json', 'full_measurement_summary_sha256')):
        if sha256_file(directory / filename) != original_hashes[key]:
            raise ValueError('full measurements changed during calibration')
    write_csv(targets[0], result, repo_root=repo_root)
    write_json(targets[1], summary, repo_root=repo_root)
    return summary


def load_full_calibrated(runtime_dir=DEFAULT_DIR, *, repo_root=ROOT):
    directory = Path(runtime_dir)
    summary = read_json(directory / 'full_calibration_summary.json')
    if summary.get('complete') is not True or summary.get('schema_version') != 1 or summary['pilot_reproduction'].get('passed') is not True:
        raise ValueError('full calibration requires successful pilot reproduction and complete summary')
    if code_hashes(SOURCE_FILES, repo_root) != summary['provenance']['code_sha256']:
        raise ValueError('full calibration implementation changed')
    verify_hashes(summary['provenance']['input_file_sha256'], repo_root)
    for filename, key in (('full_candidate_structural_metrics.csv', 'full_metrics_sha256'),
                          ('full_measurement_summary.json', 'full_measurement_summary_sha256')):
        if sha256_file(directory / filename) != summary['provenance'][key]:
            raise ValueError('full calibration source measurement checksum differs')
    path = directory / 'full_calibrated_candidates.csv'
    if sha256_file(path) != summary['full_calibrated_candidates_sha256']:
        raise ValueError('full calibration CSV checksum differs')
    frame = validate_full_calibrated(read_metadata_csv(path))
    if len(frame) != summary['candidate_count'] or list(frame.columns) != summary['columns']:
        raise ValueError('full calibration count/schema differs')
    measurements = validate_metrics(read_metadata_csv(directory / 'full_candidate_structural_metrics.csv'), candidate=True)
    controls = validate_metrics(read_metadata_csv(Path(repo_root) / 'cache/countermine_rgb/step2a/random_structural_metrics.csv'), candidate=False)
    compare_calibration_to_measurements(frame, measurements, controls)
    return frame, summary
