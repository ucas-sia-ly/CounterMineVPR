"""CPU-only joint empirical-null validation of frozen Step 2A/2B evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from countermine.mining.countermine_graph import load_graph_artifacts
from countermine.mining.graph_analysis import analyze_graphs, boolean_values, validate_graph_snapshot
from countermine.mining.structural_analysis import (
    _atomic_bytes, distribution, read_metadata_csv, sha256_file, validate_metrics, weak_ecdf,
)
from countermine.mining.structural_calibration import _compare_frozen, load_calibrated_artifacts

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    'countermine/mining/joint_null.py', 'countermine/mining/topology_null.py',
    'tools/14_analyze_joint_null.py', 'tools/15_analyze_topology_null.py',
    'tools/16_export_step2c_audit.py',
)
RANK_BINS = ('rank_1', 'rank_2_5', 'rank_6_10', 'rank_11_20', 'rank_21_50')
RELATIONS = (('same_city', True), ('cross_city', False))
QUANTILES = (('q95', .95), ('q99', .99))


def read_json(path):
    def reject(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def code_hashes(repo_root=ROOT):
    return {name: sha256_file(Path(repo_root) / name) for name in SOURCE_FILES}


def validate_step2c_snapshot(snapshot):
    validate_graph_snapshot(snapshot)
    forbidden = {'coordinates', 'matched_coordinates', 'descriptors', 'image_pixels', 'model_weights'}
    def walk(value):
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError('Step 2C snapshot contains forbidden bulk data')
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(snapshot)
    json.dumps(snapshot, allow_nan=False)


def guard_destinations(paths, repo_root=ROOT):
    """Allow writes only in the three explicitly authorized Step 2C locations."""
    repository = Path(repo_root).resolve()
    cache = repository / 'cache/countermine_rgb/step2c'
    outputs = repository / 'outputs/step2c'
    audits = repository / 'docs/audits'
    def allowed_location(path):
        return (path.is_relative_to(cache) or path.is_relative_to(outputs)
                or (path.parent == audits and path.name.startswith('step2c_')))
    for path in paths:
        raw, resolved = Path(path), Path(path).resolve()
        if not allowed_location(resolved) or not allowed_location(raw.absolute()) or raw.is_symlink():
            raise ValueError('Step 2C writes must stay in step2c cache/output or step2c_* audit files')


def recheck_inputs(provenance, repo_root=ROOT):
    for name, digest in provenance['input_file_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('source paths must remain relative to the repository')
        if sha256_file(Path(repo_root) / relative) != digest:
            raise ValueError(f'frozen source changed: {name}')
    if code_hashes(repo_root) != provenance['step2c_code_sha256']:
        raise ValueError('Step 2C implementation changed after analysis')


def load_frozen_inputs(step2b_dir='cache/countermine_rgb/step2b',
                       step2b_snapshot='docs/audits/step2b_countermine_graph_metrics.json', *, repo_root=ROOT):
    """Validate all published hashes and reproduce the frozen graph report read-only."""
    repository, directory = Path(repo_root).resolve(), Path(step2b_dir).resolve()
    snapshot_path = Path(step2b_snapshot).resolve()
    calibrated, calibration = load_calibrated_artifacts(directory, repo_root=repository)
    image_nodes, edges, place_nodes, place_edges, graph = load_graph_artifacts(directory, repo_root=repository)
    frozen = read_json(snapshot_path)
    validate_graph_snapshot(frozen)
    previous = calibration['provenance']
    for key in ('step2a_snapshot_sha256', 'step2a_candidate_metrics_sha256', 'step2a_random_metrics_sha256',
                'step2a_measurement_summary_sha256', 'step2a_measurement_config_sha256',
                'step2a_image_fingerprints_sha256', 'step2a_input_hashes', 'validation'):
        if frozen['provenance'].get(key) != previous[key]:
            raise ValueError('frozen Step 2B snapshot provenance differs from the validated source chain')
    if snapshot_path.read_bytes() != (directory / 'graph_analysis_summary.json').read_bytes():
        raise ValueError('curated Step 2B snapshot differs from its runtime snapshot')
    if len(edges) != 5000 or len(calibrated) != 5000:
        raise ValueError('Step 2C requires the frozen 5000-edge pilot')
    if frozen['graph_artifact_hashes'] != graph['output_hashes']:
        raise ValueError('Step 2B snapshot and graph CSV hashes differ')
    for name, digest in frozen['provenance']['graph_code_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(repository / name) != digest:
            raise ValueError('frozen Step 2B implementation hashes differ')
    report, _, _ = analyze_graphs(image_nodes, edges, place_nodes, place_edges)
    for name, value in report.items():
        _compare_frozen(value, frozen[name], f'step2b.{name}')
    # Also bind every scalar evidence bundle to the validated calibration CSV.
    merged = edges.merge(calibrated, on='pair_uid', suffixes=('', '_calibrated'), validate='one_to_one')
    if len(merged) != 5000:
        raise ValueError('image graph and calibration identities differ')
    for name in ('ratio_null_percentile', 'match_count_null_percentile', 'structural_bottleneck', 'structural_geomean'):
        if not np.allclose(merged[name], merged[name + '_calibrated'], rtol=0, atol=1e-12):
            raise ValueError('image graph evidence differs from calibration')
    recorded = calibration['input_files']
    step2a = (repository / recorded['runtime_dir']).resolve()
    measurements = validate_metrics(read_metadata_csv(step2a / 'candidate_structural_metrics.csv'), candidate=True)
    controls = validate_metrics(read_metadata_csv(step2a / 'random_structural_metrics.csv'), candidate=False)
    if len(controls) != 4999:
        raise ValueError('Step 2C requires 4999 measured random controls')
    source_paths = [snapshot_path, *(directory / name for name in graph['output_hashes']),
                    *(directory / name for name in ('graph_summary.json', 'calibration_summary.json',
                                                  'calibrated_candidates.csv', 'graph_analysis_summary.json')),
                    *(step2a / name for name in ('candidate_structural_metrics.csv', 'random_structural_metrics.csv',
                                               'measurement_summary.json', 'measurement_config.json', 'image_fingerprints.json')),
                    *(repository / recorded[name] for name in ('snapshot', 'manifest', 'candidates', 'candidate_summary'))]
    # Step 2B records curated figure and validation-log hashes separately.
    validation_path = directory / 'validation_summary.json'
    if validation_path.is_file():
        validation = read_json(validation_path)
        if validation['snapshot_sha256'] != sha256_file(snapshot_path):
            raise ValueError('Step 2B validation snapshot hash differs')
        for name, digest in validation.get('figure_sha256', {}).items():
            if Path(name).name != name or not name.startswith('step2b_'):
                raise ValueError('invalid Step 2B curated artifact name')
            path = repository / 'docs/audits' / name
            if sha256_file(path) != digest:
                raise ValueError(f'Step 2B figure checksum differs: {name}')
            source_paths.append(path)
        for name, digest in validation.get('logs', {}).items():
            if Path(name).name != name or sha256_file(directory / name) != digest:
                raise ValueError('Step 2B log checksum differs')
            source_paths.append(directory / name)
        source_paths.append(validation_path)
    input_hashes = {}
    for path in source_paths:
        try:
            relative = path.resolve().relative_to(repository).as_posix()
        except ValueError as error:
            raise ValueError('frozen sources must be inside the repository') from error
        input_hashes[relative] = sha256_file(path)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    provenance = {
        'step2a_snapshot_sha256': previous['step2a_snapshot_sha256'],
        'step2b_snapshot_sha256': sha256_file(snapshot_path),
        'step2a_candidate_metrics_sha256': previous['step2a_candidate_metrics_sha256'],
        'step2a_random_metrics_sha256': previous['step2a_random_metrics_sha256'],
        'graph_artifact_sha256': graph['output_hashes'], 'input_file_sha256': input_hashes,
        'step2c_code_sha256': code_hashes(repository), 'current_git_commit': commit,
        'step2a_binding_validation': previous['validation'],
        'real_rgb_only': True, 'synthetic_images_used': False, 'no_new_local_matching': True,
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
    }
    recheck_inputs(provenance, repository)
    edges = edges.sort_values('pair_uid').reset_index(drop=True)
    return {'image_nodes': image_nodes, 'image_edges': edges,
            'joint_candidates': bind_joint_measurements(edges, measurements),
            'place_nodes': place_nodes, 'place_edges': place_edges, 'random': controls,
            'step2b_snapshot': frozen, 'provenance': provenance}


def bind_joint_measurements(edges, measurements):
    """Use Step 2A raw ratios for ECDF ties, retaining frozen graph evidence.

    Graph CSV serialization and numeric parsing can slightly shift a ratio.
    An ECDF is discontinuous there: even this small change can remove an entire
    tie group. Bind by UID to the original measurements used for calibration;
    do not round values, relax percentile checks, or modify graph artifacts.
    """
    if (edges['pair_uid'].duplicated().any() or measurements['pair_uid'].duplicated().any()
            or set(edges['pair_uid']) != set(measurements['pair_uid'])):
        raise ValueError('graph and original candidate measurement identities differ')
    original = measurements.set_index('pair_uid').loc[edges['pair_uid']].reset_index(drop=True)
    result = edges.copy()
    for column in ('same_city', 'rank_bin', 'num_matches', 'exact_pixel_duplicate'):
        left, right = result[column].to_numpy(), original[column].to_numpy()
        if column in ('same_city', 'exact_pixel_duplicate'):
            left, right = boolean_values(result[column]), boolean_values(original[column])
        if not np.array_equal(left, right):
            raise ValueError(f'graph and original candidate measurements disagree: {column}')
    if not np.allclose(result['local_match_ratio'], original['local_match_ratio'], rtol=0, atol=1e-12):
        raise ValueError('graph and original candidate measurements disagree: local_match_ratio')
    # Positional assignment after explicit UID alignment also handles shuffled
    # rows and nonconsecutive graph indices without pandas index alignment.
    result['local_match_ratio'] = original['local_match_ratio'].to_numpy()
    return result


def weak_ecdf_threshold(values, quantile):
    """Smallest observed raw value with weak F(value)>=q, including all ties."""
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 < quantile <= 1:
        raise ValueError('threshold reconstruction needs finite nonempty data and q in (0,1]')
    distinct, counts = np.unique(values, return_counts=True)
    percentiles = counts.cumsum() / len(values)
    threshold = distinct[np.flatnonzero(percentiles >= quantile)[0]].item()
    return {'threshold': threshold, 'realized_tail_count': int((values >= threshold).sum()),
            'realized_tail_fraction': float((values >= threshold).mean()),
            'weak_ecdf_at_threshold': float(weak_ecdf(values, [threshold])[0])}


def random_joint_evidence(random):
    result = random.copy()
    result['same_city'] = boolean_values(result['same_city'])
    for metric in ('local_match_ratio', 'num_matches'):
        values = pd.to_numeric(result[metric], errors='raise').to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError('random metrics must be finite')
        result[metric] = values
    for prefix in ('random_', 'loo_'):
        for metric in ('ratio_null_percentile', 'match_count_null_percentile'):
            result[prefix + metric] = np.nan
    thresholds = {}
    for relation, flag in RELATIONS:
        selected = result['same_city'] == flag
        thresholds[relation] = {'count': int(selected.sum())}
        if not selected.any():
            continue
        for short, metric in (('ratio', 'local_match_ratio'), ('count', 'num_matches')):
            values = result.loc[selected, metric].to_numpy()
            positions = np.searchsorted(np.sort(values), values, side='right')
            column = 'ratio_null_percentile' if short == 'ratio' else 'match_count_null_percentile'
            result.loc[selected, 'random_' + column] = positions / len(values)
            if len(values) > 1:
                result.loc[selected, 'loo_' + column] = (positions - 1) / (len(values) - 1)
            for name, quantile in QUANTILES:
                reconstruction = weak_ecdf_threshold(values, quantile)
                if short == 'count':
                    reconstruction['threshold'] = int(reconstruction['threshold'])
                thresholds[relation][f'{short}_{name}_threshold'] = reconstruction['threshold']
                thresholds[relation][f'{short}_{name}_realized_tail'] = reconstruction
    result['random_structural_bottleneck'] = np.minimum(result['random_ratio_null_percentile'], result['random_match_count_null_percentile'])
    result['loo_structural_bottleneck'] = np.minimum(result['loo_ratio_null_percentile'], result['loo_match_count_null_percentile'])
    for name, quantile in QUANTILES:
        result['random_core_' + name] = result['random_structural_bottleneck'] >= quantile
        available = result['loo_structural_bottleneck'].notna()
        result['loo_core_' + name] = pd.Series(pd.NA, index=result.index, dtype='boolean')
        result.loc[available, 'loo_core_' + name] = result.loc[available, 'loo_structural_bottleneck'] >= quantile
    return result, thresholds


def core_rates(frame, prefix=''):
    report = {'count': len(frame)}
    for name, _ in QUANTILES:
        flags = frame[prefix + 'core_' + name]
        valid = flags.notna()
        count = int(flags.loc[valid].sum())
        report[name + '_count'] = count
        report[name + '_evaluated_count'] = int(valid.sum())
        report[name + '_fraction'] = count / int(valid.sum()) if valid.any() else None
    return report


def compare_rates(candidate_rates, random_rates):
    result = {}
    for name, _ in QUANTILES:
        candidate, random = candidate_rates[name + '_fraction'], random_rates[name + '_fraction']
        result[name + '_rate_ratio'] = candidate / random if random and candidate is not None else None
        result[name + '_absolute_difference'] = candidate - random if candidate is not None and random is not None else None
    return result


def align_paired_joint(candidates, random):
    if candidates['pair_uid'].duplicated().any() or random['candidate_pair_uid'].duplicated().any():
        raise ValueError('joint pairs require unique candidate/control alignment keys')
    if not set(random['candidate_pair_uid']).issubset(set(candidates['pair_uid'])):
        raise ValueError('random control refers to an unknown candidate pair_uid')
    candidate = candidates[['pair_uid', 'same_city', 'rank_bin', 'structural_bottleneck']].rename(
        columns={'pair_uid': 'candidate_pair_uid', 'structural_bottleneck': 'candidate_structural_bottleneck'})
    control = random[['candidate_pair_uid', 'pair_uid', 'same_city', 'rank_bin', 'random_structural_bottleneck']].rename(
        columns={'pair_uid': 'random_pair_uid'})
    joined = candidate.merge(control, on='candidate_pair_uid', suffixes=('', '_random'), validate='one_to_one')
    if not (joined['same_city'] == joined['same_city_random']).all() or not (joined['rank_bin'] == joined['rank_bin_random']).all():
        raise ValueError('candidate/control relation or rank_bin disagrees')
    joined = joined.drop(columns=['same_city_random', 'rank_bin_random'])
    joined['delta_bottleneck'] = joined['candidate_structural_bottleneck'] - joined['random_structural_bottleneck']
    if not np.isfinite(joined[['candidate_structural_bottleneck', 'random_structural_bottleneck', 'delta_bottleneck']].to_numpy()).all():
        raise ValueError('paired bottlenecks must be finite')
    return joined.sort_values('candidate_pair_uid').reset_index(drop=True)


def paired_joint_summary(paired):
    delta = paired['delta_bottleneck'].to_numpy()
    return {'count': len(paired), 'candidate_gt_random': int((delta > 0).sum()),
            'candidate_eq_random': int((delta == 0).sum()), 'candidate_lt_random': int((delta < 0).sum()),
            'fraction_candidate_gt_random': float((delta > 0).mean()) if len(delta) else None,
            'delta_bottleneck': distribution(delta, delta=True)}


def analyze_joint_null(candidates, random):
    candidates = candidates.copy()
    candidates['same_city'] = boolean_values(candidates['same_city'])
    controls, thresholds = random_joint_evidence(random)
    # Independently verify that candidates use the very same relation-specific null.
    canonical_ratio = np.zeros(len(candidates))
    canonical_count = np.zeros(len(candidates))
    for relation, flag in RELATIONS:
        c, r = candidates['same_city'] == flag, controls['same_city'] == flag
        if c.any() and not r.any():
            raise ValueError('candidate relation lacks a frozen random null')
        for short, metric in (('ratio', 'local_match_ratio'), ('match_count', 'num_matches')):
            if c.any():
                reconstructed = weak_ecdf(controls.loc[r, metric], candidates.loc[c, metric])
                if not np.allclose(candidates.loc[c, short + '_null_percentile'], reconstructed, rtol=0, atol=1e-12):
                    raise ValueError('candidate percentile differs from frozen relation-specific convention')
                (canonical_ratio if short == 'ratio' else canonical_count)[c] = reconstructed
    canonical_bottleneck = np.minimum(canonical_ratio, canonical_count)
    if not np.allclose(candidates['structural_bottleneck'], canonical_bottleneck, rtol=0, atol=1e-12):
        raise ValueError('candidate bottleneck differs from reconstructed frozen evidence')
    for name, quantile in QUANTILES:
        expected = (canonical_bottleneck >= quantile) & ~boolean_values(candidates['exact_pixel_duplicate'])
        if not np.array_equal(boolean_values(candidates['core_' + name]), expected):
            raise ValueError('frozen candidate core flags disagree with joint evidence')
    paired_candidates = candidates.copy()
    # Reconstruct identical weak-ECDF arithmetic on BOTH paired sides. This
    # avoids false wins/losses from CSV floating-point roundtrip at true ties.
    paired_candidates['structural_bottleneck'] = canonical_bottleneck
    paired = align_paired_joint(paired_candidates, controls)
    rates, loo_rates, paired_report = {}, {}, {}
    selections = [('all', np.ones(len(candidates), bool), np.ones(len(controls), bool))]
    selections += [(name, candidates['same_city'] == flag, controls['same_city'] == flag) for name, flag in RELATIONS]
    for name, c, r in selections:
        candidate_rate, random_rate = core_rates(candidates.loc[c]), core_rates(controls.loc[r], 'random_')
        rates[name] = {'candidate': candidate_rate, 'random': random_rate, **compare_rates(candidate_rate, random_rate)}
        loo_rates[name] = core_rates(controls.loc[r], 'loo_')
    for name, selected in [('all', paired), *((name, paired.loc[paired['same_city'] == flag]) for name, flag in RELATIONS)]:
        paired_report[name] = paired_joint_summary(selected)
    paired_report['rank_bins'] = {name: paired_joint_summary(paired.loc[paired['rank_bin'] == name]) for name in RANK_BINS}
    report = {
        'joint_null': {'thresholds': thresholds, 'rates': rates, 'leave_one_out_sensitivity': loo_rates,
                       'definitions': {'primary': 'same relation weak ECDF count(null <= x)/N, including self for random controls',
                                       'leave_one_out': '(count(null <= own_value)-1)/(N-1), removing the same control from both metrics',
                                       'loo_singleton_relation': 'undefined; null percentiles/flags, excluded from sensitivity denominator',
                                       'joint': 'min(ratio_null_percentile, match_count_null_percentile) >= q',
                                       'candidate_duplicates': 'candidate rates use frozen Step 2B nonduplicate core flags; random joint criterion has no extra duplicate gate',
                                       'paired_ties': 'candidate weak-ECDF bottleneck reconstructed and validated against frozen Step 2B at atol=1e-12; both sides use identical arithmetic before exact comparison',
                                       'threshold': 'smallest observed raw value whose weak ECDF >= q; no interpolated quantile',
                                       'interpretation': 'descriptive structural-evidence enrichment, not recognition performance'}},
        'paired_joint_evidence': paired_report,
    }
    return controls, paired, report


def publish_stage(output_dir, stage, tables, report, provenance, configuration, *, repo_root=ROOT):
    directory = Path(output_dir)
    summary_path = directory / (stage + '_summary.json')
    targets = [summary_path, *(directory / name for name in tables)]
    guard_destinations(targets, repo_root)
    payloads = {name: table.to_csv(index=False, float_format='%.17g').encode('utf-8') for name, table in tables.items()}
    summary = {'schema_version': 1, 'stage': stage, 'complete': True, 'provenance': provenance,
               'configuration': configuration, 'report': report,
               'artifacts': {name: {'sha256': hashlib.sha256(payload).hexdigest(), 'rows': len(tables[name]),
                                    'columns': list(tables[name].columns)} for name, payload in payloads.items()}}
    validate_step2c_snapshot(summary)
    recheck_inputs(provenance, repo_root)
    if any(path.exists() for path in targets):
        if not all(path.is_file() for path in targets) or read_json(summary_path) != summary:
            raise ValueError('existing Step 2C artifacts differ or are incomplete; use a fresh Step 2C output directory')
        for name, metadata in summary['artifacts'].items():
            if sha256_file(directory / name) != metadata['sha256']:
                raise ValueError('existing Step 2C artifact checksum differs')
        return summary
    recheck_inputs(provenance, repo_root)
    guard_destinations(targets, repo_root)
    for name, payload in payloads.items():
        _atomic_bytes(directory / name, payload)
    _atomic_bytes(summary_path, (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    return summary


def load_stage(output_dir, stage, expected_provenance, *, repo_root=ROOT):
    directory = Path(output_dir)
    summary = read_json(directory / (stage + '_summary.json'))
    validate_step2c_snapshot(summary)
    if summary.get('stage') != stage or summary.get('complete') is not True or summary.get('schema_version') != 1:
        raise ValueError('Step 2C stage needs a complete supported summary')
    if summary['provenance'] != expected_provenance:
        raise ValueError('Step 2C stage provenance differs from current frozen inputs/code')
    tables = {}
    for name, metadata in summary['artifacts'].items():
        if Path(name).name != name or sha256_file(directory / name) != metadata['sha256']:
            raise ValueError('Step 2C stage artifact checksum differs')
        table = pd.read_csv(directory / name, keep_default_na=False)
        if len(table) != metadata['rows'] or list(table.columns) != metadata['columns']:
            raise ValueError('Step 2C stage table schema/count differs')
        tables[name] = table
    recheck_inputs(expected_provenance, repo_root)
    return summary, tables


def run_joint_null(step2b_dir, step2b_snapshot, output_dir, *, repo_root=ROOT):
    guard_destinations([Path(output_dir) / name for name in (
        'joint_null_summary.json', 'random_joint_metrics.csv', 'paired_joint_evidence.csv')], repo_root)
    inputs = load_frozen_inputs(step2b_dir, step2b_snapshot, repo_root=repo_root)
    controls, paired, report = analyze_joint_null(inputs['joint_candidates'], inputs['random'])
    return publish_stage(output_dir, 'joint_null', {'random_joint_metrics.csv': controls,
                         'paired_joint_evidence.csv': paired}, report, inputs['provenance'],
                         {'seed': 42, 'candidate_count': 5000, 'random_count': 4999,
                          'quantiles': [.95, .99], 'primary_self_inclusive': True}, repo_root=repo_root)
