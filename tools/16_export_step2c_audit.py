#!/usr/bin/env python3
"""Export validated Step 2C joint-null/topology-null summaries and four CPU plots."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from countermine.mining.joint_null import (
    analyze_joint_null, guard_destinations, load_frozen_inputs, load_stage,
    recheck_inputs, validate_step2c_snapshot,
)
from countermine.mining.structural_analysis import _atomic_bytes, sha256_file
from countermine.mining.structural_calibration import _compare_frozen
from countermine.mining.topology_null import (
    BUNDLE_COLUMNS, FixedGraph, SLICES, STATISTICS, positive_integer,
    summarize_permutations, validate_observed_statistics,
)

ARTIFACT_NAMES = ('candidate_vs_random_core_rate.png', 'paired_bottleneck.png',
                  'repeated_support_null.png', 'place_component_null.png')


def validate_analysis_stages(inputs, joint, topology, tables):
    """Reproduce joint tables and null summaries without rerunning permutations."""
    if joint['configuration'] != {'seed': 42, 'candidate_count': 5000, 'random_count': 4999,
                                  'quantiles': [.95, .99], 'primary_self_inclusive': True}:
        raise ValueError('joint-null configuration differs from the fixed pilot')
    controls, paired, report = analyze_joint_null(inputs['joint_candidates'], inputs['random'])
    _compare_frozen(report, joint['report'], 'joint_null_report')
    for name, table in (('random_joint_metrics.csv', controls), ('paired_joint_evidence.csv', paired)):
        digest = hashlib.sha256(table.to_csv(index=False, float_format='%.17g').encode()).hexdigest()
        if digest != joint['artifacts'][name]['sha256']:
            raise ValueError('joint-null table differs from frozen source reproduction')
    config = topology['configuration']
    count = positive_integer(config['permutations'], 'permutations')
    positive_integer(config['seed'], 'seed', allow_zero=True)
    if set(config) != {'seed', 'permutations', 'candidate_count'} or config['candidate_count'] != 5000:
        raise ValueError('topology-null configuration differs from the fixed graph pilot')
    permutations = tables['topology_permutations.csv']
    expected_columns = ['permutation_index'] + [name + '__' + key for name in SLICES for key in STATISTICS]
    if list(permutations.columns) != expected_columns or len(permutations) != count:
        raise ValueError('permutation distribution schema/count differs')
    values = permutations.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError('topology permutation statistics must be finite nonnegative integers')
    if not np.array_equal(permutations['permutation_index'].to_numpy(), np.arange(count)):
        raise ValueError('permutation indices must be exactly 0..N-1')
    graph = FixedGraph(inputs['image_edges'])
    observed = graph.statistics()
    validate_observed_statistics(observed, inputs['step2b_snapshot'])
    saved = topology['report']['topology_null']
    _compare_frozen(observed, saved['observed_statistics'], 'observed_topology')
    _compare_frozen(graph.stratum_records, saved['stratum_counts'], 'permutation_strata')
    if saved['bundle_columns'] != list(BUNDLE_COLUMNS) or saved['graph_endpoints_fixed'] is not True:
        raise ValueError('permutation bundle or fixed-graph definition differs')
    if saved['seed'] != config['seed'] or saved['permutations'] != count:
        raise ValueError('topology report and configuration differ')
    for name in SLICES:
        if not (permutations[name + '__edge_count'] == observed[name]['edge_count']).all():
            raise ValueError('permutation distribution changed core slice totals')
    _compare_frozen(summarize_permutations(observed, permutations), saved['comparisons'], 'permutation_summary')


def create_plots(joint_report, topology_report, paired, permutations, output_dir, *, repo_root=ROOT):
    targets = [Path(output_dir) / name for name in ARTIFACT_NAMES]
    guard_destinations(targets, repo_root)
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    def save(figure, name):
        with io.BytesIO() as buffer:
            figure.savefig(buffer, format='png', dpi=160, bbox_inches='tight')
            guard_destinations([Path(output_dir) / name], repo_root)
            _atomic_bytes(Path(output_dir) / name, buffer.getvalue())
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    all_rates = []
    for axis, relation in zip(axes, ('all', 'same_city', 'cross_city')):
        rates = joint_report['joint_null']['rates'][relation]
        for offset, source, color in ((-.18, 'candidate', '#315b88'), (.18, 'random', '#bc8050')):
            values = [rates[source][name + '_fraction'] for name in ('q95', 'q99')]
            all_rates.extend(value for value in values if value is not None)
            axis.bar(np.arange(2) + offset, [value if value is not None else 0 for value in values],
                     width=.36, label=source, color=color)
        axis.set_xticks([0, 1], ['q95', 'q99'])
        axis.set_title(relation)
    axes[0].set_ylim(0, max(.01, max(all_rates, default=0) * 1.15))
    axes[0].set_ylabel('Joint core fraction')
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .91), ncol=2, frameon=False)
    figure.suptitle('Frozen candidate vs matched-random joint evidence')
    figure.tight_layout(rect=(0, 0, 1, .82))
    save(figure, ARTIFACT_NAMES[0])

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(paired['random_structural_bottleneck'].to_numpy(float),
                 paired['candidate_structural_bottleneck'].to_numpy(float), s=8, alpha=.25, color='#315b88')
    axis.plot([0, 1], [0, 1], '--', color='#a13a31', label='y = x')
    axis.set(xlabel='Matched-random bottleneck', ylabel='Candidate bottleneck',
             xlim=(-.02, 1.02), ylim=(-.02, 1.02), title='Paired joint structural evidence')
    axis.legend()
    save(figure, ARTIFACT_NAMES[1])

    comparisons = topology_report['topology_null']['comparisons']['core_q95']
    for statistic, name, label in (
        ('repeated_support_2', ARTIFACT_NAMES[2], 'Place pairs with >=2 q95 image supports'),
        ('largest_place_component', ARTIFACT_NAMES[3], 'Largest q95 place component size'),
    ):
        values = permutations['core_q95__' + statistic].to_numpy(int)
        observed = comparisons[statistic]['observed']
        low, high = min(int(values.min()), observed), max(int(values.max()), observed)
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.hist(values, bins=np.arange(low - .5, high + 1.5), color='#6687aa', edgecolor='white', label='stratified bundle permutations')
        axis.axvline(observed, color='#a13a31', linewidth=2, label=f'Observed: {observed}')
        axis.set(xlabel=label, ylabel='Permutation count', title='Fixed candidate graph topology null')
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend()
        save(figure, name)
    return {name: sha256_file(Path(output_dir) / name) for name in ARTIFACT_NAMES}


def export_audit(args, *, repo_root=ROOT):
    inputs = load_frozen_inputs(args.step2b_dir, args.step2b_snapshot, repo_root=repo_root)
    provenance = inputs['provenance']
    joint, joint_tables = load_stage(args.runtime_dir, 'joint_null', provenance, repo_root=repo_root)
    topology, topology_tables = load_stage(args.runtime_dir, 'topology_null', provenance, repo_root=repo_root)
    if set(joint_tables) != {'random_joint_metrics.csv', 'paired_joint_evidence.csv'} or set(topology_tables) != {'topology_permutations.csv'}:
        raise ValueError('Step 2C stage artifacts do not match the supported schema')
    validate_analysis_stages(inputs, joint, topology, topology_tables)
    runtime_snapshot = args.runtime_dir / 'null_topology_metrics.json'
    curated = [args.snapshot.parent / ('step2c_' + name) for name in ARTIFACT_NAMES]
    targets = [args.snapshot, runtime_snapshot, *curated, *(args.output_dir / name for name in ARTIFACT_NAMES)]
    guard_destinations(targets, repo_root)
    if args.snapshot.parent.resolve() != (Path(repo_root) / 'docs/audits').resolve() or not args.snapshot.name.startswith('step2c_') or args.snapshot.suffix != '.json':
        raise ValueError('curated snapshot must be a docs/audits/step2c_*.json file')
    stage_paths = [args.runtime_dir / (name + '_summary.json') for name in ('joint_null', 'topology_null')]
    stage_hashes = {path.name: sha256_file(path) for path in stage_paths}
    figures = create_plots(joint['report'], topology['report'], joint_tables['paired_joint_evidence.csv'],
                           topology_tables['topology_permutations.csv'], args.output_dir, repo_root=repo_root)
    snapshot = {'schema_version': 1,
                'provenance': {**provenance, 'seed': topology['configuration']['seed'],
                               'permutations': topology['configuration']['permutations'],
                               'step2c_stage_summary_sha256': stage_hashes},
                'configuration': {'joint_null': joint['configuration'], 'topology_null': topology['configuration']},
                **joint['report'], **topology['report'],
                'runtime_artifact_sha256': {name: metadata['sha256'] for stage in (joint, topology) for name, metadata in stage['artifacts'].items()},
                'figure_sha256': {'step2c_' + name: digest for name, digest in figures.items()},
                'scientific_scope': 'statistical/topological structural-confusion validation only; no semantic ground truth, training mapping, or VPR performance claim'}
    validate_step2c_snapshot(snapshot)
    # Reject stale or mixed runtime sources before any curated publication.
    recheck_inputs(provenance, repo_root)
    for path in stage_paths:
        if sha256_file(path) != stage_hashes[path.name]:
            raise ValueError('Step 2C stage summary changed during figure export')
    for name, digest in snapshot['runtime_artifact_sha256'].items():
        if sha256_file(args.runtime_dir / name) != digest:
            raise ValueError('Step 2C table changed during figure export')
    guard_destinations(targets, repo_root)
    for source, destination in zip((args.output_dir / name for name in ARTIFACT_NAMES), curated):
        _atomic_bytes(destination, source.read_bytes())
    payload = (json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    _atomic_bytes(runtime_snapshot, payload)
    _atomic_bytes(args.snapshot, payload)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step2b-dir', type=Path, default=Path('cache/countermine_rgb/step2b'))
    parser.add_argument('--step2b-snapshot', type=Path, default=Path('docs/audits/step2b_countermine_graph_metrics.json'))
    parser.add_argument('--runtime-dir', type=Path, default=Path('cache/countermine_rgb/step2c'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/step2c'))
    parser.add_argument('--snapshot', type=Path, default=Path('docs/audits/step2c_null_topology_metrics.json'))
    args = parser.parse_args()
    try:
        snapshot = export_audit(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'error: {error}\n')
    print(f"Exported {snapshot['provenance']['permutations']} permutation audit: {args.snapshot}")


if __name__ == '__main__':
    main()
