#!/usr/bin/env python3
"""Validate joint ratio/count evidence against existing matched random controls on CPU."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from countermine.mining.joint_null import run_joint_null


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step2b-dir', type=Path, default=Path('cache/countermine_rgb/step2b'))
    parser.add_argument('--step2b-snapshot', type=Path, default=Path('docs/audits/step2b_countermine_graph_metrics.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('cache/countermine_rgb/step2c'))
    args = parser.parse_args()
    try:
        summary = run_joint_null(args.step2b_dir, args.step2b_snapshot, args.output_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'error: {error}\n')
    rates = summary['report']['joint_null']['rates']['all']
    print(f"Candidates: {rates['candidate']['count']}; matched controls: {rates['random']['count']}")
    for name in ('q95', 'q99'):
        print(f"{name}: candidate={rates['candidate'][name+'_fraction']:.6f}; random={rates['random'][name+'_fraction']:.6f}")
    print(f'Saved joint-null analysis: {args.output_dir}')


if __name__ == '__main__':
    main()
