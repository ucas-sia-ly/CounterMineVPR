#!/usr/bin/env python3
"""Validate frozen candidate topology with stratified evidence-bundle permutations on CPU."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from countermine.mining.topology_null import run_topology_null


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step2b-dir', type=Path, default=Path('cache/countermine_rgb/step2b'))
    parser.add_argument('--step2b-snapshot', type=Path, default=Path('docs/audits/step2b_countermine_graph_metrics.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('cache/countermine_rgb/step2c'))
    parser.add_argument('--permutations', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        summary = run_topology_null(args.step2b_dir, args.step2b_snapshot, args.output_dir,
                                    permutations=args.permutations, seed=args.seed,
                                    progress=lambda done, total: print(f'Permutations: {done}/{total}', flush=True))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'error: {error}\n')
    print(f"Saved {summary['configuration']['permutations']} fixed-graph permutations: {args.output_dir}")


if __name__ == '__main__':
    main()
