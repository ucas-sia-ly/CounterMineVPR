#!/usr/bin/env python3
"""Build all eligible canonical frozen SALAD pairs, without sampling."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from countermine.mining.full_structural_population import run_build_full_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'cache/gsv_mini/manifest.csv')
    parser.add_argument('--candidates', type=Path, default=ROOT / 'cache/gsv_mini/rgb_candidates_raw.csv')
    parser.add_argument('--candidate-summary', type=Path, default=ROOT / 'cache/gsv_mini/rgb_candidates_raw_summary.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'cache/countermine_rgb/step2d')
    args = parser.parse_args()
    try:
        result = run_build_full_population(args.manifest, args.candidates, args.candidate_summary, args.output_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Step 2D population failed: {error}\n')
    print(f"Complete canonical population: {result['population']['full_canonical_edge_count']} pairs")
    print(f'Population artifacts: {args.output_dir}')


if __name__ == '__main__':
    main()
