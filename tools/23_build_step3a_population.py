#!/usr/bin/env python3
"""Build 500 deterministic paired hard/random diagnostic triplets and native PNGs."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from countermine.probe.step3a_pipeline import build_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/gsv_mini/manifest.csv"))
    parser.add_argument("--candidates", type=Path, default=Path("cache/gsv_mini/safe250/rgb_candidates_raw.csv"))
    parser.add_argument("--mining-summary", type=Path, default=Path("cache/gsv_mini/safe250/rgb_candidates_raw_summary.json"))
    parser.add_argument("--index-summary", type=Path, default=Path("cache/gsv_mini/index_summary.json"))
    parser.add_argument("--step2d2-snapshot", type=Path, default=Path("docs/audits/step2d2_native100_metrics.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/gsv-cities"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/step3a"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_population(args.manifest, args.candidates, args.mining_summary, args.index_summary,
                     args.step2d2_snapshot, args.dataset_root, args.cache_dir, selection_seed=args.seed)


if __name__ == "__main__":
    main()
