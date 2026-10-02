#!/usr/bin/env python3
"""Require exact 32-image feature replay and frozen 512-pair Step 2A replay."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from countermine.mining.local_feature_bank import BANK_DIR, validate_feature_bank


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "cache/gsv_mini/manifest.csv")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/gsv-cities")
    parser.add_argument("--bank-dir", type=Path, default=BANK_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-cache-size", type=int, default=128)
    return parser


def main():
    parser = build_parser()
    try:
        result = validate_feature_bank(parser.parse_args())
        print(f"PASSED: exact replay of {result['direct_feature_replay']['image_count']} images and {result['pair_replay']['pair_count']} historical candidate pairs. Full matching is now permitted.")
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
