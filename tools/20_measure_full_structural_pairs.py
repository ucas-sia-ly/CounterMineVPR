#!/usr/bin/env python3
"""Benchmark or resume deterministic full-scale frozen-LightGlue pair shards."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from countermine.mining.full_structural_io import DEFAULT_DIR
from countermine.mining.local_feature_bank import BANK_DIR
from countermine.mining.full_structural_measurement import run_full_measurement


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--bank-dir", type=Path, default=BANK_DIR)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/gsv-cities")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-cache-size", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=None)
    return parser


def main():
    parser = build_parser()
    try:
        result = run_full_measurement(parser.parse_args())
        print(f"New complete shards: {result['new_complete_shards']}; validated skips: {result['validated_skipped_shards']}; total expected: {result['total_shards']}")
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
