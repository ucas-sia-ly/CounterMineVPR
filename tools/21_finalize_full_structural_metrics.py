#!/usr/bin/env python3
"""CPU-only atomic merge after all deterministic Step 2D shards pass validation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from countermine.mining.full_structural_io import DEFAULT_DIR
from countermine.mining.full_structural_measurement import finalize_full_metrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_DIR)
    return parser


def main():
    parser = build_parser()
    try:
        summary = finalize_full_metrics(parser.parse_args().runtime_dir)
        print(f"Complete: {summary['candidate_count']} pairs, {summary['completed_shard_count']}/{summary['shard_count']} validated shards; no local models loaded.")
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
