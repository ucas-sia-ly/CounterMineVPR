#!/usr/bin/env python3
"""Freeze original RGB or diagnostic relit SALAD descriptors without training."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from countermine.probe.step3a_pipeline import freeze_descriptors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/step3a"))
    parser.add_argument("--phase", choices=("rgb", "relit"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    freeze_descriptors(args.cache_dir, phase=args.phase, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
