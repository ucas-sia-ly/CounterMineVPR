#!/usr/bin/env python3
"""Calibrate all full-scale candidates against the immutable matched-random null."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from countermine.mining.full_structural_calibration import run_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-dir', type=Path, default=ROOT / 'cache/countermine_rgb/step2d')
    args = parser.parse_args()
    try:
        result = run_calibration(args.runtime_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Step 2D calibration failed: {error}\n')
    print(f"Calibrated {result['candidate_count']} full candidates; frozen pilot reproduction passed")


if __name__ == '__main__':
    main()
