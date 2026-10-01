#!/usr/bin/env python3
"""Analyze completed Step3A observations, make figures, export portable metrics."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from countermine.probe.step3a_pipeline import analyze_and_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/step3a"))
    args = parser.parse_args()
    analyze_and_export(args.cache_dir)


if __name__ == "__main__":
    main()
