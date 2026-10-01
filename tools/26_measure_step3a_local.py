#!/usr/bin/env python3
"""Measure R8 fidelity and unregistered cross-place ALIKED/LightGlue confusion."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from countermine.probe.step3a_pipeline import measure_local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/step3a"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    measure_local(args.cache_dir, device=args.device)


if __name__ == "__main__":
    main()
