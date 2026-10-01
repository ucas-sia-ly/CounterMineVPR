#!/usr/bin/env python3
"""Generate native full_scene IC-Light probes once per independently seeded image."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from countermine.probe.step3a_pipeline import generate_probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/step3a"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    generate_probes(args.cache_dir, device=args.device)


if __name__ == "__main__":
    main()
