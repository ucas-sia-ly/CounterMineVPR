#!/usr/bin/env python3
"""Build/resume exact float32 ALIKED features once for every frozen RGB image."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from countermine.mining.local_feature_bank import BANK_DIR, build_feature_bank


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "cache/gsv_mini/manifest.csv")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/gsv-cities")
    parser.add_argument("--bank-dir", type=Path, default=BANK_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = build_feature_bank(args)
        print(f"Complete feature bank: {summary['image_count']} original RGB images; run tools/19_validate_aliked_bank.py next.")
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
