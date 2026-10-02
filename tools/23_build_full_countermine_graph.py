#!/usr/bin/env python3
"""Build full real-RGB CounterMine image/place graphs on CPU."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from countermine.mining.full_countermine_graph import run_full_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path("cache/countermine_rgb/step2d"))
    args = parser.parse_args()
    try:
        summary = run_full_graph(args.runtime_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    for name in ("image_edges.csv", "place_edges.csv"):
        print(f"{name}: {summary['artifacts'][name]['count']:,} undirected edges")


if __name__ == "__main__":
    main()
