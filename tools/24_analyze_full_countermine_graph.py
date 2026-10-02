#!/usr/bin/env python3
"""Analyze full real-RGB structural-confusion graphs and publish the audit."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from countermine.mining.full_graph_analysis import export_full_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path("cache/countermine_rgb/step2d"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/gsv-cities"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/step2d"))
    parser.add_argument("--snapshot", type=Path, default=Path("docs/audits/step2d_full_countermine_metrics.json"))
    args = parser.parse_args()
    try:
        result = export_full_audit(args.runtime_dir, output_dir=args.output_dir, snapshot=args.snapshot,
                                   dataset_root=args.dataset_root)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Analyzed {result['population']['canonical_edge_count']:,} original-RGB candidate edges")
    print(f"Saved full structural-confusion audit: {args.snapshot}")


if __name__ == "__main__":
    main()
