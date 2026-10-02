#!/usr/bin/env python3
"""Analyze a completed real-RGB Step 2A pilot on CPU and export audit evidence."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.mining.structural_analysis import export_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path("cache/countermine_rgb/step2a"))
    parser.add_argument("--manifest", type=Path, default=Path("cache/gsv_mini/manifest.csv"))
    parser.add_argument("--candidates", type=Path, default=Path("cache/gsv_mini/rgb_candidates_raw.csv"))
    parser.add_argument("--candidate-summary", type=Path, default=Path("cache/gsv_mini/rgb_candidates_raw_summary.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/gsv-cities"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/step2a"))
    parser.add_argument("--snapshot", type=Path, default=Path("docs/audits/step2a_rgb_structural_metrics.json"))
    args = parser.parse_args()
    try:
        snapshot = export_audit(
            args.runtime_dir, args.manifest, args.candidates, args.candidate_summary,
            args.dataset_root, args.output_dir, args.snapshot, repo_root=REPO_ROOT,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    paired = snapshot["paired_candidate_vs_random"]
    print(f"Completed candidates: {snapshot['candidate_summary']['count']:,}; matched random controls: {snapshot['random_summary']['count']:,}")
    print(f"Paired mean delta: {paired['delta_local_match_ratio']['mean']:.6g}; fraction candidate > random: {paired['fraction_candidate_gt_random']:.6g}")
    print(f"Saved CPU audit snapshot: {args.snapshot}")


if __name__ == "__main__":
    main()
