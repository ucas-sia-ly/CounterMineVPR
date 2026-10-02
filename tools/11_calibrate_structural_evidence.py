#!/usr/bin/env python3
"""Calibrate the completed 5000-pair real-RGB Step 2A pilot on CPU."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.mining.structural_calibration import run_calibration  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path("cache/countermine_rgb/step2a"))
    parser.add_argument("--manifest", type=Path, default=Path("cache/gsv_mini/manifest.csv"))
    parser.add_argument("--candidates", type=Path, default=Path("cache/gsv_mini/rgb_candidates_raw.csv"))
    parser.add_argument("--candidate-summary", type=Path, default=Path("cache/gsv_mini/rgb_candidates_raw_summary.json"))
    parser.add_argument("--snapshot", type=Path, default=Path("docs/audits/step2a_rgb_structural_metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/countermine_rgb/step2b"))
    args = parser.parse_args()
    try:
        summary = run_calibration(
            args.runtime_dir, args.manifest, args.candidates, args.candidate_summary,
            args.snapshot, args.output_dir, repo_root=REPO_ROOT,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    counts = summary["calibration"]["relation_null_counts"]
    print(f"Calibrated {summary['candidate_count']:,} real-RGB candidates from {summary['random_count']:,} existing matched controls on CPU.")
    print(f"Relation nulls: same_city={counts['same_city']:,}, cross_city={counts['cross_city']:,}; minimum=100.")
    print(f"Saved calibration artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
