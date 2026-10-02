"""Build the deterministic Step 2A real-RGB pilot and matched random controls."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.mining.structural_population import (  # noqa: E402
    build_structural_population,
    load_population_inputs,
    sha256_file,
    write_population_artifacts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/gsv_mini/manifest.csv"))
    parser.add_argument("--candidates", type=Path, default=Path("cache/gsv_mini/rgb_candidates_raw.csv"))
    parser.add_argument("--candidate-summary", type=Path,
                        default=Path("cache/gsv_mini/rgb_candidates_raw_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/countermine_rgb/step2a"))
    parser.add_argument("--max-pairs", type=int, default=5000,
                        help="Unique canonical pairs; default 5000, 0 means all eligible")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_pairs < 0:
        parser.error("--max-pairs must be >=0")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    input_hashes = {"manifest_sha256": sha256_file(args.manifest),
                    "candidate_csv_sha256": sha256_file(args.candidates),
                    "candidate_summary_sha256": sha256_file(args.candidate_summary)}
    manifest, candidates, _ = load_population_inputs(
        args.manifest, args.candidates, args.candidate_summary)
    population, controls, summary = build_structural_population(
        manifest, candidates, max_pairs=args.max_pairs, seed=args.seed)
    summary.update(input_hashes)
    summary = write_population_artifacts(
        population, controls, summary, args.output_dir,
        manifest_path=args.manifest, candidate_path=args.candidates,
        candidate_summary_path=args.candidate_summary)
    print(f"Validated {summary['raw_directed_candidate_rows']} raw directed candidates")
    print(f"Eligible >=250m: {summary['eligible_geo_directed_rows']} directed rows, "
          f"{summary['eligible_unique_canonical_pairs']} canonical pairs")
    print(f"Sampled {summary['sampled_pair_count']} candidates with "
          f"{summary['matched_random_count']} matched random controls")
    print(f"Population artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
