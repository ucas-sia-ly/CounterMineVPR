"""Print CPU-only diagnostics for a streamed RGB candidate CSV."""

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.mining.candidate_diagnostics import (  # noqa: E402
    CandidateDiagnostics,
    DIAGNOSTIC_COLUMNS,
    GEO_THRESHOLDS_M,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=Path("cache/gsv_mini/rgb_candidates_raw.csv")
    )
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    identifiers = {
        column: str for column in DIAGNOSTIC_COLUMNS
        if column.endswith(("_image_id", "_place_uid", "_city_id")) or column == "pair_uid"
    }
    with CandidateDiagnostics() as diagnostics:
        with pd.read_csv(
            args.candidates.expanduser(), dtype=identifiers,
            chunksize=args.chunk_size, float_precision="round_trip",
        ) as source:
            for chunk in source:
                diagnostics.add(chunk)
        summary = diagnostics.summary()

    count = summary["total_candidate_pairs"]
    print(f"Queries: {summary['number_of_queries']:,}; directed candidate rows: {count:,}")
    for label, key, names in (
        ("Similarity", "similarity_statistics", ("min", "q05", "q25", "median", "q75", "q95", "max", "mean")),
        ("Geographic distance (m)", "geo_distance_statistics", ("min", "q05", "q25", "median", "q75", "q95", "max")),
    ):
        print(f"{label}: " + ", ".join(f"{name}={summary[key][name]:.6g}" for name in names))
    print(f"Same city: {summary['fraction_same_city']:.2%} of directed candidate rows")
    for threshold in GEO_THRESHOLDS_M:
        below = summary["geo_distance_counts_below_m"][str(threshold)]
        print(f"Distance < {threshold:,} m: {below:,}/{count:,} ({below / count:.2%})")
    rank_counts = summary["rank_distribution"]
    ranks = [int(rank) for rank in rank_counts]
    if ranks == list(range(min(ranks), max(ranks) + 1)) and len(set(rank_counts.values())) == 1:
        rank_report = f"ranks {min(ranks)}..{max(ranks)}: {next(iter(rank_counts.values())):,} directed rows each"
    else:
        rank_report = ", ".join(f"{rank}: {rows:,}" for rank, rows in rank_counts.items())
    print("Rank distribution: " + rank_report)
    print(f"Unique undirected pair_uid values: {summary['unique_pair_uids']:,}")
    print(
        f"Reverse duplicated pairs: {summary['reverse_duplicate_pairs']:,} mutual undirected pairs; "
        f"{summary['reverse_duplicated_candidate_rows']:,}/{count:,} directed rows "
        f"({summary['reverse_duplicate_fraction']:.2%}) belong to those pairs"
    )
    print("Top 20 highest-similarity candidates (ties: query row, negative row):")
    highest = pd.DataFrame(summary["top_candidates"])
    columns = [
        "query_row_index", "negative_row_index",
        "query_place_uid", "negative_place_uid", "rank", "similarity", "geo_distance_m",
    ]
    print(highest.loc[:, columns].to_string(
        index=False, formatters={"similarity": "{:.7f}".format, "geo_distance_m": "{:.1f}".format}
    ))


if __name__ == "__main__":
    main()
