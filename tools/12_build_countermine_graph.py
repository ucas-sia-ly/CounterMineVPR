#!/usr/bin/env python3
"""Build the CPU-only CounterMine structural confusion pilot graphs."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.mining.structural_analysis import read_metadata_csv, sha256_file  # noqa: E402
from countermine.mining.structural_calibration import load_calibrated_artifacts  # noqa: E402
from countermine.mining.countermine_graph import (  # noqa: E402
    build_image_graph, build_place_graph, validate_graph_source_geometry, write_graph_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", "--step2b-dir", type=Path, default=Path("cache/countermine_rgb/step2b"))
    parser.add_argument("--manifest", type=Path, default=Path("cache/gsv_mini/manifest.csv"))
    args = parser.parse_args()
    try:
        calibrated, calibration = load_calibrated_artifacts(args.runtime_dir, repo_root=REPO_ROOT)
        if len(calibrated) != 5000:
            raise ValueError("the frozen Step 2B pilot requires exactly 5000 candidate edges")
        expected_manifest = calibration.get("provenance", {}).get("step2a_input_hashes", {}).get("manifest_sha256")
        if expected_manifest != sha256_file(args.manifest):
            raise ValueError("graph manifest hash differs from the frozen calibrated manifest")
        manifest = read_metadata_csv(args.manifest)
        source_validation = validate_graph_source_geometry(calibrated, manifest)
        image_nodes, image_edges = build_image_graph(calibrated, manifest)
        place_nodes, place_edges = build_place_graph(image_nodes, image_edges)
        summary = write_graph_artifacts(args.runtime_dir, image_nodes, image_edges, place_nodes, place_edges,
                                        calibration_summary=calibration, repo_root=REPO_ROOT,
                                        source_validation=source_validation)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Image graph: {summary['image_node_count']:,} nodes, {summary['image_edge_count']:,} edges")
    print(f"Place graph: {summary['place_node_count']:,} nodes, {summary['place_edge_count']:,} edges")
    print(f"Frozen core_q95 edges: {int(image_edges['core_q95'].sum()):,}; core_q99: {int(image_edges['core_q99'].sum()):,}")
    print(f"Saved graph metadata: {args.runtime_dir / 'graph_summary.json'}")


if __name__ == "__main__":
    main()
