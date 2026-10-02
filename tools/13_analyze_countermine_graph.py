#!/usr/bin/env python3
"""Analyze and export the existing Step 2B graph pilot on CPU."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from countermine.mining.countermine_graph import load_graph_artifacts
from countermine.mining.graph_analysis import analyze_graphs, build_graph_snapshot
from countermine.mining.graph_visuals import ARTIFACT_NAMES, create_graph_visuals, curate_graph_artifacts
from countermine.mining.structural_calibration import load_calibrated_artifacts
from countermine.mining.structural_analysis import _atomic_bytes, sha256_file, write_snapshot


def validate_export_inputs(args, calibration):
    """Check source identity and every write target before rendering anything."""
    sources = {key: (ROOT / value).resolve() for key, value in calibration["input_files"].items()}
    provenance = calibration["provenance"]
    if args.step2a_dir.resolve() != sources["runtime_dir"]:
        raise ValueError("visual sources must use the frozen Step 2A runtime directory")
    if sha256_file(args.manifest) != provenance["step2a_input_hashes"]["manifest_sha256"]:
        raise ValueError("visual manifest hash differs from the frozen Step 2A input")
    if sha256_file(args.step2a_dir / "image_fingerprints.json") != provenance["step2a_image_fingerprints_sha256"]:
        raise ValueError("visual image fingerprints differ from frozen Step 2A measurements")
    if not args.snapshot.name.startswith("step2b_") or args.snapshot.suffix != ".json":
        raise ValueError("the curated snapshot must have a step2b_*.json filename")
    protected_files = {path for key, path in sources.items() if key != "runtime_dir"}
    protected_files.update(path.resolve() for path in (ROOT / "docs/audits").glob("step2a_*"))
    protected_dirs = (sources["runtime_dir"], (ROOT / "cache/countermine_rgb/step2a").resolve(),
                      args.dataset_root.resolve(), (ROOT / "data/gsv-cities").resolve(),
                      (ROOT / "third_party").resolve())
    targets = [args.snapshot, args.runtime_dir / "graph_analysis_summary.json",
               args.runtime_dir / "image_node_audit.csv", args.runtime_dir / "place_node_audit.csv",
               args.runtime_dir / "place_core_q95_components.json"]
    targets.extend(args.output_dir / name for name in ARTIFACT_NAMES)
    targets.extend(args.snapshot.parent / f"step2b_{name}" for name in ARTIFACT_NAMES)
    for target in targets:
        resolved = target.resolve()
        if resolved in protected_files or any(resolved.is_relative_to(directory) for directory in protected_dirs):
            raise ValueError("Step 2B exports must not overwrite Step 2A evidence or original RGB inputs")


def export_graph_audit(args):
    _, calibration = load_calibrated_artifacts(args.runtime_dir)
    validate_export_inputs(args, calibration)
    image_nodes, image_edges, place_nodes, place_edges, graph = load_graph_artifacts(args.runtime_dir)
    report, image_audit, place_audit = analyze_graphs(image_nodes, image_edges, place_nodes, place_edges)
    visual_metadata = create_graph_visuals(
        image_nodes, image_edges, place_nodes, place_edges, report,
        args.manifest, args.dataset_root, args.step2a_dir, args.output_dir,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    snapshot = build_graph_snapshot(report, calibration, graph, repo_root=ROOT,
                                    git_commit=commit, visual_metadata=visual_metadata)
    # Publication uses the same validated source chain that produced the figures.
    _, rechecked = load_calibrated_artifacts(args.runtime_dir)
    if rechecked != calibration:
        raise ValueError("calibration changed while graph figures were rendered")
    for filename, digest in graph["output_hashes"].items():
        if sha256_file(args.runtime_dir / filename) != digest:
            raise ValueError("graph artifacts changed while graph figures were rendered")
    validate_export_inputs(args, calibration)
    for name, table in (("image_node_audit", image_audit), ("place_node_audit", place_audit)):
        _atomic_bytes(args.runtime_dir / f"{name}.csv", table.to_csv(index=False, float_format="%.17g").encode("utf-8"))
    _atomic_bytes(args.runtime_dir / "place_core_q95_components.json",
                  (json.dumps(report["place_core_q95_components"], sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    curate_graph_artifacts(args.output_dir, args.snapshot.parent)
    write_snapshot(args.runtime_dir / "graph_analysis_summary.json", snapshot)
    write_snapshot(args.snapshot, snapshot)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path("cache/countermine_rgb/step2b"))
    parser.add_argument("--step2a-dir", type=Path, default=Path("cache/countermine_rgb/step2a"))
    parser.add_argument("--manifest", type=Path, default=Path("cache/gsv_mini/manifest.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/gsv-cities"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/step2b"))
    parser.add_argument("--snapshot", type=Path, default=Path("docs/audits/step2b_countermine_graph_metrics.json"))
    args = parser.parse_args()
    try:
        snapshot = export_graph_audit(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Image graph: {snapshot['image_graph']['full']['node_count']:,} nodes / {snapshot['image_graph']['full']['edge_count']:,} edges")
    print(f"Core q95: {snapshot['image_graph']['core_q95']['edge_count']:,} image edges / {snapshot['place_graph']['core_q95']['edge_count']:,} place edges")
    print(f"Saved CPU graph audit: {args.snapshot}")


if __name__ == "__main__":
    main()
