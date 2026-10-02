#!/usr/bin/env python3
"""Optional small-set LightGlue overlays from bank features; no extraction."""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from countermine.mining.full_countermine_graph import load_full_graph
from countermine.mining.full_graph_analysis import analyze_full_graphs, recorded_bank_dir, select_full_top_edges
from countermine.mining.full_graph_visuals import BankRGBReader, _sheet, edge_caption
from countermine.mining.full_structural_io import code_hashes, guard_step2d, read_json, write_json
from countermine.mining.graph_visuals import _atomic_bytes, _font
from countermine.mining.structural_analysis import METRIC_COLUMNS, sha256_file


def select_overlay_edges(edges, place_audit):
    top = select_full_top_edges(edges, 20)
    hubs = place_audit.loc[place_audit.degree_core_q99 > 0].sort_values(
        ["degree_core_q99", "place_uid"], ascending=[False, True], kind="stable").head(5)
    selections = [top]
    core = edges.loc[edges.core_q99]
    for place in hubs.place_uid:
        selections.append(select_full_top_edges(core.loc[(core.place_uid_a == place) | (core.place_uid_b == place)], 1))
    import pandas as pd
    return pd.concat(selections, ignore_index=True).drop_duplicates("pair_uid", keep="first")


def overlay_panel(reader, row, match, maximum=100):
    lines = ["pair " + row["pair_uid"][:20]] + edge_caption(row)
    header = 10 + len(lines) * 22
    panel = Image.new("RGB", (1292, 480 + header), "#fafafa")
    draw = ImageDraw.Draw(panel)
    for number, line in enumerate(lines):
        draw.text((8, 5 + 22 * number), line, font=_font(16), fill="#151515")
    for side, x in (("a", 0), ("b", 652)):
        with reader.read(row["image_id_" + side], 640) as image:
            panel.paste(image, (x, header))
    if match.scores is None:
        raise ValueError("score-ordered visualization requires matcher confidence scores")
    scores = np.asarray(match.scores, dtype=float)
    points_a, points_b = np.asarray(match.points_a), np.asarray(match.points_b)
    if len(points_a) != len(scores) or len(points_b) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("visual correspondence output has invalid shapes/scores")
    selected = np.lexsort((np.arange(len(scores)), -scores))[:maximum]
    for number in selected:
        a, b = points_a[number], points_b[number]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("visual correspondences contain nonfinite coordinates")
        start, end = (float(a[0]), float(a[1]) + header), (float(b[0]) + 652, float(b[1]) + header)
        color = (255, 180, 35)
        draw.line((start, end), fill=color, width=1)
        for x, y in (start, end):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    return panel


def run_overlays(args, *, repo_root=ROOT):
    from countermine.mining.banked_structural_matcher import BankedStructuralMatcher
    root = Path(repo_root)
    runtime, output, dataset = (Path(value) if Path(value).is_absolute() else root / value
                                 for value in (args.runtime_dir, args.output_dir, args.dataset_root))
    image = output / "top_full_match_overlays.jpg"
    curated = root / "docs/audits/step2d_top_full_match_overlays.jpg"
    summary_path = runtime / "full_match_overlay_summary.json"
    guard_step2d([image, curated, summary_path], repo_root=root)
    nodes, edges, places, place_edges, graph = load_full_graph(runtime, repo_root=root)
    _, _, place_audit, _ = analyze_full_graphs(nodes, edges, places, place_edges)
    selected = select_overlay_edges(edges, place_audit)
    measurement = read_json(runtime / "full_measurement_summary.json")
    bank_dir = recorded_bank_dir(measurement, runtime, repo_root=root)
    reader = BankRGBReader(root / "cache/gsv_mini/manifest.csv", dataset, bank_dir)
    scientific_hashes = {name: metadata["sha256"] for name, metadata in graph["artifacts"].items()}
    scientific_hashes.update({name: sha256_file(runtime / name) for name in (
        "full_candidate_structural_metrics.csv", "full_calibrated_candidates.csv", "full_graph_summary.json")})
    matcher = BankedStructuralMatcher(bank_dir, device=args.device, seed=42,
                                      feature_cache_size=args.feature_cache_size, repo_root=root)
    panels = []
    for row in selected.to_dict("records"):
        matched = matcher.match(row["image_id_a"], row["image_id_b"])
        for field in METRIC_COLUMNS:
            actual, expected = matched.metrics[field], row[field]
            valid = actual == expected if field.startswith("num_") or field == "exact_pixel_duplicate" else abs(actual - expected) <= 1e-12
            if not valid:
                raise ValueError(f"bank-only visual replay differs from saved scientific {field}")
        panels.append(overlay_panel(reader, row, matched))
    if any(sha256_file(runtime / name) != digest for name, digest in scientific_hashes.items()):
        raise ValueError("saved scientific inputs changed during optional visualization")
    _sheet("Step 2D: selected bank-only correspondence overlays", "Top20 bottleneck edges and strongest edges from top5 q99 hubs; at most100 score-ordered matches.", panels, image)
    guard_step2d([curated, summary_path], repo_root=root)
    _atomic_bytes(curated, image.read_bytes())
    summary = {"schema_version": 1, "complete": True, "seed": 42, "visualization_only": True,
        "real_rgb_only": True, "synthetic_images_used": False, "aliked_reextraction": False,
        "pair_count": len(selected), "pair_uids": selected.pair_uid.tolist(), "max_matches_displayed_per_pair": 100,
        "feature_bank_summary_sha256": sha256_file(bank_dir / "summary.json"),
        "matcher_provenance": matcher.provenance,
        "code_sha256": code_hashes(("tools/25_visualize_full_structural_matches.py",
                                     "countermine/mining/full_graph_visuals.py",
                                     "countermine/mining/banked_structural_matcher.py"), repo_root=root),
        "scientific_artifact_sha256": scientific_hashes,
        "curated_figure_sha256": {curated.name: sha256_file(curated)}}
    write_json(summary_path, summary, repo_root=root)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path("cache/countermine_rgb/step2d"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/step2d"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/gsv-cities"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-cache-size", type=int, default=128)
    args = parser.parse_args()
    try:
        summary = run_overlays(args)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Visualized {summary['pair_count']} selected bank-only pairs; saved scientific measurements unchanged")


if __name__ == "__main__":
    main()
