#!/usr/bin/env python3
"""Render correspondence overlays for 20 completed, nonduplicate RGB pairs."""

import argparse
import colorsys
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from countermine.mining.structural_analysis import (
    curate_audit_artifacts, load_completed_metrics, select_top_structural_candidates,
)
from countermine.mining.structural_matcher import (
    StructuralMatcher, decode_original_rgb, deterministic_match_indices, sha256_file,
    source_provenance,
)


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_match_panel(pixels_a, pixels_b, result, caption, max_matches=100):
    """Paste original pixels at native geometry; overlays never transform input."""
    gap, header = 24, 80
    width, height = 640, 480
    panel = Image.new("RGB", (2 * width + gap, height + header), "#f6f6f6")
    panel.paste(Image.fromarray(pixels_a), (0, header))
    panel.paste(Image.fromarray(pixels_b), (width + gap, header))
    draw = ImageDraw.Draw(panel)
    font = _font(16)
    for index, line in enumerate(caption):
        draw.text((8, 6 + 22 * index), line, font=font, fill="#151515")
    selected = deterministic_match_indices(result.scores, len(result.points_a), max_matches)
    for order, match_index in enumerate(selected):
        point_a, point_b = result.points_a[match_index], result.points_b[match_index]
        hue = ((order * 0.6180339887498949) % 1.0)
        color = tuple(int(255 * value) for value in colorsys.hsv_to_rgb(hue, .8, 1.0))
        a = (float(point_a[0]), float(point_a[1]) + header)
        b = (float(point_b[0]) + width + gap, float(point_b[1]) + header)
        draw.line((a, b), fill=color, width=1)
        for x, y in (a, b):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color, outline="#ffffff")
    return panel


def render_overlays(args, matcher_factory=StructuralMatcher):
    candidates, _, _, config = load_completed_metrics(
        args.runtime_dir, args.manifest, args.candidates, args.candidate_summary,
    )
    selected = select_top_structural_candidates(candidates, top_n=20)
    if selected.empty:
        raise ValueError("no completed nonduplicate candidate pairs are available")
    if any(sha256_file(ROOT / path) != digest for path, digest in config["source_sha256"].items()):
        raise ValueError("overlay source implementation differs from completed measurements")
    if source_provenance(ROOT) != config["vendor_provenance"]:
        raise ValueError("overlay vendor source differs from completed measurements")
    import csv
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest = {row["image_id"]: row for row in csv.DictReader(handle)}
    fingerprints = json.loads((args.runtime_dir / "image_fingerprints.json").read_text())
    root = args.dataset_root.resolve()
    source_paths = {}
    for image_id in sorted(set(selected["image_id_a"]) | set(selected["image_id_b"])):
        relative = Path(manifest[image_id]["relative_path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
            raise ValueError("manifest image must be inside the original dataset root")
        if sha256_file(path) != fingerprints[image_id]["image_file_sha256"]:
            raise ValueError(f"original image bytes changed after measurement: {image_id}")
        source_paths[image_id] = path
    matcher = matcher_factory(device=args.device, seed=config["seed"],
                              feature_cache_size=args.feature_cache_size, repo_root=ROOT)
    if matcher.provenance != {**config["matcher_provenance"], "feature_cache_size": args.feature_cache_size}:
        raise ValueError("overlay matcher weights/runtime differ from completed measurements")
    matcher.set_image_fingerprints(fingerprints)
    panels = []
    try:
        for number, row in enumerate(selected.to_dict("records"), 1):
            a, b = row["image_id_a"], row["image_id_b"]
            result = matcher.match(a, source_paths[a], b, source_paths[b])
            if result.metrics["exact_pixel_duplicate"]:
                raise ValueError("an exact duplicate entered the nonduplicate overlay selection")
            for name in ("num_keypoints_a", "num_keypoints_b", "num_matches", "local_match_ratio"):
                if abs(float(result.metrics[name]) - float(row[name])) > 1e-10:
                    raise ValueError(f"overlay inference differs from frozen metrics: {row['pair_uid']} / {name}")
            pixels_a, fingerprint_a = decode_original_rgb(source_paths[a])
            pixels_b, fingerprint_b = decode_original_rgb(source_paths[b])
            if fingerprint_a != fingerprints[a] or fingerprint_b != fingerprints[b]:
                raise ValueError("original image changed while rendering overlay")
            caption = [
                f"#{number} {row['place_uid_a']} ({row['city_id_a']})  |  {row['place_uid_b']} ({row['city_id_b']})",
                f"geo {float(row['geo_distance_m']):.1f}m   SALAD rank {int(row['best_rgb_rank'])}   similarity {float(row['max_salad_similarity']):.6f}",
                f"keypoints {int(row['num_keypoints_a'])}/{int(row['num_keypoints_b'])}   matches {int(row['num_matches'])}   ratio {float(row['local_match_ratio']):.6f}   displayed <=100 by confidence",
            ]
            panels.append(render_match_panel(pixels_a, pixels_b, result, caption))
            print(f"Rendered structural correspondence audit: {number}/{len(selected)}", flush=True)
        margin = 12
        columns = 2 if len(panels) > 1 else 1
        rows = (len(panels) + columns - 1) // columns
        panel_width, panel_height = panels[0].size
        sheet = Image.new("RGB", (columns * (panel_width + margin) + margin,
                                 rows * (panel_height + margin) + margin), "#dddddd")
        for index, panel in enumerate(panels):
            sheet.paste(panel, (margin + (index % columns) * (panel_width + margin),
                               margin + (index // columns) * (panel_height + margin)))
        import io
        import importlib
        buffer = io.BytesIO()
        sheet.save(buffer, format="JPEG", quality=92, subsampling=0)
        measure_cli = importlib.import_module("tools.08_measure_structural_pairs")
        measure_cli.atomic_bytes(args.output, buffer.getvalue())
        sheet.close()
        curate_audit_artifacts(args.output.parent, args.audit_dir, require_overlays=True)
    finally:
        for panel in panels:
            panel.close()
    return args.output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path("cache/countermine_rgb/step2a"))
    parser.add_argument("--manifest", type=Path, default=Path("cache/gsv_mini/manifest.csv"))
    parser.add_argument("--candidates", type=Path, default=Path("cache/gsv_mini/rgb_candidates_raw.csv"))
    parser.add_argument("--candidate-summary", type=Path, default=Path("cache/gsv_mini/rgb_candidates_raw_summary.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/gsv-cities"))
    parser.add_argument("--output", type=Path, default=Path("outputs/step2a/top20_match_overlays.jpg"))
    parser.add_argument("--audit-dir", type=Path, default=Path("docs/audits"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-cache-size", type=int, default=4)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        output = render_overlays(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Saved original-RGB correspondence overlays: {output}")


if __name__ == "__main__":
    main()
