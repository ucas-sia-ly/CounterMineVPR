"""CPU-only, original-RGB inspection figures for the Step 2B pilot graph.

The figures read existing scalar measurements. They never compute image
features, correspondences, geometric registration, or optimization quantities.
Whole-frame display thumbnails are explicitly distinguished from original
640x480 source decoding and do not change any measured evidence.
"""

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw, ImageFont


ARTIFACT_NAMES = (
    "top50_structural_bottleneck.jpg",
    "ratio_vs_bottleneck_top20.jpg",
    "repeated_place_confusions.jpg",
    "largest_place_components.jpg",
    "ratio_vs_count_percentile.png",
    "bottleneck_by_rank.png",
    "place_component_sizes.png",
    "keypoints_vs_ratio.png",
)
RANK_BINS = ("rank_1", "rank_2_5", "rank_6_10", "rank_11_20", "rank_21_50")
VISUAL_CAPS = {
    "top_bottleneck_image_edges": 50,
    "comparison_image_edges_per_group": 20,
    "repeated_place_pairs": 20,
    "supporting_image_pairs_per_place_pair": 3,
    "largest_nontrivial_place_components": 5,
    "connecting_image_pairs_per_component": 3,
    "representative_place_nodes_per_component": 12,
}


def _records(rows):
    if hasattr(rows, "to_dict"):
        return rows.to_dict("records")
    return [dict(row) for row in rows]


def _true(value):
    if isinstance(value, str):
        if value.lower() not in ("true", "false"):
            raise ValueError(f"invalid Boolean value: {value}")
        return value.lower() == "true"
    return bool(value)


def _bottleneck_key(row):
    return (-float(row["structural_bottleneck"]),
            -float(row["structural_geomean"]),
            -int(row["num_matches"]), str(row["pair_uid"]))


def select_image_edges(rows, top_n=50, ranking="bottleneck"):
    """Select nonduplicates with frozen, deterministic scalar tie breaks."""
    if top_n < 0:
        raise ValueError("top_n must be nonnegative")
    records = [row for row in _records(rows) if not _true(row["exact_pixel_duplicate"])]
    if ranking == "bottleneck":
        key = _bottleneck_key
    elif ranking == "ratio":
        key = lambda row: (-float(row["local_match_ratio"]),
                           -int(row["num_matches"]), -float(row["max_salad_similarity"]), str(row["pair_uid"]))
    else:
        raise ValueError("ranking must be bottleneck or ratio")
    return sorted(records, key=key)[:top_n]


def select_repeated_place_pairs(rows, top_n=20):
    """Use the requested exclusive priority groups, then frozen tie breaks."""
    if top_n < 0:
        raise ValueError("top_n must be nonnegative")
    flags = ("independent_support_3", "independent_support_2",
             "repeated_support_3", "repeated_support_2")
    records = []
    for row in _records(rows):
        priority = next((i for i, flag in enumerate(flags) if _true(row[flag])), None)
        if priority is not None:
            records.append((priority, row))
    records.sort(key=lambda item: (item[0],
                 -float(item[1]["structural_bottleneck_max"]),
                 -int(item[1]["num_core_q95_image_edges"]),
                 str(item[1]["place_uid_a"]), str(item[1]["place_uid_b"])))
    return [row for _, row in records[:top_n]]


def place_core_components(place_nodes, place_edges):
    """Return all place components, including singletons, in stable order."""
    nodes = sorted(str(row["place_uid"]) for row in _records(place_nodes))
    if len(nodes) != len(set(nodes)):
        raise ValueError("duplicate place graph nodes")
    parent = {node: node for node in nodes}

    def find(node):
        if node not in parent:
            raise ValueError("place edge endpoint missing from place nodes")
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for row in _records(place_edges):
        if int(row["num_core_q95_image_edges"]) > 0:
            a, b = find(str(row["place_uid_a"])), find(str(row["place_uid_b"]))
            if a != b:
                parent[max(a, b)] = min(a, b)
    grouped = {}
    for node in nodes:
        grouped.setdefault(find(node), []).append(node)
    return sorted(grouped.values(), key=lambda group: (-len(group), tuple(group)))


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _save_image(image, path):
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=92, subsampling=0)
        _atomic_bytes(path, buffer.getvalue())
    image.close()


def _font(size=15):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


class OriginalRGBReader:
    """Bounded source decoding with the exact Step 2A encoded/pixel hashes."""

    def __init__(self, manifest_path, dataset_root, step2a_dir):
        with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.manifest = {row["image_id"]: row for row in rows}
        if len(self.manifest) != len(rows):
            raise ValueError("duplicate image_id in manifest")
        self.root = Path(dataset_root).resolve()
        self.fingerprints = json.loads((Path(step2a_dir) / "image_fingerprints.json").read_text())
        self.validated_ids = set()

    def validate_edge_metadata(self, row):
        for endpoint in ("a", "b"):
            image_id = row[f"image_id_{endpoint}"]
            if image_id not in self.manifest:
                raise ValueError(f"image absent from original manifest: {image_id}")
            source = self.manifest[image_id]
            for field in ("place_uid", "city_id", "row_index"):
                key = f"{field}_{endpoint}"
                if key in row and field in source and str(row[key]) != str(source[field]):
                    raise ValueError(f"graph endpoint {field} differs from original manifest: {image_id}")

    def read(self, image_id, display_width=640):
        if image_id not in self.manifest or image_id not in self.fingerprints:
            raise ValueError(f"image lacks frozen source provenance: {image_id}")
        relative = Path(self.manifest[image_id]["relative_path"])
        path = (self.root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(self.root):
            raise ValueError("manifest image must be inside the original dataset root")
        data = path.read_bytes()
        expected = self.fingerprints[image_id]
        if hashlib.sha256(data).hexdigest() != expected["image_file_sha256"]:
            raise ValueError(f"original image bytes changed after measurement: {image_id}")
        with Image.open(io.BytesIO(data)) as source:
            image = source.convert("RGB")
        if image.size != (640, 480):
            image.close()
            raise ValueError("expected original RGB 640x480")
        if hashlib.sha256(image.tobytes()).hexdigest() != expected["rgb_pixel_sha256"]:
            image.close()
            raise ValueError(f"original RGB pixels changed after measurement: {image_id}")
        self.validated_ids.add(image_id)
        if display_width != 640:
            thumbnail = image.resize((display_width, display_width * 3 // 4), Image.Resampling.LANCZOS)
            image.close()
            image = thumbnail
        return image


def _edge_caption(row, detailed=True):
    minimum = min(int(row["num_keypoints_a"]), int(row["num_keypoints_b"]))
    lines = [
        f"{row['place_uid_a']} ({row['city_id_a']}) | {row['place_uid_b']} ({row['city_id_b']})",
        f"geo {float(row['geo_distance_m']):.1f} m | SALAD rank {int(row['best_rgb_rank'])} | similarity {float(row['max_salad_similarity']):.5f}",
        f"min keypoints {minimum} | matches {int(row['num_matches'])} | ratio {float(row['local_match_ratio']):.5f}",
        f"ratio pct {float(row['ratio_null_percentile']):.5f} | count pct {float(row['match_count_null_percentile']):.5f}",
        f"bottleneck {float(row['structural_bottleneck']):.5f} | symmetric coverage {float(row['symmetric_match_coverage']):.5f}",
    ]
    if not detailed:
        return [lines[0], lines[2], lines[4]]
    return lines


def _pair_panel(reader, row, width=320, heading=None, detailed=True, highlight=False):
    reader.validate_edge_metadata(row)
    gap = 12
    lines = _edge_caption(row, detailed=detailed)
    if heading:
        lines.insert(0, heading)
    header = 12 + len(lines) * 23
    panel = Image.new("RGB", (2 * width + gap, width * 3 // 4 + header), "#fff8da" if highlight else "#fafafa")
    draw = ImageDraw.Draw(panel)
    for index, line in enumerate(lines):
        draw.text((8, 6 + index * 23), line, fill="#6f4000" if highlight else "#151515", font=_font(15 if width == 320 else 17))
    for endpoint, x in (("a", 0), ("b", width + gap)):
        with reader.read(row[f"image_id_{endpoint}"], width) as image:
            panel.paste(image, (x, header))
    if highlight:
        draw.rectangle((0, 0, panel.width - 1, panel.height - 1), outline="#bb7200", width=4)
    return panel


def _notice_panel(title, text, width=1328):
    image = Image.new("RGB", (width, 180), "#fafafa")
    draw = ImageDraw.Draw(image)
    draw.text((18, 18), title, fill="#151515", font=_font(24))
    draw.text((18, 64), text, fill="#333333", font=_font(18))
    draw.text((18, 110), "Original real RGB only; existing scalar measurements; no new local matching.", fill="#555555", font=_font(16))
    return image


def _top_sheet(reader, selected, output):
    if not selected:
        return _save_image(_notice_panel("Top structural bottleneck edges", "No nonduplicate measured candidate edges are available."), output)
    columns, gap, title_height = 2, 12, 72
    panel_width, panel_height = 1292, 630  # full native RGB endpoints, six caption lines
    sheet = Image.new("RGB", (columns * (panel_width + gap) + gap,
                     ((len(selected) + 1) // 2) * (panel_height + gap) + gap + title_height), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 8), "Step 2B: top nonduplicate structural bottleneck edges", fill="#151515", font=_font(24))
    draw.text((14, 42), "Original real RGB at native 640x480; analysis evidence only. q95/q99 are pilot slices.", fill="#333333", font=_font(17))
    for index, row in enumerate(selected):
        panel = _pair_panel(reader, row, 640, f"#{index + 1} | pair {row['pair_uid'][:16]}")
        sheet.paste(panel, (gap + (index % columns) * (panel_width + gap),
                           title_height + gap + (index // columns) * (panel_height + gap)))
        panel.close()
    _save_image(sheet, output)


def _comparison_sheet(reader, ratio, bottleneck, output):
    if not ratio and not bottleneck:
        return _save_image(_notice_panel("Ratio vs bottleneck", "No nonduplicate measured candidate edges are available."), output)
    ratio_ids, bottleneck_ids = {r["pair_uid"] for r in ratio}, {r["pair_uid"] for r in bottleneck}
    gap, width, height, title_height = 12, 652, 344, 96
    sheet = Image.new("RGB", (2 * width + 3 * gap, max(len(ratio), len(bottleneck)) * (height + gap) + title_height + gap), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 8), "Raw ratio vs calibrated bottleneck | amber = appears in only one top list", font=_font(20), fill="#151515")
    draw.text((14, 39), "Whole original frames displayed at 320x240; source decoding/hashes use native 640x480.", font=_font(16), fill="#333333")
    draw.text((14, 66), "A. Top raw local_match_ratio", font=_font(18), fill="#151515")
    draw.text((width + 2 * gap, 66), "B. Top structural_bottleneck", font=_font(18), fill="#151515")
    for column, (rows, other_ids) in enumerate(((ratio, bottleneck_ids), (bottleneck, ratio_ids))):
        for index, row in enumerate(rows):
            exclusive = row["pair_uid"] not in other_ids
            heading = f"#{index + 1} | {'ONLY THIS LIST' if exclusive else 'both lists'} | {row['pair_uid'][:12]}"
            panel = _pair_panel(reader, row, heading=heading, detailed=False, highlight=exclusive)
            sheet.paste(panel, (gap + column * (width + gap), title_height + index * (height + gap)))
            panel.close()
    _save_image(sheet, output)


def _support_rows(image_edges, place_a, place_b):
    places = {str(place_a), str(place_b)}
    return sorted([row for row in image_edges if _true(row["core_q95"])
                   and not _true(row["exact_pixel_duplicate"])
                   and {str(row["place_uid_a"]), str(row["place_uid_b"])} == places], key=_bottleneck_key)


def _repeated_sheet(reader, pairs, support, output):
    if not pairs:
        return _save_image(_notice_panel("Repeated place-confusion audit", "No place pairs meet repeated_support_2 in the frozen core_q95 slice."), output)
    gap, pair_height, section_header, width = 12, 630, 128, 1332
    height = 86 + sum(section_header + max(1, len(support[i])) * (pair_height + gap) for i in range(len(pairs)))
    sheet = Image.new("RGB", (width, height), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), "Repeated place confusion: descriptive support across real views", font=_font(24), fill="#151515")
    draw.text((12, 43), "Native 640x480 originals; at most 3 strongest core_q95 image pairs per place pair.", font=_font(15), fill="#333333")
    y = 86
    flags = ("independent_support_3", "independent_support_2", "repeated_support_3", "repeated_support_2")
    for index, row in enumerate(pairs):
        title = next(flag for flag in flags if _true(row[flag]))
        lines = [
            f"#{index + 1} {row['place_uid_a']} | {row['place_uid_b']} | {title}",
            f"image supports: all {int(row['num_supporting_image_edges'])}, core_q95 {int(row['num_core_q95_image_edges'])} | all-support unique views {int(row['num_unique_images_a'])}/{int(row['num_unique_images_b'])}",
            f"minimum geo {float(row['min_geo_distance_m']):.1f} m | bottleneck max {float(row['structural_bottleneck_max']):.5f}",
        ]
        if "num_core_q95_unique_images_a" in row:
            lines.append(f"core-only unique views {int(row['num_core_q95_unique_images_a'])}/{int(row['num_core_q95_unique_images_b'])}; independent_support above uses all-support unique views")
        for offset, line in enumerate(lines):
            draw.text((12, y + 8 + offset * 25), line, font=_font(17), fill="#151515")
        y += section_header
        for edge in support[index]:
            panel = _pair_panel(reader, edge, width=640, heading=f"core_q95 support | pair {edge['pair_uid'][:16]}")
            sheet.paste(panel, (12, y))
            panel.close()
            y += pair_height + gap
    _save_image(sheet, output)


def _component_sheet(reader, sections, representatives, output):
    if not sections:
        return _save_image(_notice_panel("Largest place core_q95 components", "No nontrivial place-level core_q95 connected components are present."), output)
    width, gap, rep_width, pair_height, header = 1332, 12, 240, 630, 118
    heights = []
    for index, section in enumerate(sections):
        representative_rows = (len(representatives[index]) + 3) // 4
        heights.append(header + representative_rows * 224 + 40 + len(section["image_edges"]) * (pair_height + gap))
    sheet = Image.new("RGB", (width, 84 + sum(heights)), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), "Largest nontrivial place core_q95 components", font=_font(24), fill="#151515")
    draw.text((12, 42), "Whole-frame representatives displayed at 240x180; connecting pair originals native 640x480. Capped views and links.", font=_font(16), fill="#333333")
    y = 84
    for index, section in enumerate(sections):
        nodes = section["place_uids"]
        draw.text((12, y + 8), f"Component #{index + 1}: {len(nodes)} places | showing {len(representatives[index])} representative place nodes", font=_font(21), fill="#151515")
        # Wrap all selected place labels so no node name disappears off canvas.
        for line_index in range(0, min(len(nodes), 12), 4):
            draw.text((12, y + 40 + (line_index // 4) * 22), " | ".join(nodes[line_index:line_index + 4]), font=_font(16), fill="#333333")
        y += header
        for number, (place_uid, image_id) in enumerate(representatives[index]):
            x, yy = 12 + (number % 4) * 324, y + (number // 4) * 224
            draw.text((x, yy), place_uid, font=_font(16), fill="#151515")
            with reader.read(image_id, rep_width) as image:
                sheet.paste(image, (x, yy + 26))
        y += ((len(representatives[index]) + 3) // 4) * 224
        draw.text((12, y + 5), "Strongest connecting core_q95 image pairs (up to 3)", font=_font(18), fill="#151515")
        y += 40
        for edge in section["image_edges"]:
            panel = _pair_panel(reader, edge, width=640, heading=f"connecting pair {edge['pair_uid'][:16]}")
            sheet.paste(panel, (12, y))
            panel.close()
            y += pair_height + gap
    _save_image(sheet, output)


def _save_plot(figure, path):
    with io.BytesIO() as buffer:
        figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        _atomic_bytes(path, buffer.getvalue())


def _plots(image_edges, component_sizes, top_ratio, top_bottleneck, output_dir):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7, 6))
    axes.scatter([r["ratio_null_percentile"] for r in image_edges],
                 [r["match_count_null_percentile"] for r in image_edges], s=9, alpha=.35, color="#3b6a9b")
    for quantile, color in ((.95, "#b35c00"), (.99, "#a02945")):
        axes.axvline(quantile, color=color, linestyle="--", linewidth=1, label=f"q{int(100 * quantile)} pilot slice")
        axes.axhline(quantile, color=color, linestyle="--", linewidth=1)
    axes.set(xlabel="Ratio null percentile", ylabel="Match-count null percentile", xlim=(-.01, 1.02), ylim=(-.01, 1.02), title="Multiple dimensions of local structural evidence")
    axes.legend(loc="upper left")
    _save_plot(figure, output_dir / "ratio_vs_count_percentile.png")
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(8, 5))
    grouped = [[float(r["structural_bottleneck"]) for r in image_edges if r["rank_bin"] == rank] for rank in RANK_BINS]
    available = [(index + 1, values) for index, values in enumerate(grouped) if values]
    if available:
        axes.boxplot([values for _, values in available], positions=[position for position, _ in available], showfliers=False)
    else:
        axes.text(.5, .5, "No measured edges", ha="center", transform=axes.transAxes)
    axes.set_xticks(range(1, 6), RANK_BINS, rotation=15)
    axes.set(xlabel="Frozen SALAD rank bin", ylabel="Structural bottleneck", ylim=(-.02, 1.02), title="Local structural evidence by global candidate rank")
    _save_plot(figure, output_dir / "bottleneck_by_rank.png")
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(7, 5))
    from collections import Counter
    counts = Counter(component_sizes)
    if counts:
        axes.bar(sorted(counts), [counts[size] for size in sorted(counts)], color="#3b6a9b", width=.8)
    else:
        axes.text(.5, .5, "No place graph nodes", ha="center", transform=axes.transAxes)
    axes.set(xlabel="Place component size (including singletons)", ylabel="Component count", title="Place-level core_q95 component sizes")
    _save_plot(figure, output_dir / "place_component_sizes.png")
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(8, 5.5))
    axes.scatter([min(int(r["num_keypoints_a"]), int(r["num_keypoints_b"])) for r in image_edges],
                 [r["local_match_ratio"] for r in image_edges], color="#888888", s=10, alpha=.3, label="All measured candidates")
    for rows, marker, color, label in ((top_ratio, "o", "#bd6500", "Top-50 raw ratio"),
                                        (top_bottleneck, "x", "#2c659d", "Top-50 bottleneck")):
        axes.scatter([min(int(r["num_keypoints_a"]), int(r["num_keypoints_b"])) for r in rows],
                     [r["local_match_ratio"] for r in rows], marker=marker, color=color, s=40,
                     facecolors="none" if marker == "o" else color, linewidths=1.2, label=label)
    axes.set(xlabel="Minimum endpoint keypoint count", ylabel="Raw local match ratio", title="Denominator audit: ratio and bottleneck rankings")
    axes.legend()
    _save_plot(figure, output_dir / "keypoints_vs_ratio.png")
    plt.close(figure)


def create_graph_visuals(image_nodes, image_edges, place_nodes, place_edges,
                         analysis_report, manifest_path, dataset_root,
                         step2a_dir, output_dir):
    """Render eight required artifacts and return path-free selection metadata.

    The analysis report is accepted for a consistent orchestration interface;
    selection is computed from the frozen graph tables, without interpreting
    learned behavior or a final mining threshold.
    """
    del analysis_report
    image_nodes, image_edges = _records(image_nodes), _records(image_edges)
    place_nodes, place_edges = _records(place_nodes), _records(place_edges)
    output_dir = Path(output_dir).resolve()
    for protected in (Path(dataset_root).resolve(), Path(step2a_dir).resolve()):
        if output_dir == protected or output_dir.is_relative_to(protected):
            raise ValueError("graph figures must not overwrite dataset or Step 2A evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = OriginalRGBReader(manifest_path, dataset_root, step2a_dir)
    top50 = select_image_edges(image_edges, VISUAL_CAPS["top_bottleneck_image_edges"])
    ratio50 = select_image_edges(image_edges, 50, "ratio")
    ratio20 = ratio50[:VISUAL_CAPS["comparison_image_edges_per_group"]]
    bottleneck20 = top50[:VISUAL_CAPS["comparison_image_edges_per_group"]]
    repeated = select_repeated_place_pairs(place_edges, VISUAL_CAPS["repeated_place_pairs"])
    support = [_support_rows(image_edges, row["place_uid_a"], row["place_uid_b"])[:VISUAL_CAPS["supporting_image_pairs_per_place_pair"]] for row in repeated]
    if any(not rows for rows in support):
        raise ValueError("repeated place-support table disagrees with core_q95 image edges")
    components = place_core_components(place_nodes, place_edges)
    nontrivial = [group for group in components if len(group) > 1][:VISUAL_CAPS["largest_nontrivial_place_components"]]
    core_edges = [row for row in image_edges if _true(row["core_q95"]) and not _true(row["exact_pixel_duplicate"])]
    sections, representatives = [], []
    for group in nontrivial:
        members = set(group)
        links = sorted([row for row in core_edges if row["place_uid_a"] in members and row["place_uid_b"] in members], key=_bottleneck_key)
        shown = links[:VISUAL_CAPS["connecting_image_pairs_per_component"]]
        if not shown:
            raise ValueError("place component disagrees with core_q95 image edges")
        # Choose each place's endpoint from its strongest connecting core edge.
        image_by_place = {}
        for row in links:
            for endpoint in ("a", "b"):
                image_by_place.setdefault(str(row[f"place_uid_{endpoint}"]), str(row[f"image_id_{endpoint}"]))
        chosen_places = sorted(image_by_place)[:VISUAL_CAPS["representative_place_nodes_per_component"]]
        sections.append({"place_uids": group, "image_edges": shown})
        representatives.append([(place, image_by_place[place]) for place in chosen_places])
    _top_sheet(reader, top50, output_dir / ARTIFACT_NAMES[0])
    _comparison_sheet(reader, ratio20, bottleneck20, output_dir / ARTIFACT_NAMES[1])
    _repeated_sheet(reader, repeated, support, output_dir / ARTIFACT_NAMES[2])
    _component_sheet(reader, sections, representatives, output_dir / ARTIFACT_NAMES[3])
    _plots(image_edges, [len(group) for group in components], ratio50, top50, output_dir)
    return {
        "caps": dict(VISUAL_CAPS),
        "real_rgb_only": True,
        "synthetic_images_used": False,
        "no_new_local_matching": True,
        "source_geometry": "Native 640x480 RGB; no EXIF transpose, crop, padding, or source transformation.",
        "display_geometry": "Top-50, repeated-support, and component-link endpoints native 640x480; comparison endpoints whole-frame 320x240; component representatives whole-frame 240x180. Display scaling never enters measurements.",
        "source_validation": {"encoded_and_rgb_pixel_sha256": True, "validated_image_count": len(reader.validated_ids)},
        "artifact_names": list(ARTIFACT_NAMES),
        "top50_bottleneck_pair_uids": [row["pair_uid"] for row in top50],
        "top50_ratio_pair_uids": [row["pair_uid"] for row in ratio50],
        "ratio_top20_pair_uids": [row["pair_uid"] for row in ratio20],
        "bottleneck_top20_pair_uids": [row["pair_uid"] for row in bottleneck20],
        "ratio_only_top20_pair_uids": [row["pair_uid"] for row in ratio20 if row["pair_uid"] not in {r["pair_uid"] for r in bottleneck20}],
        "bottleneck_only_top20_pair_uids": [row["pair_uid"] for row in bottleneck20 if row["pair_uid"] not in {r["pair_uid"] for r in ratio20}],
        "repeated_support_selection_order": "independent_support_3, independent_support_2, repeated_support_3, repeated_support_2; bottleneck max descending, core_q95 support descending, place UID lexical",
        "supporting_image_pair_selection": "Core_q95 nonduplicates; bottleneck descending, geomean descending, matches descending, pair_uid ascending",
        "repeated_place_pairs": [{"place_uid_a": row["place_uid_a"], "place_uid_b": row["place_uid_b"], "supporting_pair_uids": [edge["pair_uid"] for edge in support[index]]} for index, row in enumerate(repeated)],
        "largest_place_components": [{"place_uids": section["place_uids"], "supporting_pair_uids": [row["pair_uid"] for row in section["image_edges"]], "representative_place_uids": [place for place, _ in representatives[index]], "representative_images": [{"place_uid": place, "image_id": image_id} for place, image_id in representatives[index]]} for index, section in enumerate(sections)],
    }


def curate_graph_artifacts(output_dir, audit_dir):
    """Atomically copy each final Step 2B figure without touching Step 2A."""
    output_dir, audit_dir = Path(output_dir), Path(audit_dir)
    missing = [name for name in ARTIFACT_NAMES if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required Step 2B figures: {', '.join(missing)}")
    names = []
    for name in ARTIFACT_NAMES:
        target_name = f"step2b_{name}"
        _atomic_bytes(audit_dir / target_name, (output_dir / name).read_bytes())
        names.append(target_name)
    return names
