"""Bounded original-RGB contact sheets and CPU full-population figures.

Only selected display thumbnails are decoded. Subsampling affects scatter
display alone: every descriptive statistic and selection uses all edges.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from countermine.mining.full_countermine_graph import CORE_SLICES
from countermine.mining.full_graph_analysis import select_full_top_edges
from countermine.mining.full_structural_io import ROOT, guard_step2d, read_json
from countermine.mining.graph_analysis import RANK_BINS, boolean_values
from countermine.mining.graph_visuals import _atomic_bytes, _font, _save_image, _save_plot
from countermine.mining.structural_analysis import read_metadata_csv, sha256_file


PLOT_NAMES = (
    "full_bottleneck_distribution.png", "full_bottleneck_by_rank.png", "full_salad_vs_bottleneck.png",
    "full_core_fraction_by_rank.png", "full_place_component_sizes.png", "full_place_degree_distribution.png",
    "full_keypoints_vs_ratio.png",
)
MONTAGE_NAMES = ("top100_full_structural_edges.jpg", "top_place_hubs.jpg", "largest_place_components.jpg")
ARTIFACT_NAMES = (*PLOT_NAMES, *MONTAGE_NAMES)
VISUAL_CAPS = {"top_edges": 100, "hubs_per_slice": 5, "links_per_hub": 4,
               "components_per_slice": 3, "links_per_component": 3, "representatives_per_component": 8,
               "scatter_display_edges": 50_000}


class BankRGBReader:
    """Decode one original source image at a time, bound to bank fingerprints."""

    def __init__(self, manifest_path, dataset_root, bank_dir):
        manifest = read_metadata_csv(manifest_path)
        self.manifest = manifest.set_index("image_id").to_dict("index")
        index = read_metadata_csv(Path(bank_dir) / "index.csv")
        required = {"image_id", "image_file_sha256", "rgb_pixel_sha256"}
        if not required.issubset(index) or index.image_id.duplicated().any() or manifest.image_id.duplicated().any():
            raise ValueError("visual sources require unique manifest/bank fingerprint identities")
        if set(index.image_id) != set(manifest.image_id):
            raise ValueError("bank index must cover every frozen manifest image")
        self.fingerprints = index.set_index("image_id").to_dict("index")
        self.root = Path(dataset_root).resolve()
        self.validated_ids = set()

    def read(self, image_id, width=320):
        if image_id not in self.manifest or image_id not in self.fingerprints:
            raise ValueError("visual image is absent from manifest/bank index")
        relative = Path(self.manifest[image_id]["relative_path"])
        path = (self.root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(self.root):
            raise ValueError("visual source escapes the original dataset directory")
        raw = path.read_bytes()
        fingerprint = self.fingerprints[image_id]
        if hashlib.sha256(raw).hexdigest() != fingerprint["image_file_sha256"]:
            raise ValueError("visual original image bytes differ from bank extraction")
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
        if image.size != (640, 480) or hashlib.sha256(image.tobytes()).hexdigest() != fingerprint["rgb_pixel_sha256"]:
            image.close()
            raise ValueError("visual original RGB geometry/pixels differ from bank extraction")
        self.validated_ids.add(image_id)
        if width != 640:
            resized = image.resize((width, width * 3 // 4), Image.Resampling.LANCZOS)
            image.close()
            image = resized
        return image


def edge_caption(row):
    return [f"{row['place_uid_a']} ({row['city_id_a']}) | {row['place_uid_b']} ({row['city_id_b']})",
            f"geo {float(row['geo_distance_m']):.1f} m | SALAD rank {int(row['best_rgb_rank'])} | sim {float(row['max_salad_similarity']):.5f}",
            f"min keypoints {int(row['min_num_keypoints'])} | matches {int(row['num_matches'])} | ratio {float(row['local_match_ratio']):.5f}",
            f"ratio pct {float(row['ratio_null_percentile']):.5f} | count pct {float(row['match_count_null_percentile']):.5f}",
            f"bottleneck {float(row['structural_bottleneck']):.5f} | symmetric coverage {float(row['symmetric_match_coverage']):.5f}"]


def pair_panel(reader, row, *, heading="", width=360):
    lines = ([heading] if heading else []) + edge_caption(row)
    header = 10 + len(lines) * 19
    panel = Image.new("RGB", (2 * width + 12, header + width * 3 // 4), "#fafafa")
    draw = ImageDraw.Draw(panel)
    for number, text in enumerate(lines):
        draw.text((6, 4 + 19 * number), text, font=_font(12), fill="#151515")
    for side, x in (("a", 0), ("b", width + 12)):
        with reader.read(row["image_id_" + side], width) as image:
            panel.paste(image, (x, header))
    return panel


def _sheet(title, subtitle, panels, destination, *, columns=2, metadata=None):
    # Selected panels are already bounded; source images are released immediately.
    width = max((panel.width for panel in panels), default=732)
    height = max((panel.height for panel in panels), default=150)
    gap, header = 12, 80
    count = max(1, len(panels))
    sheet = Image.new("RGB", (columns * (width + gap) + gap, ((count + columns - 1) // columns) * (height + gap) + header), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, font=_font(23), fill="#151515")
    draw.text((12, 44), subtitle, font=_font(15), fill="#333333")
    if not panels:
        draw.text((12, header + 20), "No nontrivial selected evidence in this slice.", font=_font(18), fill="#333333")
    for number, panel in enumerate(panels):
        sheet.paste(panel, (gap + number % columns * (width + gap), header + number // columns * (height + gap)))
        panel.close()
    _save_image(sheet, destination)
    return metadata


def _top_montage(reader, selected, output):
    panels = [pair_panel(reader, row, heading=f"#{number + 1} | {row['pair_uid'][:16]}") for number, row in enumerate(selected.to_dict("records"))]
    _sheet("Step 2D: top 100 full structural edges", "Original RGB whole frames; display 360x270. Diagnostic evidence only.", panels, output)


def select_hub_sections(edges, place_audit):
    sections = []
    for flag in ("core_q95", "core_q99"):
        degree = "degree_" + flag
        hubs = place_audit.loc[place_audit[degree] > 0].sort_values([degree, "place_uid"], ascending=[False, True], kind="stable").head(VISUAL_CAPS["hubs_per_slice"])
        core = edges.loc[boolean_values(edges[flag])]
        for hub in hubs.to_dict("records"):
            place = hub["place_uid"]
            selected = select_full_top_edges(core.loc[(core.place_uid_a == place) | (core.place_uid_b == place)], VISUAL_CAPS["links_per_hub"])
            row = selected.iloc[0]
            side = "a" if row.place_uid_a == place else "b"
            sections.append({"slice": flag, "place_uid": place, "full_degree": int(hub["candidate_degree_full"]),
                "core_degree": int(hub[degree]), "structural_edge_fraction": float(hub["structural_edge_fraction_" + flag]),
                "representative_image_id": row["image_id_" + side], "image_edges": selected.to_dict("records")})
    return sections


def select_component_sections(edges, components):
    sections = []
    for flag in CORE_SLICES:
        core = edges.loc[boolean_values(edges[flag])]
        for component in components[flag][:VISUAL_CAPS["components_per_slice"]]:
            members = set(component["place_uids"])
            links = select_full_top_edges(core.loc[core.place_uid_a.isin(members) & core.place_uid_b.isin(members)], VISUAL_CAPS["links_per_component"])
            representatives = {}
            # Prefer strongest-edge representatives, then lexical places to fill the cap.
            for row in links.to_dict("records"):
                for side in ("a", "b"):
                    representatives.setdefault(row["place_uid_" + side], row["image_id_" + side])
            if len(representatives) < VISUAL_CAPS["representatives_per_component"]:
                for row in core.loc[core.place_uid_a.isin(members) & core.place_uid_b.isin(members)].sort_values("pair_uid", kind="stable").itertuples(index=False):
                    representatives.setdefault(row.place_uid_a, row.image_id_a)
                    representatives.setdefault(row.place_uid_b, row.image_id_b)
                    if len(representatives) >= VISUAL_CAPS["representatives_per_component"]:
                        break
            representatives = list(representatives.items())[:VISUAL_CAPS["representatives_per_component"]]
            sections.append({"slice": flag, "component_id": component["component_id"], "size": component["size"],
                             "edge_count": component["edge_count"], "image_edges": links.to_dict("records"),
                             "representatives": [{"place_uid": place, "image_id": image} for place, image in representatives]})
    return sections


def _section_panel(reader, section, *, hub):
    width, gap, header, rep_height = 1488, 12, 78, 184
    representatives = ([{"place_uid": section["place_uid"], "image_id": section["representative_image_id"]}] if hub else section["representatives"])
    rows = (len(representatives) + 5) // 6
    pair_rows = (len(section["image_edges"]) + 1) // 2
    pair_height = 394
    panel = Image.new("RGB", (width, header + rows * rep_height + pair_rows * (pair_height + gap)), "#fafafa")
    draw = ImageDraw.Draw(panel)
    if hub:
        title = f"{section['slice']} hub {section['place_uid']} | full degree {section['full_degree']} | core degree {section['core_degree']} | core/full {section['structural_edge_fraction']:.4f}"
    else:
        title = f"{section['slice']} {section['component_id']} | {section['size']} places | {section['edge_count']} place edges"
    draw.text((10, 10), title, font=_font(17), fill="#151515")
    draw.text((10, 43), "Original-RGB representatives and strongest links; visual inspection, no automatic classification.", font=_font(14), fill="#333333")
    for number, representative in enumerate(representatives):
        x, y = 12 + number % 6 * 246, header + number // 6 * rep_height
        draw.text((x, y), representative["place_uid"], font=_font(12), fill="#151515")
        with reader.read(representative["image_id"], 200) as image:
            panel.paste(image, (x, y + 24))
    offset = header + rows * rep_height
    for number, edge in enumerate(section["image_edges"]):
        pair = pair_panel(reader, edge, heading=f"connecting pair {edge['pair_uid'][:16]}")
        panel.paste(pair, (12 + number % 2 * 744, offset + number // 2 * (pair_height + gap)))
        pair.close()
    return panel


def _display_subset(edges):
    if len(edges) <= VISUAL_CAPS["scatter_display_edges"]:
        return edges
    # Full graph is pair_uid sorted; equally spaced indices are deterministic.
    indices = np.linspace(0, len(edges) - 1, VISUAL_CAPS["scatter_display_edges"], dtype=np.int64)
    return edges.iloc[indices]


def create_full_plots(edges, place_audit, components, output):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from collections import Counter
    colors = ("#4477aa", "#ee9944", "#aa3377", "#228855")
    def save(figure, name):
        _save_plot(figure, output / name)
        plt.close(figure)
    figure, ax = plt.subplots(figsize=(8, 5))
    ax.hist(edges.structural_bottleneck, bins=np.linspace(0, 1, 101), color="#4477aa")
    for flag, threshold, color in zip(CORE_SLICES, (.95, .975, .99, .995), colors):
        ax.axvline(threshold, color=color, ls="--", label=flag.replace("core_", ""))
    ax.set(xlabel="Structural bottleneck", ylabel="Full candidate edge count", title="Full-population structural evidence")
    ax.legend()
    save(figure, PLOT_NAMES[0])
    figure, ax = plt.subplots(figsize=(8, 5))
    groups = [(number + 1, edges.loc[edges.rank_bin == rank, "structural_bottleneck"].to_numpy()) for number, rank in enumerate(RANK_BINS)]
    available = [(position, values) for position, values in groups if len(values)]
    if available:
        ax.boxplot([values for _, values in available], positions=[position for position, _ in available], showfliers=False)
    ax.set_xticks(range(1, 6), RANK_BINS, rotation=15)
    ax.set(ylabel="Structural bottleneck", ylim=(-.02, 1.02), title="Structural evidence by frozen SALAD rank")
    save(figure, PLOT_NAMES[1])
    display = _display_subset(edges)
    figure, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(display.max_salad_similarity, display.structural_bottleneck, s=3, alpha=.15, rasterized=True)
    ax.set(xlabel="Maximum SALAD similarity", ylabel="Structural bottleneck", title=f"Full candidates ({len(display):,} display edges)")
    save(figure, PLOT_NAMES[2])
    figure, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(5)
    for flag, offset, color in (("core_q95", -.18, colors[0]), ("core_q99", .18, colors[2])):
        fractions = [float(boolean_values(edges.loc[edges.rank_bin == rank, flag]).mean()) if (edges.rank_bin == rank).any() else 0 for rank in RANK_BINS]
        ax.bar(positions + offset, fractions, width=.36, label=flag, color=color)
    ax.set_xticks(positions, RANK_BINS, rotation=15)
    ax.set(ylabel="Nonduplicate core fraction", title="Core fractions by frozen SALAD rank")
    ax.legend()
    save(figure, PLOT_NAMES[3])
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, flag, color in zip(axes.flat, CORE_SLICES, colors):
        count = Counter(row["size"] for row in components[flag])
        ax.bar(list(count), list(count.values()), color=color)
        ax.set(xlabel="Non-singleton place component size", ylabel="Component count", title=flag)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        ax.set_yscale("log") if count else None
    save(figure, PLOT_NAMES[4])
    figure, ax = plt.subplots(figsize=(8, 5))
    for flag, color in zip(CORE_SLICES, colors):
        values = place_audit["degree_" + flag].to_numpy()
        degrees, counts = np.unique(values[values > 0], return_counts=True)
        ax.step(degrees, counts, where="mid", marker="o", markersize=4, label=flag, color=color)
    ax.set(xlabel="Active place structural degree", ylabel="Place count", title="Structural-confusion hub degrees")
    if any((place_audit["degree_" + flag] > 0).any() for flag in CORE_SLICES):
        ax.set(xscale="log", yscale="log")
    ax.legend()
    save(figure, PLOT_NAMES[5])
    figure, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(display.min_num_keypoints, display.local_match_ratio, s=3, alpha=.15, color="#777777", rasterized=True, label="Candidate display subset")
    for ranking, marker, color in (("ratio", "o", colors[1]), ("bottleneck", "x", colors[0])):
        selected = select_full_top_edges(edges, ranking=ranking)
        ax.scatter(selected.min_num_keypoints, selected.local_match_ratio, marker=marker, s=25, color=color, label="Top100 " + ranking)
    ax.set(xlabel="Minimum endpoint keypoint count", ylabel="Raw local match ratio", title="Full-population denominator audit")
    ax.legend()
    save(figure, PLOT_NAMES[6])
    return len(display)


def create_full_graph_visuals(image_nodes, edges, place_nodes, place_edges, report, place_audit,
                              components, manifest_path, dataset_root, bank_dir, output_dir, *, repo_root=ROOT):
    del image_nodes, place_nodes, place_edges, report
    output = Path(output_dir)
    guard_step2d([output / name for name in ARTIFACT_NAMES], repo_root=repo_root)
    output.mkdir(parents=True, exist_ok=True)
    reader = BankRGBReader(manifest_path, dataset_root, bank_dir)
    selected = select_full_top_edges(edges)
    hubs = select_hub_sections(edges, place_audit)
    sections = select_component_sections(edges, components)
    _top_montage(reader, selected, output / MONTAGE_NAMES[0])
    _sheet("Top structural-confusion place hubs: q95 and q99", "Representative source and up to four strongest core links per hub.",
           [_section_panel(reader, section, hub=True) for section in hubs], output / MONTAGE_NAMES[1], columns=1)
    _sheet("Largest structural-confusion place components", "q95/q975/q99/q995; up to three largest components per slice, bounded contact sheets.",
           [_section_panel(reader, section, hub=False) for section in sections], output / MONTAGE_NAMES[2], columns=1)
    display_count = create_full_plots(edges, place_audit, components, output)
    return {"real_rgb_only": True, "synthetic_images_used": False, "no_new_local_matching": True,
            "caps": VISUAL_CAPS, "display_scatter_edge_count": display_count,
            "scientific_statistics_use_all_edges": True, "artifact_names": list(ARTIFACT_NAMES),
            "source_geometry": "original RGB 640x480; no EXIF transpose or measurement transformation",
            "display_geometry": "whole-frame pair thumbnails 360x270; representatives 200x150",
            "source_validation": {"encoded_and_rgb_sha256": True, "validated_image_count": len(reader.validated_ids)},
            "top100_pair_uids": selected.pair_uid.tolist(),
            "hub_audit": [{key: value for key, value in section.items() if key != "image_edges"} | {
                "supporting_pair_uids": [row["pair_uid"] for row in section["image_edges"]]} for section in hubs],
            "component_audit": [{key: value for key, value in section.items() if key != "image_edges"} | {
                "supporting_pair_uids": [row["pair_uid"] for row in section["image_edges"]]} for section in sections]}


def curate_full_graph_artifacts(output_dir, audit_dir, *, repo_root=ROOT):
    output, audit = Path(output_dir), Path(audit_dir)
    guard_step2d([audit / ("step2d_" + name) for name in ARTIFACT_NAMES], repo_root=repo_root)
    if any(not (output / name).is_file() for name in ARTIFACT_NAMES):
        raise ValueError("full graph figures are incomplete")
    for name in ARTIFACT_NAMES:
        _atomic_bytes(audit / ("step2d_" + name), (output / name).read_bytes())
    return {"step2d_" + name: sha256_file(audit / ("step2d_" + name)) for name in ARTIFACT_NAMES}
