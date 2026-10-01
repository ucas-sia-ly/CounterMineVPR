"""CPU visual audit of paired Step 3A retrieval-margin changes.

The montage is ordered only by the requested hard-negative margin change.
This module defines no mining score, decision threshold, or significance test.
Images are rendered from native 640x480 full scenes without cropping.
"""

from pathlib import Path

import numpy as np


FIGURE_NAMES = (
    "hard_vs_random_margin_shift.png",
    "margin_vs_local_confusion.png",
    "rgb_vs_relit_margin.png",
)
MONTAGE_NAME = "top_margin_collapse_50.jpg"


def _finite(record, key):
    value = record[key]
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a finite number") from error
    if not np.isfinite(result):
        raise ValueError(f"{key} must be a finite number")
    return result


def _passed(record):
    passed = record["all_images_R8_pass"]
    if not isinstance(passed, (bool, np.bool_)):
        raise ValueError("all_images_R8_pass must be a Boolean")
    return bool(passed)


def select_top_margin_collapse(records, top_n=50):
    """Pick the smallest hard margin changes among the common R8-pass subset."""
    if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    selected = []
    for record in records:
        delta = _finite(record, "delta_margin_hard")
        if _passed(record):
            selected.append((
                delta,
                str(record["q_image_id"]),
                str(record["p_image_id"]),
                str(record["n_hard_image_id"]),
                record,
            ))
    selected.sort(key=lambda item: item[:4])
    return [item[-1] for item in selected[:int(top_n)]]


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot

    return pyplot


def _scatter(records, x_key, y_key, destination, *, title, x_label, y_label, identity=False):
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(7, 6))
    try:
        points = np.array([
            [_finite(record, x_key), _finite(record, y_key)] for record in records
        ], dtype=np.float64).reshape((-1, 2))
        passed = np.array([_passed(record) for record in records], dtype=bool)
        for mask, color, label in (
            (~passed, "#aaaaaa", "At least one image fails R8"),
            (passed, "#2266aa", "All four images pass R8"),
        ):
            axis.scatter(points[mask, 0], points[mask, 1], s=22, alpha=0.72,
                         color=color, edgecolors="none", label=f"{label} (n={int(mask.sum())})")
        axis.axhline(0, color="#555555", linewidth=0.9, zorder=0)
        axis.axvline(0, color="#555555", linewidth=0.9, zorder=0)
        if identity:
            limit_min = float(min(0.0, points.min())) if len(points) else -0.1
            limit_max = float(max(0.0, points.max())) if len(points) else 0.1
            padding = max(1e-3, (limit_max - limit_min) * 0.06)
            limits = (limit_min - padding, limit_max + padding)
            axis.plot(limits, limits, color="#777777", linestyle="--", linewidth=1,
                      label="Equal values")
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
        if not len(points):
            axis.text(0.5, 0.5, "No complete triplets", ha="center", va="center",
                      transform=axis.transAxes)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(title)
        axis.grid(alpha=0.18)
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160)
    finally:
        pyplot.close(figure)


def create_step3a_figures(records, output_dir):
    """Render the three requested descriptive scatters, keeping failed probes."""
    records = list(records)
    output_dir = Path(output_dir)
    specifications = (
        (FIGURE_NAMES[0], "delta_margin_random", "delta_margin_hard",
         "Paired hard vs random retrieval-margin shift",
         "Random-negative delta margin", "Hard-negative delta margin", True),
        (FIGURE_NAMES[1], "delta_margin_hard", "delta_cross_match_ratio_hard",
         "Hard-negative margin shift and local confusion",
         "Hard-negative delta margin", "Hard-negative delta cross-match ratio", False),
        (FIGURE_NAMES[2], "margin_hard_rgb", "margin_hard_z",
         "RGB vs relit hard-negative retrieval margin",
         "RGB hard-negative margin", "Relit hard-negative margin", True),
    )
    outputs = {}
    for filename, x_key, y_key, title, x_label, y_label, identity in specifications:
        destination = output_dir / filename
        _scatter(records, x_key, y_key, destination, title=title,
                 x_label=x_label, y_label=y_label, identity=identity)
        outputs[filename] = destination
    return outputs


def _font(size):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _native_panel(path, width):
    from PIL import Image

    with Image.open(path) as image:
        if image.size != (640, 480):
            raise ValueError(f"Step 3A visual audit requires native 640x480 full scenes: {path}")
        # A 4:3 display resize preserves the entire scene and its aspect ratio.
        return image.convert("RGB").resize((width, width * 3 // 4), Image.Resampling.LANCZOS)


def create_top_margin_collapse_montage(records, image_lookup, output_path, *, top_n=50, panel_width=320):
    """Show q/p/hard RGB then q/p/hard relit for the requested worst 50 rows.

    ``image_lookup`` maps image_id to source_path and relit_path. The two paths
    must point to the frozen native probes. Each source is opened individually;
    only the montage canvas and one display panel are retained in memory.
    """
    from PIL import Image, ImageDraw

    if isinstance(panel_width, bool) or not isinstance(panel_width, int) or panel_width < 160 or panel_width % 4:
        raise ValueError("panel_width must be an integer multiple of four, at least 160")
    rows = select_top_margin_collapse(records, top_n=top_n)
    output_path = Path(output_path)
    # Reject source overwrite even when that image is outside the selected rows.
    source_paths = {
        Path(paths[key]).resolve()
        for paths in image_lookup.values() for key in ("source_path", "relit_path")
    }
    if output_path.resolve() in source_paths:
        raise ValueError("Visual audit must not overwrite a source or relit image")
    gap, header_height, line_height = 12, 84, 23
    panel_height = panel_width * 3 // 4
    annotation_height = 5 * line_height + gap
    row_height = panel_height + annotation_height + gap
    width = 6 * panel_width + 7 * gap
    height = header_height + max(1, len(rows)) * row_height
    sheet = Image.new("RGB", (width, height), "white")
    try:
        draw = ImageDraw.Draw(sheet)
        title_font, text_font = _font(21), _font(16)
        draw.text((gap, 6), f"Step 3A: {len(rows)} smallest hard-negative delta margins in the all-images-R8-pass subset",
                  font=title_font, fill="black")
        draw.text((gap, 35), "Native full-FOV 640x480; 4:3 display panels. Shared q/p controls; q/p/hard/random R8 >= 0.40.",
                  font=text_font, fill="black")
        columns = ("q RGB", "p RGB", "n_hard RGB", "q relit", "p relit", "n_hard relit")
        for column, label in enumerate(columns):
            draw.text((gap + column * (panel_width + gap), 59), label,
                      font=text_font, fill="black")
        if not rows:
            draw.text((gap, header_height + gap), "No all-images-R8-pass triplets available.",
                      font=title_font, fill="black")
        for index, row in enumerate(rows):
            top = header_height + index * row_height
            image_ids = [row[f"{role}_image_id"] for role in ("q", "p", "n_hard")]
            panels = [(image_id, key) for key in ("source_path", "relit_path") for image_id in image_ids]
            for column, (image_id, key) in enumerate(panels):
                panel = _native_panel(image_lookup[image_id][key], panel_width)
                try:
                    sheet.paste(panel, (gap + column * (panel_width + gap), top))
                finally:
                    panel.close()
            lines = (
                f"#{index + 1}  " + "   ".join(
                    f"{role}=row {row[f'{role}_row_index']}" if f"{role}_row_index" in row
                    else f"{role}={row[f'{role}_image_id'][:60]}"
                    for role in ("q", "p", "n_hard")
                ) + "   (full image identities in metrics JSON)",
                f"RGB: q-p cosine={_finite(row, 's_qp'):.6f}   q-n cosine={_finite(row, 's_qhard'):.6f}   margin={_finite(row, 'margin_hard_rgb'):+.6f}",
                f"Relit: q-p cosine={_finite(row, 'sz_qp'):.6f}   q-n cosine={_finite(row, 'sz_qhard'):.6f}   margin={_finite(row, 'margin_hard_z'):+.6f}",
                f"Delta margin={_finite(row, 'delta_margin_hard'):+.6f}   delta cross-match ratio={_finite(row, 'delta_cross_match_ratio_hard'):+.6f}",
                f"Source-relit R8: q={_finite(row, 'q_source_relit_R8'):.4f}   p={_finite(row, 'p_source_relit_R8'):.4f}   n_hard={_finite(row, 'n_hard_source_relit_R8'):.4f}",
            )
            for line_index, line in enumerate(lines):
                draw.text((gap, top + panel_height + gap + line_index * line_height), line,
                          fill="black", font=text_font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path, format="JPEG", quality=94, subsampling=0)
    finally:
        sheet.close()
    return output_path


def create_step3a_visual_outputs(records, image_lookup, output_dir):
    """Create all four requested artifacts and return their path mapping."""
    records = list(records)
    output_dir = Path(output_dir)
    outputs = create_step3a_figures(records, output_dir)
    outputs[MONTAGE_NAME] = create_top_margin_collapse_montage(
        records, image_lookup, output_dir / MONTAGE_NAME,
    )
    return outputs
