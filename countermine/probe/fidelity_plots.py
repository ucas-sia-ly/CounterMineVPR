"""CPU diagnostic figures for source-to-relight canonical-coordinate fidelity.

Plotting imports are lazy, so importing metric or CLI utilities does not load
matplotlib. Only matched coordinates and confidence scores are needed here;
feature descriptors and model objects never enter the plotting interface.
"""

from pathlib import Path

import numpy as np


RELIGHT_MODES = ("official_rmbg", "full_scene")
MODE_LABELS = {"official_rmbg": "Official RMBG", "full_scene": "Full scene"}
MODE_COLORS = {"official_rmbg": "#1f77b4", "full_scene": "#ff7f0e"}
EPSILONS = (2, 4, 8, 16)
OVERLAY_COLORS = ("#2ca02c", "#ffb000", "#d62728")
CANONICAL_SIZE = 512


def _pyplot():
    """Select the noninteractive CPU backend before importing pyplot."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot

    return pyplot


def _destination(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _matched_arrays(points_source, points_relit, match_scores=None):
    source = np.asarray(points_source, dtype=np.float64)
    relit = np.asarray(points_relit, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or relit.shape != source.shape:
        raise ValueError("Matched source and relit coordinates must both have shape [N, 2].")
    if not np.isfinite(source).all() or not np.isfinite(relit).all():
        raise ValueError("Matched coordinates must be finite.")
    scores = None
    if match_scores is not None:
        scores = np.asarray(match_scores, dtype=np.float64)
        if scores.shape != (len(source),) or not np.isfinite(scores).all():
            raise ValueError("Match scores must be finite and have shape [N].")
    return source, relit, scores


def displacement_categories(displacements):
    """Return overlay categories: 0 for <=4, 1 for (4,8], 2 for (8,16].

    Larger displacements receive -1 and must never be drawn in an overlay.
    The CDF intentionally uses the complete displacement arrays instead.
    """
    values = np.asarray(displacements, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Displacements must be a finite, nonnegative vector.")
    categories = np.full(len(values), -1, dtype=np.int8)
    categories[values <= 16] = 2
    categories[values <= 8] = 1
    categories[values <= 4] = 0
    return categories


def select_visualization_matches(points_source, points_relit, match_scores,
                                 max_matches=80):
    """Deterministically cap eligible matches without altering metric inputs.

    Only displacements <=16 canonical pixels are eligible. Rank by descending
    confidence, then ascending displacement, source/relit coordinates, and
    original index. Explicit tie resolution makes repeat runs reproducible.
    """
    if isinstance(max_matches, bool) or not isinstance(max_matches, (int, np.integer)):
        raise ValueError("max_matches must be a nonnegative integer.")
    if max_matches < 0:
        raise ValueError("max_matches must be a nonnegative integer.")
    source, relit, scores = _matched_arrays(points_source, points_relit, match_scores)
    displacement = np.linalg.norm(source - relit, axis=1)
    eligible = np.flatnonzero(displacement <= 16)
    if len(eligible) == 0 or max_matches == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((
        eligible,
        relit[eligible, 1], relit[eligible, 0],
        source[eligible, 1], source[eligible, 0],
        displacement[eligible], -scores[eligible],
    ))
    return eligible[order[:max_matches]]


def plot_repeatability(records, destination):
    """Compare per-source R2/R4/R8/R16 distributions for both relit modes."""
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(9, 5))
    try:
        for mode_index, mode in enumerate(RELIGHT_MODES):
            rows = [record for record in records if record.get("mode") == mode]
            positions = np.arange(len(EPSILONS)) + (-0.17 if mode_index == 0 else 0.17)
            values = [np.asarray([float(row[f"repeatability_min_{epsilon}px"])
                                  for row in rows], dtype=np.float64)
                      for epsilon in EPSILONS]
            if any(not np.isfinite(value).all() for value in values):
                raise ValueError("Repeatability values must be finite.")
            if rows:
                boxes = axis.boxplot(values, positions=positions, widths=0.28,
                                     patch_artist=True, manage_ticks=False,
                                     medianprops={"color": "black"},
                                     flierprops={"marker": ".", "markersize": 4})
                for box in boxes["boxes"]:
                    box.set_facecolor(MODE_COLORS[mode])
                    box.set_alpha(0.55)
                # Every point is one source image, including repeated values.
                for position, samples in zip(positions, values):
                    axis.scatter(np.full(len(samples), position), samples,
                                 s=15, color=MODE_COLORS[mode], zorder=3)
            axis.plot([], [], color=MODE_COLORS[mode], linewidth=6,
                      label=f"{MODE_LABELS[mode]} (n={len(rows)} sources)")
        axis.set_xticks(np.arange(len(EPSILONS)))
        axis.set_xticklabels([f"R{epsilon} ({epsilon} px)" for epsilon in EPSILONS])
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel("Good matches / min(source keypoints, relit keypoints)")
        axis.set_title("Source-to-relight repeatability in canonical 512 px coordinates")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(_destination(destination), dpi=160)
    finally:
        pyplot.close(figure)


def plot_displacement_cdf(displacements_by_mode, destination):
    """Plot the exact per-match ECDF, retaining all large-displacement failures."""
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(9, 5))
    maximum = 0.0
    try:
        for mode in RELIGHT_MODES:
            values = np.asarray(displacements_by_mode.get(mode, []), dtype=np.float64)
            if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("CDF displacements must be a finite, nonnegative vector.")
            values = np.sort(values)
            label = f"{MODE_LABELS[mode]} (n={len(values)} matches)"
            if len(values):
                maximum = max(maximum, float(values[-1]))
                probabilities = np.arange(1, len(values) + 1, dtype=np.float64) / len(values)
                axis.step(np.concatenate(([0.0], values)),
                          np.concatenate(([0.0], probabilities)), where="post",
                          color=MODE_COLORS[mode], label=label)
            else:
                axis.plot([], [], color=MODE_COLORS[mode], label=label)
        axis.set_xlim(0, max(1.0, maximum * 1.02))
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Matched displacement (canonical pixels)")
        axis.set_ylabel("Fraction of all matches at or below displacement")
        axis.set_title("Source-to-relight displacement CDF (all matches; no trimming)")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(_destination(destination), dpi=160)
    finally:
        pyplot.close(figure)


def _font(size):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _canonical_rgb(path):
    from PIL import Image

    with Image.open(path) as image:
        if image.size != (CANONICAL_SIZE, CANONICAL_SIZE):
            raise ValueError(f"Worst-example image must be exactly 512x512: {path}")
        return image.convert("RGB")


def plot_worst_examples(examples_by_mode, destination):
    """Render caller-selected worst cases as SOURCE | RELIT | MATCH OVERLAY.

    Each example has keys record, source_path, relit_path, points_source,
    points_relit, and match_scores. The caller selects the five lowest R8 rows
    per mode, resolving ties by audit_index and row_index, during streaming.
    This function consumes only those bounded matched-coordinate arrays.
    """
    from PIL import Image, ImageDraw

    rows = [(mode, example) for mode in RELIGHT_MODES
            for example in examples_by_mode.get(mode, [])]
    if any(len(examples_by_mode.get(mode, [])) > 5 for mode in RELIGHT_MODES):
        raise ValueError("Worst-example plots accept at most five examples per relit mode.")
    pane_size, gap, row_label_height = CANONICAL_SIZE, 12, 38
    header_height = 78
    row_height = pane_size + row_label_height + gap
    width = pane_size * 3 + gap * 4
    sheet = Image.new("RGB", (width, header_height + max(1, len(rows)) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font, label_font = _font(22), _font(17)
    for column, title in enumerate(("SOURCE", "RELIT", "MATCH OVERLAY")):
        draw.text((gap + column * (pane_size + gap), 10), title,
                  fill="black", font=title_font)
    legend_x = gap
    for color, title in zip(OVERLAY_COLORS, ("<=4 px", "(4,8] px", "(8,16] px")):
        draw.line((legend_x, 55, legend_x + 25, 55), fill=color, width=4)
        draw.text((legend_x + 31, 43), title, fill="black", font=label_font)
        legend_x += 180
    draw.text((legend_x, 43), "Overlay: source backdrop, <=16 px only, cap 80",
              fill="black", font=label_font)
    if not rows:
        draw.text((gap, header_height + 15), "No relighting examples available.",
                  fill="black", font=title_font)
    for row_index, (mode, example) in enumerate(rows):
        record = example["record"]
        source, relit, scores = _matched_arrays(example["points_source"],
                                                example["points_relit"],
                                                example["match_scores"])
        selected = select_visualization_matches(source, relit, scores)
        categories = displacement_categories(np.linalg.norm(source - relit, axis=1))
        source_image = _canonical_rgb(example["source_path"])
        relit_image = _canonical_rgb(example["relit_path"])
        overlay = source_image.convert("L").convert("RGB")
        overlay_draw = ImageDraw.Draw(overlay)
        for match_index in selected:
            x0, y0 = source[match_index]
            x1, y1 = relit[match_index]
            color = OVERLAY_COLORS[int(categories[match_index])]
            overlay_draw.line((float(x0), float(y0), float(x1), float(y1)),
                              fill=color, width=2)
            overlay_draw.ellipse((float(x0 - 2), float(y0 - 2),
                                  float(x0 + 2), float(y0 + 2)), fill=color)
            # Relit endpoint is a hollow square; source endpoint is a solid dot.
            overlay_draw.rectangle((float(x1 - 2), float(y1 - 2),
                                    float(x1 + 2), float(y1 + 2)),
                                   outline=color, width=1)
        top = header_height + row_index * row_height
        label = (f"{MODE_LABELS[mode]} | audit {record['audit_index']} | "
                 f"row {record['row_index']} | R8={float(record['repeatability_min_8px']):.4f} "
                 f"| shown {len(selected)}/{len(source)} matches")
        draw.text((gap, top + 6), label, fill="black", font=label_font)
        for column, panel in enumerate((source_image, relit_image, overlay)):
            sheet.paste(panel, (gap + column * (pane_size + gap), top + row_label_height))
            panel.close()
    sheet.save(_destination(destination), quality=92)
    sheet.close()
