"""CPU-only metrics and figures for the frozen 100-source native audit.

All sources have equal weight. No registration or geometric fitting is used.
The inherited diagnostic gate is frozen before this audit and never filters
records. Undefined zero-match displacements remain null, with missing counts.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from countermine.probe.local_fidelity import compute_fidelity_metrics


SOURCE_COUNT = 100
NATIVE_SIZE = (640, 480)
FIDELITY_METRICS = (
    "num_keypoints_source", "num_keypoints_relit", "num_matches", "match_ratio_min",
    *(f"repeatability_{epsilon}px" for epsilon in (2, 4, 8, 16)),
    *(f"precision_{epsilon}px" for epsilon in (2, 4, 8, 16)),
    "displacement_mean", "displacement_median", "displacement_q75",
    "displacement_q90", "displacement_q95", "displacement_max",
    "grid_coverage_4px", "grid_coverage_8px",
)
PIXEL_METRICS = (
    "rgb_mae_normalized", "luma_source_mean", "luma_relit_mean",
    "luma_mean_delta", "luma_mae_normalized",
)
GATE_COLUMNS = ("pass_R8", "pass_coverage8", "pass_D95", "inherited_pilot_gate_pass")
FROZEN_GATE = {
    "repeatability_8px_min": 0.40,
    "grid_coverage_8px_min": 0.60,
    "displacement_q95_max": 5.0,
    "diagnostic_only": True,
    "filters_data": False,
}
SUMMARY_METRICS = {
    "num_matches": "num_matches", "match_ratio_min": "match_ratio_min",
    **{f"R{epsilon}": f"repeatability_{epsilon}px" for epsilon in (2, 4, 8, 16)},
    "displacement_median": "displacement_median", "displacement_q95": "displacement_q95",
    "coverage4": "grid_coverage_4px", "coverage8": "grid_coverage_8px",
}
QUANTILES = (
    ("min", 0), ("q01", .01), ("q05", .05), ("q10", .10), ("q25", .25),
    ("median", .50), ("q75", .75), ("q90", .90), ("q95", .95), ("q99", .99), ("max", 1),
)
LUMA_COEFFICIENTS = (.2126, .7152, .0722)
LUMA_DEFINITION = (
    "Y'=0.2126 R+0.7152 G+0.0722 B on stored 8-bit RGB values, without gamma "
    "linearization; luma means/delta in [0,255] units, luma MAE divided by 255"
)
COMPACT_METRICS = {
    "num_matches": "num_matches", "match_ratio_min": "match_ratio_min",
    "R4": "repeatability_4px", "R8": "repeatability_8px", "R16": "repeatability_16px",
    "displacement_median": "displacement_median", "displacement_q95": "displacement_q95",
    "coverage4": "grid_coverage_4px", "coverage8": "grid_coverage_8px",
    "rgb_mae_normalized": "rgb_mae_normalized", "luma_mean_delta": "luma_mean_delta",
    "luma_mae_normalized": "luma_mae_normalized",
}
PROVENANCE_EXCLUDED = {
    "audit_index", "row_index", "image_id", "policy", "mode", "source_path", "original_path",
    "output_path", "relative_path", "source_512_path", "original_width", "original_height",
    "canonical_width", "canonical_height", "scale_x", "scale_y", "retained_area_fraction",
    "retained_long_axis_fraction", "original_format", "source_format", "original_sha256",
    "source_sha256", "output_sha256", "crop_left", "crop_top", "crop_size",
    "crop_long_axis_fraction", "crop_area_fraction", *FIDELITY_METRICS, *PIXEL_METRICS, *GATE_COLUMNS,
}


def source_identity(record):
    identity = tuple(record[name] for name in ("audit_index", "row_index"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in identity):
        raise ValueError("source identities must be nonnegative integers")
    return identity


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def compute_native100_metrics(matches):
    """Canonical and original-image pixel displacements coincide at scale 1."""
    if (matches.canonical_width, matches.canonical_height) != NATIVE_SIZE:
        raise ValueError("native matched coordinates must be exactly 640x480")
    measured = compute_fidelity_metrics(
        matches.points_source, matches.points_relit, matches.match_scores,
        matches.num_keypoints_source, matches.num_keypoints_relit,
        canonical_width=640, canonical_height=480,
    )
    result = {}
    for name in FIDELITY_METRICS:
        original = (name.replace("repeatability_", "repeatability_min_")
                    if name.startswith("repeatability_") else
                    name.replace("precision_", "precision_matches_") if name.startswith("precision_") else name)
        result[name] = measured[original]
    return result


def _native_rgb_pixels(path):
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != NATIVE_SIZE:
            raise ValueError(f"source and relit must be exactly 640x480 RGB PNG: {path}")
        return np.array(image, dtype=np.float64, copy=True)


def compute_intervention_strength(source_path, relit_path):
    """Pixel diagnostics only; no score, gate, or nuisance-equalization claim."""
    source, relit = _native_rgb_pixels(source_path), _native_rgb_pixels(relit_path)
    coefficients = np.asarray(LUMA_COEFFICIENTS, dtype=np.float64)
    luma_source = np.sum(source * coefficients, axis=2)
    luma_relit = np.sum(relit * coefficients, axis=2)
    source_mean, relit_mean = float(luma_source.mean()), float(luma_relit.mean())
    return {
        "rgb_mae_normalized": float(np.abs(relit - source).mean() / 255.0),
        "luma_source_mean": source_mean, "luma_relit_mean": relit_mean,
        "luma_mean_delta": relit_mean - source_mean,
        "luma_mae_normalized": float(np.abs(luma_relit - luma_source).mean() / 255.0),
    }


def inherited_gate(metrics):
    r8 = _finite(metrics["repeatability_8px"], "repeatability_8px")
    coverage = _finite(metrics["grid_coverage_8px"], "grid_coverage_8px")
    d95 = metrics["displacement_q95"]
    if d95 is not None:
        d95 = _finite(d95, "displacement_q95")
    result = {
        "pass_R8": r8 >= FROZEN_GATE["repeatability_8px_min"],
        "pass_coverage8": coverage >= FROZEN_GATE["grid_coverage_8px_min"],
        "pass_D95": d95 is not None and d95 <= FROZEN_GATE["displacement_q95_max"],
    }
    result["inherited_pilot_gate_pass"] = all(result.values())
    return result


def wilson_interval(pass_count, total):
    """Two-sided 95% Wilson score interval, without SciPy or continuity correction."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (pass_count, total)):
        raise ValueError("Wilson counts must be integers")
    if total < 1 or not 0 <= pass_count <= total:
        raise ValueError("Wilson interval requires 0 <= pass_count <= positive total")
    z = 1.959963984540054
    proportion = pass_count / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return {"lower": 0.0 if pass_count == 0 else max(0.0, center - radius),
            "upper": 1.0 if pass_count == total else min(1.0, center + radius),
            "confidence_level": .95, "z": z}


def validate_fidelity_record(record):
    source_identity(record)
    if not isinstance(record.get("image_id"), str) or not record["image_id"]:
        raise ValueError("image_id is required")
    counts = [record[name] for name in FIDELITY_METRICS[:3]]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("keypoint and match counts must be nonnegative integers")
    denominator = min(counts[:2])
    if counts[2] > denominator:
        raise ValueError("num_matches cannot exceed either keypoint count")
    for name in FIDELITY_METRICS:
        value = record[name]
        if name.startswith("displacement_") and value is None:
            if counts[2]:
                raise ValueError("displacements may be null only for zero-match pairs")
            continue
        value = _finite(value, name)
        if value < 0 or (name.startswith(("repeatability_", "precision_", "match_ratio_", "grid_coverage_")) and value > 1):
            raise ValueError(f"{name} is outside its valid range")
        if name.startswith("displacement_") and not counts[2]:
            raise ValueError("zero-match displacement must remain null")
    ratio = counts[2] / denominator if denominator else 0.0
    if not math.isclose(record["match_ratio_min"], ratio, rel_tol=0, abs_tol=1e-12):
        raise ValueError("match ratio disagrees with keypoint/match counts")
    previous = -1.0
    for epsilon in (2, 4, 8, 16):
        repeatability = record[f"repeatability_{epsilon}px"]
        precision = record[f"precision_{epsilon}px"]
        if repeatability > ratio + 1e-12 or repeatability < previous:
            raise ValueError("repeatability must be monotone and cannot exceed match ratio")
        expected = repeatability * denominator / counts[2] if counts[2] else 0.0
        if not math.isclose(precision, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("precision disagrees with repeatability and counts")
        previous = repeatability
    for name in PIXEL_METRICS:
        value = _finite(record[name], name)
        bound = 1.0 if name.endswith("normalized") else 255.0
        lower = -255.0 if name == "luma_mean_delta" else 0.0
        if value < lower or value > bound:
            raise ValueError(f"{name} is outside its valid range")
    if not math.isclose(record["luma_mean_delta"], record["luma_relit_mean"] - record["luma_source_mean"], rel_tol=0, abs_tol=1e-10):
        raise ValueError("luma delta disagrees with image means")
    expected_gate = inherited_gate(record)
    if any(not isinstance(record[name], bool) or record[name] != expected_gate[name] for name in GATE_COLUMNS):
        raise ValueError("saved inherited gate flags disagree with frozen thresholds")


def validate_population_records(records):
    if len(records) != SOURCE_COUNT:
        raise ValueError("fidelity dataset must contain exactly 100 records; never filter failures")
    for record in records:
        validate_fidelity_record(record)
    for name in ("audit_index", "row_index", "image_id"):
        if len({record[name] for record in records}) != SOURCE_COUNT:
            raise ValueError(f"fidelity dataset requires 100 distinct {name} identities")


def summarize_distribution(values):
    valid = [_finite(value, "distribution value") for value in values if value is not None]
    result = {name: float(np.quantile(valid, probability)) if valid else None for name, probability in QUANTILES}
    result.update(total_count=len(values), valid_count=len(valid), missing_count=len(values) - len(valid))
    return result


def compact_record(record):
    compact = {name: record[name] for name in ("audit_index", "row_index", "image_id")}
    compact.update({name: record[source] for name, source in COMPACT_METRICS.items()})
    compact.update({name: record[name] for name in GATE_COLUMNS})
    compact.update({name: value for name, value in record.items() if name not in PROVENANCE_EXCLUDED
                    and not name.endswith(("_path", "_sha256"))})
    return compact


def bottom_tail_tables(records):
    """Raw tables retain overlaps; union uses identity and priority order.

    Undefined D95 from zero-match pairs ranks first in the highest-D95 table.
    It is not assigned an invented finite displacement or silently excluded.
    """
    lowest_r8 = sorted(records, key=lambda row: (row["repeatability_8px"], *source_identity(row)))[:20]
    highest_d95 = sorted(records, key=lambda row: (
        row["displacement_q95"] is not None,
        -row["displacement_q95"] if row["displacement_q95"] is not None else 0,
        *source_identity(row),
    ))[:20]
    lowest_coverage = sorted(records, key=lambda row: (row["grid_coverage_8px"], *source_identity(row)))[:20]
    failures = sorted((row for row in records if not row["inherited_pilot_gate_pass"]), key=source_identity)
    tables = {"lowest_R8": lowest_r8, "highest_D95": highest_d95,
              "lowest_coverage8": lowest_coverage, "gate_failures": failures}
    union, seen = [], set()
    for row in [*failures, *lowest_r8, *highest_d95, *lowest_coverage]:
        identity = source_identity(row)
        if identity not in seen:
            union.append(row)
            seen.add(identity)
    tables["deduplicated_manual_audit_set"] = union
    return {name: [compact_record(row) for row in rows] for name, rows in tables.items()}


def summarize_native100(records):
    validate_population_records(records)
    passes = sum(row["inherited_pilot_gate_pass"] for row in records)
    return {
        "fidelity_summary": {name: summarize_distribution([row[metric] for row in records])
                             for name, metric in SUMMARY_METRICS.items()},
        "gate": {"name": "inherited_pilot_gate_pass", "thresholds": dict(FROZEN_GATE),
                 "pass_count": passes, "fail_count": len(records) - passes,
                 "acceptance_fraction": passes / len(records), "wilson_95": wilson_interval(passes, len(records))},
        "intervention_strength": {name: summarize_distribution([row[name] for row in records]) for name in PIXEL_METRICS},
        "bottom_tail": bottom_tail_tables(records),
    }


def plot_fidelity_distribution(records, destination):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 8), constrained_layout=True)
    FigureCanvasAgg(figure)
    for axis, (name, title, threshold) in zip(figure.subplots(2, 2).flat, (
        ("repeatability_4px", "R4 (4 original-image pixels)", None),
        ("repeatability_8px", "R8 (8 original-image pixels)", .40),
        ("displacement_q95", "D95 (original-image pixels)", 5.0),
        ("grid_coverage_8px", "8px coverage (normalized 8x8 grid)", .60),
    )):
        values = [row[name] for row in records if row[name] is not None]
        axis.hist(values, bins=20, color="#3b729b", edgecolor="white")
        if threshold is not None:
            axis.axvline(threshold, color="#b33333", linestyle="--", label="Inherited pilot threshold")
            axis.legend(fontsize=8)
        axis.set_title(title)
        axis.set_ylabel("Source count (equal weight)")
        axis.set_xlabel("Original-image pixels" if name == "displacement_q95" else "Fraction")
        if name != "displacement_q95":
            axis.set_xlim(0, 1)
        else:
            axis.text(.98, .06, f"Undefined (zero matches): {len(records)-len(values)}", transform=axis.transAxes,
                      ha="right", va="bottom", fontsize=8,
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("100 native 640x480 full-scene probes: fixed shared illumination intervention")
    figure.savefig(destination, dpi=150)


def plot_intervention_strength(records, destination):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 8), constrained_layout=True)
    FigureCanvasAgg(figure)
    for axis, name in zip(figure.subplots(2, 3).flat, PIXEL_METRICS):
        axis.hist([row[name] for row in records], bins=20, color="#4b8879", edgecolor="white")
        axis.set_title(name)
        axis.set_xlabel("Fraction of 255" if name.endswith("normalized") else "Stored RGB luma units")
        axis.set_ylabel("Source count (equal weight)")
        axis.grid(axis="y", alpha=.2)
    figure.axes[-1].axis("off")
    figure.axes[-1].text(.05, .8, "CPU pixel diagnostics only\nNo gate or mining score\n\n" +
                         "Y' = 0.2126 R + 0.7152 G + 0.0722 B\nStored 8-bit RGB; no gamma linearization",
                         va="top", fontsize=10)
    figure.suptitle("Fixed shared illumination intervention: appearance-change diagnostics")
    figure.savefig(destination, dpi=150)


def plot_worst_contact_sheet(records, rows, outputs, destination):
    """Twenty unique sources in failure/R8/D95/coverage priority; no stretching."""
    tails = bottom_tail_tables(records)
    selected = tails["deduplicated_manual_audit_set"][:20]
    lookup = {source_identity(row): row for row in rows}
    width, height, margin, gap, caption, header = 320, 240, 16, 12, 44, 32
    # Two sources per row, each with source and relit columns.
    columns, row_count = 4, (len(selected) + 1) // 2
    sheet = Image.new("RGB", (margin * 2 + columns * width + (columns - 1) * gap,
                              header + row_count * (height + caption + gap) + margin), "#f7f7f7")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for column in range(columns):
        draw.text((margin + column * (width + gap), 9), "SOURCE" if column % 2 == 0 else "RELIT",
                  fill="#111111", font=font)
    for index, record in enumerate(selected):
        identity = source_identity(record)
        top, base_column = header + (index // 2) * (height + caption + gap), (index % 2) * 2
        for offset, path in enumerate((lookup[identity]["source_path"], outputs[identity])):
            left = margin + (base_column + offset) * (width + gap)
            with Image.open(path) as image:
                display = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
                sheet.paste(display, (left + (width - display.width) // 2, top + (height - display.height) // 2))
        left = margin + base_column * (width + gap)
        d95 = "null" if record["displacement_q95"] is None else f"{record['displacement_q95']:.2f}"
        text = (f"audit_index={identity[0]} row_index={identity[1]}  gate={'PASS' if record['inherited_pilot_gate_pass'] else 'FAIL'}\n"
                f"R4={record['R4']:.3f} R8={record['R8']:.3f} D95={d95} coverage8={record['coverage8']:.3f}")
        draw.text((left, top + height + 4), text, fill="#111111", font=font)
    sheet.save(destination, format="JPEG", quality=92)
    sheet.close()
