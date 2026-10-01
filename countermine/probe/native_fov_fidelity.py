"""Pure metrics, frozen comparisons and figures for native full-FOV probes."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from countermine.probe.local_fidelity import compute_fidelity_metrics


NATIVE_POLICY = "native_full_fov"
HISTORICAL_POLICIES = ("square_crop_512", "full_fov_512")
NATIVE_METRICS = (
    "num_keypoints_source", "num_keypoints_relit", "num_matches", "match_ratio_min",
    *(f"repeatability_{epsilon}px" for epsilon in (2, 4, 8, 16)),
    "displacement_median", "displacement_q95", "grid_coverage_4px", "grid_coverage_8px",
)
FROZEN_METRICS = {
    **{name: name for name in ("num_keypoints_source", "num_keypoints_relit", "num_matches",
                              "match_ratio_min", "grid_coverage_4px", "grid_coverage_8px")},
    **{f"repeatability_{epsilon}px": f"repeatability_original_{epsilon}px" for epsilon in (2, 4, 8, 16)},
    "displacement_median": "displacement_original_median",
    "displacement_q95": "displacement_original_q95",
}
COMPARISON_METRICS = {
    "square_crop_512": {
        "delta_R4": "repeatability_4px", "delta_R8": "repeatability_8px",
        "delta_D95": "displacement_q95", "delta_coverage8": "grid_coverage_8px",
    },
    "full_fov_512": {
        "delta_R4": "repeatability_4px", "delta_R8": "repeatability_8px",
        "delta_D95": "displacement_q95",
    },
}
PAIRED_COLUMNS = (
    "audit_index", "row_index",
    *(f"{delta}_vs_{policy}" for policy, metrics in COMPARISON_METRICS.items() for delta in metrics),
)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _identity(record):
    values = []
    for name in ("audit_index", "row_index"):
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        values.append(value)
    return tuple(values)


def _validate_metrics(record, label):
    counts = []
    for name in ("num_keypoints_source", "num_keypoints_relit", "num_matches"):
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label}.{name} must be a nonnegative integer")
        counts.append(value)
    denominator = min(counts[:2])
    if counts[2] > denominator:
        raise ValueError(f"{label}: matches cannot exceed keypoints")
    for name in NATIVE_METRICS:
        value = record[name]
        if name.startswith("displacement_"):
            if value is None:
                if counts[2]:
                    raise ValueError(f"{label}.{name} is only undefined with no matches")
                continue
            if not counts[2]:
                raise ValueError(f"{label}.{name} must be undefined with no matches")
        _finite(value, f"{label}.{name}")
        if value < 0:
            raise ValueError(f"{label}.{name} must be nonnegative")
        if name.startswith(("repeatability_", "match_ratio_", "grid_coverage_")) and value > 1:
            raise ValueError(f"{label}.{name} must be in [0,1]")
    ratio = counts[2] / denominator if denominator else 0.0
    if not math.isclose(record["match_ratio_min"], ratio, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}.match_ratio_min disagrees with counts")
    if any(record[f"repeatability_{epsilon}px"] > ratio + 1e-12 for epsilon in (2, 4, 8, 16)):
        raise ValueError(f"{label}: repeatability cannot exceed the match ratio")


def load_historical_metrics(snapshot_path):
    """Read all ten frozen sources and both original-pixel policy records."""
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("per_source"), list):
        raise ValueError("historical Step 2D0 snapshot requires per_source records")
    records, seen_audits, seen_rows, seen_images = {}, set(), set(), set()
    for source in snapshot["per_source"]:
        if not isinstance(source, dict):
            raise ValueError("historical source records must be objects")
        identity = _identity(source)
        image_id = source.get("image_id")
        if (identity in records or identity[0] in seen_audits or identity[1] in seen_rows
                or not isinstance(image_id, str) or not image_id or image_id in seen_images):
            raise ValueError("historical snapshot must identify ten distinct original images")
        policies = source.get("policies")
        if not isinstance(policies, dict) or set(policies) != set(HISTORICAL_POLICIES):
            raise ValueError("historical snapshot must contain both frozen geometry policies")
        converted = {}
        for policy in HISTORICAL_POLICIES:
            metrics = policies[policy]
            if not isinstance(metrics, dict):
                raise ValueError("historical policy metrics must be objects")
            converted[policy] = {name: metrics[old_name] for name, old_name in FROZEN_METRICS.items()}
            _validate_metrics(converted[policy], f"historical.{policy}")
        records[identity] = {"image_id": image_id, "policies": converted}
        seen_audits.add(identity[0])
        seen_rows.add(identity[1])
        seen_images.add(image_id)
    if len(records) != 10:
        raise ValueError("historical snapshot must contain exactly ten sources")
    return records, snapshot


def compute_native_metrics(matches):
    """Use the unchanged output-pixel metrics; native scale is exactly one."""
    if (matches.canonical_width, matches.canonical_height) != (640, 480):
        raise ValueError("native matched coordinates must have canonical dimensions 640x480")
    metrics = compute_fidelity_metrics(
        matches.points_source, matches.points_relit, matches.match_scores,
        matches.num_keypoints_source, matches.num_keypoints_relit,
        canonical_width=640, canonical_height=480,
    )
    return {
        name: metrics[f"repeatability_min_{name.removeprefix('repeatability_')}"]
        if name.startswith("repeatability_") else metrics[name]
        for name in NATIVE_METRICS
    }


def paired_native_comparisons(records, historical):
    lookup = {}
    for record in records:
        identity = _identity(record)
        if identity in lookup or identity not in historical:
            raise ValueError("native comparisons require unique frozen source identities")
        if record["image_id"] != historical[identity]["image_id"]:
            raise ValueError("native and historical image identities must agree")
        _validate_metrics(record, "native")
        lookup[identity] = record
    if len(lookup) != 10 or set(lookup) != set(historical):
        raise ValueError("native comparisons require all ten frozen sources")
    paired = []
    for identity in sorted(lookup):
        native = lookup[identity]
        result = {"audit_index": identity[0], "row_index": identity[1]}
        for policy, metrics in COMPARISON_METRICS.items():
            frozen = historical[identity]["policies"][policy]
            for delta, metric in metrics.items():
                current, previous = native[metric], frozen[metric]
                result[f"{delta}_vs_{policy}"] = None if current is None or previous is None else float(current - previous)
        paired.append(result)
    return paired


def summarize_values(values, *, signs=False):
    finite = [value for value in values if value is not None]
    if not np.isfinite(finite).all():
        raise ValueError("summary values must be finite or genuinely undefined")
    probabilities = (("median", .5), ("q25", .25), ("q75", .75)) if signs else (
        ("median", .5), ("q05", .05), ("q25", .25), ("q75", .75), ("q95", .95),
    )
    summary = {name: float(np.quantile(finite, probability)) if finite else None
               for name, probability in probabilities}
    summary.update(min=min(finite) if finite else None, max=max(finite) if finite else None,
                   valid_count=len(finite), missing_count=len(values) - len(finite))
    if signs:
        summary.update(num_positive=sum(value > 0 for value in finite),
                       num_zero=sum(value == 0 for value in finite),
                       num_negative=sum(value < 0 for value in finite))
    return summary


def summarize_native(records, paired):
    if len(records) != 10 or len(paired) != 10:
        raise ValueError("native summaries require ten source records and ten paired rows")
    return {
        "native_metrics": {name: summarize_values([row[name] for row in records]) for name in NATIVE_METRICS},
        "paired_comparisons": {
            policy: {delta: summarize_values([row[f"{delta}_vs_{policy}"] for row in paired], signs=True)
                     for delta in metrics}
            for policy, metrics in COMPARISON_METRICS.items()
        },
    }


def plot_native_fidelity(records, historical, destination):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    records = sorted(records, key=_identity)
    figure = Figure(figsize=(14, 4.6), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    for axis, metric, title in zip(axes, ("repeatability_4px", "repeatability_8px", "displacement_q95"),
                                   ("R4 (4 original-image pixels)", "R8 (8 original-image pixels)",
                                    "D95 (original-image pixels)")):
        for policy in (*HISTORICAL_POLICIES, NATIVE_POLICY):
            values = [record[metric] if policy == NATIVE_POLICY else historical[_identity(record)]["policies"][policy][metric]
                      for record in records]
            axis.plot(range(len(records)), [np.nan if value is None else value for value in values],
                      marker="o", label=policy)
        axis.set_title(title)
        axis.set_xticks(range(len(records)), [str(record["row_index"]) for record in records], rotation=60)
        axis.set_xlabel("Frozen source row_index")
        axis.set_ylabel("original-image pixels" if metric == "displacement_q95" else "Repeatability")
        axis.grid(alpha=.25)
        if metric.startswith("repeatability_"):
            axis.set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    figure.savefig(destination, dpi=150)


def plot_native_contact_sheet(rows, native_outputs, historical_outputs, destination):
    width, height, gap, margin, header, caption = 224, 192, 12, 16, 40, 28
    sheet = Image.new("RGB", (2 * margin + 5 * width + 4 * gap,
                             header + len(rows) * (height + caption + gap) + margin), "#f7f7f7")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    labels = ("ORIGINAL", "SQUARE-512 RELIT", "FULL-FOV-512 RELIT", "NATIVE-640x480 SOURCE", "NATIVE-640x480 RELIT")
    for column, label in enumerate(labels):
        draw.text((margin + column * (width + gap), 12), label, fill="#111111", font=font)
    for index, row in enumerate(sorted(rows, key=_identity)):
        identity = _identity(row)
        paths = (row["original_path"], historical_outputs[(*identity, HISTORICAL_POLICIES[0])],
                 historical_outputs[(*identity, HISTORICAL_POLICIES[1])], row["source_path"], native_outputs[identity])
        top = header + index * (height + caption + gap)
        for column, path in enumerate(paths):
            left = margin + column * (width + gap)
            with Image.open(path) as image:
                display = ImageOps.contain(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
                sheet.paste(display, (left + (width - display.width) // 2,
                                      top + (height - display.height) // 2))
                draw.text((left, top + height + 3), f"row {row['row_index']} | {image.width}x{image.height}",
                          fill="#111111", font=font)
    sheet.save(destination, format="JPEG", quality=90)
    sheet.close()
