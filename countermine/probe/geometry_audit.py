"""Metadata, paired statistics, and figures for the frozen Step 2D0 audit.

No model is constructed here. Paths in CSV files are relative to the invocation
working directory, following the existing audit tools.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("square_crop_512", "full_fov_512")
MANIFEST_COLUMNS = (
    "audit_index", "row_index", "image_id", "policy", "source_path", "original_path",
    "original_width", "original_height", "canonical_width", "canonical_height",
    "scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction",
)
SUMMARY_METRICS = (
    "num_keypoints_source", "num_keypoints_relit", "num_matches", "match_ratio_min",
    *(f"repeatability_min_{epsilon}px" for epsilon in (2, 4, 8, 16)),
    *(f"precision_matches_{epsilon}px" for epsilon in (2, 4, 8, 16)),
    "displacement_median", "displacement_q95", "grid_coverage_4px", "grid_coverage_8px",
)
ORIGINAL_METRIC_COLUMNS = (
    *(f"repeatability_original_{epsilon}px" for epsilon in (2, 4, 8, 16)),
    *(f"precision_original_{epsilon}px" for epsilon in (2, 4, 8, 16)),
    "displacement_original_mean", "displacement_original_median",
    "displacement_original_q75", "displacement_original_q90",
    "displacement_original_q95", "displacement_original_max",
)
FAIR_SUMMARY_METRICS = (
    "num_keypoints_source", "num_keypoints_relit", "num_matches", "match_ratio_min",
    *ORIGINAL_METRIC_COLUMNS, "grid_coverage_4px", "grid_coverage_8px",
)
DELTA_METRICS = {
    "delta_R4_original": "repeatability_original_4px",
    "delta_R8_original": "repeatability_original_8px",
    "delta_displacement_original_q95": "displacement_original_q95",
    "delta_grid_coverage_8": "grid_coverage_8px",
}
LEGACY_DELTA_METRICS = {
    "delta_R4": "repeatability_min_4px",
    "delta_R8": "repeatability_min_8px",
    "delta_displacement_q95": "displacement_q95",
    "delta_grid_coverage_8": "grid_coverage_8px",
}
OUTPUT_PIXEL_COMPARABILITY_REASON = (
    "Not directly comparable across geometry policies because canonical scale differs."
)
ISOTROPIC_SCALE_TOLERANCE = 1e-12
ORIGINAL_THRESHOLD_ROUNDOFF_TOLERANCE = 1e-12
PILOT_GATE = {
    "repeatability_min_8px_min": 0.40,
    "grid_coverage_8px_min": 0.60,
    "displacement_q95_max": 5.0,
    "diagnostic_only": True,
    "filters_data": False,
}


def reference(path):
    return Path(os.path.relpath(path, Path.cwd())).as_posix()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or set(required).difference(columns):
            raise ValueError(f"{Path(path).name}: missing or duplicate CSV columns")
        rows = list(reader)
    if any(None in row or any(row[key] is None for key in required) for row in rows):
        raise ValueError(f"{Path(path).name}: malformed CSV rows")
    return rows


def frozen_identities(snapshot_path):
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise ValueError("Step 2C snapshot must be a JSON object")
    rows = snapshot.get("per_source_compact", [])
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("Step 2C snapshot must contain 20 relighting records")
    identities, seen = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("snapshot compact records must be objects")
        identity = tuple(row.get(key) for key in ("audit_index", "row_index"))
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in identity):
            raise ValueError("snapshot source identities must be nonnegative integers")
        mode = row.get("mode")
        key = (*identity, mode)
        if not isinstance(mode, str) or mode not in ("official_rmbg", "full_scene") or key in seen:
            raise ValueError("snapshot contains duplicate or unsupported source/mode records")
        seen.add(key)
        identities.add(identity)
    if (len(identities) != 10 or len({item[0] for item in identities}) != 10
            or len({item[1] for item in identities}) != 10):
        raise ValueError("snapshot must contain exactly 10 distinct source identities")
    return identities


def load_manifest(path, snapshot_path):
    identities = frozen_identities(snapshot_path)
    rows = read_csv(path, MANIFEST_COLUMNS)
    seen, source_paths = set(), set()
    for row in rows:
        for key in ("audit_index", "row_index", "original_width", "original_height",
                    "canonical_width", "canonical_height"):
            row[key] = int(row[key])
        identity = (row["audit_index"], row["row_index"])
        key = (*identity, row["policy"])
        if identity not in identities or row["policy"] not in POLICIES or key in seen:
            raise ValueError("geometry manifest has duplicate or non-frozen identities/policies")
        seen.add(key)
        for name in ("source_path", "original_path"):
            if not row[name] or Path(row[name]).is_absolute():
                raise ValueError(f"{name} must be a relative reference")
            row[name] = Path(row[name]).resolve()
        if row["source_path"] in source_paths or row["source_path"] == row["original_path"]:
            raise ValueError("canonical source paths must be distinct from originals and each other")
        source_paths.add(row["source_path"])
        width, height = row["canonical_width"], row["canonical_height"]
        if any(value < 256 or value % 64 for value in (width, height)):
            raise ValueError("canonical dimensions must be >=256 and divisible by 64")
        if row["policy"] == "square_crop_512" and (width, height) != (512, 512):
            raise ValueError("square_crop_512 must remain 512x512")
        if row["policy"] == "full_fov_512" and max(width, height) != 512:
            raise ValueError("full_fov_512 must have longest side 512")
        for name in ("scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction"):
            row[name] = float(row[name])
            if not math.isfinite(row[name]) or row[name] <= 0:
                raise ValueError(f"invalid geometry metadata: {name}")
        ow, oh = row["original_width"], row["original_height"]
        if min(ow, oh) <= 0 or not row["image_id"]:
            raise ValueError("original dimensions and image_id are required")
        if row["policy"] == "full_fov_512" and (
            width * oh != height * ow or row["retained_area_fraction"] != 1.0
            or row["retained_long_axis_fraction"] != 1.0
        ):
            raise ValueError("full-FOV geometry must preserve aspect ratio and complete content")
        short = min(ow, oh)
        expected = ((512 / short, 512 / short, short ** 2 / (ow * oh), short / max(ow, oh))
                    if row["policy"] == "square_crop_512" else (width / ow, height / oh, 1.0, 1.0))
        for name, value in zip(("scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction"), expected):
            if row[name] != value:
                raise ValueError(f"transform metadata disagrees with the canonical policy: {name}")
    if len(rows) != 20 or seen != {(*identity, policy) for identity in identities for policy in POLICIES}:
        raise ValueError("geometry manifest requires exactly two policies for each of 10 sources")
    for identity in identities:
        pair = [row for row in rows if (row["audit_index"], row["row_index"]) == identity]
        for name in ("image_id", "original_path", "original_width", "original_height"):
            if pair[0][name] != pair[1][name]:
                raise ValueError("paired policies must reference the same original image")
    originals = [row for row in rows if row["policy"] == POLICIES[0]]
    if len({row["original_path"] for row in originals}) != 10 or len({row["image_id"] for row in originals}) != 10:
        raise ValueError("frozen identities must reference ten distinct original paths and image IDs")
    return sorted(rows, key=lambda row: (row["audit_index"], POLICIES.index(row["policy"])))


def validate_png(path, size):
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != tuple(size):
            raise ValueError(f"expected actual RGB PNG at {tuple(size)}: {path}")
        image.verify()
    with Image.open(path) as image:
        image.load()


def validate_destinations(destinations, inputs):
    protected = [(REPO_ROOT / name).resolve() for name in
                 ("salad", "third_party", "cache/generator_audit", "docs/audits")]
    destinations = [Path(path).resolve() for path in destinations]
    inputs = [Path(path).resolve() for path in inputs]
    if len(destinations) != len(set(destinations)):
        raise ValueError("audit destinations must be distinct")
    if any(destination.is_relative_to(root) or root.is_relative_to(destination)
           for destination in destinations for root in protected):
        raise ValueError("geometry outputs must not modify protected or frozen Step 2C artifacts")
    if any(path == destination or path.is_relative_to(destination)
           for path in inputs for destination in destinations):
        raise ValueError("geometry outputs must not overwrite their input artifacts")


def publish(artifacts, staging):
    """Publish only managed artifacts, restoring previous outputs on failure."""
    backup = Path(staging) / "previous"
    backup.mkdir()
    saved, published = [], []
    try:
        for index, (source, destination) in enumerate(artifacts):
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            old = backup / str(index)
            if destination.exists() or destination.is_symlink():
                os.replace(destination, old)
                saved.append((old, destination))
            os.replace(source, destination)
            published.append(destination)
    except OSError:
        for destination in reversed(published):
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink(missing_ok=True)
        for old, destination in reversed(saved):
            os.replace(old, destination)
        raise


def write_csv(path, columns, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                          encoding="utf-8")


def validate_isotropic_scale(scale_x, scale_y):
    """Return the canonical/original scale, rejecting anisotropic transforms."""
    values = []
    for value, name in ((scale_x, "scale_x"), (scale_y, "scale_y")):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be finite and positive")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be finite and positive") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        values.append(value)
    if abs(values[0] - values[1]) > ISOTROPIC_SCALE_TOLERANCE:
        raise ValueError("original-pixel fidelity requires isotropic canonical scaling")
    return values[0]


def compute_original_pixel_metrics(
    points_source, points_relit, num_keypoints_source, num_keypoints_relit,
    *, scale_x, scale_y,
):
    """Measure the unchanged correspondences in common original-image pixels.

    Both matched point arrays are divided by the same isotropic canonical scale.
    The center-crop translation cancels in the point difference. No transform is
    registered, and this function does not recompute normalized grid coverage.
    Inclusive thresholds use an absolute 1e-12 original-pixel float64 roundoff
    guard (zero relative tolerance), so equivalent boundary correspondences do
    not acquire different classifications through scale division cancellation.
    """
    scale = validate_isotropic_scale(scale_x, scale_y)
    points = []
    for value, name in ((points_source, "points_source"), (points_relit, "points_relit")):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
            raise ValueError(f"{name} must have shape (N, 2) and finite coordinates")
        points.append(array)
    source, relit = points
    if source.shape != relit.shape:
        raise ValueError("Source and relit matched points must have equal shapes")
    counts = []
    for value, name in ((num_keypoints_source, "num_keypoints_source"),
                        (num_keypoints_relit, "num_keypoints_relit")):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
        counts.append(int(value))
    num_matches = len(source)
    denominator = min(counts)
    if num_matches > denominator:
        raise ValueError("num_matches cannot exceed either keypoint count")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        source_original = source / scale
        relit_original = relit / scale
        displacement = np.linalg.norm(source_original - relit_original, axis=1)
    if not (np.isfinite(source_original).all() and np.isfinite(relit_original).all()
            and np.isfinite(displacement).all()):
        raise ValueError("original-pixel coordinates and displacements must be finite")
    result = {}
    for epsilon in (2, 4, 8, 16):
        good = int(np.count_nonzero(
            (displacement <= epsilon) | np.isclose(
                displacement, epsilon, rtol=0.0, atol=ORIGINAL_THRESHOLD_ROUNDOFF_TOLERANCE,
            )
        ))
        result[f"repeatability_original_{epsilon}px"] = float(good / denominator) if denominator else 0.0
        result[f"precision_original_{epsilon}px"] = float(good / num_matches) if num_matches else 0.0
    result.update({
        "displacement_original_mean": float(displacement.mean()) if num_matches else None,
        "displacement_original_median": float(np.median(displacement)) if num_matches else None,
        **{
            f"displacement_original_q{quantile}": float(np.quantile(displacement, quantile / 100.0))
            if num_matches else None for quantile in (75, 90, 95)
        },
        "displacement_original_max": float(displacement.max()) if num_matches else None,
    })
    return {name: result[name] for name in ORIGINAL_METRIC_COLUMNS}


def pilot_gate_pass(record):
    """Historical output-pixel gate; never a cross-policy acceptance criterion."""
    names = ("repeatability_min_8px", "grid_coverage_8px", "displacement_q95")
    values = [record[name] for name in names]
    if any(value is None for value in values):
        return False
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pilot gate inputs must be finite or undefined")
    r8, coverage, displacement = values
    return bool(r8 >= 0.40 and coverage >= 0.60 and displacement <= 5.0)


def paired_deltas(records, *, metric_map=None):
    """Compute full-FOV minus square deltas, defaulting to original-pixel metrics."""
    metric_map = DELTA_METRICS if metric_map is None else metric_map
    grouped = {}
    for record in records:
        key = (record["audit_index"], record["row_index"])
        policy = record["policy"]
        pair = grouped.setdefault(key, {})
        if policy not in POLICIES or policy in pair:
            raise ValueError("paired analysis requires distinct known policies")
        pair[policy] = record
    if len(grouped) != 10 or any(set(pair) != set(POLICIES) for pair in grouped.values()):
        raise ValueError("paired analysis requires both policies for exactly 10 sources")
    paired = []
    for (audit_index, row_index), pair in sorted(grouped.items()):
        result = {"audit_index": audit_index, "row_index": row_index}
        for delta, metric in metric_map.items():
            square, full = (pair[policy][metric] for policy in POLICIES)
            if any(value is not None and not math.isfinite(value) for value in (square, full)):
                raise ValueError(f"nonfinite paired metric: {metric}")
            result[delta] = None if square is None or full is None else float(full - square)
        paired.append(result)
    return paired


def legacy_paired_deltas(records):
    return paired_deltas(records, metric_map=LEGACY_DELTA_METRICS)


def _summarize_policies(records, metric_names):
    policies = {}
    for policy in POLICIES:
        rows = [row for row in records if row["policy"] == policy]
        metrics = {}
        for metric in metric_names:
            values = [row[metric] for row in rows if row[metric] is not None]
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite summary metric: {metric}")
            metrics[metric] = {
                key: float(np.quantile(values, probability)) if values else None
                for key, probability in (("median", .5), ("q05", .05), ("q25", .25),
                                         ("q75", .75), ("q95", .95))
            }
            metrics[metric].update(valid_count=len(values), missing_count=len(rows) - len(values))
        policies[policy] = {"count_pairs": len(rows), "metrics": metrics}
    return policies


def _summarize_deltas(paired, metric_names):
    deltas = {}
    for metric in metric_names:
        values = [row[metric] for row in paired if row[metric] is not None]
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite paired delta: {metric}")
        deltas[metric] = {
            "median": float(np.median(values)) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "valid_count": len(values), "missing_count": len(paired) - len(values),
        }
    return deltas


def summarize(records, paired):
    """Keep fair comparisons and historical output-pixel diagnostics explicit."""
    legacy_pairs = legacy_paired_deltas(records)
    gates = {}
    for policy in POLICIES:
        rows = [row for row in records if row["policy"] == policy]
        passed = sum(pilot_gate_pass(row) for row in rows)
        gates[policy] = {"pass_count": passed, "total_count": len(rows),
                         "acceptance_fraction": passed / len(rows)}
    return {
        "fair_original_pixel_metrics": _summarize_policies(records, FAIR_SUMMARY_METRICS),
        "paired_original_pixel_deltas": _summarize_deltas(paired, DELTA_METRICS),
        "output_pixel_metrics": {
            "cross_policy_comparable": False, "reason": OUTPUT_PIXEL_COMPARABILITY_REASON,
            "policies": _summarize_policies(records, SUMMARY_METRICS),
        },
        "legacy_output_pixel_deltas": {
            "cross_policy_comparable": False, "reason": OUTPUT_PIXEL_COMPARABILITY_REASON,
            "metrics": _summarize_deltas(legacy_pairs, LEGACY_DELTA_METRICS),
        },
        "legacy_output_pixel_gate": {
            "cross_policy_comparable": False, "reason": OUTPUT_PIXEL_COMPARABILITY_REASON,
            "thresholds": PILOT_GATE.copy(), "policies": gates,
        },
        "coverage": {
            "coordinate_system": "normalized canonical image coordinates",
            "grid_shape": [8, 8], "role": "spatial-distribution diagnostic only",
            "definition": "occupied normalized source-side cells / 64",
            "qualifying_matches": "unchanged canonical-output-pixel tolerances of 4 and 8 pixels",
            "recomputed_in_original_coordinates": False,
        },
    }


def plot_contact_sheet(rows, outputs, destination):
    """Fit display copies into tiles while preserving every image's aspect ratio."""
    width, height, gap, margin, header, caption = 224, 192, 12, 16, 40, 28
    sources = [row for row in rows if row["policy"] == POLICIES[0]]
    lookup = {(row["row_index"], row["policy"]): row for row in rows}
    sheet = Image.new("RGB", (margin * 2 + 5 * width + 4 * gap,
                             header + len(sources) * (height + caption + gap) + margin), "#f7f7f7")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    labels = ("ORIGINAL", "SQUARE SOURCE", "SQUARE RELIT", "FULL-FOV SOURCE", "FULL-FOV RELIT")
    for column, label in enumerate(labels):
        draw.text((margin + column * (width + gap), 12), label, fill="#111111", font=font)
    for index, square in enumerate(sources):
        row_index = square["row_index"]
        full = lookup[(row_index, POLICIES[1])]
        paths = (square["original_path"], square["source_path"], outputs[(row_index, POLICIES[0])],
                 full["source_path"], outputs[(row_index, POLICIES[1])])
        top = header + index * (height + caption + gap)
        for column, path in enumerate(paths):
            left = margin + column * (width + gap)
            with Image.open(path) as image:
                display = ImageOps.contain(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
                sheet.paste(display, (left + (width - display.width) // 2,
                                      top + (height - display.height) // 2))
                draw.text((left, top + height + 3), f"row {row_index} | {image.width}x{image.height}",
                          fill="#111111", font=font)
    sheet.save(destination, format="JPEG", quality=90)
    sheet.close()


def plot_fidelity(records, destination):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14, 4.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    for axis, metric, title in zip(
        axes,
        ("repeatability_original_4px", "repeatability_original_8px", "displacement_original_q95"),
        ("R4 (4 original-image pixels)", "R8 (8 original-image pixels)",
         "Displacement q95 (original-image pixels)"),
    ):
        for policy in POLICIES:
            rows = sorted((row for row in records if row["policy"] == policy),
                          key=lambda row: (row["audit_index"], row["row_index"]))
            values = [row[metric] if row[metric] is not None else np.nan for row in rows]
            axis.plot(range(len(rows)), values, marker="o", label=policy)
        axis.set_xticks(range(len(rows)), [str(row["row_index"]) for row in rows], rotation=60)
        axis.set_title(title)
        axis.set_xlabel("Paired source row_index")
        axis.grid(alpha=.25)
        if metric.startswith("repeatability"):
            axis.set_ylim(0, 1)
            axis.set_ylabel("Repeatability at the original-image pixel tolerance")
        else:
            axis.set_ylabel("original-image pixels")
    axes[0].legend(fontsize=8)
    figure.savefig(destination, dpi=150)
