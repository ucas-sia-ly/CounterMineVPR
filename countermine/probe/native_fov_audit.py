"""Frozen identities and lossless source provenance for the native FOV audit."""

import json
import math
from pathlib import Path, PureWindowsPath

from PIL import Image

from countermine.probe.geometry_audit import (
    MANIFEST_COLUMNS as GEOMETRY_COLUMNS, REPO_ROOT, read_csv, sha256,
    validate_destinations, validate_png,
)


POLICY = "native_full_fov"
NATIVE_SIZE = (640, 480)
SOURCE_COUNT = 10
MANIFEST_COLUMNS = (*GEOMETRY_COLUMNS, "original_format", "source_format",
                    "original_sha256", "source_sha256")


def integer(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a nonnegative integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.isdecimal():
        result = int(value)
    else:
        raise ValueError(f"{label} must be a nonnegative integer")
    if result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def relative_path(value, label, *, allow_parent=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty relative reference")
    path, windows = Path(value), PureWindowsPath(value)
    if path.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"{label} must be a relative reference")
    if not allow_parent and (".." in path.parts or ".." in windows.parts):
        raise ValueError(f"{label} must not escape the dataset")
    return path


def frozen_sources(snapshot_path):
    """Return the exact ten original identities in the immutable Step 2D0 snapshot."""
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    records = snapshot.get("per_source") if isinstance(snapshot, dict) else None
    if not isinstance(records, list) or len(records) != SOURCE_COUNT:
        raise ValueError("the historical Step 2D0 snapshot must contain exactly ten sources")
    rows, audits, image_ids = set(), set(), set()
    sources = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("historical source records must be objects")
        audit = integer(record.get("audit_index"), "audit_index")
        row = integer(record.get("row_index"), "row_index")
        image_id = record.get("image_id")
        relative_path(image_id, "image_id")
        if audit in audits or row in rows or image_id in image_ids:
            raise ValueError("historical snapshot contains duplicate original identities")
        if not isinstance(record.get("policies"), dict) or set(record["policies"]) != {"square_crop_512", "full_fov_512"}:
            raise ValueError("historical sources require both completed geometry policies")
        rows.add(row)
        audits.add(audit)
        image_ids.add(image_id)
        sources.append({"audit_index": audit, "row_index": row, "image_id": image_id})
    return sorted(sources, key=lambda row: (row["audit_index"], row["row_index"]))


def validate_native_destinations(destinations, inputs, *, extra_protected=()):
    """Keep native writes away from original data and historical audit artifacts."""
    validate_destinations(destinations, inputs)
    protected = [REPO_ROOT / "cache/geometry_audit", REPO_ROOT / "outputs/step2",
                 REPO_ROOT / "outputs/step2d0", REPO_ROOT / "data", *extra_protected]
    for destination in destinations:
        resolved = Path(destination).expanduser().resolve()
        if any(resolved.is_relative_to(Path(root).resolve()) or Path(root).resolve().is_relative_to(resolved)
               for root in protected):
            raise ValueError("native audit outputs must not overwrite data or historical artifacts")


def validate_native_source(row):
    """Verify the native PNG is exactly the original JPEG's decoded RGB pixels."""
    if sha256(row["original_path"]) != row["original_sha256"] or sha256(row["source_path"]) != row["source_sha256"]:
        raise ValueError("native original or PNG source changed since the build")
    validate_png(row["source_path"], NATIVE_SIZE)
    with Image.open(row["original_path"]) as original, Image.open(row["source_path"]) as source:
        if original.format != "JPEG" or original.size != NATIVE_SIZE:
            raise ValueError("the frozen native experiment requires actual 640x480 JPEG originals")
        rgb = original.convert("RGB")
        try:
            if rgb.tobytes() != source.tobytes():
                raise ValueError("native PNG must retain every decoded original RGB pixel")
        finally:
            rgb.close()


def load_native_manifest(path, snapshot_path):
    frozen = {row["row_index"]: row for row in frozen_sources(snapshot_path)}
    rows = read_csv(path, MANIFEST_COLUMNS)
    seen, originals, sources = set(), set(), set()
    for row in rows:
        for name in ("audit_index", "row_index", "original_width", "original_height",
                     "canonical_width", "canonical_height"):
            row[name] = integer(row[name], name)
        identity = frozen.get(row["row_index"])
        if identity is None or any(row[name] != identity[name] for name in ("audit_index", "image_id")):
            raise ValueError("native manifest identities disagree with the historical snapshot")
        if row["row_index"] in seen or row["policy"] != POLICY:
            raise ValueError("native manifest must contain one native_full_fov record per source")
        seen.add(row["row_index"])
        if ((row["original_width"], row["original_height"]) != NATIVE_SIZE
                or (row["canonical_width"], row["canonical_height"]) != NATIVE_SIZE
                or row["original_format"] != "JPEG" or row["source_format"] != "PNG"):
            raise ValueError("native geometry must remain 640x480 JPEG-to-PNG")
        for name in ("scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction"):
            row[name] = float(row[name])
            if not math.isfinite(row[name]) or row[name] != 1.0:
                raise ValueError(f"native {name} must equal 1.0")
        for name in ("source_path", "original_path"):
            row[name] = relative_path(row[name], name, allow_parent=True).resolve()
        if row["source_path"] == row["original_path"] or row["source_path"] in sources or row["original_path"] in originals:
            raise ValueError("native source and original paths must be distinct for every identity")
        sources.add(row["source_path"])
        originals.add(row["original_path"])
    if len(rows) != SOURCE_COUNT or seen != set(frozen) or sources.intersection(originals):
        raise ValueError("native manifest requires exactly ten distinct frozen originals and PNGs")
    for row in rows:
        validate_native_source(row)
    return sorted(rows, key=lambda row: (row["audit_index"], row["row_index"]))
