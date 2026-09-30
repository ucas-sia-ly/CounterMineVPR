#!/usr/bin/env python3
"""Build paired source geometries for the frozen ten-image Step 2D0 audit.

This CPU-only tool decodes one original image at a time. It never selects new
identities or runs generation, feature extraction, matching, or training.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import sys
import tempfile

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.probe.canonical import canonicalize_probe  # noqa: E402


POLICIES = ("square_crop_512", "full_fov_512")
SOURCE_COUNT = 10
SEED = 42
RELIGHTING_MODES = frozenset(("full_scene", "official_rmbg"))
MANIFEST_COLUMNS = (
    "audit_index", "row_index", "image_id", "policy", "source_path", "original_path",
    "original_width", "original_height", "canonical_width", "canonical_height",
    "scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction",
)
AUDIT_REQUIRED_COLUMNS = ("audit_index", "row_index", "image_id", "relative_path")
MANAGED_PATHS = tuple(f"{policy}/source" for policy in POLICIES) + (
    "geometry_manifest.csv", "geometry_build_summary.json",
)


def _integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.isdecimal():
        result = int(value)
    else:
        raise ValueError(f"{name} must be a nonnegative integer")
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _source_relative(value: str, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty relative path")
    path = Path(value)
    windows = PureWindowsPath(value)
    if path.is_absolute() or windows.is_absolute() or windows.drive or ".." in path.parts:
        raise ValueError(f"{name} must be a relative path without '..'")
    if ".." in windows.parts:
        raise ValueError(f"{name} must be a relative path without '..'")
    return path


def _reference(path: Path) -> str:
    """Keep artifact references portable and relative to the invocation cwd."""
    return Path(os.path.relpath(path, Path.cwd())).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_identities(snapshot_path: str | Path) -> list[dict[str, int]]:
    """Recover exactly ten source identities from both frozen relighting modes."""
    path = Path(snapshot_path)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise ValueError("Step 2C snapshot must be a JSON object")
    records = snapshot.get("per_source_compact")
    if not isinstance(records, list) or len(records) != SOURCE_COUNT * 2:
        raise ValueError("Step 2C per_source_compact must contain exactly 20 records for 10 sources")
    sources, audits = {}, {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Step 2C compact records must be objects")
        audit_index = _integer(record.get("audit_index"), "audit_index")
        row_index = _integer(record.get("row_index"), "row_index")
        mode = record.get("mode")
        if not isinstance(mode, str) or mode not in RELIGHTING_MODES:
            raise ValueError("Step 2C compact records must contain the two relighting modes only")
        if row_index in sources:
            source = sources[row_index]
            if source["audit_index"] != audit_index or mode in source["modes"]:
                raise ValueError("Step 2C has duplicate modes or inconsistent source identities")
            source["modes"].add(mode)
        else:
            if audit_index in audits:
                raise ValueError("Step 2C audit_index is shared by different source identities")
            sources[row_index] = {"audit_index": audit_index, "modes": {mode}}
            audits[audit_index] = row_index
    if len(sources) != SOURCE_COUNT or any(
        source["modes"] != RELIGHTING_MODES for source in sources.values()
    ):
        raise ValueError("Step 2C must contain exactly 10 sources with both relighting modes")
    return sorted(
        ({"audit_index": source["audit_index"], "row_index": row_index}
         for row_index, source in sources.items()),
        key=lambda source: (source["audit_index"], source["row_index"]),
    )


def load_audit_sources(audit_manifest: str | Path, identities: list[dict]) -> list[dict]:
    """Join frozen identities to original-image paths without sampling."""
    by_row = {source["row_index"]: source for source in identities}
    selected = {}
    seen_rows, seen_audits, seen_ids = set(), set(), set()
    with Path(audit_manifest).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError("audit manifest has duplicate column names")
        missing = set(AUDIT_REQUIRED_COLUMNS).difference(columns)
        if missing:
            raise ValueError(f"audit manifest lacks required columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                audit_index = _integer(row["audit_index"], "audit_index")
                row_index = _integer(row["row_index"], "row_index")
                image_id = row["image_id"]
                _source_relative(image_id, "image_id")
                _source_relative(row["relative_path"], "relative_path")
                if row_index in seen_rows or audit_index in seen_audits or image_id in seen_ids:
                    raise ValueError("audit manifest has duplicate row_index, audit_index, or image_id")
                seen_rows.add(row_index)
                seen_audits.add(audit_index)
                seen_ids.add(image_id)
                if row_index in by_row:
                    if audit_index != by_row[row_index]["audit_index"]:
                        raise ValueError("audit_index disagrees with the frozen Step 2C snapshot")
                    selected[row_index] = {
                        "audit_index": audit_index, "row_index": row_index,
                        "image_id": image_id, "relative_path": row["relative_path"],
                    }
            except (ValueError, TypeError) as error:
                raise ValueError(f"audit manifest CSV line {line_number}: {error}") from error
    missing_rows = set(by_row).difference(selected)
    if missing_rows:
        raise ValueError(f"audit manifest is missing frozen row_index values: {sorted(missing_rows)}")
    return [selected[source["row_index"]] for source in identities]


def _validate_output_location(
    output_dir: Path, dataset_root: Path, snapshot_path: Path, audit_manifest: Path,
) -> None:
    protected_roots = (
        dataset_root, REPO_ROOT / "cache/generator_audit", REPO_ROOT / "docs/audits",
        REPO_ROOT / "outputs/step2", REPO_ROOT / "salad", REPO_ROOT / "third_party",
    )
    managed = tuple(output_dir / name for name in MANAGED_PATHS)
    for protected in protected_roots:
        protected = protected.resolve()
        if output_dir == protected or output_dir.is_relative_to(protected):
            raise ValueError("output_dir must be outside the dataset and frozen/protected artifacts")
        for path in managed:
            resolved = path.resolve()
            if resolved == protected or resolved.is_relative_to(protected) or protected.is_relative_to(resolved):
                raise ValueError("managed outputs must not overwrite dataset or frozen/protected artifacts")
    for input_path in (snapshot_path, audit_manifest):
        if input_path == output_dir or any(
            input_path == path.resolve() or input_path.is_relative_to(path.resolve())
            for path in managed
        ):
            raise ValueError("managed outputs must not overwrite input metadata")
    for policy in POLICIES:
        parent = output_dir / policy
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ValueError("managed policy directories must be real directories")
        source = parent / "source"
        if source.is_symlink():
            raise ValueError("managed source directories must not be symlinks")


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _publish(staging: Path, output_dir: Path) -> None:
    """Replace only managed sources/metadata and roll back publication failures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = staging / "previous"
    previous.mkdir()
    saved, published = [], []
    try:
        for name in MANAGED_PATHS:
            destination = output_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            old = previous / name
            old.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                os.replace(destination, old)
                saved.append(name)
            os.replace(staging / name, destination)
            published.append(name)
    except OSError:
        for name in reversed(published):
            _remove(output_dir / name)
        for name in reversed(saved):
            os.replace(previous / name, output_dir / name)
        raise


def build_geometry_audit(
    snapshot_path: str | Path,
    audit_manifest: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = SEED,
) -> dict:
    """Write both source policies for the exact frozen Step 2C identities."""
    snapshot_path = Path(snapshot_path).expanduser().resolve()
    audit_manifest = Path(audit_manifest).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    seed = _integer(seed, "seed")
    _validate_output_location(output_dir, dataset_root, snapshot_path, audit_manifest)
    identities = load_frozen_identities(snapshot_path)
    selected = load_audit_sources(audit_manifest, identities)
    originals = []
    for row in selected:
        path = (dataset_root / _source_relative(row["relative_path"], "relative_path")).resolve()
        if not path.is_relative_to(dataset_root):
            raise ValueError(f"source image escapes dataset root: {row['relative_path']}")
        if not path.is_file():
            raise FileNotFoundError(f"source image is missing: {row['relative_path']}")
        if path in originals:
            raise ValueError("frozen source identities must refer to ten distinct original image files")
        originals.append(path)
    summary = {
        "number_of_sources": SOURCE_COUNT,
        "number_of_records": SOURCE_COUNT * len(POLICIES),
        "policies": list(POLICIES),
        "seed": seed,
        "snapshot_reference": _reference(snapshot_path),
        "snapshot_sha256": _sha256(snapshot_path),
        "audit_manifest_reference": _reference(audit_manifest),
        "audit_manifest_sha256": _sha256(audit_manifest),
        "source_identities": identities,
        "original_source_sha256": [
            {"row_index": row["row_index"], "sha256": _sha256(path)}
            for row, path in zip(selected, originals)
        ],
        "config": {
            "dataset_root": _reference(dataset_root),
            "output_dir": _reference(output_dir),
            "seed": seed,
            "selection": "exact source identities from frozen Step 2C per_source_compact; no sampling",
            "policies": list(POLICIES),
            "long_side_pixels": 512,
            "minimum_dimension": 256,
            "dimension_multiple": 64,
            "source_coordinate_system": "decoded pixels, without EXIF transposition",
            "source_image_format": "PNG",
            "source_image_encoding": "lossless",
            "resampling": "LANCZOS",
            "path_reference_base": "invocation working directory",
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        for policy in POLICIES:
            (staging / policy / "source").mkdir(parents=True)
        with (staging / "geometry_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for row, path in zip(selected, originals):
                try:
                    with Image.open(path) as original:
                        for policy in POLICIES:
                            canonical, metadata = canonicalize_probe(original, policy=policy)
                            try:
                                width, height = canonical.size
                                if min(width, height) < 256 or width % 64 or height % 64:
                                    raise ValueError("canonical geometry requires dimensions >= 256 and divisible by 64")
                                if (width, height) != (metadata["output_width"], metadata["output_height"]):
                                    raise ValueError("canonical image dimensions disagree with transform metadata")
                                if policy == "square_crop_512" and (width, height) != (512, 512):
                                    raise ValueError("square_crop_512 must produce 512x512")
                                if max(width, height) != 512:
                                    raise ValueError("canonical longest side must be 512")
                                relative = Path(policy) / "source" / f"{row['row_index']:08d}.png"
                                canonical.save(staging / relative, format="PNG")
                            finally:
                                canonical.close()
                            writer.writerow({
                                "audit_index": row["audit_index"],
                                "row_index": row["row_index"],
                                "image_id": row["image_id"],
                                "policy": policy,
                                "source_path": _reference(output_dir / relative),
                                "original_path": _reference(path),
                                "original_width": metadata["original_width"],
                                "original_height": metadata["original_height"],
                                "canonical_width": width,
                                "canonical_height": height,
                                "scale_x": metadata["scale_x"],
                                "scale_y": metadata["scale_y"],
                                "retained_area_fraction": metadata["retained_area_fraction"],
                                "retained_long_axis_fraction": metadata["retained_long_axis_fraction"],
                            })
                except (OSError, ValueError) as error:
                    raise ValueError(f"could not preprocess frozen source {row['relative_path']}: {error}") from error
        (staging / "geometry_build_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _publish(staging, output_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("docs/audits/step2c_metrics.json"))
    parser.add_argument("--audit-manifest", type=Path, default=Path("cache/generator_audit/audit_manifest.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/gsv-cities"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/geometry_audit"))
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = build_geometry_audit(
            args.snapshot, args.audit_manifest, args.dataset_root, args.output_dir, seed=args.seed,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Wrote {summary['number_of_records']} paired PNG sources for {summary['number_of_sources']} frozen identities")
    print(f"Manifest: {args.output_dir / 'geometry_manifest.csv'}")
    print(f"Summary: {args.output_dir / 'geometry_build_summary.json'}")


if __name__ == "__main__":
    main()
