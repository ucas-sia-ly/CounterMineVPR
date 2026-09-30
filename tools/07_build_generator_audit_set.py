#!/usr/bin/env python3
"""Build the deterministic Step 2A real-image generator audit sources.

Only crop/resize preprocessing runs here. Candidate metadata is streamed into
a temporary disk-backed sort; source images are decoded one at a time.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import tempfile

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.probe.canonical import canonicalize_probe  # noqa: E402


GROUP_SIZE = 50
SEED = 42
MANIFEST_COLUMNS = ("row_index", "image_id", "relative_path", "place_uid", "city_id")
CANDIDATE_COLUMNS = (
    "query_row_index", "negative_row_index", "query_image_id", "negative_image_id",
    "similarity",
)
AUDIT_COLUMNS = (
    "audit_index", "group", "row_index", "image_id", "relative_path", "place_uid",
    "city_id", "source_512_path", "original_width", "original_height",
    "crop_left", "crop_top", "crop_size",
)


def _check_columns(reader: csv.DictReader, required: tuple[str, ...], path: Path) -> None:
    columns = reader.fieldnames or []
    if len(columns) != len(set(columns)):
        raise ValueError(f"{path} contains duplicate column names")
    missing = set(required).difference(columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {', '.join(sorted(missing))}")


def _relative_path(value: str, name: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a relative path without '..': {value}")
    return path


def load_manifest(path: str | Path) -> list[dict]:
    """Load the mini manifest, rejecting ambiguous source identities."""
    path = Path(path)
    rows = []
    seen_indices, seen_ids = set(), set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _check_columns(reader, MANIFEST_COLUMNS, path)
        for line_number, row in enumerate(reader, start=2):
            try:
                if any(not row[column] or not row[column].strip() for column in MANIFEST_COLUMNS):
                    raise ValueError("required metadata is missing")
                metadata = {column: row[column] for column in MANIFEST_COLUMNS}
                index = metadata["row_index"] = int(row["row_index"])
                if index < 0:
                    raise ValueError("row_index must be nonnegative")
                if index in seen_indices or metadata["image_id"] in seen_ids:
                    raise ValueError("duplicate row_index or image_id in manifest")
                _relative_path(metadata["relative_path"], "relative_path")
                if Path(metadata["image_id"]).is_absolute():
                    raise ValueError("image_id must not embed an absolute source path")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}, CSV line {line_number}: {error}") from error
            seen_indices.add(index)
            seen_ids.add(metadata["image_id"])
            rows.append(metadata)
    if not rows:
        raise ValueError("manifest contains no images")
    return sorted(rows, key=lambda row: row["row_index"])


def select_audit_sources(
    manifest: list[dict], candidates_path: str | Path, seed: int = SEED,
) -> list[dict]:
    """Select 50 random images, then 50 distinct highest-similarity endpoints.

    Ties retain original candidate CSV order. For each pair, visit query then
    negative so that both sides contribute whenever they are still unseen.
    The full pair sort lives on disk instead of retaining the candidate CSV.
    """
    ordered = sorted(manifest, key=lambda row: row["row_index"])
    by_index = {row["row_index"]: row for row in ordered}
    if len(by_index) != len(ordered) or len({row["image_id"] for row in ordered}) != len(ordered):
        raise ValueError("duplicate row_index or image_id in manifest")
    if len(ordered) < GROUP_SIZE:
        raise ValueError(f"Group A requires {GROUP_SIZE} unique manifest images; found {len(ordered)}")
    selected = [dict(row, group="random") for row in random.Random(seed).sample(ordered, GROUP_SIZE)]
    seen_indices = {row["row_index"] for row in selected}
    seen_ids = {row["image_id"] for row in selected}
    candidates_path = Path(candidates_path)

    with tempfile.TemporaryDirectory(prefix="countermine-audit-sort-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "candidates.sqlite3")
        try:
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute("PRAGMA cache_size = -8192")
            connection.execute(
                "CREATE TABLE pairs (csv_order INTEGER PRIMARY KEY, "
                "similarity REAL NOT NULL, query_index INTEGER NOT NULL, "
                "negative_index INTEGER NOT NULL)"
            )
            batch = []
            with candidates_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                _check_columns(reader, CANDIDATE_COLUMNS, candidates_path)
                for csv_order, row in enumerate(reader):
                    try:
                        similarity = float(row["similarity"])
                        if not math.isfinite(similarity):
                            raise ValueError("similarity must be finite")
                        indices = []
                        for side in ("query", "negative"):
                            index = int(row[f"{side}_row_index"])
                            metadata = by_index.get(index)
                            if metadata is None:
                                raise ValueError(f"{side}_row_index {index} is outside the manifest")
                            for column in ("image_id", "place_uid", "city_id"):
                                candidate_column = f"{side}_{column}"
                                if candidate_column in row and row[candidate_column] != metadata[column]:
                                    raise ValueError(f"{candidate_column} does not match manifest row_index {index}")
                            indices.append(index)
                        batch.append((csv_order, similarity, *indices))
                    except (ValueError, TypeError) as error:
                        raise ValueError(f"{candidates_path}, CSV line {csv_order + 2}: {error}") from error
                    if len(batch) == 4096:
                        connection.executemany("INSERT INTO pairs VALUES (?, ?, ?, ?)", batch)
                        batch.clear()
                if batch:
                    connection.executemany("INSERT INTO pairs VALUES (?, ?, ?, ?)", batch)
            connection.commit()
            cursor = connection.execute(
                "SELECT query_index, negative_index FROM pairs "
                "ORDER BY similarity DESC, csv_order ASC"
            )
            for pair in cursor:
                for index in pair:
                    metadata = by_index[index]
                    if index in seen_indices or metadata["image_id"] in seen_ids:
                        continue
                    selected.append(dict(metadata, group="hard_candidate"))
                    seen_indices.add(index)
                    seen_ids.add(metadata["image_id"])
                    if len(selected) == 2 * GROUP_SIZE:
                        return selected
        finally:
            connection.close()

    hard_count = len(selected) - GROUP_SIZE
    raise ValueError(
        f"Group B requires {GROUP_SIZE} unique hard-candidate images outside Group A; "
        f"found only {hard_count}"
    )


def _reference(path: Path) -> str:
    """Use portable references relative to the invocation working directory."""
    return Path(os.path.relpath(path, Path.cwd())).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _publish(staging: Path, output_dir: Path) -> None:
    """Replace only managed artifacts, restoring previous files on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    backup = staging / "previous"
    backup.mkdir()
    saved, published = [], []
    names = ("source_512", "audit_manifest.csv", "audit_summary.json")
    try:
        for name in names:
            destination = output_dir / name
            if destination.exists() or destination.is_symlink():
                os.replace(destination, backup / name)
                saved.append(name)
            os.replace(staging / name, destination)
            published.append(name)
    except OSError:
        for name in reversed(published):
            _remove(output_dir / name)
        for name in reversed(saved):
            os.replace(backup / name, output_dir / name)
        raise


def build_audit_set(
    manifest_path: str | Path,
    candidates_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    seed: int = SEED,
) -> dict:
    """Decode/canonicalize exactly 100 real images and save the audit metadata."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    candidates_path = Path(candidates_path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir == root or output_dir.is_relative_to(root):
        raise ValueError("output_dir must be outside the source dataset")
    managed_paths = tuple(
        output_dir / name for name in ("source_512", "audit_manifest.csv", "audit_summary.json")
    )
    if any(root == path or root.is_relative_to(path) for path in managed_paths):
        raise ValueError("output artifacts must not overwrite the source dataset")
    for input_path in (manifest_path, candidates_path):
        if input_path == output_dir or any(
            input_path == path or input_path.is_relative_to(path) for path in managed_paths
        ):
            raise ValueError("output artifacts must not overwrite input metadata")

    manifest = load_manifest(manifest_path)
    selected = select_audit_sources(manifest, candidates_path, seed)
    source_paths = []
    for row in selected:
        path = (root / _relative_path(row["relative_path"], "relative_path")).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"source image escapes dataset root: {row['relative_path']}")
        if not path.is_file():
            raise FileNotFoundError(f"source image is missing: {row['relative_path']}")
        source_paths.append(path)

    summary = {
        "number_of_images": len(selected),
        "count_random": GROUP_SIZE,
        "count_hard_candidate": GROUP_SIZE,
        "seed": seed,
        "canonical_resolution": [512, 512],
        "manifest_reference": _reference(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "rgb_candidate_reference": _reference(candidates_path),
        "rgb_candidate_sha256": _sha256(candidates_path),
        "config": {
            "dataset_root": _reference(root),
            "output_dir": _reference(output_dir),
            "seed": seed,
            "random_count": GROUP_SIZE,
            "hard_candidate_count": GROUP_SIZE,
            "random_selection": "random.Random(seed).sample; manifest sorted by row_index",
            "candidate_order": "similarity descending, CSV row order ascending; query then negative",
            "canonical_transform": "largest center square crop with floor offsets, then LANCZOS resize",
            "canonical_resolution": [512, 512],
            "source_coordinate_system": "decoded pixels, without EXIF transposition",
            "jpeg_quality": 95,
            "jpeg_subsampling": 0,
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        (staging / "source_512").mkdir()
        with (staging / "audit_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
            writer.writeheader()
            for audit_index, (row, path) in enumerate(zip(selected, source_paths)):
                filename = f"{row['row_index']:08d}.jpg"
                try:
                    with Image.open(path) as original:
                        rgb = original.convert("RGB")
                        try:
                            canonical, crop_metadata = canonicalize_probe(rgb)
                        finally:
                            rgb.close()
                    try:
                        canonical.save(staging / "source_512" / filename, format="JPEG", quality=95, subsampling=0)
                    finally:
                        canonical.close()
                except (OSError, ValueError) as error:
                    raise ValueError(f"could not preprocess source {row['relative_path']}: {error}") from error
                record = {
                    "audit_index": audit_index,
                    **row,
                    "source_512_path": _reference(output_dir / "source_512" / filename),
                    **{column: crop_metadata[column] for column in AUDIT_COLUMNS if column in crop_metadata},
                }
                writer.writerow(record)
        (staging / "audit_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        _publish(staging, output_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/gsv_mini/manifest.csv"))
    parser.add_argument("--candidates", type=Path, default=Path("cache/gsv_mini/rgb_candidates_raw.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/GSVCities"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/generator_audit"))
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = build_audit_set(
            args.manifest, args.candidates, args.dataset_root, args.output_dir, args.seed,
        )
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"error: {error}\n")
    print(
        f"Wrote {summary['number_of_images']} canonical RGB sources "
        f"({summary['count_random']} random, {summary['count_hard_candidate']} hard_candidate), "
        f"seed={summary['seed']}"
    )
    print(f"Manifest: {args.output_dir / 'audit_manifest.csv'}")
    print(f"Summary: {args.output_dir / 'audit_summary.json'}")


if __name__ == "__main__":
    main()
