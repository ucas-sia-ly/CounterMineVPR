#!/usr/bin/env python3
"""Copy the existing 100 frozen audit originals into lossless native RGB PNGs."""

import argparse
from pathlib import Path
import sys
import tempfile

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.probe.canonical import canonicalize_probe  # noqa: E402
from countermine.probe.geometry_audit import publish, read_csv, reference, sha256, validate_png, write_csv, write_json  # noqa: E402
from countermine.probe.native100_audit import (  # noqa: E402
    MANIFEST_COLUMNS, NATIVE_SIZE, POLICY, SOURCE_COUNT, load_frozen_population,
    population_provenance, validate_manifest_metadata, validate_native100_destinations,
)
from countermine.probe.native_fov_audit import integer, relative_path  # noqa: E402


def _originals(population, summary, dataset_root):
    """Resolve identities against the same hashed mini manifest used by Step 2."""
    mini_path = relative_path(summary["manifest_reference"], "manifest_reference", allow_parent=True).resolve()
    if sha256(mini_path) != summary["manifest_sha256"]:
        raise ValueError("the original Step 2 mini manifest changed since source selection")
    mini = read_csv(mini_path, ("row_index", "image_id", "relative_path", "place_uid", "city_id"))
    lookup = {}
    for row in mini:
        index = integer(row["row_index"], "row_index")
        if index in lookup:
            raise ValueError("historical mini manifest contains duplicate row identities")
        lookup[index] = row
    paths = []
    for row in population:
        old = lookup.get(row["row_index"])
        if old is None or any(old[name] != row.get(name) for name in ("image_id", "relative_path", "place_uid", "city_id")):
            raise ValueError("audit source does not resolve to the original Step 2 mini-manifest image")
        path = (dataset_root / relative_path(row["relative_path"], "relative_path")).resolve()
        if not path.is_relative_to(dataset_root) or not path.is_file():
            raise ValueError("historical original is missing or escapes dataset_root")
        with Image.open(path) as original:
            if original.size != NATIVE_SIZE or original.format != "JPEG":
                raise ValueError("every stored original must be exactly 640x480 JPEG; no resizing")
            original.load()
        paths.append(path)
    if len(set(paths)) != SOURCE_COUNT:
        raise ValueError("100 image identities must resolve to 100 distinct originals")
    return paths


def _verify_historical_pixels(original, row):
    """Verify identity using the old decoded PNG; this is not a geometry experiment.

    The earlier manifest has no original-image digest. Reproducing its frozen
    preprocessing verifies the stored original still yields its earlier source.
    This integrity check neither generates a probe nor computes comparisons.
    """
    old_path = relative_path(row["source_512_path"], "source_512_path", allow_parent=True).resolve()
    validate_png(old_path, (512, 512))
    rgb = original.convert("RGB")
    try:
        historical, _ = canonicalize_probe(rgb, policy="square_crop_512")
    finally:
        rgb.close()
    try:
        with Image.open(old_path) as old:
            if historical.tobytes() != old.tobytes():
                raise ValueError("stored original no longer matches the earlier Step 2 source pixels")
    finally:
        historical.close()


def build_native100_audit(audit_manifest, audit_summary, dataset_root, output_dir):
    audit_manifest, audit_summary, dataset_root, output_dir = (
        Path(p).expanduser().resolve() for p in (audit_manifest, audit_summary, dataset_root, output_dir)
    )
    population, old_summary = load_frozen_population(audit_manifest, audit_summary)
    originals = _originals(population, old_summary, dataset_root)
    names = ("source", "native100_manifest.csv", "native100_build_summary.json")
    inputs = [audit_manifest, audit_summary, *originals,
              *[Path(row["source_512_path"]).resolve() for row in population]]
    validate_native100_destinations([output_dir / name for name in names], inputs, extra_protected=[dataset_root])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native100-build-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        (staging / "source").mkdir()
        rows = []
        for row, original_path in zip(population, originals):
            filename = f"{row['row_index']:08d}.png"
            staged = staging / "source" / filename
            with Image.open(original_path) as original:
                _verify_historical_pixels(original, row)
                source, metadata = canonicalize_probe(original, policy=POLICY)
                try:
                    if source.size != NATIVE_SIZE or source.mode != "RGB":
                        raise ValueError("native preprocessing must preserve exact 640x480 RGB")
                    source.save(staged, format="PNG")
                finally:
                    source.close()
            rows.append({**row, "policy": POLICY, "original_path": reference(original_path),
                         "source_path": reference(output_dir / "source" / filename),
                         "original_width": 640, "original_height": 480,
                         "canonical_width": 640, "canonical_height": 480,
                         **{name: metadata[name] for name in ("scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction")},
                         "original_format": "JPEG", "source_format": "PNG",
                         "original_sha256": sha256(original_path), "source_sha256": sha256(staged)})
        validate_manifest_metadata(rows, population)
        columns = tuple(dict.fromkeys((*population[0].keys(), *MANIFEST_COLUMNS)))
        write_csv(staging / "native100_manifest.csv", columns, rows)
        summary = {
            "number_of_sources": SOURCE_COUNT, "number_of_records": len(rows),
            "manifest_sha256": sha256(staging / "native100_manifest.csv"),
            "source_population_provenance": population_provenance(audit_manifest, audit_summary, old_summary),
            "config": {"seed": 42, "policy": POLICY, "original_size": list(NATIVE_SIZE),
                       "canonical_size": list(NATIVE_SIZE), "scale_x": 1.0, "scale_y": 1.0,
                       "retained_area_fraction": 1.0, "retained_long_axis_fraction": 1.0,
                       "pixel_transform": "decoded RGB copy; no crop, resize, padding, stretch, or EXIF transpose",
                       "source_format": "PNG", "source_encoding": "lossless",
                       "historical_identity_verification": "hashed original mini manifest and exact decoded earlier source PNG pixels",
                       "dataset_root": reference(dataset_root)},
        }
        write_json(staging / "native100_build_summary.json", summary)
        publish([(staging / name, output_dir / name) for name in names], staging)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", type=Path, default=Path("cache/generator_audit/audit_manifest.csv"))
    parser.add_argument("--audit-summary", type=Path, default=Path("cache/generator_audit/audit_summary.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/gsv-cities"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/native100_audit"))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = build_native100_audit(args.audit_manifest, args.audit_summary, args.dataset_root, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Wrote {summary['number_of_sources']} frozen native 640x480 RGB PNG sources; no resampling")


if __name__ == "__main__":
    main()
