#!/usr/bin/env python3
"""Build lossless native 640x480 sources for the frozen Step 2D0 originals."""

import argparse
from pathlib import Path
import sys
import tempfile

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.probe.canonical import canonicalize_probe  # noqa: E402
from countermine.probe.geometry_audit import publish, read_csv, reference, sha256, write_csv, write_json  # noqa: E402
from countermine.probe.native_fov_audit import (  # noqa: E402
    MANIFEST_COLUMNS, NATIVE_SIZE, POLICY, frozen_sources, integer, relative_path,
    validate_native_destinations,
)


def _join_originals(audit_manifest, dataset_root, frozen):
    rows = read_csv(audit_manifest, ("audit_index", "row_index", "image_id", "relative_path"))
    selected, seen = {}, set()
    lookup = {source["row_index"]: source for source in frozen}
    for row in rows:
        row_index = integer(row["row_index"], "row_index")
        audit_index = integer(row["audit_index"], "audit_index")
        if row_index in seen:
            raise ValueError("audit manifest contains duplicate row identities")
        seen.add(row_index)
        if row_index not in lookup:
            continue
        identity = lookup[row_index]
        if audit_index != identity["audit_index"] or row["image_id"] != identity["image_id"]:
            raise ValueError("original audit manifest disagrees with the frozen source identity")
        original = (dataset_root / relative_path(row["relative_path"], "relative_path")).resolve()
        if not original.is_relative_to(dataset_root) or not original.is_file():
            raise ValueError("frozen original is missing or escapes the dataset root")
        selected[row_index] = original
    if set(selected) != set(lookup) or len(set(selected.values())) != 10:
        raise ValueError("audit manifest must identify all ten distinct frozen originals")
    return [selected[source["row_index"]] for source in frozen]


def build_native_fov_audit(snapshot_path, audit_manifest, dataset_root, output_dir, *, seed=42):
    paths = (snapshot_path, audit_manifest, dataset_root, output_dir)
    snapshot_path, audit_manifest, dataset_root, output_dir = (Path(path).expanduser().resolve() for path in paths)
    seed = integer(seed, "seed")
    identities = frozen_sources(snapshot_path)
    originals = _join_originals(audit_manifest, dataset_root, identities)
    managed = [output_dir / POLICY / "source", output_dir / "native_manifest.csv",
               output_dir / "native_build_summary.json"]
    validate_native_destinations(managed, [snapshot_path, audit_manifest, *originals],
                                 extra_protected=[dataset_root])
    policy_dir = output_dir / POLICY
    if policy_dir.is_symlink() or (policy_dir.exists() and not policy_dir.is_dir()):
        raise ValueError("managed native policy path must be a real directory")
    if (policy_dir / "source").is_symlink():
        raise ValueError("managed native source directory must not be a symlink")
    for path in originals:
        with Image.open(path) as image:
            if image.format != "JPEG" or image.size != NATIVE_SIZE:
                raise ValueError("native_full_fov requires actual 640x480 JPEG originals")
            image.load()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-build-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        (staging / POLICY / "source").mkdir(parents=True)
        rows = []
        for identity, original_path in zip(identities, originals):
            name = f"{identity['row_index']:08d}.png"
            staged = staging / POLICY / "source" / name
            destination = output_dir / POLICY / "source" / name
            with Image.open(original_path) as original:
                canonical, metadata = canonicalize_probe(original, policy=POLICY)
                try:
                    if canonical.mode != "RGB" or canonical.size != NATIVE_SIZE:
                        raise ValueError("native preprocessing must produce exact 640x480 RGB")
                    expected_metadata = {
                        "policy": POLICY, "original_width": 640, "original_height": 480,
                        "output_width": 640, "output_height": 480, "scale_x": 1.0, "scale_y": 1.0,
                        "retained_area_fraction": 1.0, "retained_long_axis_fraction": 1.0,
                    }
                    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                        raise ValueError("native transform metadata must preserve exact original geometry and FOV")
                    decoded = original.convert("RGB")
                    try:
                        if canonical.tobytes() != decoded.tobytes():
                            raise ValueError("native preprocessing changed decoded original pixels")
                    finally:
                        decoded.close()
                    canonical.save(staged, format="PNG")
                finally:
                    canonical.close()
            rows.append({**identity, "policy": POLICY,
                         "source_path": reference(destination), "original_path": reference(original_path),
                         "original_width": metadata["original_width"], "original_height": metadata["original_height"],
                         "canonical_width": metadata["output_width"], "canonical_height": metadata["output_height"],
                         **{name: metadata[name] for name in
                            ("scale_x", "scale_y", "retained_area_fraction", "retained_long_axis_fraction")},
                         "original_format": "JPEG", "source_format": "PNG",
                         "original_sha256": sha256(original_path), "source_sha256": sha256(staged)})
        summary = {
            "number_of_sources": 10, "number_of_records": 10,
            "historical_snapshot_reference": reference(snapshot_path),
            "historical_snapshot_sha256": sha256(snapshot_path),
            "audit_manifest_reference": reference(audit_manifest), "audit_manifest_sha256": sha256(audit_manifest),
            "source_identities": identities,
            "config": {"seed": seed, "policy": POLICY, "original_size": list(NATIVE_SIZE),
                       "canonical_size": list(NATIVE_SIZE), "scale_x": 1.0, "scale_y": 1.0,
                       "retained_area_fraction": 1.0, "retained_long_axis_fraction": 1.0,
                       "selection": "exact ten original identities from frozen Step 2D0; no sampling",
                       "pixel_transform": "decoded RGB pixel copy; no crop, resize, pad, stretch, or EXIF transpose",
                       "original_format": "JPEG", "source_format": "PNG", "source_encoding": "lossless",
                       "dataset_root": reference(dataset_root)},
        }
        write_csv(staging / "native_manifest.csv", MANIFEST_COLUMNS, rows)
        summary["manifest_sha256"] = sha256(staging / "native_manifest.csv")
        write_json(staging / "native_build_summary.json", summary)
        publish([(staging / POLICY / "source", output_dir / POLICY / "source"),
                 (staging / "native_manifest.csv", output_dir / "native_manifest.csv"),
                 (staging / "native_build_summary.json", output_dir / "native_build_summary.json")], staging)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("docs/audits/step2d0_metrics.json"))
    parser.add_argument("--audit-manifest", type=Path, default=Path("cache/generator_audit/audit_manifest.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/gsv-cities"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/native_fov_audit"))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = build_native_fov_audit(args.snapshot, args.audit_manifest, args.dataset_root,
                                         args.output_dir, seed=args.seed)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Wrote {summary['number_of_sources']} exact native 640x480 PNG sources")


if __name__ == "__main__":
    main()
