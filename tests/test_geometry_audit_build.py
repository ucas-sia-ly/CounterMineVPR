"""CPU-only tests for the frozen Step 2D0 paired-source builder."""

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "geometry_audit_builder", REPO_ROOT / "tools/11_build_geometry_audit.py",
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GeometryAuditBuilderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        self.output = self.root / "geometry_audit"
        self.snapshot = self.root / "step2c_metrics.json"
        self.audit_manifest = self.root / "audit_manifest.csv"
        self.row_indices = (101, 5, 90, 63, 28, 57, 3, 89, 11, 44)
        self.records = [
            {"audit_index": audit_index, "row_index": row_index, "mode": mode}
            for audit_index, row_index in enumerate(self.row_indices)
            for mode in ("official_rmbg", "full_scene")
        ]
        self.manifest = [
            {"audit_index": audit_index, "row_index": row_index,
             "image_id": f"source-{row_index}", "relative_path": f"Images/{row_index}.png"}
            for audit_index, row_index in enumerate(self.row_indices)
        ]
        # These extra historical manifest rows must never be decoded or sampled.
        self.manifest.extend([
            {"audit_index": 10, "row_index": 501, "image_id": "unselected-501", "relative_path": "missing-501.png"},
            {"audit_index": 11, "row_index": 502, "image_id": "unselected-502", "relative_path": "missing-502.png"},
        ])
        self.save_snapshot()
        write_csv(self.audit_manifest, list(reversed(self.manifest)))
        for row_index in self.row_indices:
            path = self.dataset / "Images" / f"{row_index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            with Image.new("RGB", (80, 60), "green") as image:
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 19, 59), fill=(255, row_index, 0))
                draw.rectangle((60, 0, 79, 59), fill=(0, row_index, 255))
                image.save(path)

    def save_snapshot(self):
        self.snapshot.write_text(json.dumps({"per_source_compact": self.records}), encoding="utf-8")

    def build(self, seed=42):
        return builder.build_geometry_audit(
            self.snapshot, self.audit_manifest, self.dataset, self.output, seed=seed,
        )

    def output_checksums(self):
        return {path.relative_to(self.output).as_posix(): checksum(path)
                for path in self.output.rglob("*") if path.is_file()}

    def test_build_uses_exact_frozen_identities_and_records_both_geometries(self):
        before = {path: checksum(path) for path in (
            self.snapshot, self.audit_manifest, *self.dataset.rglob("*.png"),
        )}
        summary = self.build()
        with (self.output / "geometry_manifest.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, list(builder.MANIFEST_COLUMNS))
            rows = list(reader)
        self.assertEqual(len(rows), 20)
        self.assertEqual([int(row["row_index"]) for row in rows],
                         [index for index in self.row_indices for _ in range(2)])
        self.assertEqual([row["policy"] for row in rows], list(builder.POLICIES) * 10)
        for row in rows:
            row_index = int(row["row_index"])
            policy = row["policy"]
            width, height = (512, 512) if policy == "square_crop_512" else (512, 384)
            fraction = 0.75 if policy == "square_crop_512" else 1.0
            self.assertEqual((int(row["original_width"]), int(row["original_height"])), (80, 60))
            self.assertEqual((int(row["canonical_width"]), int(row["canonical_height"])), (width, height))
            self.assertEqual(float(row["retained_area_fraction"]), fraction)
            self.assertEqual(float(row["retained_long_axis_fraction"]), fraction)
            self.assertFalse(Path(row["source_path"]).is_absolute())
            self.assertFalse(Path(row["original_path"]).is_absolute())
            source_path = self.output / policy / "source" / f"{row_index:08d}.png"
            self.assertEqual(row["source_path"], Path(os.path.relpath(source_path, Path.cwd())).as_posix())
            with Image.open(source_path) as output:
                self.assertEqual(output.size, (width, height))
                self.assertEqual(output.mode, "RGB")
                self.assertEqual(output.format, "PNG")
                with Image.open(self.dataset / "Images" / f"{row_index}.png") as original:
                    if policy == "square_crop_512":
                        # Reproduce the historical crop/convert/resize directly.
                        expected = original.crop((10, 0, 70, 60)).convert("RGB").resize(
                            (512, 512), Image.Resampling.LANCZOS,
                        )
                    else:
                        expected = original.convert("RGB").resize((512, 384), Image.Resampling.LANCZOS)
                    try:
                        self.assertEqual(output.tobytes(), expected.tobytes())
                    finally:
                        expected.close()
        saved = json.loads((self.output / "geometry_build_summary.json").read_text())
        self.assertEqual(summary, saved)
        self.assertEqual(saved["number_of_sources"], 10)
        self.assertEqual(saved["number_of_records"], 20)
        self.assertEqual(saved["seed"], 42)
        self.assertEqual(saved["snapshot_sha256"], checksum(self.snapshot))
        self.assertEqual(saved["audit_manifest_sha256"], checksum(self.audit_manifest))
        self.assertEqual(saved["config"]["source_image_format"], "PNG")
        self.assertEqual(before, {path: checksum(path) for path in before})

    def test_repeated_build_is_byte_identical_and_preserves_downstream_artifacts(self):
        self.build()
        downstream = self.output / "full_fov_512" / "relit_full_scene" / "existing.png"
        downstream.parent.mkdir(parents=True)
        downstream.write_bytes(b"completed downstream experiment")
        historical = self.output / "geometry_fidelity.csv"
        historical.write_bytes(b"preserved measurement")
        before = self.output_checksums()
        self.build()
        self.assertEqual(before, self.output_checksums())

    def test_seed_and_snapshot_record_order_do_not_resample_identities(self):
        self.build()
        sources = {path: checksum(path) for path in self.output.rglob("*.png")}
        self.records.reverse()
        self.save_snapshot()
        summary = self.build(seed=99)
        self.assertEqual(sources, {path: checksum(path) for path in sources})
        self.assertEqual([row["row_index"] for row in summary["source_identities"]], list(self.row_indices))
        self.assertEqual(summary["config"]["seed"], 99)

    def test_wrong_source_count_or_duplicate_mode_is_rejected(self):
        original = list(self.records)
        corruptions = (original[:-2], original + original[:2], [dict(row) for row in original])
        corruptions[2][1]["mode"] = corruptions[2][0]["mode"]
        for records in corruptions:
            with self.subTest(count=len(records)):
                self.records = records
                self.save_snapshot()
                with self.assertRaisesRegex(ValueError, "(10|20|duplicate)"):
                    self.build()
                self.assertFalse(self.output.exists())

    def test_manifest_must_match_the_frozen_audit_identity(self):
        for change in ("missing", "audit_index", "duplicate_image_id"):
            with self.subTest(change=change):
                rows = [dict(row) for row in self.manifest]
                if change == "missing":
                    rows.pop(0)
                elif change == "audit_index":
                    rows[0]["audit_index"] = 99
                else:
                    rows[-1]["image_id"] = rows[0]["image_id"]
                write_csv(self.audit_manifest, rows)
                with self.assertRaises(ValueError):
                    self.build()
                self.assertFalse(self.output.exists())

    def test_bad_geometry_or_missing_source_preserves_published_results(self):
        self.build()
        before = self.output_checksums()
        path = self.dataset / "Images" / f"{self.row_indices[-1]}.png"
        original = path.read_bytes()
        for corruption in ("missing", "bad_geometry", "too_small_canonical"):
            with self.subTest(corruption=corruption):
                if corruption == "missing":
                    path.unlink()
                else:
                    dimensions = (80, 51) if corruption == "bad_geometry" else (80, 20)
                    with Image.new("RGB", dimensions) as image:
                        image.save(path)
                try:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        self.build()
                    self.assertEqual(before, self.output_checksums())
                finally:
                    path.write_bytes(original)

    def test_unsafe_source_paths_are_rejected(self):
        for value in ("../outside.png", "/absolute.png", "C:\\outside.png", "..\\outside.png"):
            with self.subTest(value=value):
                rows = [dict(row) for row in self.manifest]
                rows[0]["relative_path"] = value
                write_csv(self.audit_manifest, rows)
                with self.assertRaisesRegex(ValueError, "relative path"):
                    self.build()
                self.assertFalse(self.output.exists())

    def test_dataset_and_frozen_protected_output_locations_are_rejected(self):
        unsafe = (
            self.dataset,
            self.dataset / "new_outputs",
            REPO_ROOT / "cache/generator_audit/geometry",
            REPO_ROOT / "docs/audits/geometry",
            REPO_ROOT / "outputs/step2/geometry",
            REPO_ROOT / "third_party/geometry",
            REPO_ROOT / "salad/geometry",
        )
        for output in unsafe:
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, "(outside|overwrite)"):
                    builder.build_geometry_audit(
                        self.snapshot, self.audit_manifest, self.dataset, output,
                    )
                self.assertFalse(output.exists() and output != self.dataset)

    def test_managed_directory_cannot_contain_the_input_dataset_or_metadata(self):
        moved_dataset = self.output / "square_crop_512/source/child_dataset"
        moved_dataset.parent.mkdir(parents=True)
        self.dataset.rename(moved_dataset)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            builder.build_geometry_audit(
                self.snapshot, self.audit_manifest, moved_dataset, self.output,
            )
        self.assertTrue((moved_dataset / "Images").is_dir())
        moved_snapshot = self.output / "geometry_build_summary.json"
        moved_snapshot.write_bytes(self.snapshot.read_bytes())
        with self.assertRaisesRegex(ValueError, "overwrite"):
            builder.build_geometry_audit(
                moved_snapshot, self.audit_manifest, self.dataset, self.output,
            )

    def test_symlink_source_escape_and_policy_directory_are_rejected(self):
        original = self.dataset / "Images" / f"{self.row_indices[0]}.png"
        outside = self.root / "outside.png"
        original.rename(outside)
        original.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.build()
        original.unlink()
        outside.rename(original)
        self.output.mkdir()
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        (self.output / "square_crop_512").symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "real directories"):
            self.build()

    def test_publish_failure_rolls_back_managed_sources_and_metadata(self):
        self.build()
        before = self.output_checksums()
        real_replace = os.replace
        failed = False

        def fail_once(source, destination):
            nonlocal failed
            if Path(destination) == self.output / "geometry_build_summary.json" and not failed:
                failed = True
                raise OSError("simulated atomic publication failure")
            return real_replace(source, destination)

        with patch.object(builder.os, "replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "publication failure"):
                self.build()
        self.assertTrue(failed)
        self.assertEqual(before, self.output_checksums())


if __name__ == "__main__":
    unittest.main()
