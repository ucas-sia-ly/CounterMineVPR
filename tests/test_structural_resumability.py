"""CPU-only durable checkpoints, provenance, and changed-image tests."""

import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

from countermine.mining.candidate_miner import haversine_distance_m, stable_pair_uid
from countermine.mining.structural_matcher import (
    LocalMatchResult, canonical_json_bytes, structural_metrics,
)


CLI = importlib.import_module("tools.08_measure_structural_pairs")


def fixture_metrics(fingerprints, a="a", b="b"):
    metrics = structural_metrics(8, 10, [[10, 10], [90, 70]], [[40, 40], [160, 120]])
    metrics.update({
        "exact_pixel_duplicate": fingerprints[a]["rgb_pixel_sha256"] == fingerprints[b]["rgb_pixel_sha256"],
        "rgb_pixel_sha_a": fingerprints[a]["rgb_pixel_sha256"], "rgb_pixel_sha_b": fingerprints[b]["rgb_pixel_sha256"],
        "image_file_sha_a": fingerprints[a]["image_file_sha256"], "image_file_sha_b": fingerprints[b]["image_file_sha256"],
    })
    return metrics


class JournalTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.fingerprints = {image: {"rgb_pixel_sha256": image * 64, "image_file_sha256": image * 64}
                             for image in ("a", "b", "c")}
        self.config = {"input_hashes": {"manifest_sha256": "1" * 64}, "matcher": {"threshold": .1}}
        self.tasks = [
            ("candidate", "uid", {"pair_uid": "uid", "image_id_a": "a", "image_id_b": "b"}, "a", "b"),
            ("random", "uid", {"pair_uid": "uid", "image_id_a": "a", "image_id_b": "c"}, "a", "c"),
        ]

    def journal(self, config=None, fingerprints=None):
        return CLI.MeasurementJournal(self.root, config or self.config, self.tasks,
                                      fingerprints or self.fingerprints)

    def test_partial_resume_never_duplicates_completed_pairs(self):
        first = self.journal()
        first.record("candidate", "uid", fixture_metrics(self.fingerprints))
        summary = first.publish()
        self.assertFalse(summary["complete"])
        resumed = self.journal()
        self.assertEqual(set(resumed.completed), {("candidate", "uid")})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            resumed.record("candidate", "uid", fixture_metrics(self.fingerprints))
        resumed.record("random", "uid", fixture_metrics(self.fingerprints, "a", "c"))
        summary = resumed.publish()
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["random_count"], 1)
        _, candidate_rows = CLI.read_csv(self.root / "candidate_structural_metrics.csv")
        self.assertEqual(len(candidate_rows), 1)

    def test_resume_rejects_configuration_input_hash_and_image_fingerprint_change(self):
        journal = self.journal()
        journal.record("candidate", "uid", fixture_metrics(self.fingerprints))
        for changed in ({**self.config, "matcher": {"threshold": .2}},
                        {**self.config, "input_hashes": {"manifest_sha256": "2" * 64}}):
            with self.subTest(config=changed), self.assertRaisesRegex(ValueError, "provenance"):
                self.journal(config=changed)
        changed_images = {**self.fingerprints, "a": {"rgb_pixel_sha256": "x" * 64, "image_file_sha256": "x" * 64}}
        with self.assertRaisesRegex(ValueError, "image hash"):
            self.journal(fingerprints=changed_images)

    def test_resume_rejects_checkpoint_metadata_metric_and_uid_corruption(self):
        journal = self.journal()
        journal.record("candidate", "uid", fixture_metrics(self.fingerprints))
        path = next((self.root / ".measurements").glob("*.json"))
        original = path.read_bytes()
        for key, value in (("image_id_b", "other"), ("local_match_ratio", .99), ("exact_pixel_duplicate", True)):
            record = json.loads(original)
            record["row"][key] = value
            path.write_bytes(canonical_json_bytes(record))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.journal()
        path.write_bytes(original)
        record = json.loads(original)
        record["pair_uid"] = "other"
        path.write_bytes(canonical_json_bytes(record))
        with self.assertRaises(ValueError):
            self.journal()

    def test_completed_metrics_are_protected_but_partial_publication_can_recover(self):
        journal = self.journal()
        journal.record("candidate", "uid", fixture_metrics(self.fingerprints))
        journal.publish()
        # Simulate interruption between CSV replacements and summary replacement.
        (self.root / "candidate_structural_metrics.csv").write_text("torn-partial-view\n")
        resumed = self.journal()
        resumed.record("random", "uid", fixture_metrics(self.fingerprints, "a", "c"))
        resumed.publish()
        (self.root / "candidate_structural_metrics.csv").write_text("changed-complete-view\n")
        with self.assertRaisesRegex(ValueError, "CSV changed"):
            self.journal()

    def test_atomic_failure_preserves_existing_output_and_cleans_temporary_file(self):
        output = self.root / "output.json"
        output.write_bytes(b"published")
        with mock.patch.object(CLI.os, "replace", side_effect=OSError("simulated interruption")):
            with self.assertRaises(OSError):
                CLI.atomic_bytes(output, b"new")
        self.assertEqual(output.read_bytes(), b"published")
        self.assertEqual(list(self.root.iterdir()), [output])

    def test_orphan_outputs_without_provenance_are_rejected(self):
        (self.root / "candidate_structural_metrics.csv").write_text("orphan")
        with self.assertRaisesRegex(ValueError, "without provenance"):
            self.journal()


class MeasurementResumeIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        runtime = root / "runtime"
        runtime.mkdir()
        self.args = SimpleNamespace(
            runtime_dir=runtime, manifest=root / "manifest.csv", candidates=root / "raw.csv",
            candidate_summary=root / "raw_summary.json", dataset_root=root / "dataset",
            seed=42, device="cpu", feature_cache_size=2, checkpoint_every=1,
        )
        self.args.dataset_root.mkdir()
        self.manifest = []
        for index, image_id in enumerate(("a", "b", "c")):
            Image.new("RGB", (640, 480), (index * 70, 20, 30)).save(self.args.dataset_root / f"{image_id}.png")
            self.manifest.append({"row_index": index, "image_id": image_id,
                                  "relative_path": f"{image_id}.png", "place_uid": f"p{index}",
                                  "city_id": "city", "lat": index * .003, "lon": 0.0})
        uid = stable_pair_uid("a", "b")
        self.distance = float(haversine_distance_m(0, 0, .003, 0))
        random_distance = float(haversine_distance_m(0, 0, .006, 0))
        # Keep the random target in the same frozen distance bin.
        self.manifest[2]["lat"] = .0035
        random_distance = float(haversine_distance_m(0, 0, .0035, 0))
        population = {"pair_uid": uid, "image_id_a": "a", "image_id_b": "b", "row_index_a": "0", "row_index_b": "1",
                      "place_uid_a": "p0", "place_uid_b": "p1", "city_id_a": "city", "city_id_b": "city",
                      "geo_distance_m": str(self.distance), "same_city": "True", "rank_bin": "rank_1",
                      "anchor_query_image_id": "a", "anchor_negative_image_id": "b"}
        control = {"pair_uid": uid, "candidate_pair_uid": uid, "random_pair_uid": stable_pair_uid("a", "c"),
                   "random_control_available": "True", "anchor_query_image_id": "a", "anchor_query_row_index": "0",
                   "anchor_query_place_uid": "p0", "anchor_query_city_id": "city", "candidate_negative_image_id": "b",
                   "candidate_negative_row_index": "1", "candidate_negative_place_uid": "p1", "candidate_negative_city_id": "city",
                   "random_negative_image_id": "c", "random_negative_row_index": "2", "random_negative_place_uid": "p2",
                   "random_negative_city_id": "city", "candidate_geo_distance_m": str(self.distance),
                   "random_geo_distance_m": str(random_distance), "candidate_same_city": "True", "random_same_city": "True",
                   "same_city": "True", "rank_bin": "rank_1", "geo_distance_bin": "250_500"}
        import csv
        for path, rows in ((runtime / "population.csv", [population]), (runtime / "random_controls.csv", [control]),
                           (self.args.manifest, self.manifest), (self.args.candidates, [{"query_image_id": "a", "negative_image_id": "b"}])):
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        self.args.candidate_summary.write_text('{}\n')
        summary = {"seed": 42}
        for key, path in (("manifest_sha256", self.args.manifest), ("candidate_csv_sha256", self.args.candidates),
                          ("candidate_summary_sha256", self.args.candidate_summary),
                          ("population_sha256", runtime / "population.csv"), ("random_controls_sha256", runtime / "random_controls.csv")):
            summary[key] = CLI.sha256_file(path)
        CLI.atomic_json(runtime / "population_summary.json", summary)

    def loader(self, *args):
        return self.manifest, [{"query_image_id": "a", "negative_image_id": "b"}], {}

    def factory(self, fail_second=False):
        constructed = []

        class FakeMatcher:
            def __init__(self, **kwargs):
                self.provenance, self.calls = {"test_fixture": True}, []
                constructed.append(self)

            def set_image_fingerprints(self, fingerprints):
                self.fingerprints = fingerprints

            def match(self, a, path_a, b, path_b):
                self.calls.append((a, b))
                if fail_second and len(self.calls) == 2:
                    raise RuntimeError("interrupted fixture")
                return LocalMatchResult(fixture_metrics(self.fingerprints, a, b), None, None, None)

        return FakeMatcher, constructed

    def test_streamed_interruption_resume_and_completed_noop(self):
        factory, constructed = self.factory(fail_second=True)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            CLI.run_measurement(self.args, matcher_factory=factory, input_loader=self.loader)
        marker = json.loads((self.args.runtime_dir / "measurement_summary.json").read_text())
        self.assertFalse(marker["complete"])
        factory, constructed = self.factory()
        summary = CLI.run_measurement(self.args, matcher_factory=factory, input_loader=self.loader)
        self.assertTrue(summary["complete"])
        self.assertEqual(len(constructed[0].calls), 1)
        CLI.run_measurement(self.args, matcher_factory=factory, input_loader=self.loader)
        self.assertEqual(constructed[1].calls, [])

    def test_changed_original_bytes_are_rejected_before_matcher_creation(self):
        factory, constructed = self.factory()
        CLI.run_measurement(self.args, matcher_factory=factory, input_loader=self.loader)
        Image.new("RGB", (640, 480), (90, 90, 90)).save(self.args.dataset_root / "a.png")
        with self.assertRaisesRegex(ValueError, "original image bytes"):
            CLI.run_measurement(self.args, matcher_factory=factory, input_loader=self.loader)
        self.assertEqual(len(constructed), 1)

    def test_checkpoints_remain_incomplete_until_final_input_bytes_validation(self):
        factory, _ = self.factory()
        published_completion = []
        original = CLI.MeasurementJournal.publish

        def traced_publish(journal, force_incomplete=False):
            result = original(journal, force_incomplete=force_incomplete)
            published_completion.append(result["complete"])
            return result

        with mock.patch.object(CLI.MeasurementJournal, "publish", new=traced_publish):
            CLI.run_measurement(self.args, matcher_factory=factory, input_loader=self.loader)
        self.assertEqual(published_completion, [False, False, True])


if __name__ == "__main__":
    unittest.main()
