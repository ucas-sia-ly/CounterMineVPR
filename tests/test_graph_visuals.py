"""CPU-only selection, original-source fidelity, and complete figure checks."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from countermine.mining import graph_visuals


class TestGraphVisualSelections(unittest.TestCase):
    @staticmethod
    def edge(uid, ratio=.5, bottleneck=.5, matches=50, similarity=.8, duplicate=False):
        return {"pair_uid": uid, "local_match_ratio": ratio,
                "structural_bottleneck": bottleneck, "structural_geomean": bottleneck,
                "num_matches": matches, "max_salad_similarity": similarity,
                "exact_pixel_duplicate": duplicate}

    def test_bottleneck_order_and_duplicate_exclusion(self):
        rows = [self.edge("duplicate", bottleneck=1, duplicate=True),
                self.edge("b", bottleneck=.9), self.edge("a", bottleneck=.9),
                self.edge("count", bottleneck=.9, matches=51), self.edge("lower", bottleneck=.8)]
        rows[-1]["structural_geomean"] = .99
        rows[1]["structural_geomean"] = .91
        self.assertEqual([r["pair_uid"] for r in graph_visuals.select_image_edges(rows)],
                         ["b", "count", "a", "lower"])
        self.assertEqual([r["pair_uid"] for r in graph_visuals.select_image_edges(list(reversed(rows)))],
                         ["b", "count", "a", "lower"])

    def test_raw_ratio_preserves_step2a_ties(self):
        rows = [self.edge("b"), self.edge("a"), self.edge("salad", similarity=.9),
                self.edge("matches", matches=51), self.edge("ratio", ratio=.6),
                self.edge("duplicate", ratio=1, duplicate=True)]
        self.assertEqual([r["pair_uid"] for r in graph_visuals.select_image_edges(rows, ranking="ratio")],
                         ["ratio", "matches", "salad", "a", "b"])

    def test_repeated_pair_exclusive_priority_then_evidence(self):
        flags = ("independent_support_3", "independent_support_2", "repeated_support_3", "repeated_support_2")
        rows = []
        for index, active in enumerate(flags):
            row = {flag: False for flag in flags}
            row.update({"place_uid_a": f"p{index}", "place_uid_b": "z",
                        "structural_bottleneck_max": .9 + index * .01,
                        "num_core_q95_image_edges": 3})
            row[active] = True
            rows.append(row)
        rows[0]["repeated_support_2"] = True
        self.assertEqual([r["place_uid_a"] for r in graph_visuals.select_repeated_place_pairs(rows[::-1])],
                         ["p0", "p1", "p2", "p3"])
        tied = dict(rows[-1], place_uid_a="aaa")
        self.assertEqual([r["place_uid_a"] for r in graph_visuals.select_repeated_place_pairs([rows[-1], tied])], ["aaa", "p3"])

    def test_components_preserve_singletons_and_stable_lexical_ties(self):
        nodes = [{"place_uid": uid} for uid in ("e", "d", "c", "b", "a")]
        edges = [{"place_uid_a": "b", "place_uid_b": "a", "num_core_q95_image_edges": 2},
                 {"place_uid_a": "d", "place_uid_b": "c", "num_core_q95_image_edges": 1},
                 {"place_uid_a": "b", "place_uid_b": "c", "num_core_q95_image_edges": 0}]
        self.assertEqual(graph_visuals.place_core_components(nodes, edges), [["a", "b"], ["c", "d"], ["e"]])


class TestOriginalRGBGraphVisuals(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.step2a = self.root / "step2a"
        self.step2a.mkdir()
        self.manifest = self.root / "manifest.csv"
        self.output = self.root / "figures"
        self.image_nodes, fingerprints, rows = [], {}, []
        for index, color in enumerate(("red", "orange", "blue", "purple", "green", "gray")):
            filename = f"image{index}.png"
            path = self.dataset / filename
            with Image.new("RGB", (640, 480), color) as image:
                image.save(path)
                pixels_hash = hashlib.sha256(image.tobytes()).hexdigest()
            image_id, place_uid = f"i{index}", f"city:p{index // 2}"
            row = {"image_id": image_id, "row_index": index, "place_uid": place_uid,
                   "city_id": "city", "relative_path": filename}
            rows.append(row)
            self.image_nodes.append({k: v for k, v in row.items() if k != "relative_path"})
            fingerprints[image_id] = {"image_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "rgb_pixel_sha256": pixels_hash}
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (self.step2a / "image_fingerprints.json").write_text(json.dumps(fingerprints))
        self.edges = [self.edge("b", 0, 2, .97), self.edge("a", 1, 3, .98),
                      self.edge("c", 2, 4, .2)]
        self.place_nodes = [{"place_uid": f"city:p{i}", "city_id": "city"} for i in range(3)]
        self.place_edges = [self.place_edge("city:p0", "city:p1", 2),
                            self.place_edge("city:p1", "city:p2", 0)]

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def edge(uid, a, b, score):
        return {"pair_uid": uid, "image_id_a": f"i{a}", "image_id_b": f"i{b}",
                "place_uid_a": f"city:p{a // 2}", "place_uid_b": f"city:p{b // 2}",
                "city_id_a": "city", "city_id_b": "city", "geo_distance_m": 900.,
                "best_rgb_rank": 1, "max_salad_similarity": .8, "num_keypoints_a": 500,
                "num_keypoints_b": 600, "num_matches": 100, "local_match_ratio": .2,
                "ratio_null_percentile": score, "match_count_null_percentile": score,
                "structural_bottleneck": score, "structural_geomean": score,
                "symmetric_match_coverage": .5, "rank_bin": "rank_1",
                "core_q95": score >= .95, "exact_pixel_duplicate": False}

    @staticmethod
    def place_edge(a, b, core):
        return {"place_uid_a": a, "place_uid_b": b,
                "num_supporting_image_edges": max(1, core), "num_core_q95_image_edges": core,
                "num_unique_images_a": 2, "num_unique_images_b": 2,
                "num_core_q95_unique_images_a": 2 if core else 0,
                "num_core_q95_unique_images_b": 2 if core else 0,
                "structural_bottleneck_max": .98 if core else .2, "min_geo_distance_m": 900.,
                "independent_support_3": core >= 3, "independent_support_2": core >= 2,
                "repeated_support_3": core >= 3, "repeated_support_2": core >= 2}

    def create(self):
        return graph_visuals.create_graph_visuals(self.image_nodes, self.edges, self.place_nodes,
              self.place_edges, {}, self.manifest, self.dataset, self.step2a, self.output)

    def test_all_artifacts_native_source_preservation_and_curated_prefixes(self):
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.dataset.iterdir()}
        frozen = (self.step2a / "image_fingerprints.json").read_bytes()
        metadata = self.create()
        self.assertEqual(metadata["top50_bottleneck_pair_uids"], ["a", "b", "c"])
        self.assertEqual(metadata["repeated_place_pairs"][0]["supporting_pair_uids"], ["a", "b"])
        self.assertEqual(metadata["largest_place_components"][0]["place_uids"], ["city:p0", "city:p1"])
        self.assertTrue(metadata["real_rgb_only"])
        self.assertFalse(metadata["synthetic_images_used"])
        self.assertTrue(metadata["no_new_local_matching"])
        self.assertNotIn(str(self.root), json.dumps(metadata, allow_nan=False))
        self.assertEqual(metadata["source_validation"]["validated_image_count"], 5)
        for name in graph_visuals.ARTIFACT_NAMES:
            with Image.open(self.output / name) as image:
                image.verify()
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before})
        self.assertEqual(frozen, (self.step2a / "image_fingerprints.json").read_bytes())
        audit = self.root / "audit"
        audit.mkdir()
        sentinel = audit / "step2a_sentinel.jpg"
        sentinel.write_bytes(b"historical evidence")
        names = graph_visuals.curate_graph_artifacts(self.output, audit)
        self.assertEqual(len(names), 8)
        self.assertTrue(all(name.startswith("step2b_") for name in names))
        for name in names:
            self.assertEqual((audit / name).read_bytes(), (self.output / name.removeprefix("step2b_")).read_bytes())
        self.assertEqual(sentinel.read_bytes(), b"historical evidence")

    def test_original_file_hash_mismatch_is_rejected(self):
        (self.dataset / "image0.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "original image bytes changed"):
            self.create()

    def test_original_decoded_pixel_hash_mismatch_is_rejected(self):
        path = self.step2a / "image_fingerprints.json"
        value = json.loads(path.read_text())
        value["i0"]["rgb_pixel_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        reader = graph_visuals.OriginalRGBReader(self.manifest, self.dataset, self.step2a)
        with self.assertRaisesRegex(ValueError, "original RGB pixels changed"):
            reader.read("i0")

    def test_native_size_is_required(self):
        path = self.dataset / "image0.png"
        with Image.new("RGB", (320, 240), "red") as image:
            image.save(path)
        fingerprints_path = self.step2a / "image_fingerprints.json"
        fingerprints = json.loads(fingerprints_path.read_text())
        fingerprints["i0"]["image_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        fingerprints_path.write_text(json.dumps(fingerprints))
        with self.assertRaisesRegex(ValueError, "original RGB 640x480"):
            graph_visuals.OriginalRGBReader(self.manifest, self.dataset, self.step2a).read("i0")

    def test_empty_repeated_and_component_slices_have_panels(self):
        self.edges = [dict(row, core_q95=False) for row in self.edges]
        self.place_edges = [self.place_edge("city:p0", "city:p1", 0),
                            self.place_edge("city:p1", "city:p2", 0)]
        metadata = self.create()
        self.assertEqual(metadata["repeated_place_pairs"], [])
        self.assertEqual(metadata["largest_place_components"], [])
        for name in ("repeated_place_confusions.jpg", "largest_place_components.jpg"):
            with Image.open(self.output / name) as image:
                self.assertEqual(image.height, 180)

    def test_output_cannot_replace_original_or_step2a_evidence(self):
        for protected in (self.dataset, self.step2a):
            self.output = protected / "figures"
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                self.create()
            self.assertFalse(self.output.exists())

    def test_graph_endpoint_metadata_must_match_original_manifest(self):
        self.edges[0]["place_uid_a"] = "wrong-place"
        with self.assertRaisesRegex(ValueError, "place_uid differs from original manifest"):
            self.create()

    def test_source_relative_path_cannot_escape_dataset(self):
        reader = graph_visuals.OriginalRGBReader(self.manifest, self.dataset, self.step2a)
        reader.manifest["i0"]["relative_path"] = "../outside.png"
        with self.assertRaisesRegex(ValueError, "inside the original dataset"):
            reader.read("i0")


if __name__ == "__main__":
    unittest.main()
