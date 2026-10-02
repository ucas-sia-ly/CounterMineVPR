"""CPU bank metadata, exact tensor serialization and bank-only cache tests."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from countermine.mining.banked_structural_matcher import BankFeatureCache, BankedStructuralMatcher
from countermine.mining.full_structural_io import read_json, write_csv, write_json
from countermine.mining.local_feature_bank import (
    FEATURE_KEYS, INDEX_COLUMNS, atomic_feature_file, compare_feature_replay,
    build_feature_bank, deterministic_replay_ids, feature_tensor_schema, load_feature_bank,
    load_tensor_features, require_bank_validation, validate_bank_index,
    verify_source_images,
)
from countermine.mining.structural_matcher import canonical_json_bytes, decode_original_rgb, sha256_file, structural_metrics


def numpy_features(count=3):
    return {
        "keypoints": np.asarray([[[10, 10], [100, 100], [200, 200]][:count]], dtype=np.float32).reshape(1, count, 2),
        "descriptors": np.arange(count * 128, dtype=np.float32).reshape(1, count, 128),
        "keypoint_scores": np.ones((1, count), dtype=np.float32),
        "image_size": np.asarray([[640, 480]], dtype=np.float32),
    }


class FeatureTensorTests(unittest.TestCase):
    def test_actual_extractor_tensor_fields_and_no_downcast(self):
        features = numpy_features()
        schema = feature_tensor_schema(features, tensor_type=np.ndarray)
        self.assertEqual(tuple(sorted(schema)), FEATURE_KEYS)
        self.assertEqual(schema["descriptors"], {"shape": [1, 3, 128], "dtype": "float32"})
        for mutate in (
            lambda d: d.pop("keypoint_scores"),
            lambda d: d.update(descriptors=d["descriptors"].astype(np.float16)),
            lambda d: d.update(descriptors=np.zeros((1, 3, 64), np.float32)),
            lambda d: d.update(image_size=np.asarray([[512, 512]], np.float32)),
            lambda d: d.update(keypoint_scores=np.full((1, 3), np.nan, np.float32)),
        ):
            candidate = numpy_features()
            mutate(candidate)
            with self.assertRaises(ValueError):
                feature_tensor_schema(candidate, tensor_type=np.ndarray)

    def test_exact_replay_fails_on_smallest_float32_drift(self):
        a, b = numpy_features(), numpy_features()
        self.assertEqual(set(compare_feature_replay(a, b, tensor_type=np.ndarray).values()), {0.0})
        b["keypoints"][0, 0, 0] = np.nextafter(b["keypoints"][0, 0, 0], np.float32(np.inf))
        with self.assertRaisesRegex(ValueError, "max_absolute_difference=.*STOP"):
            compare_feature_replay(a, b, tensor_type=np.ndarray)

    def test_atomic_torch_serialization_preserves_dtype_shape_and_values(self):
        import torch  # Explicit CPU tensor-cache test; no GPU/model construction.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank = root / "cache/countermine_rgb/step2d/aliked_bank"
            destination = bank / "features/000000.pt"
            features = {key: torch.from_numpy(value) for key, value in numpy_features().items()}
            schema = atomic_feature_file(destination, features, repo_root=root)
            record = {"image_id": "a", "feature_file": "features/000000.pt",
                      "feature_file_sha256": sha256_file(destination), "num_keypoints": 3}
            loaded = load_tensor_features(bank, record)
            for key in FEATURE_KEYS:
                self.assertTrue(torch.equal(features[key], loaded[key]))
                self.assertEqual(features[key].dtype, loaded[key].dtype)
                self.assertEqual(tuple(features[key].shape), tuple(loaded[key].shape))
                self.assertEqual(loaded[key].device.type, "cpu")
            self.assertEqual(schema["keypoints"]["dtype"], "torch.float32")
            self.assertFalse(list(destination.parent.glob("*.tmp")))
            destination.write_bytes(b"corrupt generated cache")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                load_tensor_features(bank, record)

    def test_replay_selection_exact_requested_digest_and_order_independent(self):
        identities = [f"image{i}" for i in range(80)]
        selected = deterministic_replay_ids(identities, "CounterMineVPR-Step2D-feature-replay", 32)
        expected = sorted(identities, key=lambda value: (
            hashlib.sha256(("CounterMineVPR-Step2D-feature-replay|42|" + value).encode()).hexdigest(), value,
        ))[:32]
        self.assertEqual(selected, expected)
        self.assertEqual(selected, deterministic_replay_ids(list(reversed(identities)), "CounterMineVPR-Step2D-feature-replay", 32))
        with self.assertRaises(ValueError):
            deterministic_replay_ids(["a", "a"], "prefix", 1)

    def test_bank_lru_never_preloads_and_handles_zero_capacity(self):
        cache = BankFeatureCache(2)
        cache.put("a", "cpu-a")
        cache.put("b", "cpu-b")
        self.assertEqual(cache.get("a"), "cpu-a")
        cache.put("c", "cpu-c")
        self.assertIsNone(cache.get("b"))
        self.assertEqual(len(cache), 2)
        disabled = BankFeatureCache(0)
        disabled.put("a", "cpu-a")
        self.assertEqual(len(disabled), 0)
        for capacity in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                BankFeatureCache(capacity)

    def test_bank_only_matcher_reuses_exact_step2a_formulas_on_cpu_tensors(self):
        import torch
        matcher = BankedStructuralMatcher.__new__(BankedStructuralMatcher)
        matcher.torch, matcher.device = torch, torch.device("cpu")
        a = {key: torch.from_numpy(value) for key, value in numpy_features().items()}
        b = {key: torch.from_numpy(value) for key, value in numpy_features().items()}
        b["keypoints"] += 5
        matcher.records = {"a": {"rgb_pixel_sha256": "a" * 64}, "b": {"rgb_pixel_sha256": "b" * 64}}
        matcher._cpu_features = lambda identity: a if identity == "a" else b
        matcher.matcher = lambda values: {"matches": torch.tensor([[[0, 1], [2, 2]]]), "scores": torch.tensor([[.8, .7]])}
        result = matcher.match("a", "b")
        expected = structural_metrics(3, 3, a["keypoints"][0].numpy()[[0, 2]], b["keypoints"][0].numpy()[[1, 2]])
        expected.update(min_num_keypoints=3, exact_pixel_duplicate=False)
        self.assertEqual(result.metrics, expected)
        np.testing.assert_array_equal(result.points_a, [[10, 10], [200, 200]])
        np.testing.assert_array_equal(result.points_b, [[105, 105], [205, 205]])
        np.testing.assert_allclose(result.scores, [.8, .7])
        self.assertEqual(a["descriptors"].dtype, torch.float32)
        self.assertEqual(a["descriptors"].device.type, "cpu")

    def test_zero_keypoints_skip_matcher_and_nonunique_matches_fail(self):
        import torch
        matcher = BankedStructuralMatcher.__new__(BankedStructuralMatcher)
        matcher.torch, matcher.device = torch, torch.device("cpu")
        a = {key: torch.from_numpy(value) for key, value in numpy_features(0).items()}
        b = {key: torch.from_numpy(value) for key, value in numpy_features().items()}
        matcher.records = {"a": {"rgb_pixel_sha256": "a" * 64}, "b": {"rgb_pixel_sha256": "a" * 64}}
        matcher._cpu_features = lambda identity: a if identity == "a" else b
        matcher.matcher = lambda values: self.fail("zero-keypoint pair invoked LightGlue")
        result = matcher.match("a", "b")
        self.assertEqual(result.metrics["local_match_ratio"], 0)
        self.assertEqual(result.metrics["num_matches"], 0)
        self.assertTrue(result.metrics["exact_pixel_duplicate"])
        a = b
        matcher.matcher = lambda values: {"matches": torch.tensor([[[0, 1], [0, 2]]])}
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            matcher.match("a", "b")


class BankIndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bank = self.root / "cache/countermine_rgb/step2d/aliked_bank"
        self.manifest_path = self.root / "cache/gsv_mini/manifest.csv"
        self.manifest_path.parent.mkdir(parents=True)
        self.manifest = pd.DataFrame({"row_index": [0, 1], "image_id": ["a", "b"], "relative_path": ["a.png", "b.png"]})
        self.manifest.to_csv(self.manifest_path, index=False)
        self.static_code = {"source.py": "c" * 64}
        self.scientific = {"step2a_snapshot_sha256": "f" * 64}
        runtime = {"torch_version": "fixture", "aliked_weights_sha256": "w" * 64,
                   "aliked_effective_config": {"max_num_keypoints": 2048}}
        self.frozen_config = {"matcher_provenance": runtime}
        self.configuration = {
            "manifest_file": "cache/gsv_mini/manifest.csv", "manifest_sha256": sha256_file(self.manifest_path),
            "code_sha256": self.static_code, "frozen_provenance": self.scientific,
            "extractor_provenance": runtime,
        }
        self.config_hash = hashlib.sha256(canonical_json_bytes(self.configuration)).hexdigest()
        rows = []
        self.historical = {}
        for row in self.manifest.to_dict("records"):
            feature_file = f"features/{row['row_index']:06d}.pt"
            path = self.bank / feature_file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"CPU fixture {row['image_id']}".encode())
            identity = {"row_index": row["row_index"], "image_id": row["image_id"],
                        "image_file_sha256": "a" * 64, "rgb_pixel_sha256": "b" * 64,
                        "num_keypoints": 3, "feature_file": feature_file, "feature_file_sha256": sha256_file(path)}
            rows.append(identity)
            self.historical[row["image_id"]] = {"image_file_sha256": "a" * 64, "rgb_pixel_sha256": "b" * 64}
            record = {**identity, "schema_version": 1, "complete": True,
                      "configuration_sha256": self.config_hash,
                      "tensor_schema": feature_tensor_schema(numpy_features(), tensor_type=np.ndarray)}
            write_json(self.bank / "records" / f"{row['row_index']:06d}.json", record, repo_root=self.root)
        self.index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
        write_csv(self.bank / "index.csv", self.index, repo_root=self.root)
        write_json(self.bank / "configuration.json", self.configuration, repo_root=self.root)
        self.summary = {"schema_version": 1, "complete": True, "image_count": len(self.index),
                        "configuration": self.configuration, "configuration_sha256": self.config_hash,
                        "index_sha256": sha256_file(self.bank / "index.csv")}
        write_json(self.bank / "summary.json", self.summary, repo_root=self.root)
        self.addCleanup(patch.stopall)
        patch("countermine.mining.local_feature_bank.code_hashes", return_value=self.static_code).start()
        patch("countermine.mining.local_feature_bank.frozen_provenance", return_value=self.scientific).start()
        patch("countermine.mining.local_feature_bank.load_bank_manifest", side_effect=lambda *a, **k: (self.manifest, self.frozen_config, self.historical)).start()

    def test_bank_index_identity_schema_and_atomic_marker(self):
        index, summary = load_feature_bank(self.bank, repo_root=self.root)
        self.assertEqual(index.to_dict("records"), self.index.to_dict("records"))
        self.assertEqual(summary, self.summary)
        self.summary["complete"] = False
        write_json(self.bank / "summary.json", self.summary, repo_root=self.root)
        with self.assertRaisesRegex(ValueError, "completion marker"):
            load_feature_bank(self.bank, repo_root=self.root)

    def test_missing_corrupt_feature_and_changed_image_hash_fail_closed(self):
        (self.bank / "features/000000.pt").write_bytes(b"different cache")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            load_feature_bank(self.bank, repo_root=self.root)
        (self.bank / "features/000000.pt").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            load_feature_bank(self.bank, repo_root=self.root)

    def test_historical_source_and_extraction_code_binding(self):
        self.historical["a"]["image_file_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "historical"):
            load_feature_bank(self.bank, repo_root=self.root)
        self.historical["a"]["image_file_sha256"] = "a" * 64
        with patch("countermine.mining.local_feature_bank.code_hashes", return_value={"source.py": "e" * 64}):
            with self.assertRaisesRegex(ValueError, "code provenance"):
                load_feature_bank(self.bank, repo_root=self.root)

    def test_index_schema_count_identity_and_filename_checks(self):
        validate_bank_index(self.index, self.manifest)
        for change in (
            lambda frame: frame.drop(columns="rgb_pixel_sha256"),
            lambda frame: frame.assign(row_index=[1, 0]),
            lambda frame: frame.assign(image_id=["b", "a"]),
            lambda frame: frame.assign(num_keypoints=[2049, 3]),
            lambda frame: frame.assign(feature_file=["features/a.pt", "features/b.pt"]),
            lambda frame: frame.assign(feature_file_sha256=["bad", "bad"]),
        ):
            with self.assertRaises(ValueError):
                validate_bank_index(change(self.index.copy()), self.manifest)

    def test_matching_requires_valid_bank_replay_marker(self):
        with self.assertRaisesRegex(ValueError, "replay first"):
            require_bank_validation(self.bank, self.summary, repo_root=self.root)
        marker = {"schema_version": 1, "complete": True, "passed": True,
                  "bank_summary_sha256": sha256_file(self.bank / "summary.json"),
                  "configuration_sha256": self.summary["configuration_sha256"],
                  "code_sha256": self.static_code,
                  "direct_feature_replay": {"image_count": 32, "passed": True, "exact_tensor_equality": True,
                                             "maximum_absolute_difference": {key: 0.0 for key in FEATURE_KEYS}},
                  "pair_replay": {"pair_count": 512, "passed": True}}
        write_json(self.bank / "validation.json", marker, repo_root=self.root)
        require_bank_validation(self.bank, self.summary, repo_root=self.root)
        for field, value in (("passed", False), ("bank_summary_sha256", "changed"), ("code_sha256", {})):
            write_json(self.bank / "validation.json", {**marker, field: value}, repo_root=self.root)
            with self.assertRaises(ValueError):
                require_bank_validation(self.bank, self.summary, repo_root=self.root)

    def test_current_source_bytes_are_checked_without_model_imports(self):
        dataset = self.root / "dataset"
        dataset.mkdir()
        for filename in ("a.png", "b.png"):
            Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(dataset / filename)
        self.index["image_file_sha256"] = [sha256_file(dataset / name) for name in ("a.png", "b.png")]
        verify_source_images(self.index, self.manifest, dataset)
        (dataset / "b.png").write_bytes(b"different encoded original")
        with self.assertRaisesRegex(ValueError, "source bytes differ"):
            verify_source_images(self.index, self.manifest, dataset)

    def test_bank_builder_resumes_completed_images_and_rejects_source_drift(self):
        import torch
        from types import SimpleNamespace
        dataset = self.root / "builder_dataset"
        dataset.mkdir()
        for filename in ("a.png", "b.png"):
            Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(dataset / filename)
        self.historical = {row["image_id"]: decode_original_rgb(dataset / row["relative_path"])[1]
                           for row in self.manifest.to_dict("records")}
        self.frozen_config.update(seed=42, image_policy={"geometry": [640, 480], "local_resize": None,
                                                       "real_rgb_only": True, "synthetic_images_used": False},
                                  vendor_provenance={"git_commit": "historical", "source_sha256": {}})
        calls = []
        runtime = self.frozen_config["matcher_provenance"]
        class FakeExtractor:
            def __init__(self, **kwargs):
                self.torch = torch
                self.provenance = runtime
            def extract_pixels(self, pixels):
                calls.append("extract")
                return {key: torch.from_numpy(value) for key, value in numpy_features().items()}
        fresh_bank = self.root / "cache/countermine_rgb/step2d/builder_bank"
        args = SimpleNamespace(bank_dir=fresh_bank, manifest=self.manifest_path,
                               dataset_root=dataset, device="cpu", seed=42)
        with patch("countermine.mining.local_feature_bank.source_provenance", return_value={"git_commit": "current", "source_sha256": {}}):
            first = build_feature_bank(args, repo_root=self.root, extractor_factory=FakeExtractor)
            self.assertEqual(first["image_count"], 2)
            self.assertEqual(len(calls), 2)
            self.assertEqual(build_feature_bank(args, repo_root=self.root, extractor_factory=FakeExtractor), first)
            self.assertEqual(len(calls), 2)
            Image.fromarray(np.full((480, 640, 3), 255, np.uint8)).save(dataset / "a.png")
            with self.assertRaisesRegex(ValueError, "differs from frozen Step 2A"):
                build_feature_bank(args, repo_root=self.root, extractor_factory=FakeExtractor)


if __name__ == "__main__":
    unittest.main()
