"""CPU-only checks for exact, deterministic RGB candidate generation."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from countermine.mining.candidate_miner import (
    ChunkedCandidateMiner,
    haversine_distance_m,
    mine_candidates,
    stable_pair_uid,
)

# CandidateMinerTests是一个测试类，用于测试候选挖掘器
class CandidateMinerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        rng = np.random.default_rng(104)
        descriptors = rng.normal(size=(8, 5)).astype(np.float32)
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
        self.path = self.save_descriptors(descriptors)
        self.manifest = self.make_manifest(8)
        self.manifest["place_uid"] = [f"place-{i // 2}" for i in range(8)]

# save_descriptors方法用于将描述符保存到临时目录中
    def save_descriptors(self, descriptors, name="descriptors_fp16.npy"):
        path = self.root / name
        np.save(path, np.asarray(descriptors, dtype=np.float16))
        return path

    @staticmethod
    def make_manifest(count):
        return pd.DataFrame(
            {
                "row_index": np.arange(count, dtype=np.int64),
                "image_id": [f"image-{i:03d}" for i in range(count)],
                "place_uid": [f"place-{i}" for i in range(count)],
                "city_id": ["Boston" if i % 2 == 0 else "London" for i in range(count)],
                "lat": np.arange(count, dtype=np.float64) * 0.01,
                "lon": np.arange(count, dtype=np.float64) * 0.02,
            }
        )

    @staticmethod
    def scalar_distance(lat1, lon1, lat2, lon2):
        """Independent haversine oracle, used only for these tiny test inputs."""
        lat1, lon1, lat2, lon2 = np.deg2rad([lat1, lon1, lat2, lon2])
        a = np.sin((lat2 - lat1) / 2) ** 2
        a += np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        return 6_371_000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def brute_force(self, path, manifest, top_k, min_distance=0.0):
        """A full similarity matrix is acceptable only in this tiny oracle."""
        descriptors = np.load(path).astype(np.float32)
        similarities = descriptors @ descriptors.T
        rows = []
        for query in range(len(manifest)):
            candidates = []
            for negative in range(len(manifest)):
                if manifest.iloc[query]["place_uid"] == manifest.iloc[negative]["place_uid"]:
                    continue
                distance = self.scalar_distance(
                    manifest.iloc[query]["lat"],
                    manifest.iloc[query]["lon"],
                    manifest.iloc[negative]["lat"],
                    manifest.iloc[negative]["lon"],
                )
                if distance < min_distance:
                    continue
                candidates.append((negative, float(similarities[query, negative])))
            candidates.sort(key=lambda candidate: (-candidate[1], candidate[0]))
            self.assertGreaterEqual(len(candidates), top_k)
            rows.extend((query, negative, rank, similarity) for rank, (negative, similarity)
                        in enumerate(candidates[:top_k], start=1))
        return pd.DataFrame(
            rows, columns=["query_row_index", "negative_row_index", "rank", "similarity"]
        )

    def mine(self, **kwargs):
        arguments = {
            "top_k": 3,
            "query_chunk_size": 2,
            "reference_chunk_size": 3,
            "device": "cpu",
        }
        arguments.update(kwargs)
        return mine_candidates(self.path, self.manifest, **arguments)

    def assert_matches_oracle(self, actual, expected):
        for column in ("query_row_index", "negative_row_index", "rank"):
            np.testing.assert_array_equal(actual[column].to_numpy(), expected[column].to_numpy())
        np.testing.assert_allclose(actual["similarity"], expected["similarity"], rtol=1e-6, atol=1e-7)

    def test_chunked_retrieval_matches_brute_force_and_rank_order(self):
        actual = self.mine()
        expected = self.brute_force(self.path, self.manifest, top_k=3)
        self.assert_matches_oracle(actual, expected)
        self.assertEqual(len(actual), len(self.manifest) * 3)
        self.assertTrue(np.isfinite(actual["similarity"]).all())
        for _, group in actual.groupby("query_row_index", sort=False):
            self.assertEqual(group["rank"].tolist(), [1, 2, 3])
            self.assertEqual(
                list(zip(group["similarity"], group["negative_row_index"])),
                sorted(zip(group["similarity"], group["negative_row_index"]),
                       key=lambda candidate: (-candidate[0], candidate[1])),
            )

    def test_same_place_and_self_are_always_excluded(self):
        actual = self.mine(top_k=6, reference_chunk_size=1)
        self.assertFalse((actual["query_place_uid"] == actual["negative_place_uid"]).any())
        self.assertFalse((actual["query_row_index"] == actual["negative_row_index"]).any())
        for query, group in actual.groupby("query_row_index"):
            self.assertEqual(len(group), 6)
            self.assertEqual(
                set(group["negative_row_index"]),
                set(self.manifest.loc[self.manifest["place_uid"] != self.manifest.iloc[query]["place_uid"],
                                      "row_index"]),
            )

    def test_candidate_table_metadata_remains_aligned_with_descriptor_rows(self):
        # Pandas' own index is unrelated to the descriptor row_index column.
        self.manifest.index = np.arange(100, 108)
        actual = self.mine()
        self.assertEqual(
            set(actual.columns),
            {
                "query_row_index", "negative_row_index", "query_image_id", "negative_image_id",
                "query_place_uid", "negative_place_uid", "query_city_id", "negative_city_id",
                "rank", "similarity", "geo_distance_m", "same_city", "pair_uid",
            },
        )
        self.assertEqual(actual["query_row_index"].tolist(), np.repeat(np.arange(8), 3).tolist())
        for row in actual.itertuples(index=False):
            query = self.manifest.iloc[row.query_row_index]
            negative = self.manifest.iloc[row.negative_row_index]
            self.assertEqual(row.query_image_id, query["image_id"])
            self.assertEqual(row.negative_image_id, negative["image_id"])
            self.assertEqual(row.query_place_uid, query["place_uid"])
            self.assertEqual(row.negative_place_uid, negative["place_uid"])
            self.assertEqual(row.query_city_id, query["city_id"])
            self.assertEqual(row.negative_city_id, negative["city_id"])
            self.assertEqual(row.same_city, query["city_id"] == negative["city_id"])
            np.testing.assert_allclose(
                row.geo_distance_m,
                self.scalar_distance(query["lat"], query["lon"], negative["lat"], negative["lon"]),
                rtol=2e-6,
                atol=1e-6,
            )
            self.assertEqual(row.pair_uid, stable_pair_uid(row.query_image_id, row.negative_image_id))

    def test_geographic_distance_filter_matches_brute_force(self):
        self.manifest = self.make_manifest(8)
        self.manifest["lat"] = [0.0, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        self.manifest["lon"] = 0.0
        actual = self.mine(min_geo_distance_m=500.0)
        expected = self.brute_force(self.path, self.manifest, top_k=3, min_distance=500.0)
        self.assert_matches_oracle(actual, expected)
        self.assertTrue((actual["geo_distance_m"] >= 500.0).all())
        forbidden = ((actual["query_row_index"] == 0) & (actual["negative_row_index"] == 1))
        forbidden |= ((actual["query_row_index"] == 1) & (actual["negative_row_index"] == 0))
        self.assertFalse(forbidden.any())
        # A zero threshold permits zero-distance images from different places.
        self.manifest["lat"] = 0.0
        self.manifest["lon"] = 0.0
        zero_threshold = self.mine(top_k=7, min_geo_distance_m=0.0)
        self.assertEqual(len(zero_threshold), 56)
        self.assertTrue((zero_threshold["geo_distance_m"] == 0.0).all())

    def test_haversine_supports_broadcasting_and_known_distances(self):
        distances = haversine_distance_m(
            np.array([0.0, 45.0])[:, None],
            np.array([0.0, 12.0])[:, None],
            np.array([0.0, 45.0])[None, :],
            np.array([1.0, 12.0])[None, :],
        )
        self.assertEqual(distances.shape, (2, 2))
        # Common mean-earth radii (6371 km or 6371.0088 km) are both valid.
        np.testing.assert_allclose(distances[0, 0], 111_194.92664455874, rtol=2e-6)
        self.assertEqual(distances[1, 1], 0.0)
        for i, (lat, lon) in enumerate(((0.0, 0.0), (45.0, 12.0))):
            for j, (other_lat, other_lon) in enumerate(((0.0, 1.0), (45.0, 12.0))):
                np.testing.assert_allclose(
                    distances[i, j], self.scalar_distance(lat, lon, other_lat, other_lon),
                    rtol=2e-6, atol=1e-6,
                )

    def test_pair_uid_is_symmetric_stable_sha256(self):
        uid = stable_pair_uid("image-a", "image-b")
        self.assertEqual(uid, stable_pair_uid("image-b", "image-a"))
        self.assertEqual(uid, stable_pair_uid("image-a", "image-b"))
        self.assertRegex(uid, r"^[0-9a-f]{64}$")
        self.assertNotEqual(uid, stable_pair_uid("image-a", "image-c"))

    def test_ties_and_different_chunk_sizes_produce_identical_results(self):
        # Exact unit axes create tied scores on both sides of a block boundary.
        descriptors = np.eye(4, dtype=np.float32)[[0, 0, 0, 1, 1, 2, 2, 3]]
        path = self.save_descriptors(descriptors, name="tied.npy")
        manifest = self.make_manifest(8)
        expected = self.brute_force(path, manifest, top_k=4)
        reference = None
        for query_size, reference_size in ((1, 1), (2, 3), (3, 2), (8, 4), (20, 20)):
            with self.subTest(query_chunk_size=query_size, reference_chunk_size=reference_size):
                actual = mine_candidates(
                    path, manifest, top_k=4, query_chunk_size=query_size,
                    reference_chunk_size=reference_size, device="cpu",
                )
                self.assert_matches_oracle(actual, expected)
                if reference is None:
                    reference = actual
                else:
                    pd.testing.assert_frame_equal(actual, reference, check_exact=True)
        self.assertEqual(reference.loc[reference["query_row_index"] == 0, "negative_row_index"].tolist(),
                         [1, 2, 3, 4])

    def test_streamed_chunks_equal_collected_result(self):
        miner = ChunkedCandidateMiner(
            self.path, self.manifest, top_k=3, query_chunk_size=3,
            reference_chunk_size=2, device="cpu",
        )
        chunks = list(miner.iter_candidates())
        self.assertEqual([chunk["query_row_index"].nunique() for chunk in chunks], [3, 3, 2])
        actual = pd.concat(chunks, ignore_index=True)
        pd.testing.assert_frame_equal(actual, miner.mine(), check_exact=True)

    def test_random_duplicate_descriptors_keep_tie_order_across_chunk_sizes(self):
        rng = np.random.default_rng(212)
        sources = rng.normal(size=(24, 64)).astype(np.float32)
        sources /= np.linalg.norm(sources, axis=1, keepdims=True)
        # Identical reference vectors appear in several differently sized blocks.
        descriptors = np.concatenate((sources, sources[:16]), axis=0)
        path = self.save_descriptors(descriptors, name="random_duplicates.npy")
        manifest = self.make_manifest(len(descriptors))
        expected = self.brute_force(path, manifest, top_k=7)
        reference = None
        for query_size, reference_size in ((1, 1), (3, 4), (6, 9), (13, 17), (40, 40)):
            with self.subTest(query_chunk_size=query_size, reference_chunk_size=reference_size):
                actual = mine_candidates(
                    path, manifest, top_k=7, query_chunk_size=query_size,
                    reference_chunk_size=reference_size, device="cpu",
                )
                self.assert_matches_oracle(actual, expected)
                if reference is None:
                    reference = actual
                else:
                    pd.testing.assert_frame_equal(actual, reference, check_exact=True)

    def test_descriptors_are_opened_as_read_only_memmap(self):
        original_load = np.load
        with mock.patch("countermine.mining.candidate_miner.np.load", wraps=original_load) as load:
            self.mine()
        self.assertTrue(load.call_args_list)
        for call in load.call_args_list:
            self.assertEqual(call.kwargs.get("mmap_mode"), "r")

    def test_invalid_manifest_alignment_and_required_columns_fail(self):
        shuffled = self.manifest.iloc[::-1].reset_index(drop=True)
        duplicate = self.manifest.copy()
        duplicate.loc[1, "row_index"] = 0
        gap = self.manifest.copy()
        gap.loc[7, "row_index"] = 9
        for manifest in (shuffled, duplicate, gap):
            with self.subTest(indices=manifest["row_index"].tolist()):
                with self.assertRaisesRegex(ValueError, "row_index"):
                    ChunkedCandidateMiner(self.path, manifest, top_k=3, device="cpu")
        for column in ("row_index", "image_id", "place_uid", "city_id", "lat", "lon"):
            with self.subTest(missing_column=column):
                with self.assertRaisesRegex(ValueError, column):
                    ChunkedCandidateMiner(self.path, self.manifest.drop(columns=column),
                                          top_k=3, device="cpu")

    def test_descriptor_shape_and_parameter_errors_fail_clearly(self):
        short_path = self.save_descriptors(np.eye(5), name="short.npy")
        with self.assertRaisesRegex(ValueError, "(row|manifest|shape)"):
            ChunkedCandidateMiner(short_path, self.manifest, top_k=3, device="cpu")
        vector_path = self.save_descriptors(np.ones(8), name="vector.npy")
        with self.assertRaises(ValueError):
            ChunkedCandidateMiner(vector_path, self.manifest, top_k=3, device="cpu")
        for parameter in ("top_k", "query_chunk_size", "reference_chunk_size"):
            with self.subTest(parameter=parameter):
                with self.assertRaisesRegex(ValueError, parameter):
                    self.mine(**{parameter: 0})
        with self.assertRaisesRegex(ValueError, "min_geo_distance_m"):
            self.mine(min_geo_distance_m=-1)

    def test_insufficient_valid_references_fail_instead_of_emitting_invalid_pairs(self):
        with self.assertRaises(ValueError):
            self.mine(top_k=7)
        with self.assertRaises(ValueError):
            self.mine(top_k=1, min_geo_distance_m=20_000_000.0)


if __name__ == "__main__":
    unittest.main()
