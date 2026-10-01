"""CPU scalar cache checks for complete and interrupted local measurements."""

import copy
from types import SimpleNamespace
import unittest

import numpy as np

from countermine.probe.step3a_pipeline import source_relit_R8, validate_local_records


class Step3ALocalCacheTests(unittest.TestCase):
    def setUp(self):
        self.population = {"images": [{"image_id": name} for name in ("q", "p", "h", "r")],
                           "triplets": [{"q_image_id": "q", "p_image_id": "p",
                                         "n_hard_image_id": "h", "n_random_image_id": "r"}]}
        self.saved = {"per_image": [{"image_id": name, "source_relit_R8": .4,
                                     "fidelity_R8_pass": True, "rgb_mae_normalized": .2,
                                     "abs_luma_mean_delta": 10.0} for name in ("q", "p", "h", "r")],
                      "cross_place": [{"q_image_id": "q", "negative_image_id": negative,
                                       "phase": phase, "num_matches": 2, "num_keypoints_source": 8,
                                       "num_keypoints_target": 4, "cross_match_ratio": .5,
                                       "matched_source_cell_coverage": 2 / 64,
                                       "matched_target_cell_coverage": 1 / 64}
                                      for negative in ("h", "r") for phase in ("rgb", "z")]}

    def test_complete_and_partial_scalar_caches(self):
        validate_local_records(self.saved, self.population, complete=True)
        partial = {"per_image": self.saved["per_image"][:1], "cross_place": []}
        validate_local_records(partial, self.population, complete=False)
        with self.assertRaises(ValueError):
            validate_local_records(partial, self.population, complete=True)

    def test_R8_uses_frozen_float64_distance_at_boundary(self):
        source = np.array([[0, 0], [0, 0]], dtype=np.float32)
        relit = np.array([[8, .001], [8, 0]], dtype=np.float32)
        self.assertEqual(np.linalg.norm(relit[0]).item(), 8.0)
        matches = SimpleNamespace(points_source=source, points_relit=relit,
                                  num_keypoints_source=2, num_keypoints_relit=2)
        self.assertEqual(source_relit_R8(matches), .5)

    def test_duplicate_unknown_images_or_pairs_rejected(self):
        for kind in ("per_image", "cross_place"):
            for operation in ("duplicate", "unknown"):
                with self.subTest(kind=kind, operation=operation):
                    changed = copy.deepcopy(self.saved)
                    if operation == "duplicate":
                        changed[kind].append(changed[kind][0])
                    else:
                        changed[kind][0]["image_id" if kind == "per_image" else "negative_image_id"] = "unknown"
                    with self.assertRaises(ValueError):
                        validate_local_records(changed, self.population, complete=False)

    def test_fidelity_and_cross_support_definitions_rejected_if_changed(self):
        changes = (("per_image", "source_relit_R8", float("nan")),
                   ("per_image", "fidelity_R8_pass", False),
                   ("per_image", "rgb_mae_normalized", -1),
                   ("cross_place", "num_matches", 5),
                   ("cross_place", "num_keypoints_source", True),
                   ("cross_place", "cross_match_ratio", .4),
                   ("cross_place", "matched_target_cell_coverage", .1))
        for kind, key, value in changes:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.saved)
                changed[kind][0][key] = value
                with self.assertRaises(ValueError):
                    validate_local_records(changed, self.population, complete=False)


if __name__ == "__main__":
    unittest.main()
