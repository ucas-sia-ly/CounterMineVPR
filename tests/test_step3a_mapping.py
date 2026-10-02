"""CPU-only tests for metadata-derived training place identities."""

from dataclasses import FrozenInstanceError
import unittest

import pandas as pd

from countermine.training.gsv_place_mapping import PlaceRecord, build_place_mapping


class MetadataDataset:
    """Numeric labels deliberately do not encode original place IDs."""

    def __init__(self, rows, places=None):
        self.dataframe = pd.DataFrame(rows).set_index("internal_id")
        self.places_ids = list(places if places is not None else pd.unique(self.dataframe.index))

    def __len__(self):
        return len(self.places_ids)

    def get_img_name(self, row):
        return f"{row['city_id']}_{int(row['original_id']):07d}_2021_05_view.jpg"


class PlaceMappingTests(unittest.TestCase):
    def dataset(self):
        return MetadataDataset([
            {"internal_id": 987654, "city_id": "Boston", "original_id": 5994},
            {"internal_id": 987654, "city_id": "Boston", "original_id": 5994},
            {"internal_id": 222222, "city_id": "London", "original_id": 2184},
            {"internal_id": 222222, "city_id": "London", "original_id": 2184},
        ])

    def test_canonical_identity_comes_from_dataset_image_metadata(self):
        records, audit = build_place_mapping(self.dataset(), ("Boston:0005994", "London:0002184"))
        self.assertEqual(records, (
            PlaceRecord(0, 987654, "Boston", "Boston:0005994"),
            PlaceRecord(1, 222222, "London", "London:0002184"),
        ))
        self.assertEqual(audit["total_training_places"], 2)
        self.assertEqual(audit["boston_training_places"], 1)
        self.assertEqual(audit["london_training_places"], 1)
        self.assertEqual(audit["mapped_graph_places"], 2)
        self.assertEqual(audit["unmapped_graph_places"], 0)
        self.assertTrue(audit["one_to_one_mapping"])
        with self.assertRaises(FrozenInstanceError):
            records[0].city_id = "London"

    def test_duplicate_canonical_mapping_rejected(self):
        dataset = MetadataDataset([
            {"internal_id": 7, "city_id": "Boston", "original_id": 1},
            {"internal_id": 8, "city_id": "Boston", "original_id": 1},
        ])
        with self.assertRaisesRegex(ValueError, "Duplicate canonical"):
            build_place_mapping(dataset)

    def test_duplicate_internal_id_rejected(self):
        dataset = self.dataset()
        dataset.places_ids = [987654, 987654]
        with self.assertRaisesRegex(ValueError, "Duplicate internal"):
            build_place_mapping(dataset)

    def test_missing_graph_place_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unmapped CounterMine"):
            build_place_mapping(self.dataset(), ("Boston:0000000",))

    def test_every_row_city_checked(self):
        dataset = self.dataset()
        dataset.dataframe.loc[987654, "city_id"] = ["Boston", "London"]
        with self.assertRaisesRegex(ValueError, "Rows disagree"):
            build_place_mapping(dataset)

    def test_every_row_original_identity_checked(self):
        dataset = self.dataset()
        dataset.dataframe.loc[987654, "original_id"] = [5994, 5995]
        with self.assertRaisesRegex(ValueError, "Rows disagree"):
            build_place_mapping(dataset)

    def test_image_name_city_must_agree(self):
        dataset = self.dataset()
        dataset.get_img_name = lambda row: "London_0005994_2021_05_view.jpg"
        with self.assertRaisesRegex(ValueError, "Image-name city"):
            build_place_mapping(dataset)

    def test_uid_format_must_match_frozen_graph(self):
        with self.assertRaisesRegex(ValueError, "Invalid canonical"):
            build_place_mapping(self.dataset(), ("Boston:5994",))

    def test_single_row_place_is_supported(self):
        dataset = MetadataDataset([{"internal_id": 3, "city_id": "London", "original_id": 5}])
        records, _ = build_place_mapping(dataset)
        self.assertEqual(records[0].canonical_place_uid, "London:0000005")


if __name__ == "__main__":
    unittest.main()
