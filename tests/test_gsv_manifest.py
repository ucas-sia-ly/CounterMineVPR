"""CPU-only tests for the portable GSV-Cities mini manifest."""

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from countermine.mining.gsv_manifest import build_gsv_manifest, reconstruct_filename


FIELDNAMES = (
    "place_id",
    "city_id",
    "year",
    "month",
    "northdeg",
    "lat",
    "lon",
    "panoid",
)


def salad_filename(row):
    """Build fixture paths independently from the implementation under test."""
    return (
        f"{row['city_id']}_{int(row['place_id']):07d}_"
        f"{int(row['year']):04d}_{int(row['month']):02d}_"
        f"{int(row['northdeg']):03d}_{row['lat']}_{row['lon']}_"
        f"{row['panoid']}.jpg"
    )


class GSVManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        (self.root / "Dataframes").mkdir()
        (self.root / "Images").mkdir()
        self.rows = {}
        for city in ("London", "Boston"):
            self.write_city(city)

    def write_city(self, city, place_ids=(13, 25, 42, 77), images_per_place=6):
        rows = []
        image_directory = self.root / "Images" / city
        image_directory.mkdir(exist_ok=True)
        for place_id in place_ids:
            for image_number in range(images_per_place):
                row = {
                    "place_id": place_id,
                    "city_id": city,
                    "year": 2020,
                    "month": image_number + 1,
                    "northdeg": image_number,
                    "lat": 51.5,
                    "lon": -0.12,
                    "panoid": f"pano_{place_id}_{image_number}",
                }
                rows.append(row)
                (image_directory / salad_filename(row)).touch()
        self.rows[city] = rows
        self.write_csv(city, rows)
        return rows

    def write_csv(self, city, rows):
        with (self.root / "Dataframes" / f"{city}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as output:
            writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def test_filename_reconstruction_matches_salad_convention(self):
        row = {
            "place_id": 13,
            "city_id": "London",
            "year": 2020,
            "month": 3,
            "northdeg": 7,
            "lat": 51.5,
            "lon": -0.12,
            "panoid": "panoA",
        }
        self.assertEqual(
            reconstruct_filename(row),
            "London_0000013_2020_03_007_51.5_-0.12_panoA.jpg",
        )

    def test_place_uid_uses_original_place_id_and_city_id(self):
        manifest = build_gsv_manifest(
            self.root, cities=("London", "Boston"), places_per_city=4
        )
        rows_with_place_13 = manifest[manifest["original_place_id"] == 13]
        self.assertEqual(
            set(rows_with_place_13["place_uid"]),
            {"London:0000013", "Boston:0000013"},
        )
        self.assertEqual(set(manifest["image_id"]), set(manifest["relative_path"]))
        self.assertTrue(
            manifest["relative_path"].str.startswith("Images/").all()
        )
        self.assertFalse(manifest["relative_path"].str.startswith("/").any())

        reversed_cities = build_gsv_manifest(
            self.root, cities=("Boston", "London"), places_per_city=4
        )
        pd.testing.assert_frame_equal(manifest, reversed_cities)

    def test_sampling_is_invariant_to_csv_row_order(self):
        original = build_gsv_manifest(
            self.root, places_per_city=2, images_per_place=4, seed=42
        )
        for city in ("London", "Boston"):
            self.write_csv(city, list(reversed(self.rows[city])))
        reordered = build_gsv_manifest(
            self.root, places_per_city=2, images_per_place=4, seed=42
        )
        pd.testing.assert_frame_equal(original, reordered)
        self.assertEqual(len(original), 2 * 2 * 4)
        self.assertEqual(original.groupby("place_uid").size().unique().tolist(), [4])

    def test_row_index_matches_deterministic_descriptor_order(self):
        manifest = build_gsv_manifest(
            self.root, places_per_city=2, images_per_place=4, seed=42
        )
        self.assertEqual(manifest["row_index"].tolist(), list(range(len(manifest))))
        expected_order = manifest.sort_values(
            ["city_id", "place_uid", "relative_path"], kind="stable"
        )["relative_path"].tolist()
        self.assertEqual(manifest["relative_path"].tolist(), expected_order)

    def test_missing_selected_image_fails_by_default(self):
        rows = self.write_city("London", place_ids=(13,), images_per_place=4)
        (self.root / "Images" / "London" / salad_filename(rows[0])).unlink()
        with self.assertRaises(FileNotFoundError):
            build_gsv_manifest(
                self.root, cities=("London",), places_per_city=1, images_per_place=4
            )

    def test_allow_missing_discards_incomplete_place(self):
        rows = self.write_city("London", place_ids=(13, 25), images_per_place=4)
        (self.root / "Images" / "London" / salad_filename(rows[0])).unlink()
        manifest = build_gsv_manifest(
            self.root,
            cities=("London",),
            places_per_city=2,
            images_per_place=4,
            allow_missing=True,
        )
        self.assertEqual(manifest["place_uid"].unique().tolist(), ["London:0000025"])
        self.assertEqual(manifest["row_index"].tolist(), [0, 1, 2, 3])
        self.assertEqual(manifest.attrs["missing_image_count"], 1)

    def test_same_seed_produces_identical_manifest(self):
        first = build_gsv_manifest(
            self.root, places_per_city=2, images_per_place=4, seed=17
        )
        second = build_gsv_manifest(
            self.root, places_per_city=2, images_per_place=4, seed=17
        )
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
