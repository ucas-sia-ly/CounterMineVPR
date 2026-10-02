"""CPU-only original-RGB figure/montage selection, bounds and hash binding."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image

from test_countermine_graph import fixture
from test_full_graph_analysis import full_fixture
from countermine.mining.full_graph_analysis import analyze_full_graphs
from countermine.mining.full_graph_visuals import (
    ARTIFACT_NAMES, BankRGBReader, create_full_graph_visuals, curate_full_graph_artifacts,
    select_component_sections, select_hub_sections,
)


class FullVisualTests(unittest.TestCase):
    def test_hubs_and_components_use_strong_core_edges_and_bounded_representatives(self):
        tables = full_fixture()
        report, _, place_audit, components = analyze_full_graphs(*tables)
        hubs = select_hub_sections(tables[1], place_audit)
        self.assertEqual({section["slice"] for section in hubs}, {"core_q95", "core_q99"})
        for section in hubs:
            self.assertEqual(section["core_degree"], 1)
            self.assertLessEqual(len(section["image_edges"]), 4)
            for row in section["image_edges"]:
                self.assertIn(section["place_uid"], (row["place_uid_a"], row["place_uid_b"]))
                self.assertTrue(row[section["slice"]])
        sections = select_component_sections(tables[1], components)
        self.assertEqual({section["slice"] for section in sections}, {"core_q95", "core_q975", "core_q99", "core_q995"})
        self.assertTrue(all(len(section["image_edges"]) <= 3 for section in sections))
        self.assertTrue(all(len(section["representatives"]) <= 8 for section in sections))

    def prepare(self, root):
        tables = full_fixture()
        _, manifest = fixture()
        dataset = root / "data/gsv-cities"
        dataset.mkdir(parents=True)
        rows = []
        for number, row in enumerate(manifest.to_dict("records")):
            path = dataset / (row["image_id"] + ".png")
            with Image.new("RGB", (640, 480), (number * 20, 100, 180)) as image:
                image.save(path)
                pixels = hashlib.sha256(image.tobytes()).hexdigest()
            rows.append({"image_id": row["image_id"], "image_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "rgb_pixel_sha256": pixels})
        manifest["relative_path"] = manifest.image_id + ".png"
        manifest_path = root / "cache/gsv_mini/manifest.csv"
        manifest_path.parent.mkdir(parents=True)
        manifest.to_csv(manifest_path, index=False)
        bank = root / "cache/countermine_rgb/step2d/aliked_bank"
        bank.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(bank / "index.csv", index=False)
        return tables, manifest_path, dataset, bank

    def test_all_ten_figures_render_and_curate_under_step2d_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, manifest, dataset, bank = self.prepare(root)
            report, _, place_audit, components = analyze_full_graphs(*tables)
            output = root / "outputs/step2d"
            metadata = create_full_graph_visuals(*tables, report, place_audit, components,
                                                 manifest, dataset, bank, output, repo_root=root)
            self.assertEqual(len(metadata["artifact_names"]), 10)
            self.assertEqual(len(metadata["top100_pair_uids"]), 4)
            self.assertTrue(metadata["scientific_statistics_use_all_edges"])
            for name in ARTIFACT_NAMES:
                with Image.open(output / name) as image:
                    self.assertGreater(image.width, 500)
                    self.assertGreater(image.height, 100)
            hashes = curate_full_graph_artifacts(output, root / "docs/audits", repo_root=root)
            self.assertEqual(len(hashes), 10)
            self.assertTrue(all(name.startswith("step2d_") for name in hashes))
            with self.assertRaisesRegex(ValueError, "Step 2D writes"):
                curate_full_graph_artifacts(output, root / "outputs/step2c", repo_root=root)

    def test_reader_detects_changed_original_pixels_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, dataset, bank = self.prepare(root)
            reader = BankRGBReader(manifest, dataset, bank)
            with reader.read("a", 320) as image:
                self.assertEqual(image.size, (320, 240))
            with Image.new("RGB", (640, 480), "red") as changed:
                changed.save(dataset / "a.png")
            with self.assertRaisesRegex(ValueError, "image bytes differ"):
                reader.read("a")


if __name__ == "__main__":
    unittest.main()
