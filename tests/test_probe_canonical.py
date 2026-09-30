"""Checks for the shared square coordinate system used by diagnostic probes."""

import unittest

from PIL import Image, ImageDraw

from countermine.probe.canonical import canonicalize_probe


class ProbeCanonicalTest(unittest.TestCase):
    def test_returns_512_rgb_for_different_shapes_and_modes(self) -> None:
        for size in [(1, 1), (13, 9), (9, 13), (512, 512), (1024, 768)]:
            for mode in ["RGB", "L", "RGBA"]:
                with self.subTest(size=size, mode=mode):
                    result, metadata = canonicalize_probe(Image.new(mode, size))
                    self.assertEqual(result.size, (512, 512))
                    self.assertEqual(result.mode, "RGB")
                    self.assertEqual(metadata["output_width"], 512)
                    self.assertEqual(metadata["output_height"], 512)

    def test_wide_image_center_crop(self) -> None:
        image = Image.new("RGB", (1024, 512), "magenta")
        draw = ImageDraw.Draw(image)
        draw.rectangle((256, 0, 511, 511), fill="red")
        draw.rectangle((512, 0, 767, 511), fill="blue")

        result, metadata = canonicalize_probe(image)

        self.assertEqual(metadata, {
            "original_width": 1024,
            "original_height": 512,
            "crop_left": 256,
            "crop_top": 0,
            "crop_size": 512,
            "crop_long_axis_fraction": 0.5,
            "crop_area_fraction": 0.5,
            "output_width": 512,
            "output_height": 512,
        })
        expected = Image.new("RGB", (512, 512), "blue")
        ImageDraw.Draw(expected).rectangle((0, 0, 255, 511), fill="red")
        self.assertEqual(result.tobytes(), expected.tobytes())
        self.assertEqual(image.size, (1024, 512))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 255))

    def test_tall_image_center_crop(self) -> None:
        image = Image.new("RGB", (512, 1024), "magenta")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 256, 511, 511), fill="red")
        draw.rectangle((0, 512, 511, 767), fill="blue")

        result, metadata = canonicalize_probe(image)

        self.assertEqual(metadata["crop_left"], 0)
        self.assertEqual(metadata["crop_top"], 256)
        self.assertEqual(metadata["crop_size"], 512)
        self.assertEqual(metadata["original_width"], 512)
        self.assertEqual(metadata["original_height"], 1024)
        expected = Image.new("RGB", (512, 512), "blue")
        ImageDraw.Draw(expected).rectangle((0, 0, 511, 255), fill="red")
        self.assertEqual(result.tobytes(), expected.tobytes())

    def test_crop_retention_fractions_for_square_wide_tall_and_odd_dimensions(self) -> None:
        for size in [(512, 512), (1024, 512), (512, 1024), (13, 9), (9, 13)]:
            with self.subTest(size=size):
                width, height = size
                crop_size = min(size)
                result, metadata = canonicalize_probe(Image.new("RGB", size))
                self.addCleanup(result.close)
                self.assertEqual(metadata["crop_long_axis_fraction"], crop_size / max(size))
                self.assertEqual(metadata["crop_area_fraction"], crop_size ** 2 / (width * height))

    def test_square_image_is_unchanged_before_resize(self) -> None:
        image = Image.new("RGB", (512, 512), "blue")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 0, 511), fill="red")
        draw.rectangle((511, 0, 511, 511), fill="green")
        draw.rectangle((0, 0, 511, 0), fill="yellow")
        draw.rectangle((0, 511, 511, 511), fill="white")

        result, metadata = canonicalize_probe(image)

        self.assertEqual(result.tobytes(), image.tobytes())
        self.assertEqual(metadata["crop_left"], 0)
        self.assertEqual(metadata["crop_top"], 0)
        self.assertEqual(metadata["crop_size"], 512)

    def test_odd_excess_uses_floor_offset_deterministically(self) -> None:
        for size, expected_offset in [((517, 512), (2, 0)), ((512, 517), (0, 2))]:
            with self.subTest(size=size):
                image = Image.new("RGB", size, "blue")
                first_image, first_metadata = canonicalize_probe(image)
                second_image, second_metadata = canonicalize_probe(image)
                self.assertEqual(
                    (first_metadata["crop_left"], first_metadata["crop_top"]),
                    expected_offset,
                )
                self.assertEqual(first_metadata, second_metadata)
                self.assertEqual(first_image.tobytes(), second_image.tobytes())


if __name__ == "__main__":
    unittest.main()
