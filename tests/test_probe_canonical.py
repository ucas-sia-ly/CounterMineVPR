"""CPU checks for deterministic canonical geometry policies."""

import unittest

from PIL import Image, ImageDraw

from countermine.probe.canonical import canonicalize_probe, full_fov_512, square_crop_512


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
            "policy": "square_crop_512",
            "original_width": 1024,
            "original_height": 512,
            "crop_left": 256,
            "crop_top": 0,
            "crop_size": 512,
            "crop_long_axis_fraction": 0.5,
            "crop_area_fraction": 0.5,
            "output_width": 512,
            "output_height": 512,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "retained_area_fraction": 0.5,
            "retained_long_axis_fraction": 0.5,
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

    def test_square_policy_preserves_historical_pixel_transform_exactly(self) -> None:
        for size in [(13, 9), (9, 13), (1024, 768), (517, 512)]:
            for mode in ("RGB", "L", "RGBA"):
                with self.subTest(size=size, mode=mode):
                    image = Image.new(mode, size)
                    ImageDraw.Draw(image).rectangle((0, 0, size[0] // 2, size[1] - 1), fill=255)
                    width, height = size
                    side = min(size)
                    left, top = (width - side) // 2, (height - side) // 2
                    expected = image.crop((left, top, left + side, top + side)).convert("RGB").resize(
                        (512, 512), Image.Resampling.LANCZOS,
                    )
                    default, default_metadata = canonicalize_probe(image)
                    explicit, explicit_metadata = canonicalize_probe(image, "square_crop_512")
                    named, named_metadata = square_crop_512(image)
                    self.assertEqual(default.tobytes(), expected.tobytes())
                    self.assertEqual(explicit.tobytes(), expected.tobytes())
                    self.assertEqual(named.tobytes(), expected.tobytes())
                    self.assertEqual(default_metadata, explicit_metadata)
                    self.assertEqual(default_metadata, named_metadata)

    def test_full_fov_four_three_becomes_512_by_384_with_exact_metadata(self) -> None:
        for mode in ("RGB", "L", "RGBA"):
            with self.subTest(mode=mode):
                image = Image.new(mode, (1024, 768))
                result, metadata = full_fov_512(image)
                self.assertEqual(result.size, (512, 384))
                self.assertEqual(result.mode, "RGB")
                self.assertEqual(metadata, {
                    "policy": "full_fov_512",
                    "original_width": 1024,
                    "original_height": 768,
                    "output_width": 512,
                    "output_height": 384,
                    "scale_x": 0.5,
                    "scale_y": 0.5,
                    "retained_area_fraction": 1.0,
                    "retained_long_axis_fraction": 1.0,
                })

    def test_full_fov_keeps_both_edges_and_matches_whole_image_resize(self) -> None:
        image = Image.new("RGB", (1024, 768), "blue")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 127, 767), fill="red")
        draw.rectangle((896, 0, 1023, 767), fill="green")
        original = image.tobytes()
        result, _ = canonicalize_probe(image, "full_fov_512")
        expected = image.convert("RGB").resize((512, 384), Image.Resampling.LANCZOS)
        self.assertEqual(result.tobytes(), expected.tobytes())
        self.assertEqual(result.getpixel((0, 192)), (255, 0, 0))
        self.assertEqual(result.getpixel((511, 192)), (0, 128, 0))
        self.assertEqual(image.tobytes(), original)

    def test_full_fov_portrait_and_square_keep_aspect_ratio(self) -> None:
        for input_size, output_size in [((768, 1024), (384, 512)), ((640, 640), (512, 512))]:
            with self.subTest(input_size=input_size):
                result, metadata = full_fov_512(Image.new("RGB", input_size))
                self.assertEqual(result.size, output_size)
                self.assertEqual(metadata["scale_x"], metadata["scale_y"])
                self.assertEqual(metadata["retained_area_fraction"], 1.0)

    def test_full_fov_uses_stored_pixels_without_exif_transposition(self) -> None:
        image = Image.new("RGB", (512, 384), "blue")
        ImageDraw.Draw(image).rectangle((0, 0, 63, 383), fill="red")
        image.getexif()[274] = 6
        result, metadata = full_fov_512(image)
        self.assertEqual(result.size, (512, 384))
        self.assertEqual(result.tobytes(), image.tobytes())
        self.assertEqual(metadata["original_width"], 512)
        self.assertEqual(metadata["original_height"], 384)

    def test_full_fov_rejects_unsupported_geometry_without_fallback(self) -> None:
        for size in ((1024, 769), (640, 480 + 1), (1000, 500 + 1), (512, 300)):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "exact aspect-preserving"):
                    full_fov_512(Image.new("RGB", size))

    def test_invalid_policy_and_nonimage_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown canonical policy"):
            canonicalize_probe(Image.new("RGB", (512, 384)), "unknown")
        for function in (canonicalize_probe, square_crop_512, full_fov_512):
            with self.subTest(function=function.__name__):
                with self.assertRaises(TypeError):
                    function(None)


if __name__ == "__main__":
    unittest.main()
