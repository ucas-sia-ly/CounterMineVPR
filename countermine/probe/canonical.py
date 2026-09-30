"""Deterministic image geometry for CounterMine diagnostic probes."""

from PIL import Image


CANONICAL_SIZE = 512


def canonicalize_probe(image: Image.Image) -> tuple[Image.Image, dict[str, int]]:
    """Center-crop to the largest square, then resize to 512 x 512 RGB.

    Odd excess pixels are retained on the right or bottom side of the source
    outside the crop. Crop coordinates refer to the original image pixels.
    The input image is not modified.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("canonicalize_probe requires a PIL image")

    original_width, original_height = image.size
    crop_size = min(original_width, original_height)
    if crop_size <= 0:
        raise ValueError("canonicalize_probe requires positive image dimensions")

    crop_left = (original_width - crop_size) // 2
    crop_top = (original_height - crop_size) // 2
    square = image.crop(
        (crop_left, crop_top, crop_left + crop_size, crop_top + crop_size)
    )
    output = square.convert("RGB").resize(
        (CANONICAL_SIZE, CANONICAL_SIZE), Image.Resampling.LANCZOS
    )
    metadata = {
        "original_width": original_width,
        "original_height": original_height,
        "crop_left": crop_left,
        "crop_top": crop_top,
        "crop_size": crop_size,
        "output_width": CANONICAL_SIZE,
        "output_height": CANONICAL_SIZE,
    }
    return output, metadata
