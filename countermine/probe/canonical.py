"""Deterministic image geometry for CounterMine diagnostic probes."""

from PIL import Image


CANONICAL_SIZE = 512
CANONICAL_POLICIES = ("square_crop_512", "full_fov_512", "native_full_fov")


def _validate_image(image: Image.Image) -> tuple[int, int]:
    if not isinstance(image, Image.Image):
        raise TypeError("canonicalize_probe requires a PIL image")
    if min(image.size) <= 0:
        raise ValueError("canonicalize_probe requires positive image dimensions")
    return image.size


def square_crop_512(image: Image.Image) -> tuple[Image.Image, dict[str, str | int | float]]:
    """Center-crop to the largest square, then resize to 512 x 512 RGB.

    Odd excess pixels are retained on the right or bottom side of the source
    outside the crop. Crop coordinates refer to the original image pixels.
    The input image is not modified.
    """
    original_width, original_height = _validate_image(image)
    crop_size = min(original_width, original_height)

    crop_left = (original_width - crop_size) // 2
    crop_top = (original_height - crop_size) // 2
    square = image.crop(
        (crop_left, crop_top, crop_left + crop_size, crop_top + crop_size)
    )
    output = square.convert("RGB").resize(
        (CANONICAL_SIZE, CANONICAL_SIZE), Image.Resampling.LANCZOS
    )
    metadata = {
        "policy": "square_crop_512",
        "original_width": original_width,
        "original_height": original_height,
        "crop_left": crop_left,
        "crop_top": crop_top,
        "crop_size": crop_size,
        "crop_long_axis_fraction": crop_size / max(original_width, original_height),
        "crop_area_fraction": crop_size ** 2 / (original_width * original_height),
        "output_width": CANONICAL_SIZE,
        "output_height": CANONICAL_SIZE,
        "scale_x": CANONICAL_SIZE / crop_size,
        "scale_y": CANONICAL_SIZE / crop_size,
        "retained_area_fraction": crop_size ** 2 / (original_width * original_height),
        "retained_long_axis_fraction": crop_size / max(original_width, original_height),
    }
    return output, metadata


def full_fov_512(image: Image.Image) -> tuple[Image.Image, dict[str, str | int | float]]:
    """Resize the complete RGB image with exact aspect ratio and a 512 long side.

    The shorter side must be an exact integer multiple of 64. Unsupported
    aspect ratios fail instead of changing the field of view, adding padding,
    or introducing stretch. Coordinates use the stored pixels; EXIF orientation
    is deliberately never applied. The input image is not modified.
    """
    original_width, original_height = _validate_image(image)
    longest = max(original_width, original_height)
    output_width, width_remainder = divmod(original_width * CANONICAL_SIZE, longest)
    output_height, height_remainder = divmod(original_height * CANONICAL_SIZE, longest)
    if width_remainder or height_remainder or output_width % 64 or output_height % 64:
        raise ValueError(
            "full_fov_512 requires an exact aspect-preserving 512-long-side "
            "geometry with width and height divisible by 64"
        )
    output = image.convert("RGB").resize(
        (output_width, output_height), Image.Resampling.LANCZOS
    )
    metadata = {
        "policy": "full_fov_512",
        "original_width": original_width,
        "original_height": original_height,
        "output_width": output_width,
        "output_height": output_height,
        "scale_x": output_width / original_width,
        "scale_y": output_height / original_height,
        "retained_area_fraction": 1.0,
        "retained_long_axis_fraction": 1.0,
    }
    return output, metadata


def native_full_fov(image: Image.Image) -> tuple[Image.Image, dict[str, str | int | float]]:
    """Copy the frozen GSV-Cities 640 x 480 stored pixels into RGB.

    No EXIF transposition, crop, resize, padding, or stretch is applied. Other
    source geometries fail rather than silently changing the experiment.
    """
    original_width, original_height = _validate_image(image)
    if (original_width, original_height) != (640, 480):
        raise ValueError(
            "native_full_fov requires source dimensions exactly 640x480; "
            f"received {original_width}x{original_height}"
        )
    output = image.copy() if image.mode == "RGB" else image.convert("RGB")
    metadata = {
        "policy": "native_full_fov",
        "original_width": original_width,
        "original_height": original_height,
        "output_width": original_width,
        "output_height": original_height,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "retained_area_fraction": 1.0,
        "retained_long_axis_fraction": 1.0,
    }
    return output, metadata


def canonicalize_probe(
    image: Image.Image, policy: str = "square_crop_512",
) -> tuple[Image.Image, dict[str, str | int | float]]:
    """Apply an explicit geometry policy; the historical square default is kept.

    All policies operate on stored pixels without EXIF transposition.
    """
    if policy == "square_crop_512":
        return square_crop_512(image)
    if policy == "full_fov_512":
        return full_fov_512(image)
    if policy == "native_full_fov":
        return native_full_fov(image)
    raise ValueError(f"Unknown canonical policy {policy!r}; expected one of {CANONICAL_POLICIES}")
