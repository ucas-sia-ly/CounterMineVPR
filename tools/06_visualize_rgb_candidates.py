#!/usr/bin/env python3
"""Render real-image RGB candidate pairs as a Pillow contact sheet.

CSV selection is streamed, retaining only the requested highest-similarity
pairs. Images are loaded from manifest paths and resized in memory only.
"""

import argparse
import csv
import heapq
import math
import os
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw, ImageFont, ImageOps


MANIFEST_COLUMNS = ("row_index", "image_id", "relative_path", "place_uid")
CANDIDATE_COLUMNS = (
    "query_row_index", "negative_row_index", "query_image_id", "negative_image_id",
    "query_place_uid", "negative_place_uid", "rank", "similarity", "geo_distance_m",
)


def _check_columns(reader: csv.DictReader, required: tuple[str, ...], path: Path) -> None:
    columns = reader.fieldnames or []
    if len(columns) != len(set(columns)):
        raise ValueError(f"{path} contains duplicate column names")
    missing = set(required).difference(columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {', '.join(sorted(missing))}")


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read metadata in original descriptor order; never silently reorder it."""
    path = Path(path)
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _check_columns(reader, MANIFEST_COLUMNS, path)
        for expected_index, row in enumerate(reader):
            try:
                row_index = int(row["row_index"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"manifest row {expected_index} has invalid row_index") from error
            if row_index != expected_index:
                raise ValueError("manifest row_index must equal range(N) in existing row order")
            if any(not row[column] for column in MANIFEST_COLUMNS):
                raise ValueError(f"manifest row_index {row_index} contains missing metadata")
            rows.append({column: row[column] for column in MANIFEST_COLUMNS})
    if not rows:
        raise ValueError("manifest contains no images")
    return rows


def select_top_candidates(path: str | Path, top_n: int = 30) -> list[dict]:
    """Keep the best N rows by similarity descending, then query/negative index."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    path = Path(path)
    heap = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _check_columns(reader, CANDIDATE_COLUMNS, path)
        for line_number, row in enumerate(reader, start=2):
            try:
                candidate = {column: row[column] for column in CANDIDATE_COLUMNS}
                for column in ("query_row_index", "negative_row_index", "rank"):
                    candidate[column] = int(candidate[column])
                for column in ("similarity", "geo_distance_m"):
                    candidate[column] = float(candidate[column])
                if (
                    min(candidate["query_row_index"], candidate["negative_row_index"]) < 0
                    or candidate["rank"] < 1
                    or not math.isfinite(candidate["similarity"])
                    or not math.isfinite(candidate["geo_distance_m"])
                    or candidate["geo_distance_m"] < 0
                ):
                    raise ValueError("indices/rank or similarity/distance are invalid")
                if any(not candidate[column] for column in (
                    "query_image_id", "negative_image_id", "query_place_uid", "negative_place_uid",
                )):
                    raise ValueError("image/place metadata is missing")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}, CSV line {line_number}: {error}") from error
            # Worst retained row is at the root; earlier rows resolve exact ties.
            key = (
                candidate["similarity"], -candidate["query_row_index"],
                -candidate["negative_row_index"], -line_number,
            )
            entry = (key, candidate)
            if len(heap) < top_n:
                heapq.heappush(heap, entry)
            elif key > heap[0][0]:
                heapq.heapreplace(heap, entry)
    if not heap:
        raise ValueError(f"{path} contains no candidate pairs")
    return [entry[1] for entry in sorted(heap, key=lambda entry: entry[0], reverse=True)]


def _font(size: int) -> ImageFont.ImageFont:
    for name in (
        "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "...", font=font) > width:
        text = text[:-1]
    return text + "..."


def _image_path(root: Path, metadata: dict[str, str]) -> Path:
    relative_path = Path(metadata["relative_path"])
    if relative_path.is_absolute():
        raise ValueError(f"manifest relative_path must be relative: {relative_path}")
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"manifest image path is outside dataset root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"source image is missing: {path}")
    return path


def create_contact_sheet(
    manifest_path: str | Path,
    candidates_path: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
    top_n: int = 30,
    image_width: int = 360,
    image_height: int = 240,
) -> Path:
    """Render selected directed pairs without altering any dataset image."""
    if image_width < 160 or image_height < 80:
        raise ValueError("image_width must be >= 160 and image_height must be >= 80")
    output_path = Path(output_path).expanduser().resolve()
    formats = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
    image_format = formats.get(output_path.suffix.lower())
    if image_format is None:
        raise ValueError("output extension must be .jpg, .jpeg, or .png")
    manifest = load_manifest(manifest_path)
    candidates = select_top_candidates(candidates_path, top_n)
    root = Path(dataset_root).expanduser().resolve()
    if output_path.is_relative_to(root) and any(
        (root / row["relative_path"]).resolve() == output_path for row in manifest
    ):
        raise ValueError("output path must not overwrite a source image")

    # Validate selected metadata and all source paths before creating output.
    image_pairs = []
    for candidate in candidates:
        pair = []
        for role in ("query", "negative"):
            index = candidate[f"{role}_row_index"]
            if index >= len(manifest):
                raise ValueError(f"{role}_row_index {index} is outside manifest range")
            metadata = manifest[index]
            for column in ("image_id", "place_uid"):
                if candidate[f"{role}_{column}"] != metadata[column]:
                    raise ValueError(f"{role} {column} does not match manifest row_index {index}")
            pair.append(_image_path(root, metadata))
        if output_path in pair:
            raise ValueError("output path must not overwrite a source image")
        image_pairs.append(pair)

    margin, gap, header_height, caption_height = 16, 16, 40, 70
    row_height = image_height + caption_height
    width = 2 * margin + 2 * image_width + gap
    height = header_height + len(candidates) * row_height + margin
    sheet = Image.new("RGB", (width, height), "#f7f7f7")
    draw = ImageDraw.Draw(sheet)
    font, header_font = _font(15), _font(19)
    for offset, title in ((0, "Query"), (image_width + gap, "Negative")):
        draw.text((margin + offset, 9), title, fill="#111111", font=header_font)

    for pair_index, (candidate, paths) in enumerate(zip(candidates, image_pairs)):
        y = header_height + pair_index * row_height
        for column, path in enumerate(paths):
            x = margin + column * (image_width + gap)
            try:
                with Image.open(path) as original:
                    visual = ImageOps.exif_transpose(original).convert("RGB")
                    visual = ImageOps.contain(
                        visual, (image_width, image_height), method=Image.Resampling.LANCZOS,
                    )
            except (OSError, ValueError) as error:
                raise ValueError(f"could not decode source image {path}: {error}") from error
            draw.rectangle((x, y, x + image_width - 1, y + image_height - 1), fill="#dddddd")
            sheet.paste(visual, (x + (image_width - visual.width) // 2, y + (image_height - visual.height) // 2))
            label = f"{('Query', 'Negative')[column]} place_uid: {candidate[('query', 'negative')[column] + '_place_uid']}"
            draw.text((x, y + image_height + 7), _fit_text(draw, label, font, image_width), fill="#111111", font=font)
        details = (
            f"#{pair_index + 1}   rank {candidate['rank']}   "
            f"similarity {candidate['similarity']:.6f}   "
            f"geographic distance {candidate['geo_distance_m']:.1f} m"
        )
        draw.text((margin, y + image_height + 31), _fit_text(draw, details, font, width - 2 * margin), fill="#333333", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
        options = {"quality": 92, "subsampling": 0} if image_format == "JPEG" else {}
        sheet.save(temporary_path, format=image_format, **options)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        sheet.close()
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/gsv_mini/manifest.csv"))
    parser.add_argument("--candidates", type=Path, default=Path("cache/gsv_mini/rgb_candidates_raw.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/gsv-cities"))
    parser.add_argument("--output", type=Path, default=Path("outputs/step1c/top30_rgb_candidates.jpg"))
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--image-width", type=int, default=360)
    parser.add_argument("--image-height", type=int, default=240)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        output = create_contact_sheet(
            args.manifest, args.candidates, args.dataset_root, args.output,
            args.top_n, args.image_width, args.image_height,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Saved RGB candidate contact sheet: {output}")


if __name__ == "__main__":
    main()
