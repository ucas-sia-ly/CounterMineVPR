#!/usr/bin/env python3
"""Run a user-invoked, at-most-10-source IC-Light preprocessing-mode audit.

Importing this module or requesting --help never loads a model. Actual
generation happens only when run_smoke/main is explicitly called.
"""

import argparse
from contextlib import ExitStack
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.probe.iclight_adapter import (  # noqa: E402
    ICLightAdapter, ICLightConfig, MODES,
)


SMOKE_COLUMNS = (
    "audit_index", "row_index", "mode", "source_512_path", "output_path", "seed",
    "prompt", "width", "height", "steps", "cfg", "highres_scale", "elapsed_seconds",
)
REQUIRED_COLUMNS = ("audit_index", "row_index", "image_id", "source_512_path")
MAX_SMOKE_SOURCES = 10


def load_smoke_sources(manifest_path: str | Path, count: int = 10) -> list[dict]:
    """Select the first rows, rejecting legacy/mixed formats across the manifest."""
    if not isinstance(count, int) or not 1 <= count <= MAX_SMOKE_SOURCES:
        raise ValueError("smoke count must be between 1 and 10; larger audits are not enabled")
    manifest_path = Path(manifest_path)
    selected = []
    seen_indices, seen_ids = set(), set()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError("audit manifest contains duplicate column names")
        missing = set(REQUIRED_COLUMNS).difference(columns)
        if missing:
            raise ValueError(f"audit manifest lacks required columns: {', '.join(sorted(missing))}")
        for expected_index, row in enumerate(reader):
            try:
                if any(not row[column] or not row[column].strip() for column in REQUIRED_COLUMNS):
                    raise ValueError("required source metadata is missing")
                audit_index, row_index = int(row["audit_index"]), int(row["row_index"])
                if audit_index != expected_index or row_index < 0:
                    raise ValueError("audit_index must follow existing zero-based row order; row_index must be nonnegative")
                if row_index in seen_indices or row["image_id"] in seen_ids:
                    raise ValueError("duplicate row_index or image_id among smoke sources")
                if Path(row["source_512_path"]).is_absolute():
                    raise ValueError("source_512_path must be a relative reference")
                if Path(row["source_512_path"]).suffix != ".png":
                    raise ValueError(
                        "source_512_path must reference a lossless .png; rebuild "
                        "the audit set with tools/07_build_generator_audit_set.py"
                    )
            except (ValueError, TypeError) as error:
                raise ValueError(f"audit manifest CSV line {expected_index + 2}: {error}") from error
            if len(selected) < count:
                selected.append({**row, "audit_index": audit_index, "row_index": row_index})
            seen_indices.add(row_index)
            seen_ids.add(row["image_id"])
    if len(selected) != count:
        raise ValueError(f"smoke audit requires {count} sources; manifest contains only {len(selected)}")
    return selected


def _validate_image(image: Image.Image) -> None:
    if not isinstance(image, Image.Image):
        raise ValueError("IC-Light output must be a PIL image")
    if image.size != (512, 512) or image.mode != "RGB":
        raise ValueError(f"audit image must be 512x512 RGB, got {image.size} {image.mode}")
    image.load()
    if not np.isfinite(np.asarray(image)).all():
        raise ValueError("audit image contains non-finite pixels")


def _validate_file(path: Path) -> None:
    """Require actual lossless PNG data, file structure, and a full decode."""
    try:
        if path.suffix != ".png":
            raise ValueError("canonical audit probes must use .png")
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("canonical audit probe is not encoded as PNG; rebuild the audit set")
            image.verify()
        with Image.open(path) as image:
            _validate_image(image)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid audit image {path.name}: {error}") from error


def _reference(path: Path) -> str:
    return Path(os.path.relpath(path, Path.cwd())).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _contact_sheet(sources: list[dict], staged_output: Path, destination: Path) -> None:
    margin, gap, tile, header, caption = 16, 16, 256, 44, 28
    row_height = tile + caption + gap
    size = (2 * margin + 3 * tile + 2 * gap, header + len(sources) * row_height + margin)
    with Image.new("RGB", size, "#f7f7f7") as sheet:
        draw = ImageDraw.Draw(sheet)
        for column, label in enumerate(("SOURCE", "OFFICIAL_RMBG", "FULL_SCENE")):
            draw.text((margin + column * (tile + gap), 12), label, font=_font(18), fill="#111111")
        for position, row in enumerate(sources):
            filename = f"{row['row_index']:08d}.png"
            paths = [Path(row["source_512_path"])] + [staged_output / "relit" / mode / filename for mode in MODES]
            top = header + position * row_height
            for column, path in enumerate(paths):
                left = margin + column * (tile + gap)
                with Image.open(path) as image:
                    with image.resize((tile, tile), Image.Resampling.LANCZOS) as thumbnail:
                        sheet.paste(thumbnail, (left, top))
                draw.text(
                    (left, top + tile + 5), f"audit {row['audit_index']} | row {row['row_index']}",
                    font=_font(14), fill="#333333",
                )
        sheet.save(destination, format="JPEG", quality=95, subsampling=0)
    with Image.open(destination) as check:
        check.verify()
    with Image.open(destination) as check:
        check.load()
        if check.mode != "RGB" or check.size != size:
            raise ValueError("contact sheet failed validation")


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _publish(artifacts: list[tuple[Path, Path]]) -> None:
    """Publish only completed artifacts, with rollback on a failed replacement."""
    with ExitStack() as stack:
        saved, published = [], []
        try:
            for staged, destination in artifacts:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() or destination.is_symlink():
                    temporary = stack.enter_context(tempfile.TemporaryDirectory(
                        prefix=".iclight-previous-", dir=destination.parent,
                    ))
                    backup = Path(temporary) / destination.name
                    os.replace(destination, backup)
                    saved.append((backup, destination))
                os.replace(staged, destination)
                published.append(destination)
        except OSError:
            for destination in reversed(published):
                _remove(destination)
            for backup, destination in reversed(saved):
                os.replace(backup, destination)
            raise


def run_smoke(
    manifest_path: str | Path,
    output_dir: str | Path,
    contact_sheet_path: str | Path,
    count: int = 10,
    config: ICLightConfig | None = None,
    *,
    adapter_factory=ICLightAdapter,
) -> dict:
    """Run sequential paired-mode inference only when explicitly invoked."""
    config = config if config is not None else ICLightConfig()
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    contact_sheet_path = Path(contact_sheet_path).expanduser().resolve()
    if contact_sheet_path.suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError("contact sheet must use .jpg or .jpeg")
    sources = load_smoke_sources(manifest_path, count)
    managed = [output_dir / "relit" / mode for mode in MODES] + [
        output_dir / "iclight_smoke.csv", output_dir / "iclight_smoke_summary.json",
    ]
    if any(contact_sheet_path == path or contact_sheet_path.is_relative_to(path) for path in managed):
        raise ValueError("contact sheet must be outside the other managed smoke outputs")
    managed.append(contact_sheet_path)
    for path in [manifest_path, *(Path(row["source_512_path"]).resolve() for row in sources)]:
        if any(path == destination or path.is_relative_to(destination) for destination in managed):
            raise ValueError("smoke outputs must not overwrite the manifest or source images")
    # Validate every selected source before constructing even a lazy adapter.
    for directory in {Path(row["source_512_path"]).resolve().parent for row in sources}:
        if any(path.suffix.lower() in (".jpg", ".jpeg") for path in directory.iterdir()):
            raise ValueError(
                "legacy JPEG canonical sources found; rebuild the audit set with "
                "tools/07_build_generator_audit_set.py instead of mixing JPG and PNG"
            )
    for row in sources:
        _validate_file(Path(row["source_512_path"]))

    source_hashes = {row["source_512_path"]: _sha256(Path(row["source_512_path"])) for row in sources}
    config_record = json.loads(json.dumps(asdict(config), default=str))
    summary = {
        "number_of_sources": count,
        "count_outputs": count * len(MODES),
        "modes": list(MODES),
        "config": config_record,
        "probe_image_format": "PNG",
        "probe_image_encoding": "lossless",
        "manifest_reference": _reference(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_sha256": source_hashes,
        "runs": [],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".iclight-smoke-", dir=output_dir.parent) as temporary, \
            tempfile.TemporaryDirectory(prefix=".iclight-sheet-", dir=contact_sheet_path.parent) as sheet_temporary:
        staging = Path(temporary)
        staged_sheet = Path(sheet_temporary) / contact_sheet_path.name
        for mode in MODES:
            (staging / "relit" / mode).mkdir(parents=True)
        adapter = adapter_factory(config)
        with (staging / "iclight_smoke.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SMOKE_COLUMNS)
            writer.writeheader()
            for row in sources:
                with Image.open(row["source_512_path"]) as source:
                    source.load()
                    for mode in MODES:
                        filename = f"{row['row_index']:08d}.png"
                        output_path = output_dir / "relit" / mode / filename
                        with source.copy() as inference_input:
                            original_pixels = inference_input.tobytes()
                            start = time.perf_counter()
                            result = adapter.relight(inference_input, mode)
                            elapsed = time.perf_counter() - start
                            try:
                                if inference_input.tobytes() != original_pixels:
                                    raise ValueError("IC-Light adapter modified its input image")
                                _validate_image(result)
                                result.save(staging / "relit" / mode / filename, format="PNG")
                            finally:
                                if isinstance(result, Image.Image):
                                    result.close()
                        _validate_file(staging / "relit" / mode / filename)
                        if not math.isfinite(elapsed) or elapsed < 0:
                            raise ValueError("invalid inference elapsed time")
                        writer.writerow({
                            "audit_index": row["audit_index"], "row_index": row["row_index"],
                            "mode": mode, "source_512_path": row["source_512_path"],
                            "output_path": _reference(output_path), "seed": config.seed,
                            "prompt": config.prompt, "width": config.width, "height": config.height,
                            "steps": config.steps, "cfg": config.cfg, "highres_scale": config.highres_scale,
                            "elapsed_seconds": f"{elapsed:.6f}",
                        })
                        summary["runs"].append({
                            "audit_index": row["audit_index"], "row_index": row["row_index"],
                            "mode": mode, "output_path": _reference(output_path),
                            "output_sha256": _sha256(staging / "relit" / mode / filename),
                            "elapsed_seconds": elapsed,
                            "adapter_stats": dict(getattr(adapter, "last_run_stats", {})),
                        })
                        print(f"audit {row['audit_index']} | row {row['row_index']} | {mode} | {elapsed:.2f}s", flush=True)
        _contact_sheet(sources, staging, staged_sheet)
        (staging / "iclight_smoke_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
        )
        artifacts = [(staging / "relit" / mode, output_dir / "relit" / mode) for mode in MODES]
        artifacts += [(staging / name, output_dir / name) for name in ("iclight_smoke.csv", "iclight_smoke_summary.json")]
        artifacts.append((staged_sheet, contact_sheet_path))
        _publish(artifacts)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/generator_audit/audit_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/generator_audit"))
    parser.add_argument("--contact-sheet", type=Path, default=Path("outputs/step2/iclight_smoke_10.jpg"))
    parser.add_argument("--count", type=int, default=10, help="1..10; full-dataset generation is disabled")
    parser.add_argument("--checkpoint-path", type=Path, help="optional local iclight_sd15_fc.safetensors")
    parser.add_argument("--cache-dir", type=Path, help="optional model cache outside third_party")
    parser.add_argument("--local-files-only", action="store_true", help="load only already-cached weights")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = ICLightConfig(
            checkpoint_path=str(args.checkpoint_path) if args.checkpoint_path else None,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            local_files_only=args.local_files_only,
        )
        summary = run_smoke(args.manifest, args.output_dir, args.contact_sheet, args.count, config)
    except (OSError, ValueError, RuntimeError, AssertionError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Saved {summary['count_outputs']} outputs for {summary['number_of_sources']} sources")
    print(f"Audit CSV: {args.output_dir / 'iclight_smoke.csv'}")
    print(f"Configuration and memory statistics: {args.output_dir / 'iclight_smoke_summary.json'}")
    print(f"Contact sheet: {args.contact_sheet}")


if __name__ == "__main__":
    main()
