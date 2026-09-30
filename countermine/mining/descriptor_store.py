"""Small, CPU-only helpers for streaming a manifest-aligned descriptor index."""

import csv
from pathlib import Path, PurePosixPath
import tempfile
from typing import Callable, Iterable

import numpy as np


def load_manifest_paths(manifest_path: str | Path) -> list[str]:
    """Return portable image paths in exact ``row_index`` order.

    The index is small enough to keep paths in memory; image pixels and
    descriptors are streamed elsewhere.
    """
    with Path(manifest_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"row_index", "relative_path"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("manifest requires row_index and relative_path columns")
        rows: list[tuple[int, str]] = []
        for record in reader:
            raw_index = record["row_index"]
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid manifest row_index: {raw_index!r}") from error
            if index < 0 or str(index) != raw_index:
                raise ValueError(f"invalid manifest row_index: {raw_index!r}")

            relative_path = record["relative_path"]
            if relative_path is None or "\\" in relative_path:
                raise ValueError(f"invalid manifest relative_path: {relative_path!r}")
            path = PurePosixPath(relative_path)
            parts = path.parts
            if (
                path.is_absolute()
                or len(parts) != 3
                or parts[0] != "Images"
                or path.as_posix() != relative_path
                or any(part in ("", ".", "..") for part in parts)
            ):
                raise ValueError(f"invalid manifest relative_path: {relative_path!r}")
            rows.append((index, relative_path))

    rows.sort(key=lambda row: row[0])
    if [index for index, _ in rows] != list(range(len(rows))):
        raise ValueError("manifest row_index must be unique and contiguous from zero")
    if not rows:
        raise ValueError("manifest contains no images")
    return [path for _, path in rows]


def write_descriptor_memmap(
    batches: Iterable[tuple[int, np.ndarray]],
    output_path: str | Path,
    row_count: int,
    flush_every: int = 16,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[int, int]:
    """Stream contiguous descriptor batches to an atomically installed .npy file.

    A temporary memmap is removed if extraction fails. The final path appears
    only when every manifest row has been written and flushed.
    """
    if row_count <= 0 or flush_every <= 0:
        raise ValueError("row_count and flush_every must be positive")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    memmap: np.memmap | None = None
    completed = 0
    dimension: int | None = None
    batch_number = 0

    try:
        for start, descriptors in batches:
            if isinstance(start, bool) or not isinstance(start, int) or start != completed:
                raise ValueError(f"descriptor batch starts at {start}, expected {completed}")
            batch = np.asarray(descriptors)
            if batch.ndim != 2 or batch.shape[0] == 0 or batch.shape[1] == 0:
                raise ValueError("descriptor batches must be nonempty 2D arrays")
            real_numeric = np.issubdtype(batch.dtype, np.integer) or np.issubdtype(
                batch.dtype, np.floating
            )
            if not real_numeric or not np.isfinite(batch).all():
                raise ValueError("descriptor batch contains non-finite or nonnumeric values")
            if completed + batch.shape[0] > row_count:
                raise ValueError("descriptor batches exceed manifest row count")
            if dimension is None:
                dimension = batch.shape[1]
                with tempfile.NamedTemporaryFile(
                    dir=output.parent,
                    prefix=f".{output.name}.",
                    suffix=".partial.npy",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                memmap = np.lib.format.open_memmap(
                    temporary_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(row_count, dimension),
                )
            elif batch.shape[1] != dimension:
                raise ValueError(
                    f"descriptor dimension changed from {dimension} to {batch.shape[1]}"
                )

            with np.errstate(over="ignore", invalid="ignore"):
                stored = batch.astype(np.float16)
            if not np.isfinite(stored).all():
                raise ValueError("descriptor values overflow float16 storage")
            assert memmap is not None
            memmap[completed : completed + batch.shape[0]] = stored
            completed += batch.shape[0]
            batch_number += 1
            if batch_number % flush_every == 0:
                memmap.flush()
            if progress_callback is not None:
                progress_callback(completed)

        if completed != row_count or dimension is None:
            raise ValueError(f"wrote {completed} descriptor rows; expected {row_count}")
        assert memmap is not None and temporary_path is not None
        memmap.flush()
        del memmap
        memmap = None
        temporary_path.replace(output)
        temporary_path = None
        return completed, dimension
    finally:
        if memmap is not None:
            del memmap
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def verify_descriptor_memmap(
    path: str | Path,
    expected_rows: int,
    sample_size: int = 128,
    norm_atol: float = 0.02,
) -> tuple[int, int]:
    """Check shape, dtype, and deterministic sampled finite unit descriptors."""
    if expected_rows <= 0 or sample_size <= 0 or norm_atol < 0:
        raise ValueError("expected_rows and sample_size must be positive; norm_atol >= 0")
    descriptors = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        descriptors.ndim != 2
        or descriptors.shape[0] != expected_rows
        or descriptors.shape[1] == 0
        or descriptors.dtype != np.float16
    ):
        raise ValueError("descriptor file has the wrong shape or dtype")
    row_ids = np.unique(
        np.linspace(0, expected_rows - 1, min(sample_size, expected_rows), dtype=np.int64)
    )
    sampled = np.asarray(descriptors[row_ids], dtype=np.float32)
    if not np.isfinite(sampled).all():
        raise ValueError("sampled descriptors contain NaN or Inf")
    norms = np.linalg.norm(sampled, axis=1)
    if not np.all(np.abs(norms - 1.0) <= norm_atol):
        raise ValueError("sampled descriptor L2 norms are not approximately one")
    return descriptors.shape
