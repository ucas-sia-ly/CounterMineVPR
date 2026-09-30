"""Exact, streamed RGB candidate retrieval from a manifest-aligned descriptor bank.

Only real-image descriptors are used here. Scores are float32 dot products of
the already normalized descriptors; this stage does not compute CounterMine
scores or local matches. Use ``iter_candidates`` for bounded output memory.
"""

import hashlib
import json
import math
import operator
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch


REQUIRED_COLUMNS = ("row_index", "image_id", "place_uid", "city_id", "lat", "lon")
CANDIDATE_COLUMNS = (
    "query_row_index",
    "negative_row_index",
    "query_image_id",
    "negative_image_id",
    "query_place_uid",
    "negative_place_uid",
    "query_city_id",
    "negative_city_id",
    "rank",
    "similarity",
    "geo_distance_m",
    "same_city",
    "pair_uid",
)
EARTH_RADIUS_M = 6_371_008.8
# Fixed GEMM dimensions and globally anchored positions keep reduction order
# independent of caller chunk boundaries, including short final blocks.
_DOT_QUERY_TILE_SIZE = 32
_DOT_REFERENCE_TILE_SIZE = 128


def stable_pair_uid(image_id_a: str, image_id_b: str) -> str:
    """SHA256 of an unambiguous, lexicographically sorted pair of image IDs."""
    payload = json.dumps(
        sorted((image_id_a, image_id_b)), ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def haversine_distance_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distances in meters, with NumPy broadcasting.

    Coordinates are in degrees. Pass ``[:, None]`` query coordinates and
    ``[None, :]`` reference coordinates for a block of pairwise distances, or
    equally shaped vectors to measure only the selected candidate pairs.
    """
    lat1, lon1, lat2, lon2 = (
        np.deg2rad(np.asarray(value, dtype=np.float64))
        for value in (lat1, lon1, lat2, lon2)
    )
    haversine = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def _positive_integer(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if isinstance(value, (bool, np.bool_)) or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


class ChunkedCandidateMiner:
    """Retain only block similarities and a running Top-K for each query.

    Descriptor memory is O((Q + R + 160) * D); similarity/selection memory is
    O(Q * (R + K)). Metadata is O(N). Neither a full descriptor copy nor an
    N-by-N similarity matrix is allocated. Oversized chunk requests are capped
    at N - 1, including for tiny banks.

    Manifest rows must already be ordered by ``row_index``. They are never
    silently reordered. Equal similarities are ordered by reference row index.
    Dot products use fixed, globally anchored internal tiles so changing the
    requested chunk sizes does not change the matrix kernel or a pair's position
    within it. Scores are not rounded or perturbed to break ties. Different
    devices/backends can still differ in float32 roundoff.
    """

    def __init__(
        self,
        descriptor_path: str | Path,
        manifest: pd.DataFrame,
        top_k: int = 50,
        query_chunk_size: int = 128,
        reference_chunk_size: int = 1024,
        device: str | torch.device | None = None,
        min_geo_distance_m: float = 0.0,
    ) -> None:
        self.top_k = _positive_integer(top_k, "top_k")
        self.query_chunk_size = _positive_integer(query_chunk_size, "query_chunk_size")
        self.reference_chunk_size = _positive_integer(
            reference_chunk_size, "reference_chunk_size"
        )
        try:
            self.min_geo_distance_m = float(min_geo_distance_m)
        except (TypeError, ValueError) as error:
            raise ValueError("min_geo_distance_m must be finite and >= 0") from error
        if not math.isfinite(self.min_geo_distance_m) or self.min_geo_distance_m < 0:
            raise ValueError("min_geo_distance_m must be finite and >= 0")

        missing = set(REQUIRED_COLUMNS).difference(manifest.columns)
        if missing:
            raise ValueError(f"manifest lacks required columns: {', '.join(sorted(missing))}")
        if not manifest.columns.is_unique:
            raise ValueError("manifest column names must be unique")
        self.row_count = len(manifest)
        if self.row_count == 0:
            raise ValueError("manifest contains no images")
        row_indices = manifest["row_index"]
        if (
            not pd.api.types.is_integer_dtype(row_indices)
            or row_indices.isna().any()
            or not np.array_equal(row_indices.to_numpy(), np.arange(self.row_count))
        ):
            raise ValueError("manifest row_index must equal range(N) in existing row order")
        if manifest.loc[:, REQUIRED_COLUMNS].isna().any().any():
            raise ValueError("manifest contains missing required metadata")
        if not manifest["image_id"].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError("manifest image_id values must be nonempty strings")
        if not manifest["image_id"].is_unique:
            raise ValueError("manifest image_id values must be unique")

        self.image_ids = manifest["image_id"].to_numpy(copy=True)
        self.place_uids = manifest["place_uid"].to_numpy(copy=True)
        self.city_ids = manifest["city_id"].to_numpy(copy=True)
        self._place_codes = pd.factorize(manifest["place_uid"], sort=False)[0]
        try:
            self.latitudes = manifest["lat"].to_numpy(dtype=np.float64, copy=True)
            self.longitudes = manifest["lon"].to_numpy(dtype=np.float64, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("manifest lat/lon must be numeric coordinates") from error
        if (
            not np.isfinite(self.latitudes).all()
            or not np.isfinite(self.longitudes).all()
            or (np.abs(self.latitudes) > 90).any()
            or (np.abs(self.longitudes) > 180).any()
        ):
            raise ValueError("manifest lat/lon must be finite coordinates within valid ranges")

        self.descriptor_path = Path(descriptor_path).expanduser().resolve()
        self._descriptors = np.load(self.descriptor_path, mmap_mode="r", allow_pickle=False)
        if (
            self._descriptors.ndim != 2
            or self._descriptors.shape[0] != self.row_count
            or self._descriptors.shape[1] == 0
        ):
            raise ValueError("descriptor shape must be (len(manifest), nonzero dimension)")
        if not np.issubdtype(self._descriptors.dtype, np.floating):
            raise ValueError("descriptor dtype must be floating point")
        self.descriptor_shape = self._descriptors.shape

        place_counts = np.bincount(self._place_codes)
        valid_counts = self.row_count - place_counts[self._place_codes]
        insufficient = np.flatnonzero(valid_counts < self.top_k)
        if insufficient.size:
            row = int(insufficient[0])
            raise ValueError(
                f"query row_index {row} has only {valid_counts[row]} valid references "
                f"after same-place exclusion; top_k={self.top_k}"
            )

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")
        self.peak_cuda_memory_mb = 0.0
        self.dot_product_tile_shape = (_DOT_QUERY_TILE_SIZE, _DOT_REFERENCE_TILE_SIZE)

    def _descriptor_block(self, start: int, stop: int) -> torch.Tensor:
        # Copy only the active slice: torch must not wrap a read-only memmap.
        block = np.array(self._descriptors[start:stop], dtype=np.float32, copy=True, order="C")
        if not np.isfinite(block).all():
            raise ValueError(f"descriptor rows [{start}, {stop}) contain non-finite values")
        return torch.from_numpy(block).to(self.device)

    @staticmethod
    def _dot_products(
        queries: torch.Tensor,
        references: torch.Tensor,
        query_start: int,
        reference_start: int,
    ) -> torch.Tensor:
        """Compute q @ r.T with a chunk-independent float32 reduction layout.

        Plain GEMM can switch reduction kernels with block shape, altering the
        last bits of even duplicate descriptors' scores. Padding small internal
        tiles and anchoring their positions to global row indices avoids that
        problem without rounding similarities or allocating a descriptor bank.
        """
        query_tile = queries.new_zeros((_DOT_QUERY_TILE_SIZE, queries.shape[1]))
        reference_tile = references.new_zeros((_DOT_REFERENCE_TILE_SIZE, references.shape[1]))
        similarities = queries.new_empty((len(queries), len(references)))
        query_offset = 0
        while query_offset < len(queries):
            query_position = (query_start + query_offset) % _DOT_QUERY_TILE_SIZE
            query_count = min(_DOT_QUERY_TILE_SIZE - query_position, len(queries) - query_offset)
            query_tile.zero_()
            query_slice = slice(query_position, query_position + query_count)
            query_tile[query_slice] = queries[query_offset : query_offset + query_count]
            reference_offset = 0
            while reference_offset < len(references):
                reference_position = (reference_start + reference_offset) % _DOT_REFERENCE_TILE_SIZE
                reference_count = min(
                    _DOT_REFERENCE_TILE_SIZE - reference_position, len(references) - reference_offset
                )
                reference_tile.zero_()
                reference_slice = slice(reference_position, reference_position + reference_count)
                reference_tile[reference_slice] = references[
                    reference_offset : reference_offset + reference_count
                ]
                tile_scores = query_tile @ reference_tile.T
                similarities[
                    query_offset : query_offset + query_count,
                    reference_offset : reference_offset + reference_count,
                ] = tile_scores[query_slice, reference_slice]
                reference_offset += reference_count
            query_offset += query_count
        return similarities

    @staticmethod
    def _merge_topk(
        scores: torch.Tensor, indices: torch.Tensor, top_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Secondary key first, then stable sorting on the primary key. Using
        # torch.topk alone would choose arbitrary members of boundary ties.
        by_index = torch.argsort(indices, dim=1, stable=True)
        scores = scores.gather(1, by_index)
        indices = indices.gather(1, by_index)
        by_score = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :top_k]
        return scores.gather(1, by_score), indices.gather(1, by_score)

    @torch.inference_mode()
    def _retrieve_chunk(self, query_start: int, query_stop: int) -> tuple[np.ndarray, np.ndarray]:
        queries = self._descriptor_block(query_start, query_stop)
        query_count = query_stop - query_start
        best_scores = torch.empty((query_count, 0), dtype=torch.float32, device=self.device)
        best_indices = torch.empty((query_count, 0), dtype=torch.int64, device=self.device)
        query_places = torch.as_tensor(
            self._place_codes[query_start:query_stop], device=self.device
        )
        reference_size = min(self.reference_chunk_size, self.row_count - 1)
        for reference_start in range(0, self.row_count, reference_size):
            reference_stop = min(reference_start + reference_size, self.row_count)
            references = self._descriptor_block(reference_start, reference_stop)
            similarities = self._dot_products(queries, references, query_start, reference_start)
            reference_places = torch.as_tensor(
                self._place_codes[reference_start:reference_stop], device=self.device
            )
            forbidden = query_places[:, None] == reference_places[None, :]
            if self.min_geo_distance_m > 0:
                distances = haversine_distance_m(
                    self.latitudes[query_start:query_stop, None],
                    self.longitudes[query_start:query_stop, None],
                    self.latitudes[None, reference_start:reference_stop],
                    self.longitudes[None, reference_start:reference_stop],
                )
                forbidden |= torch.as_tensor(distances < self.min_geo_distance_m, device=self.device)
                del distances
            similarities.masked_fill_(forbidden, -torch.inf)
            # Reference columns are ascending row indices, so a stable score
            # sort also handles the secondary key before truncating this block.
            block_order = torch.argsort(
                similarities, dim=1, descending=True, stable=True
            )[:, : self.top_k]
            block_scores = similarities.gather(1, block_order)
            block_indices = block_order + reference_start
            best_scores, best_indices = self._merge_topk(
                torch.cat((best_scores, block_scores), dim=1),
                torch.cat((best_indices, block_indices), dim=1),
                self.top_k,
            )
            del references, similarities, forbidden, reference_places
            del block_order, block_scores, block_indices

        scores = best_scores.cpu().numpy()
        indices = best_indices.cpu().numpy()
        valid_counts = np.isfinite(scores).sum(axis=1)
        insufficient = np.flatnonzero(valid_counts < self.top_k)
        if insufficient.size:
            offset = int(insufficient[0])
            raise ValueError(
                f"query row_index {query_start + offset} has only {valid_counts[offset]} "
                f"valid references after exclusions; top_k={self.top_k}, "
                f"min_geo_distance_m={self.min_geo_distance_m}"
            )
        return indices, scores

    def _candidate_table(
        self, query_start: int, indices: np.ndarray, scores: np.ndarray
    ) -> pd.DataFrame:
        queries = np.repeat(np.arange(query_start, query_start + len(indices)), self.top_k)
        negatives = indices.reshape(-1)
        query_ids, negative_ids = self.image_ids[queries], self.image_ids[negatives]
        return pd.DataFrame(
            {
                "query_row_index": queries,
                "negative_row_index": negatives,
                "query_image_id": query_ids,
                "negative_image_id": negative_ids,
                "query_place_uid": self.place_uids[queries],
                "negative_place_uid": self.place_uids[negatives],
                "query_city_id": self.city_ids[queries],
                "negative_city_id": self.city_ids[negatives],
                "rank": np.tile(np.arange(1, self.top_k + 1), len(indices)),
                "similarity": scores.reshape(-1),
                "geo_distance_m": haversine_distance_m(
                    self.latitudes[queries], self.longitudes[queries],
                    self.latitudes[negatives], self.longitudes[negatives],
                ),
                "same_city": self.city_ids[queries] == self.city_ids[negatives],
                "pair_uid": [stable_pair_uid(a, b) for a, b in zip(query_ids, negative_ids)],
            },
            columns=CANDIDATE_COLUMNS,
        )

    def iter_candidates(self) -> Iterator[pd.DataFrame]:
        """Yield ascending query chunks, each containing exactly K valid rows/query.

        A later chunk can fail if geographic exclusions leave too few negatives;
        streaming consumers should publish their output only after exhaustion.
        CUDA uses full float32 precision (TF32 disabled) and reports peak memory.
        """
        use_cuda = self.device.type == "cuda"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.peak_cuda_memory_mb = 0.0
        query_size = min(self.query_chunk_size, self.row_count - 1)
        try:
            for query_start in range(0, self.row_count, query_size):
                query_stop = min(query_start + query_size, self.row_count)
                # Restore caller precision settings before yielding control.
                previous_precision = torch.get_float32_matmul_precision()
                try:
                    torch.set_float32_matmul_precision("highest")
                    indices, scores = self._retrieve_chunk(query_start, query_stop)
                finally:
                    torch.set_float32_matmul_precision(previous_precision)
                yield self._candidate_table(query_start, indices, scores)
        finally:
            if use_cuda:
                self.peak_cuda_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                print(f"Peak CUDA memory: {self.peak_cuda_memory_mb:.1f} MB", flush=True)

    def mine(self) -> pd.DataFrame:
        """Materialize the O(N*K) candidate table; descriptors remain streamed."""
        return pd.concat(self.iter_candidates(), ignore_index=True)


def mine_candidates(
    descriptor_path: str | Path,
    manifest: pd.DataFrame,
    top_k: int = 50,
    query_chunk_size: int = 128,
    reference_chunk_size: int = 1024,
    device: str | torch.device | None = None,
    min_geo_distance_m: float = 0.0,
) -> pd.DataFrame:
    """Convenience wrapper; use ``ChunkedCandidateMiner`` to stream large tables."""
    return ChunkedCandidateMiner(
        descriptor_path, manifest, top_k, query_chunk_size, reference_chunk_size,
        device, min_geo_distance_m,
    ).mine()
