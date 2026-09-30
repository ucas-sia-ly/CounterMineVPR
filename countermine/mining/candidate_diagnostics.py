"""Bounded-memory, CPU-only diagnostics for streamed RGB candidate tables."""

from collections import Counter
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import pandas as pd


# Keep this module independent of candidate_miner, which imports Torch.
DIAGNOSTIC_COLUMNS = (
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
GEO_THRESHOLDS_M = (150, 200, 250, 500, 1000)


class CandidateDiagnostics:
    """Accumulate exact statistics without holding the candidate CSV in RAM.

    A temporary SQLite index detects duplicate directed pairs and counts
    undirected/reverse pairs. Two temporary float64 files hold similarities and
    distances; exact quantiles partition their writable memory maps in place.
    Only the current input chunk and the 20 highest-scoring rows remain in RAM.
    ``summary`` finalizes the accumulator, after which ``add`` is unavailable.
    Use this class as a context manager to remove its temporary files.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix="rgb_candidate_diagnostics_", dir=directory
        )
        self._root = Path(self._temporary.name)
        self._connection = sqlite3.connect(self._root / "pairs.sqlite3")
        self._connection.execute("PRAGMA cache_size = -8192")
        self._connection.execute("PRAGMA temp_store = FILE")
        self._connection.execute(
            "CREATE TABLE pairs (query_row INTEGER NOT NULL, negative_row INTEGER NOT NULL, "
            "pair_uid TEXT NOT NULL, PRIMARY KEY (query_row, negative_row)) WITHOUT ROWID"
        )
        self._connection.execute("CREATE INDEX pairs_uid ON pairs (pair_uid)")
        self._similarity_file = (self._root / "similarity.bin").open("wb")
        self._distance_file = (self._root / "distance.bin").open("wb")
        self._count = 0
        self._same_city_count = 0
        self._below_counts = {threshold: 0 for threshold in GEO_THRESHOLDS_M}
        self._rank_counts = Counter()
        self._similarity_sum = 0.0
        self._similarity_min = float("inf")
        self._similarity_max = float("-inf")
        self._distance_min = float("inf")
        self._distance_max = float("-inf")
        self._top_candidates = pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)
        self._result = None
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        """Close handles and remove disk-backed temporary data."""
        if not self._closed:
            self._similarity_file.close()
            self._distance_file.close()
            self._connection.close()
            self._temporary.cleanup()
            self._closed = True

    @staticmethod
    def _integers(series: pd.Series, name: str, minimum: int) -> np.ndarray:
        if pd.api.types.is_bool_dtype(series):
            raise ValueError(f"{name} must contain integers >= {minimum}")
        try:
            values = pd.to_numeric(series, errors="raise").to_numpy()
            valid = (
                np.isfinite(values).all()
                and (values >= minimum).all()
                and (values <= np.iinfo(np.int64).max).all()
                and (values == np.floor(values)).all()
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must contain integers >= {minimum}") from error
        if not valid:
            raise ValueError(f"{name} must contain integers >= {minimum}")
        return values.astype(np.int64)

    def add(self, table: pd.DataFrame) -> None:
        """Add a bounded chunk; duplicate directed pairs fail across chunks too."""
        if self._closed or self._result is not None:
            raise RuntimeError("candidate diagnostics is closed or finalized")
        if not table.columns.is_unique:
            raise ValueError("candidate column names must be unique")
        missing = set(DIAGNOSTIC_COLUMNS).difference(table.columns)
        if missing:
            raise ValueError(f"candidate table lacks required columns: {', '.join(sorted(missing))}")
        if table.empty:
            return
        query_rows = self._integers(table["query_row_index"], "query_row_index", 0)
        negative_rows = self._integers(table["negative_row_index"], "negative_row_index", 0)
        ranks = self._integers(table["rank"], "rank", 1)
        try:
            similarities = table["similarity"].to_numpy(dtype=np.float64)
            distances = table["geo_distance_m"].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("similarity and geo_distance_m must contain finite numbers") from error
        if not np.isfinite(similarities).all():
            raise ValueError("all candidate similarities must be finite")
        if not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError("geo_distance_m must contain finite nonnegative numbers")
        if not table["pair_uid"].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError("every pair_uid must be a nonempty string")
        same_city = table["same_city"]
        if same_city.isna().any() or not same_city.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError("same_city must contain boolean values")

        # Commit before appending numeric files so a rejected duplicate chunk
        # leaves statistics untouched and the accumulator can still be used.
        try:
            with self._connection:
                self._connection.executemany(
                    "INSERT INTO pairs VALUES (?, ?, ?)",
                    (
                        (int(query), int(negative), uid)
                        for query, negative, uid in zip(
                            query_rows, negative_rows, table["pair_uid"]
                        )
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError("duplicated (query_row_index, negative_row_index) candidate") from error

        similarities.tofile(self._similarity_file)
        distances.tofile(self._distance_file)
        self._count += len(table)
        self._same_city_count += int(same_city.sum())
        for threshold in GEO_THRESHOLDS_M:
            self._below_counts[threshold] += int(np.count_nonzero(distances < threshold))
        self._rank_counts.update(int(rank) for rank in ranks)
        self._similarity_sum += float(np.sum(similarities, dtype=np.float64))
        self._similarity_min = min(self._similarity_min, float(similarities.min()))
        self._similarity_max = max(self._similarity_max, float(similarities.max()))
        self._distance_min = min(self._distance_min, float(distances.min()))
        self._distance_max = max(self._distance_max, float(distances.max()))
        top_chunk = table.loc[:, DIAGNOSTIC_COLUMNS].copy()
        top_chunk["query_row_index"] = query_rows
        top_chunk["negative_row_index"] = negative_rows
        top_chunk["rank"] = ranks
        top_chunk["similarity"] = similarities
        top_chunk["geo_distance_m"] = distances
        top_chunk = top_chunk.sort_values(
            ["similarity", "query_row_index", "negative_row_index"],
            ascending=[False, True, True],
            kind="stable",
        ).head(20)
        pool = top_chunk if self._top_candidates.empty else pd.concat(
            [self._top_candidates, top_chunk], ignore_index=True
        )
        self._top_candidates = pool.sort_values(
            ["similarity", "query_row_index", "negative_row_index"],
            ascending=[False, True, True],
            kind="stable",
        ).head(20).copy()

    def _quantiles(self, filename: str) -> dict[str, float]:
        values = np.memmap(self._root / filename, dtype=np.float64, mode="r+", shape=(self._count,))
        try:
            results = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95], overwrite_input=True)
            return dict(zip(("q05", "q25", "median", "q75", "q95"), map(float, results)))
        finally:
            values.flush()
            values._mmap.close()

    def summary(self) -> dict:
        """Return exact diagnostic statistics with documented reverse counts.

        ``reverse_duplicate_pairs`` counts undirected pairs whose two directions
        both appear. ``reverse_duplicated_candidate_rows`` is twice that count,
        and its fraction uses all directed candidate rows as the denominator.
        """
        if self._closed:
            raise RuntimeError("candidate diagnostics is closed")
        if self._result is not None:
            return self._result
        if self._count == 0:
            raise ValueError("candidate table contains no rows")
        self._similarity_file.flush()
        self._distance_file.flush()
        similarity = self._quantiles("similarity.bin")
        similarity.update(
            min=self._similarity_min,
            max=self._similarity_max,
            mean=self._similarity_sum / self._count,
        )
        geographic = self._quantiles("distance.bin")
        geographic.update(min=self._distance_min, max=self._distance_max)
        queries = self._connection.execute("SELECT COUNT(DISTINCT query_row) FROM pairs").fetchone()[0]
        unique_uids = self._connection.execute("SELECT COUNT(DISTINCT pair_uid) FROM pairs").fetchone()[0]
        reverse_pairs = self._connection.execute(
            "SELECT COUNT(*) FROM pairs a WHERE a.query_row < a.negative_row "
            "AND EXISTS (SELECT 1 FROM pairs b "
            "WHERE b.query_row = a.negative_row AND b.negative_row = a.query_row)"
        ).fetchone()[0]
        top_candidates = self._top_candidates.to_dict(orient="records")
        self._result = {
            "number_of_queries": int(queries),
            "total_candidate_pairs": self._count,
            "similarity_statistics": similarity,
            "geo_distance_statistics": geographic,
            "fraction_same_city": self._same_city_count / self._count,
            "geo_distance_counts_below_m": {str(key): value for key, value in self._below_counts.items()},
            "rank_distribution": {str(rank): count for rank, count in sorted(self._rank_counts.items())},
            "unique_pair_uids": int(unique_uids),
            "reverse_duplicate_pairs": int(reverse_pairs),
            "reverse_duplicated_candidate_rows": int(2 * reverse_pairs),
            "reverse_duplicate_fraction": 2 * reverse_pairs / self._count,
            "top_candidates": top_candidates,
        }
        return self._result
