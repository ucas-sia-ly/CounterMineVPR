"""Mine and validate raw RGB candidates, then save diagnostic statistics."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.mining.candidate_diagnostics import CandidateDiagnostics  # noqa: E402
from countermine.mining.candidate_miner import (  # noqa: E402
    CANDIDATE_COLUMNS,
    ChunkedCandidateMiner,
)


def validate_candidate_chunk(
    candidates: pd.DataFrame,
    manifest: pd.DataFrame,
    top_k: int,
    expected_query_start: int,
) -> int:
    """Validate complete, consecutive query groups before writing a CSV chunk.

    These checks remain active under ``python -O``. Consecutive query groups
    prevent a directed pair from recurring in a later chunk; the diagnostic
    accumulator also checks directed-pair uniqueness on disk.
    """
    if tuple(candidates.columns) != CANDIDATE_COLUMNS:
        raise ValueError("candidate columns must match CANDIDATE_COLUMNS exactly")
    if candidates.empty or len(candidates) % top_k:
        raise ValueError("each candidate chunk must contain complete Top-K query groups")
    query_count = len(candidates) // top_k
    if expected_query_start + query_count > len(manifest):
        raise ValueError("candidate query_row_index exceeds the manifest range")
    for name in ("query_row_index", "negative_row_index", "rank"):
        column = candidates[name]
        if not pd.api.types.is_integer_dtype(column) or column.isna().any():
            raise ValueError(f"candidate {name} must contain integers")
    queries = candidates["query_row_index"].to_numpy()
    negatives = candidates["negative_row_index"].to_numpy()
    if (queries < 0).any() or (queries >= len(manifest)).any():
        raise ValueError("candidate query_row_index is outside the manifest range")
    if (negatives < 0).any() or (negatives >= len(manifest)).any():
        raise ValueError("candidate negative_row_index is outside the manifest range")
    expected_queries = np.repeat(
        np.arange(expected_query_start, expected_query_start + query_count), top_k
    )
    if not np.array_equal(queries, expected_queries):
        raise ValueError("candidate queries must be consecutive and ascending without repeats")
    expected_ranks = np.tile(np.arange(1, top_k + 1), query_count)
    if not np.array_equal(candidates["rank"].to_numpy(), expected_ranks):
        raise ValueError("candidate rank must range exactly 1..K for each query")
    if candidates.duplicated(["query_row_index", "negative_row_index"]).any():
        raise ValueError("duplicated (query_row_index, negative_row_index) candidate")
    if not np.isfinite(candidates["similarity"].to_numpy(dtype=np.float64)).all():
        raise ValueError("candidate similarities must all be finite")
    if not candidates["pair_uid"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("every candidate pair_uid must be nonempty")
    if candidates["query_place_uid"].eq(candidates["negative_place_uid"]).any():
        raise ValueError("same-place candidates are forbidden")

    queries = queries.astype(np.int64, copy=False)
    negatives = negatives.astype(np.int64, copy=False)
    places = manifest["place_uid"].to_numpy()
    if (places[queries] == places[negatives]).any():
        raise ValueError("same-place candidates are forbidden by manifest metadata")
    for name in ("image_id", "place_uid", "city_id"):
        values = manifest[name].to_numpy()
        if not np.array_equal(candidates["query_" + name].to_numpy(), values[queries]):
            raise ValueError(f"query_{name} is not aligned with manifest row_index")
        if not np.array_equal(candidates["negative_" + name].to_numpy(), values[negatives]):
            raise ValueError(f"negative_{name} is not aligned with manifest row_index")
    cities = manifest["city_id"].to_numpy()
    if (
        not pd.api.types.is_bool_dtype(candidates["same_city"])
        or not np.array_equal(candidates["same_city"].to_numpy(), cities[queries] == cities[negatives])
    ):
        raise ValueError("candidate same_city must agree with manifest metadata")
    return query_count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/gsv_mini/manifest.csv"))
    parser.add_argument(
        "--descriptors", type=Path, default=Path("cache/gsv_mini/descriptors_fp16.npy")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("cache/gsv_mini"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--reference-chunk-size", type=int, default=1024)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--min-geo-distance-m", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")

    manifest_path = args.manifest.expanduser().resolve()
    descriptor_path = args.descriptors.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / "rgb_candidates_raw.csv"
    summary_path = output_dir / "rgb_candidates_raw_summary.json"
    if any(path in (manifest_path, descriptor_path) for path in (output_path, summary_path)):
        parser.error("output paths must be distinct from the input files")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = pd.read_csv(
        manifest_path, dtype={"image_id": str, "place_uid": str, "city_id": str}
    )
    miner = ChunkedCandidateMiner(
        descriptor_path, manifest, top_k=args.top_k,
        query_chunk_size=args.query_chunk_size,
        reference_chunk_size=args.reference_chunk_size,
        device=None if args.device == "auto" else args.device,
        min_geo_distance_m=args.min_geo_distance_m,
    )
    if miner.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    descriptors = np.load(descriptor_path, mmap_mode="r", allow_pickle=False)
    descriptor_dtype = str(descriptors.dtype)
    del descriptors
    manifest_sha256 = _sha256_file(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_csv = None
    temporary_summary = None
    query_count = 0
    started = time.perf_counter()
    with CandidateDiagnostics(directory=output_dir) as diagnostics:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=output_dir,
                prefix=output_path.name + ".", suffix=".partial", delete=False,
            ) as destination:
                temporary_csv = Path(destination.name)
                pd.DataFrame(columns=CANDIDATE_COLUMNS).to_csv(destination, index=False)
                for candidates in miner.iter_candidates():
                    chunk_queries = validate_candidate_chunk(
                        candidates, manifest, args.top_k, query_count
                    )
                    diagnostics.add(candidates)
                    # Preserve the original float32 scores as exact float64
                    # decimals so CSV diagnostics agree with the saved JSON.
                    candidates.to_csv(destination, index=False, header=False, float_format="%.17g")
                    query_count += chunk_queries
                    print(f"{query_count}/{len(manifest)} queries validated", flush=True)
            if query_count != len(manifest):
                raise ValueError("candidate output must include Top-K for every manifest query")
            summary = diagnostics.summary()
            if summary["total_candidate_pairs"] != len(manifest) * args.top_k:
                raise ValueError("candidate pair count does not equal number_of_queries * top_k")
            summary.update(
                {
                    "stage": "1C raw RGB diagnostic candidates",
                    "top_k": args.top_k,
                    "query_chunk_size": args.query_chunk_size,
                    "reference_chunk_size": args.reference_chunk_size,
                    "effective_query_chunk_size": min(args.query_chunk_size, len(manifest) - 1),
                    "effective_reference_chunk_size": min(args.reference_chunk_size, len(manifest) - 1),
                    "dot_product_tile_shape": list(miner.dot_product_tile_shape),
                    "descriptor_shape": list(miner.descriptor_shape),
                    "descriptor_dtype": descriptor_dtype,
                    "device": str(miner.device),
                    "requested_device": args.device,
                    "min_geo_distance_m": args.min_geo_distance_m,
                    "peak_cuda_memory_mb": miner.peak_cuda_memory_mb,
                    "manifest_sha256": manifest_sha256,
                    "manifest_path": str(manifest_path),
                    "descriptor_path": str(descriptor_path),
                    "counterminevpr_git_commit": _git_commit(),
                    "seed": args.seed,
                    "torch_num_threads": torch.get_num_threads(),
                    "numpy_version": np.__version__,
                    "pandas_version": pd.__version__,
                    "torch_version": str(torch.__version__),
                    "elapsed_seconds": time.perf_counter() - started,
                    "output_file": str(output_path),
                }
            )
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output_dir,
                prefix=summary_path.name + ".", suffix=".partial", delete=False,
            ) as destination:
                temporary_summary = Path(destination.name)
                destination.write(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
            # No published output is touched until all queries and statistics
            # have passed validation and both temporary artifacts are ready.
            temporary_csv.replace(output_path)
            temporary_summary.replace(summary_path)
        finally:
            for path in (temporary_csv, temporary_summary):
                if path is not None:
                    path.unlink(missing_ok=True)

    print(f"Wrote {summary['total_candidate_pairs']} candidates from {query_count} queries")
    print(f"Raw CSV: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
