"""Stream exact RGB Top-K negative candidates from an existing descriptor index."""

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

from countermine.mining.candidate_miner import (  # noqa: E402
    CANDIDATE_COLUMNS,
    ChunkedCandidateMiner,
)


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
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
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
    parser.add_argument("--output", type=Path, default=Path("cache/gsv_mini/rgb_candidates.csv"))
    parser.add_argument(
        "--summary", type=Path, help="Default: OUTPUT_STEM_summary.json beside the CSV"
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--reference-chunk-size", type=int, default=1024)
    parser.add_argument("--device", help="Torch device; default: CUDA when available, else CPU")
    parser.add_argument("--min-geo-distance-m", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")

    manifest_path = args.manifest.expanduser().resolve()
    descriptor_path = args.descriptors.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    summary_path = (
        args.summary.expanduser().resolve()
        if args.summary is not None
        else output_path.with_name(output_path.stem + "_summary.json")
    )
    if output_path == summary_path or any(
        path in (manifest_path, descriptor_path) for path in (output_path, summary_path)
    ):
        parser.error("output CSV and summary must be distinct from one another and the inputs")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest = pd.read_csv(
        manifest_path, dtype={"image_id": str, "place_uid": str, "city_id": str}
    )
    miner = ChunkedCandidateMiner(
        descriptor_path,
        manifest,
        top_k=args.top_k,
        query_chunk_size=args.query_chunk_size,
        reference_chunk_size=args.reference_chunk_size,
        device=args.device,
        min_geo_distance_m=args.min_geo_distance_m,
    )
    descriptors = np.load(descriptor_path, mmap_mode="r")
    descriptor_dtype = str(descriptors.dtype)
    del descriptors
    descriptor_stat = descriptor_path.stat()
    manifest_sha256 = _sha256_file(manifest_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    partial_csv = None
    partial_summary = None
    started = time.perf_counter()
    query_count = 0
    pair_count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=output_path.name + ".",
            suffix=".partial",
            delete=False,
        ) as destination:
            partial_csv = Path(destination.name)
            pd.DataFrame(columns=CANDIDATE_COLUMNS).to_csv(destination, index=False)
            for candidates in miner.iter_candidates():
                candidates.to_csv(destination, index=False, header=False)
                pair_count += len(candidates)
                query_count += candidates["query_row_index"].nunique()
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"{query_count}/{len(manifest)} queries | {pair_count} pairs | "
                    f"{query_count / elapsed:.1f} queries/sec",
                    flush=True,
                )

        if query_count != len(manifest) or pair_count != len(manifest) * args.top_k:
            raise RuntimeError("candidate retrieval did not produce Top-K for every manifest row")
        summary = {
            "stage": "1C RGB candidate generation",
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "descriptor_path": str(descriptor_path),
            "descriptor_shape": list(miner.descriptor_shape),
            "descriptor_dtype": descriptor_dtype,
            "descriptor_size_bytes": descriptor_stat.st_size,
            "descriptor_mtime_ns": descriptor_stat.st_mtime_ns,
            "top_k": args.top_k,
            "query_chunk_size": args.query_chunk_size,
            "reference_chunk_size": args.reference_chunk_size,
            "effective_query_chunk_size": min(args.query_chunk_size, len(manifest) - 1),
            "effective_reference_chunk_size": min(args.reference_chunk_size, len(manifest) - 1),
            "dot_product_tile_shape": list(miner.dot_product_tile_shape),
            "min_geo_distance_m": args.min_geo_distance_m,
            "same_place_excluded": True,
            "device": str(miner.device),
            "seed": args.seed,
            "number_of_queries": query_count,
            "number_of_pairs": pair_count,
            "peak_cuda_memory_mb": miner.peak_cuda_memory_mb,
            "elapsed_seconds": time.perf_counter() - started,
            "output_file": str(output_path),
            "output_size_bytes": partial_csv.stat().st_size,
            "counterminevpr_git_commit": _git_commit(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "torch_version": str(torch.__version__),
            "torch_num_threads": torch.get_num_threads(),
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=summary_path.parent,
            prefix=summary_path.name + ".",
            suffix=".partial",
            delete=False,
        ) as destination:
            partial_summary = Path(destination.name)
            destination.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        partial_csv.replace(output_path)
        partial_summary.replace(summary_path)
    finally:
        for path in (partial_csv, partial_summary):
            if path is not None:
                path.unlink(missing_ok=True)

    print(f"Candidate CSV: {output_path}")
    print(f"Saved configuration and summary: {summary_path}")


if __name__ == "__main__":
    main()
