"""Stream manifest images through SALAD into a float16 descriptor memmap."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.mining.descriptor_store import (  # noqa: E402
    load_manifest_paths,
    verify_descriptor_memmap,
    write_descriptor_memmap,
)
from countermine.mining.salad_encoder import SALAD_DIR, SaladEncoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("cache/gsv_mini/manifest.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/GSVCities"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/gsv_mini"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    manifest_path = args.manifest.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    relative_paths = load_manifest_paths(manifest_path)
    image_count = len(relative_paths)
    if image_count == 0:
        raise ValueError("manifest contains no images")

    manifest_metadata_path = manifest_path.with_name("manifest_summary.json")
    manifest_metadata = (
        json.loads(manifest_metadata_path.read_text(encoding="utf-8"))
        if manifest_metadata_path.is_file()
        else {}
    )
    seed = int(manifest_metadata.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    manifest_hash = hashlib.sha256()
    with manifest_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            manifest_hash.update(chunk)

    encoder = SaladEncoder(batch_size=args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "descriptors_fp16.npy"
    partial_path = args.output_dir / "descriptors_fp16.npy.partial"
    summary_path = args.output_dir / "index_summary.json"
    summary_path.unlink(missing_ok=True)

    started = time.perf_counter()
    last_report = 0

    def report(completed: int) -> None:
        nonlocal last_report
        if completed == image_count or completed - last_report >= args.batch_size * 10:
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"{completed}/{image_count} images | {completed / elapsed:.1f} images/sec",
                flush=True,
            )
            last_report = completed

    batches = (
        (start, descriptors.numpy())
        for start, descriptors in encoder.iter_encode(
            dataset_root / path for path in relative_paths
        )
    )
    shape = write_descriptor_memmap(
        batches,
        partial_path,
        row_count=image_count,
        progress_callback=report,
    )
    verified_shape = verify_descriptor_memmap(partial_path, expected_rows=image_count)
    if verified_shape != shape:
        raise AssertionError("written and verified descriptor shapes differ")
    partial_path.replace(output_path)
    verify_descriptor_memmap(output_path, expected_rows=image_count)

    commits = {}
    for key, directory in (
        ("salad_git_commit", SALAD_DIR),
        ("counterminevpr_git_commit", REPO_ROOT),
    ):
        try:
            result = subprocess.run(
                ["git", "-C", str(directory), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            commits[key] = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            commits[key] = None

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if encoder.device.type == "cuda" else 0.0
    summary = {
        "descriptor_shape": list(shape),
        "dtype": "float16",
        "number_of_images": image_count,
        "salad_image_size": [encoder.image_size, encoder.image_size],
        "salad_model_name": encoder.MODEL_NAME,
        "batch_size": args.batch_size,
        "device": str(encoder.device),
        "seed": seed,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash.hexdigest(),
        "dataset_root": str(dataset_root),
        "output_file": str(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "peak_cuda_memory_mb": round(peak_mb, 1),
        **commits,
    }
    pending_summary_path = summary_path.with_name(summary_path.name + ".partial")
    pending_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pending_summary_path.replace(summary_path)

    print(f"Peak GPU memory: {peak_mb:.1f} MB")
    print(f"Final descriptor shape: {shape}")
    print(f"Output file size: {output_path.stat().st_size / (1024**2):.1f} MB")
    print(f"Verified descriptor file: {output_path}")


if __name__ == "__main__":
    main()
