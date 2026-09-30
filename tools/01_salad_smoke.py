"""Smoke test for pretrained SALAD inference on two repository images."""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from countermine.mining.salad_encoder import SaladEncoder  # noqa: E402


DEFAULT_IMAGES = [
    REPO_ROOT / "third_party/LightGlue/assets/sacre_coeur1.jpg",
    REPO_ROOT / "third_party/LightGlue/assets/sacre_coeur2.jpg",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path, default=DEFAULT_IMAGES)
    args = parser.parse_args()
    if len(args.images) != 2:
        parser.error("provide exactly two image paths")

    print(f"torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if cuda_available else 'N/A'}")
    for path in args.images:
        print(f"input image: {path}")
        with Image.open(path) as image:
            image.verify()

    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    encoder = SaladEncoder()
    descriptors = encoder.encode(args.images)
    norms = torch.linalg.vector_norm(descriptors, dim=1)
    similarity = F.cosine_similarity(descriptors[0:1], descriptors[1:2]).item()
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if cuda_available else 0.0

    print(f"descriptor tensor shape: {tuple(descriptors.shape)}")
    print(f"descriptor dtype: {descriptors.dtype}")
    print(f"descriptor L2 norms: {norms.tolist()}")
    print(f"cosine similarity: {similarity:.6f}")
    print(f"peak CUDA memory allocated: {peak_mb:.1f} MB")

    assert descriptors.ndim == 2 and descriptors.shape[0] == 2
    assert torch.isfinite(descriptors).all().item()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3, rtol=0)


if __name__ == "__main__":
    main()
