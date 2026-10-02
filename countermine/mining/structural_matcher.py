"""Frozen original-RGB local matching and descriptive Step 2A metrics.

Torch and the vendored matcher are imported only when a matcher is created.
The metric and pixel helpers remain CPU-only and never load model weights.
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from PIL import Image


IMAGE_SIZE = (640, 480)
GRID_SIZE = 8
ALIKED_CONFIG = {"max_num_keypoints": 2048, "detection_threshold": 0.2}
LIGHTGLUE_CONFIG = {
    "features": "aliked", "depth_confidence": -1, "width_confidence": -1,
    "filter_threshold": 0.1, "mp": False,
}
METRIC_COLUMNS = (
    "num_keypoints_a", "num_keypoints_b", "num_matches", "local_match_ratio",
    "matched_source_cell_coverage", "matched_target_cell_coverage",
    "symmetric_match_coverage", "source_max_cell_match_fraction",
    "target_max_cell_match_fraction", "source_match_entropy",
    "target_match_entropy", "symmetric_match_entropy", "exact_pixel_duplicate",
    "rgb_pixel_sha_a", "rgb_pixel_sha_b", "image_file_sha_a", "image_file_sha_b",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def decode_original_rgb(path: str | Path) -> tuple[np.ndarray, dict[str, str]]:
    """Decode pixels without orientation, size, crop, or padding operations.

    The byte hash fingerprints the very same encoded bytes used by the decoder.
    RGB conversion is explicit; only the resulting RGB 640x480 pixels are used.
    """
    path = Path(path)
    data = path.read_bytes()
    try:
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
            if rgb.size != IMAGE_SIZE:
                raise ValueError(f"expected original RGB 640x480, got {rgb.size}")
            pixels = np.array(rgb, dtype=np.uint8, order="C", copy=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid original image {path}: {error}") from error
    return pixels, {
        "image_file_sha256": hashlib.sha256(data).hexdigest(),
        "rgb_pixel_sha256": hashlib.sha256(pixels.tobytes(order="C")).hexdigest(),
    }


def rgb_pixel_sha256(path: str | Path) -> str:
    return decode_original_rgb(path)[1]["rgb_pixel_sha256"]


def normalized_grid_cells(points, width=640, height=480) -> np.ndarray:
    """Map independent image endpoints to 8x8 cells, tolerating roundoff only."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        points = points.reshape(0, 2)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("matched points must be a finite Nx2 array")
    if width <= 0 or height <= 0:
        raise ValueError("image geometry must be positive")
    bounds = np.array([width, height], dtype=np.float64)
    # A subpixel shift is not roundoff. This tolerance only admits numerical
    # boundary noise (including exactly W/H from an upstream float conversion).
    tolerance = np.maximum(bounds * 1e-7, 1e-6)
    if np.any(points < -tolerance) or np.any(points > bounds + tolerance):
        raise ValueError("matched coordinates are outside the original image")
    clipped = np.maximum(0.0, np.minimum(points, np.nextafter(bounds, 0.0)))
    cells = np.floor(GRID_SIZE * clipped / bounds).astype(np.int64)
    return cells[:, 1] * GRID_SIZE + cells[:, 0]


def _spatial_diagnostics(points) -> tuple[float, float, float]:
    cells = normalized_grid_cells(points)
    if len(cells) == 0:
        return 0.0, 0.0, 0.0
    counts = np.bincount(cells, minlength=GRID_SIZE**2)
    probabilities = counts[counts > 0].astype(np.float64) / len(cells)
    return (
        float(np.count_nonzero(counts) / GRID_SIZE**2),
        float(counts.max() / len(cells)),
        float(-np.sum(probabilities * np.log(probabilities)) / np.log(GRID_SIZE**2)),
    )


def structural_metrics(num_keypoints_a: int, num_keypoints_b: int,
                       matched_points_a, matched_points_b) -> dict:
    """Describe correspondences without cross-place coordinate displacement."""
    for value in (num_keypoints_a, num_keypoints_b):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("keypoint counts must be nonnegative integers")
    a, b = np.asarray(matched_points_a), np.asarray(matched_points_b)
    if len(a) != len(b):
        raise ValueError("matched endpoint counts disagree")
    count = len(a)
    denominator = min(num_keypoints_a, num_keypoints_b)
    if count > denominator:
        raise ValueError("match count exceeds the smaller keypoint count")
    source = _spatial_diagnostics(a)
    target = _spatial_diagnostics(b)
    return {
        "num_keypoints_a": int(num_keypoints_a),
        "num_keypoints_b": int(num_keypoints_b), "num_matches": count,
        "local_match_ratio": float(count / denominator) if denominator else 0.0,
        "matched_source_cell_coverage": source[0],
        "matched_target_cell_coverage": target[0],
        "symmetric_match_coverage": min(source[0], target[0]),
        "source_max_cell_match_fraction": source[1],
        "target_max_cell_match_fraction": target[1],
        "source_match_entropy": source[2], "target_match_entropy": target[2],
        "symmetric_match_entropy": min(source[2], target[2]),
    }


@dataclass
class LocalMatchResult:
    metrics: dict
    points_a: np.ndarray
    points_b: np.ndarray
    scores: np.ndarray | None


class BoundedFeatureCache:
    """Small LRU with an explicit feature-count bound; no disk feature bank."""

    def __init__(self, capacity=4):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 0 <= capacity <= 32:
            raise ValueError("feature cache capacity must be between 0 and 32")
        self.capacity = capacity
        self._entries = OrderedDict()

    def get(self, image_id):
        if image_id not in self._entries:
            return None
        self._entries.move_to_end(image_id)
        return self._entries[image_id]

    def put(self, image_id, features):
        if self.capacity == 0:
            return
        self._entries[image_id] = features
        self._entries.move_to_end(image_id)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def __len__(self):
        return len(self._entries)


def source_provenance(repo_root: Path) -> dict:
    vendor = repo_root / "third_party/LightGlue"
    paths = sorted(vendor.rglob("*.py"))
    paths += [vendor / "pyproject.toml", vendor / "requirements.txt"]
    hashes = {path.relative_to(repo_root).as_posix(): sha256_file(path) for path in paths}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "source_sha256": hashes}


def _state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(canonical_json_bytes([name, str(value.dtype), list(value.shape)]))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


class StructuralMatcher:
    """Frozen ALIKED / LightGlue, using original 640x480 RGB tensors only."""

    def __init__(self, device="cuda", seed=42, feature_cache_size=4, repo_root=None):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        vendor = (root / "third_party/LightGlue").resolve()
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        from lightglue import ALIKED, LightGlue
        import lightglue

        if not Path(lightglue.__file__).resolve().is_relative_to(vendor):
            raise RuntimeError("LightGlue import did not resolve to the read-only vendor")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.device = torch.device(device)
        self.extractor = ALIKED(**ALIKED_CONFIG).eval().to(self.device)
        self.matcher = LightGlue(**LIGHTGLUE_CONFIG).eval().to(self.device)
        self.cache = BoundedFeatureCache(feature_cache_size)
        self.fingerprints = {}
        self.provenance = {
            "aliked_weights_sha256": _state_sha256(self.extractor),
            "lightglue_weights_sha256": _state_sha256(self.matcher),
            "aliked_effective_config": vars(self.extractor.conf),
            "lightglue_effective_config": vars(self.matcher.conf),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "device": str(self.device),
            "deterministic_algorithms": True, "feature_cache_size": feature_cache_size,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
            "package_versions": {name: importlib.metadata.version(name) for name in (
                "torch", "torchvision", "numpy", "Pillow", "kornia", "opencv-python",
            )},
        }

    def set_image_fingerprints(self, fingerprints):
        self.fingerprints = fingerprints

    def _features(self, image_id: str, path: Path):
        cached = self.cache.get(image_id)
        if cached is not None:
            return cached
        pixels, fingerprint = decode_original_rgb(path)
        if image_id in self.fingerprints and fingerprint != self.fingerprints[image_id]:
            raise ValueError(f"input image bytes changed during measurement: {image_id}")
        self.fingerprints[image_id] = fingerprint
        image = self.torch.from_numpy(pixels.transpose(2, 0, 1).copy()).to(
            device=self.device, dtype=self.torch.float32,
        ) / 255.0
        with self.torch.inference_mode():
            features = self.extractor.extract(image, resize=None)
        size = features["image_size"].detach().cpu().numpy()
        if size.shape != (1, 2) or not np.array_equal(size[0], IMAGE_SIZE):
            raise ValueError(f"ALIKED feature image_size disagrees with original input: {size}")
        self.cache.put(image_id, features)
        return features

    def match(self, image_id_a: str, path_a: Path, image_id_b: str, path_b: Path) -> LocalMatchResult:
        with self.torch.inference_mode():
            a = self._features(image_id_a, path_a)
            b = self._features(image_id_b, path_b)
            count_a, count_b = int(a["keypoints"].shape[1]), int(b["keypoints"].shape[1])
            if not min(count_a, count_b):
                points_a = points_b = np.empty((0, 2), dtype=np.float32)
                scores = np.empty(0, dtype=np.float32)
            else:
                prediction = self.matcher({"image0": a, "image1": b})
                indices = prediction["matches"][0].detach().cpu().numpy()
                if indices.ndim != 2 or indices.shape[1] != 2:
                    raise ValueError("LightGlue returned an invalid match index array")
                if (indices < 0).any() or (indices[:, 0] >= count_a).any() or (indices[:, 1] >= count_b).any():
                    raise ValueError("LightGlue returned an out-of-range match index")
                if len(np.unique(indices[:, 0])) != len(indices) or len(np.unique(indices[:, 1])) != len(indices):
                    raise ValueError("LightGlue correspondences are not one-to-one")
                points_a = a["keypoints"][0].detach().cpu().numpy()[indices[:, 0]]
                points_b = b["keypoints"][0].detach().cpu().numpy()[indices[:, 1]]
                scores = prediction.get("scores")
                scores = scores[0].detach().cpu().numpy() if scores is not None else None
                if scores is not None and (len(scores) != len(indices) or not np.isfinite(scores).all()):
                    raise ValueError("LightGlue returned invalid match scores")
        metrics = structural_metrics(count_a, count_b, points_a, points_b)
        fingerprint_a, fingerprint_b = self.fingerprints[image_id_a], self.fingerprints[image_id_b]
        metrics.update({
            "exact_pixel_duplicate": fingerprint_a["rgb_pixel_sha256"] == fingerprint_b["rgb_pixel_sha256"],
            "rgb_pixel_sha_a": fingerprint_a["rgb_pixel_sha256"],
            "rgb_pixel_sha_b": fingerprint_b["rgb_pixel_sha256"],
            "image_file_sha_a": fingerprint_a["image_file_sha256"],
            "image_file_sha_b": fingerprint_b["image_file_sha256"],
        })
        return LocalMatchResult(metrics, points_a, points_b, scores)


def deterministic_match_indices(scores, num_matches: int, max_matches=100) -> np.ndarray:
    """Select by descending confidence then original correspondence index."""
    if num_matches < 0 or not 0 <= max_matches <= 100:
        raise ValueError("invalid match count or visualization limit (maximum 100)")
    indices = np.arange(num_matches)
    if scores is not None:
        scores = np.asarray(scores)
        if scores.shape != (num_matches,) or not np.isfinite(scores).all():
            raise ValueError("match scores must be finite and aligned")
        indices = np.lexsort((indices, -scores))
    return indices[:max_matches]
