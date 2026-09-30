"""Same-coordinate local-structure fidelity for canonical diagnostic probes.

Metrics are pure NumPy and never register or warp either image. The model
adapter imports the read-only vendored ALIKED/LightGlue lazily, preserving the
512-pixel coordinate system by explicitly disabling extractor resizing.
"""

from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
import random
import re
import sys

import numpy as np
from PIL import Image


CANONICAL_SIZE = 512
GRID_SIZE = 8
EPSILONS = (2, 4, 8, 16)
METRIC_COLUMNS = (
    "num_keypoints_source", "num_keypoints_relit", "num_matches", "match_ratio_min",
    *(f"good_matches_{eps}px" for eps in EPSILONS),
    *(f"repeatability_min_{eps}px" for eps in EPSILONS),
    *(f"precision_matches_{eps}px" for eps in EPSILONS),
    "displacement_mean", "displacement_median", "displacement_q75",
    "displacement_q90", "displacement_q95", "displacement_max",
    "match_score_mean", "match_score_median",
    "occupied_grid_cells_4px", "occupied_grid_cells_8px",
    "grid_coverage_4px", "grid_coverage_8px",
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDORED_ROOT = _REPO_ROOT / "third_party" / "LightGlue"
_VENDORED_PACKAGE = "_countermine_vendored_lightglue"
_EXTRACTOR_SETTINGS = {"detection_threshold": 0.2}
_MATCHER_SETTINGS = {
    "features": "aliked", "depth_confidence": -1, "width_confidence": -1,
    "filter_threshold": 0.1, "mp": False,
}


def _keypoint_count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _canonical_points(value, name):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), including (0, 2) when empty")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain finite coordinates")
    if np.any(points < 0) or np.any(points >= CANONICAL_SIZE):
        raise ValueError(f"{name} must lie inside the canonical 512x512 image")
    return points


def compute_fidelity_metrics(
    points_source, points_relit, match_scores,
    num_keypoints_source, num_keypoints_relit,
):
    """Measure matched displacement directly in canonical pixel coordinates.

    Epsilon thresholds are inclusive. All zero-denominator ratios are 0.0;
    statistics of empty displacement or confidence samples are ``None``.
    Quantiles use NumPy's linear interpolation. Coverage counts occupied cells
    on the source side, with 64-pixel, half-open grid cells.
    """
    source = _canonical_points(points_source, "points_source")
    relit = _canonical_points(points_relit, "points_relit")
    if source.shape != relit.shape:
        raise ValueError("Source and relit matched points must have equal shapes")
    scores = np.asarray(match_scores, dtype=np.float64)
    if scores.shape != (len(source),) or not np.isfinite(scores).all():
        raise ValueError("match_scores must be finite with shape (num_matches,)")
    n_source = _keypoint_count(num_keypoints_source, "num_keypoints_source")
    n_relit = _keypoint_count(num_keypoints_relit, "num_keypoints_relit")
    n_matches = len(source)
    denominator = min(n_source, n_relit)
    if n_matches > denominator:
        raise ValueError("num_matches cannot exceed either keypoint count")
    displacement = np.linalg.norm(source - relit, axis=1)
    result = {
        "num_keypoints_source": n_source,
        "num_keypoints_relit": n_relit,
        "num_matches": n_matches,
        "match_ratio_min": float(n_matches / denominator) if denominator else 0.0,
    }
    for eps in EPSILONS:
        good = int(np.count_nonzero(displacement <= eps))
        result[f"good_matches_{eps}px"] = good
        result[f"repeatability_min_{eps}px"] = float(good / denominator) if denominator else 0.0
        result[f"precision_matches_{eps}px"] = float(good / n_matches) if n_matches else 0.0
    result.update({
        "displacement_mean": float(displacement.mean()) if n_matches else None,
        "displacement_median": float(np.median(displacement)) if n_matches else None,
        **{
            f"displacement_q{q}": float(np.quantile(displacement, q / 100.0)) if n_matches else None
            for q in (75, 90, 95)
        },
        "displacement_max": float(displacement.max()) if n_matches else None,
        "match_score_mean": float(scores.mean()) if n_matches else None,
        "match_score_median": float(np.median(scores)) if n_matches else None,
    })
    for eps in (4, 8):
        good_source = source[displacement <= eps]
        cells = np.floor(good_source / (CANONICAL_SIZE / GRID_SIZE)).astype(np.int64)
        occupied = int(len(np.unique(cells, axis=0)))
        result[f"occupied_grid_cells_{eps}px"] = occupied
        result[f"grid_coverage_{eps}px"] = float(occupied / GRID_SIZE**2)
    return {column: result[column] for column in METRIC_COLUMNS}


@dataclass(frozen=True)
class LocalFidelityConfig:
    """Fixed audit model configuration; construction does not load weights."""

    device: str = "auto"
    max_keypoints: int = 2048
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.device, str) or not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", self.device):
            raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
        if isinstance(self.max_keypoints, bool) or not isinstance(self.max_keypoints, int) or self.max_keypoints < 1:
            raise ValueError("max_keypoints must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")


@dataclass(frozen=True)
class MatchResult:
    """CPU-only arrays for one image pair, with no retained GPU match tensors."""

    num_keypoints_source: int
    num_keypoints_relit: int
    points_source: np.ndarray
    points_relit: np.ndarray
    match_scores: np.ndarray

    @property
    def displacements(self):
        source = np.asarray(self.points_source, dtype=np.float64)
        relit = np.asarray(self.points_relit, dtype=np.float64)
        return np.linalg.norm(source - relit, axis=1)


def _import_vendored_models():
    """Import only required vendored modules, without writing their bytecode."""
    package_dir = _VENDORED_ROOT / "lightglue"
    for source in ("__init__.py", "aliked.py", "utils.py", "lightglue.py"):
        if not (package_dir / source).is_file():
            raise FileNotFoundError(f"Vendored LightGlue source is missing: {package_dir / source}")
    existing = sys.modules.get(_VENDORED_PACKAGE)
    if existing is not None:
        if Path(existing.__file__).resolve() != (package_dir / "__init__.py").resolve():
            raise RuntimeError("Vendored LightGlue package namespace has an unexpected origin")
    else:
        spec = importlib.util.spec_from_file_location(
            _VENDORED_PACKAGE, package_dir / "__init__.py",
            submodule_search_locations=[str(package_dir)],
        )
        # Avoid executing __init__, which imports unrelated feature extractors.
        sys.modules[_VENDORED_PACKAGE] = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        aliked_module = importlib.import_module(f"{_VENDORED_PACKAGE}.aliked")
        lightglue_module = importlib.import_module(f"{_VENDORED_PACKAGE}.lightglue")
        for module, filename in ((aliked_module, "aliked.py"), (lightglue_module, "lightglue.py")):
            if Path(module.__file__).resolve() != (package_dir / filename).resolve():
                raise RuntimeError("Local-fidelity models must come from vendored LightGlue")
        return aliked_module.ALIKED, lightglue_module.LightGlue
    finally:
        sys.dont_write_bytecode = previous


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


class LocalFidelityMatcher:
    """Lazy ALIKED/LightGlue adapter for one-source-at-a-time audit processing."""

    def __init__(self, config=None):
        self.config = config or LocalFidelityConfig()
        if not isinstance(self.config, LocalFidelityConfig):
            raise TypeError("config must be LocalFidelityConfig")
        self.device = None
        self.extractor = None
        self.matcher = None
        self._torch = None
        self._model_metadata = None

    def _ensure_models(self):
        if self.extractor is not None and self.matcher is not None:
            return
        import torch

        device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self.config.device == "auto" else self.config.device
        )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        checkpoint_dir = (Path(torch.hub.get_dir()).expanduser() / "checkpoints").resolve()
        if any(checkpoint_dir.is_relative_to((_REPO_ROOT / name).resolve())
               for name in ("third_party", "salad")):
            raise ValueError("Model caches must be outside read-only third_party/ and salad/")
        ALIKED, LightGlue = _import_vendored_models()
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.device = device
        self._torch = torch
        self.extractor = ALIKED(
            max_num_keypoints=self.config.max_keypoints, **_EXTRACTOR_SETTINGS,
        ).eval().to(device)
        self.matcher = LightGlue(**_MATCHER_SETTINGS).eval().to(device)
        extractor_conf = vars(self.extractor.conf).copy()
        matcher_conf = vars(self.matcher.conf).copy()
        extractor_url = ALIKED.checkpoint_url.format(extractor_conf["model_name"])
        matcher_url = LightGlue.url.format(LightGlue.version, matcher_conf["weights"])
        checkpoint_names = {
            "aliked": (extractor_url, f"{extractor_conf['model_name']}.pth"),
            "lightglue": (
                matcher_url, f"{matcher_conf['weights']}_{LightGlue.version.replace('.', '-')}.pth",
            ),
        }
        checkpoint_metadata = {}
        for name, (url, filename) in checkpoint_names.items():
            path = checkpoint_dir / filename
            checkpoint_metadata[name] = {
                "url": url, "cache_path": str(path),
                "sha256": _sha256(path) if path.is_file() else None,
            }
        self._model_metadata = {
            "extractor_config": extractor_conf,
            "matcher_config": matcher_conf,
            "lightglue_weights_version": LightGlue.version,
            "checkpoints": checkpoint_metadata,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }

    def extract(self, path):
        """Extract a strict canonical RGB PNG without any crop or resize."""
        path = Path(path)
        if path.suffix.lower() != ".png":
            raise ValueError(f"Canonical fidelity probes must use PNG, got {path}")
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512):
                raise ValueError(f"Expected RGB PNG exactly 512x512, got {path}: {image.format}, {image.mode}, {image.size}")
            pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
        self._ensure_models()
        torch = self._torch
        image_tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).to(self.device)
        if tuple(image_tensor.shape) != (3, 512, 512):
            raise ValueError("Fidelity extraction requires tensor shape [3,512,512]")
        with torch.inference_mode():
            features = self.extractor.extract(image_tensor, resize=None)
        keypoints = features["keypoints"]
        if keypoints.ndim != 3 or keypoints.shape[0] != 1 or keypoints.shape[2] != 2:
            raise ValueError("ALIKED must return batched keypoints with shape [1,N,2]")
        _canonical_points(keypoints[0].detach().cpu().numpy(), "extracted keypoints")
        if "image_size" not in features or not torch.equal(
            features["image_size"], torch.tensor([[512, 512]], device=self.device).to(features["image_size"]),
        ):
            raise ValueError("ALIKED features must retain canonical image_size [512,512]")
        return features

    def match(self, features_source, features_relit):
        """Return matched CPU coordinates and confidences without registration."""
        self._ensure_models()
        torch = self._torch
        source = features_source["keypoints"]
        relit = features_relit["keypoints"]
        for keypoints in (source, relit):
            if keypoints.ndim != 3 or keypoints.shape[0] != 1 or keypoints.shape[2] != 2:
                raise ValueError("Matching requires batched keypoints with shape [1,N,2]")
        n_source, n_relit = int(source.shape[1]), int(relit.shape[1])
        if min(n_source, n_relit) == 0:
            return MatchResult(n_source, n_relit, np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
        with torch.inference_mode():
            prediction = self.matcher({"image0": features_source, "image1": features_relit})
            matches = prediction["matches"][0]
            scores = prediction["scores"][0]
            if matches.ndim != 2 or matches.shape[1] != 2 or scores.shape != (len(matches),):
                raise ValueError("LightGlue must return matches [N,2] and scores [N]")
            if matches.dtype not in (torch.int32, torch.int64):
                raise ValueError("LightGlue match indices must be integer tensors")
            if len(matches) and (
                bool((matches < 0).any()) or bool((matches[:, 0] >= n_source).any())
                or bool((matches[:, 1] >= n_relit).any())
            ):
                raise ValueError("LightGlue match indices are outside extracted keypoints")
            result = MatchResult(
                n_source, n_relit,
                source[0, matches[:, 0]].detach().cpu().numpy().copy(),
                relit[0, matches[:, 1]].detach().cpu().numpy().copy(),
                scores.detach().cpu().numpy().copy(),
            )
        compute_fidelity_metrics(
            result.points_source, result.points_relit, result.match_scores,
            result.num_keypoints_source, result.num_keypoints_relit,
        )
        return result

    def runtime_metadata(self):
        """JSON-ready provenance and settings, without loading models or weights."""
        return {
            "config": asdict(self.config),
            "resolved_device": str(self.device) if self.device is not None else None,
            "canonical_size": [512, 512],
            "extract_resize": None,
            "compiled": False,
            "epsilon_pixels": list(EPSILONS),
            "epsilon_comparison": "inclusive <=",
            "grid_shape": [GRID_SIZE, GRID_SIZE],
            "quantile_method": "linear",
            "registration": None,
            "extractor_settings": {"max_num_keypoints": self.config.max_keypoints, **_EXTRACTOR_SETTINGS},
            "matcher_settings": _MATCHER_SETTINGS.copy(),
            "vendored_root": str(_VENDORED_ROOT),
            "vendored_source_sha256": {
                filename: _sha256(_VENDORED_ROOT / filename)
                for filename in ("lightglue/aliked.py", "lightglue/utils.py", "lightglue/lightglue.py")
                if (_VENDORED_ROOT / filename).is_file()
            },
            "versions": {name: _version(name) for name in ("torch", "torchvision", "kornia", "numpy", "pillow")},
            "models": self._model_metadata,
        }
