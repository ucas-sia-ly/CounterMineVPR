"""Unregistered cross-place matching for the diagnostic Step 3A pilot.

Different places have independent coordinate systems. These metrics count
matches and occupied normalized cells only; they never compare matched point
displacement or fit a geometric transform. Feature extraction and matching use
the frozen, lazy ALIKED/LightGlue adapter without caching a dataset's features.
"""

import numpy as np

from countermine.probe.local_fidelity import (
    LocalFidelityMatcher,
    MatchResult,
)


NATIVE_WIDTH = 640
NATIVE_HEIGHT = 480
GRID_SIZE = 8


def _dimension(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _points(value, name, width, height):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), including (0, 2) when empty")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain finite coordinates")
    if np.any(points < 0) or np.any(points >= np.array([width, height])):
        raise ValueError(f"{name} must lie inside its {width}x{height} image")
    return points


def normalized_matched_cell_coverage(points, *, width=NATIVE_WIDTH, height=NATIVE_HEIGHT):
    """Fraction of occupied half-open 8x8 cells in one image's coordinates."""
    width = _dimension(width, "width")
    height = _dimension(height, "height")
    points = _points(points, "matched points", width, height)
    cells = np.floor(points * GRID_SIZE / np.array([width, height])).astype(np.int64)
    return float(len(np.unique(cells, axis=0)) / (GRID_SIZE ** 2))


def compute_cross_place_metrics(
    points_source,
    points_target,
    num_keypoints_source,
    num_keypoints_target,
    *,
    width_source=NATIVE_WIDTH,
    height_source=NATIVE_HEIGHT,
    width_target=NATIVE_WIDTH,
    height_target=NATIVE_HEIGHT,
):
    """Measure unregistered match support, independently on both image sides.

    Empty match support has ratio and coverages 0.0, including when either
    extractor returned no keypoints. Coordinates may differ arbitrarily
    between the two images; no pixel-distance criterion applies.
    """
    width_source = _dimension(width_source, "width_source")
    height_source = _dimension(height_source, "height_source")
    width_target = _dimension(width_target, "width_target")
    height_target = _dimension(height_target, "height_target")
    source = _points(points_source, "points_source", width_source, height_source)
    target = _points(points_target, "points_target", width_target, height_target)
    if source.shape != target.shape:
        raise ValueError("Matched source and target points must have equal shapes")
    n_source = _count(num_keypoints_source, "num_keypoints_source")
    n_target = _count(num_keypoints_target, "num_keypoints_target")
    denominator = min(n_source, n_target)
    num_matches = len(source)
    if num_matches > denominator:
        raise ValueError("num_matches cannot exceed either keypoint count")
    return {
        "num_keypoints_source": n_source,
        "num_keypoints_target": n_target,
        "num_matches": num_matches,
        "cross_match_ratio": float(num_matches / denominator) if denominator else 0.0,
        "matched_source_cell_coverage": normalized_matched_cell_coverage(
            source, width=width_source, height=height_source,
        ),
        "matched_target_cell_coverage": normalized_matched_cell_coverage(
            target, width=width_target, height=height_target,
        ),
    }


def cross_place_metrics_from_match(result):
    """Consume matched CPU arrays without reading ``MatchResult.displacements``."""
    if not isinstance(result, MatchResult):
        raise TypeError("result must be MatchResult")
    return compute_cross_place_metrics(
        result.points_source,
        result.points_relit,
        result.num_keypoints_source,
        result.num_keypoints_relit,
        width_source=result.canonical_width,
        height_source=result.canonical_height,
        width_target=result.canonical_width,
        height_target=result.canonical_height,
    )


class Step3ALocalMatcher(LocalFidelityMatcher):
    """Share one model load between fidelity and unrelated-place matching.

    Inherited ``match`` remains the source-to-relight fidelity path.
    ``match_cross_place`` bypasses that path entirely: no displacement, R4/R8,
    homography, or fundamental matrix is computed for unrelated places.
    """

    def __init__(self, config=None):
        super().__init__(config)
        if self.config.max_keypoints != 2048:
            raise ValueError("Step 3A freezes ALIKED max_num_keypoints at 2048")

    def match_cross_place(self, features_source, features_target):
        """Return matches in each image's native full-FOV coordinate system."""
        for features in (features_source, features_target):
            if self._feature_dimensions(features) != (NATIVE_WIDTH, NATIVE_HEIGHT):
                raise ValueError("Step 3A cross-place matching requires native 640x480 images")
        self._ensure_models()
        torch = self._torch
        source = features_source["keypoints"]
        target = features_target["keypoints"]
        for keypoints in (source, target):
            if keypoints.ndim != 3 or keypoints.shape[0] != 1 or keypoints.shape[2] != 2:
                raise ValueError("Matching requires batched keypoints with shape [1,N,2]")
            _points(keypoints[0].detach().cpu().numpy(), "extracted keypoints", NATIVE_WIDTH, NATIVE_HEIGHT)
        n_source, n_target = int(source.shape[1]), int(target.shape[1])
        if min(n_source, n_target) == 0:
            return MatchResult(
                n_source, n_target, np.empty((0, 2)), np.empty((0, 2)), np.empty(0),
                NATIVE_WIDTH, NATIVE_HEIGHT,
            )
        with torch.inference_mode():
            prediction = self.matcher({"image0": features_source, "image1": features_target})
            matches = prediction["matches"][0]
            scores = prediction["scores"][0]
            if matches.ndim != 2 or matches.shape[1] != 2 or scores.shape != (len(matches),):
                raise ValueError("LightGlue must return matches [N,2] and scores [N]")
            if matches.dtype not in (torch.int32, torch.int64):
                raise ValueError("LightGlue match indices must be integer tensors")
            if len(matches) and (
                bool((matches < 0).any()) or bool((matches[:, 0] >= n_source).any())
                or bool((matches[:, 1] >= n_target).any())
            ):
                raise ValueError("LightGlue match indices are outside extracted keypoints")
            result = MatchResult(
                n_source, n_target,
                source[0, matches[:, 0]].detach().cpu().numpy().copy(),
                target[0, matches[:, 1]].detach().cpu().numpy().copy(),
                scores.detach().cpu().numpy().copy(),
                NATIVE_WIDTH, NATIVE_HEIGHT,
            )
        # Pure support validation; never call the same-coordinate fidelity API.
        cross_place_metrics_from_match(result)
        if not np.isfinite(result.match_scores).all():
            raise ValueError("Match confidences must be finite")
        return result
