"""Frozen LightGlue using bounded CPU ALIKED bank reads, without extraction."""

from collections import OrderedDict
from pathlib import Path

import numpy as np

from countermine.mining.full_structural_io import ROOT, read_json
from countermine.mining.local_feature_bank import (
    BANK_DIR, bind_frozen_runtime, load_feature_bank, load_tensor_features,
    positive_integer, require_bank_validation, setup_torch,
)
from countermine.mining.structural_matcher import (
    LIGHTGLUE_CONFIG, LocalMatchResult, _state_sha256, structural_metrics,
)


class BankFeatureCache:
    """CPU-only LRU bounded by number of images, including zero-cache mode."""
    def __init__(self, capacity=128):
        self.capacity = positive_integer(capacity, "feature_cache_size")
        self.entries = OrderedDict()

    def get(self, identity):
        if identity not in self.entries:
            return None
        self.entries.move_to_end(identity)
        return self.entries[identity]

    def put(self, identity, features):
        if not self.capacity:
            return
        self.entries[identity] = features
        self.entries.move_to_end(identity)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)

    def __len__(self):
        return len(self.entries)


class BankedStructuralMatcher:
    def __init__(self, bank_dir=BANK_DIR, device="cuda", seed=42,
                 feature_cache_size=128, repo_root=ROOT, require_validation=True):
        feature_cache_size = positive_integer(feature_cache_size, "feature_cache_size")
        self.bank_dir = Path(bank_dir)
        index, self.bank_summary = load_feature_bank(self.bank_dir, repo_root=repo_root)
        if seed != self.bank_summary["configuration"]["seed"]:
            raise ValueError("LightGlue seed differs from validated ALIKED bank seed")
        self.validation = require_bank_validation(self.bank_dir, self.bank_summary, repo_root=repo_root) if require_validation else None
        self.records = {row["image_id"]: row for row in index.to_dict("records")}
        if feature_cache_size >= len(self.records):
            raise ValueError("feature cache must be smaller than the full image population; never cache the complete bank in RAM")
        self.torch, vendor, self.device, self.provenance = setup_torch(device, seed, repo_root)
        self.matcher = vendor.LightGlue(**LIGHTGLUE_CONFIG).eval().to(self.device)
        self.cache = BankFeatureCache(feature_cache_size)
        self.provenance.update({
            "lightglue_weights_sha256": _state_sha256(self.matcher),
            "lightglue_effective_config": vars(self.matcher.conf),
            "feature_cache_size": feature_cache_size,
            "feature_source": "validated_disk_bank_only",
        })
        frozen = read_json(Path(repo_root) / "cache/countermine_rgb/step2a/measurement_config.json")
        bind_frozen_runtime(self.provenance, frozen, "lightglue")

    def _cpu_features(self, image_id):
        if image_id not in self.records:
            raise ValueError(f"pair image is absent from the validated ALIKED bank: {image_id}")
        cached = self.cache.get(image_id)
        if cached is None:
            cached = load_tensor_features(self.bank_dir, self.records[image_id], torch_module=self.torch)
            self.cache.put(image_id, cached)
        return cached

    def match(self, image_id_a, image_id_b):
        with self.torch.inference_mode():
            cpu_a, cpu_b = self._cpu_features(image_id_a), self._cpu_features(image_id_b)
            a = {key: value.to(self.device) for key, value in cpu_a.items()}
            b = {key: value.to(self.device) for key, value in cpu_b.items()}
            count_a, count_b = int(a["keypoints"].shape[1]), int(b["keypoints"].shape[1])
            if not min(count_a, count_b):
                points_a = points_b = np.empty((0, 2), dtype=np.float32)
                scores = np.empty(0, dtype=np.float32)
            else:
                prediction = self.matcher({"image0": a, "image1": b})
                indices = prediction["matches"][0].detach().cpu().numpy()
                if indices.ndim != 2 or indices.shape[1] != 2 or not np.issubdtype(indices.dtype, np.integer):
                    raise ValueError("LightGlue returned invalid correspondence indices")
                if (indices < 0).any() or (indices[:, 0] >= count_a).any() or (indices[:, 1] >= count_b).any():
                    raise ValueError("LightGlue returned an out-of-range correspondence")
                if len(np.unique(indices[:, 0])) != len(indices) or len(np.unique(indices[:, 1])) != len(indices):
                    raise ValueError("LightGlue correspondences are not one-to-one")
                points_a = cpu_a["keypoints"][0].numpy()[indices[:, 0]]
                points_b = cpu_b["keypoints"][0].numpy()[indices[:, 1]]
                score_tensor = prediction.get("scores")
                scores = score_tensor[0].detach().cpu().numpy() if score_tensor is not None else None
                if scores is not None and (len(scores) != len(indices) or not np.isfinite(scores).all()):
                    raise ValueError("LightGlue returned invalid correspondence confidence")
                del prediction, score_tensor
            del a, b
        metrics = structural_metrics(count_a, count_b, points_a, points_b)
        metrics["min_num_keypoints"] = min(count_a, count_b)
        metrics["exact_pixel_duplicate"] = self.records[image_id_a]["rgb_pixel_sha256"] == self.records[image_id_b]["rgb_pixel_sha256"]
        return LocalMatchResult(metrics, points_a, points_b, scores)
