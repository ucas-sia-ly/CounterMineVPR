"""Use existing DINOv2 hub assets without GitHub branch discovery.

This scoped adapter leaves upstream SALAD and the model constructor untouched.
It neither downloads assets nor changes pretrained/model arguments or seeds.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from .step3a_config import sha256_file


@contextmanager
def cached_dinov2_runtime(*, torch_module=None):
    if torch_module is None:
        import torch as torch_module

    hub = torch_module.hub
    hub_dir = Path(hub.get_dir())
    repository = hub_dir / "facebookresearch_dinov2_main"
    checkpoint = hub_dir / "checkpoints/dinov2_vitb14_pretrain.pth"
    if not (repository / "hubconf.py").is_file() or not (repository / "dinov2").is_dir():
        raise RuntimeError(
            "Step 3A requires the existing torch hub cache facebookresearch_dinov2_main "
            "(hubconf.py and dinov2/). Restore the DINOv2 cache used for preparation; "
            "this launcher does not download or select another backbone."
        )
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(
            "Step 3A requires the cached checkpoints/dinov2_vitb14_pretrain.pth. "
            "Restore the pretrained cache used for preparation; no download is attempted."
        )
    sources = [repository / "hubconf.py", *sorted((repository / "dinov2").rglob("*.py"))]
    hashes = sorted((path.relative_to(repository).as_posix(), sha256_file(path)) for path in sources)
    report = {
        "mode": "existing_local_torch_hub_cache",
        "repository": "facebookresearch/dinov2:main", "backbone": "dinov2_vitb14",
        "network_downloads": False,
        "source_file_count": len(hashes),
        "source_inventory_sha256": hashlib.sha256(
            json.dumps(hashes, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "source_inventory_format": "SHA256 of compact JSON sorted (relative path, file SHA256) pairs",
        "pretrained_checkpoint": checkpoint.name,
        "pretrained_checkpoint_sha256": sha256_file(checkpoint),
    }
    original_load = hub.load

    def load(repo, model, *args, **kwargs):
        if repo != "facebookresearch/dinov2" or model != "dinov2_vitb14":
            raise RuntimeError("Unexpected torch hub model request in frozen Step 3A initialization")
        if "source" in kwargs or kwargs.get("force_reload"):
            raise RuntimeError("Unexpected torch hub source/reload override in frozen Step 3A initialization")
        return original_load(str(repository), model, *args, source="local", **kwargs)

    def reject_network(*args, **kwargs):
        raise RuntimeError("Network access is disabled during cached Step 3A model initialization")

    with patch.object(hub, "load", side_effect=load), \
            patch.object(hub, "urlopen", side_effect=reject_network), \
            patch.object(hub, "download_url_to_file", side_effect=reject_network):
        yield report
