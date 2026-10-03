"""CPU checks for cached initialization; no training or network requests."""

from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from countermine.training.cached_dinov2 import cached_dinov2_runtime
from countermine.training.step3a_config import sha256_file


class CachedDINOv2Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.hub_dir = Path(self.directory.name)
        self.repository = self.hub_dir / "facebookresearch_dinov2_main"
        (self.repository / "dinov2").mkdir(parents=True)
        (self.repository / "hubconf.py").write_text("# cached hub entrypoint\n")
        (self.repository / "dinov2/backbone.py").write_text("# cached implementation\n")
        self.checkpoint = self.hub_dir / "checkpoints/dinov2_vitb14_pretrain.pth"
        self.checkpoint.parent.mkdir()
        self.checkpoint.write_bytes(b"cached pretrained weight fixture")
        self.model = object()
        self.hub = SimpleNamespace(get_dir=Mock(return_value=str(self.hub_dir)),
                                   load=Mock(return_value=self.model), urlopen=Mock(),
                                   download_url_to_file=Mock())
        self.torch = SimpleNamespace(hub=self.hub)

    def test_exact_upstream_request_uses_local_source_without_changing_model_kwargs(self):
        original_load = self.hub.load
        state = random.getstate()
        with cached_dinov2_runtime(torch_module=self.torch) as report:
            actual = self.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.assertIs(actual, self.model)
        original_load.assert_called_once_with(str(self.repository), "dinov2_vitb14", source="local")
        self.assertIs(self.hub.load, original_load)
        self.assertEqual(random.getstate(), state)
        self.assertFalse(report["network_downloads"])
        self.assertEqual(report["pretrained_checkpoint_sha256"], sha256_file(self.checkpoint))
        self.assertEqual(report["source_file_count"], 2)
        self.assertNotIn(str(self.hub_dir), str(report))
        self.hub.urlopen.assert_not_called()
        self.hub.download_url_to_file.assert_not_called()

    def test_asset_evidence_is_reproducible_and_detects_source_or_weight_changes(self):
        with cached_dinov2_runtime(torch_module=self.torch) as first:
            pass
        with cached_dinov2_runtime(torch_module=self.torch) as repeated:
            pass
        self.assertEqual(first, repeated)
        (self.repository / "dinov2/backbone.py").write_text("# changed source\n")
        with cached_dinov2_runtime(torch_module=self.torch) as changed_source:
            pass
        self.assertNotEqual(first["source_inventory_sha256"], changed_source["source_inventory_sha256"])
        self.checkpoint.write_bytes(b"changed weights")
        with cached_dinov2_runtime(torch_module=self.torch) as changed_weights:
            pass
        self.assertNotEqual(first["pretrained_checkpoint_sha256"], changed_weights["pretrained_checkpoint_sha256"])

    def test_missing_or_empty_assets_fail_before_model_loading_and_never_download(self):
        for kind in ("hubconf", "package", "weights", "empty_weights"):
            with self.subTest(kind=kind):
                if kind == "hubconf":
                    path = self.repository / "hubconf.py"
                    before = path.read_bytes()
                    path.unlink()
                elif kind == "package":
                    path = self.repository / "dinov2"
                    path.rename(self.repository / "temporarily_moved")
                else:
                    path = self.checkpoint
                    before = path.read_bytes()
                    if kind == "weights":
                        path.unlink()
                    else:
                        path.write_bytes(b"")
                with self.assertRaisesRegex(RuntimeError, "Restore"):
                    with cached_dinov2_runtime(torch_module=self.torch):
                        self.fail("Missing assets reached model construction")
                if kind == "package":
                    (self.repository / "temporarily_moved").rename(path)
                else:
                    path.write_bytes(before)
                self.hub.load.assert_not_called()
                self.hub.download_url_to_file.assert_not_called()

    def test_network_attempts_are_blocked_and_runtime_functions_are_restored_on_error(self):
        original_load, original_open, original_download = self.hub.load, self.hub.urlopen, self.hub.download_url_to_file
        for name in ("urlopen", "download_url_to_file"):
            with self.subTest(function=name):
                original_load.side_effect = lambda *args, **kwargs: getattr(self.hub, name)("https://example.invalid")
                with self.assertRaisesRegex(RuntimeError, "Network access is disabled"):
                    with cached_dinov2_runtime(torch_module=self.torch):
                        self.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
                self.assertIs(self.hub.load, original_load)
                self.assertIs(self.hub.urlopen, original_open)
                self.assertIs(self.hub.download_url_to_file, original_download)
                original_open.assert_not_called()
                original_download.assert_not_called()

    def test_unexpected_repo_model_or_override_is_rejected(self):
        requests = [(("unrelated/repo", "dinov2_vitb14"), {}),
                    (("facebookresearch/dinov2", "dinov2_vitg14"), {}),
                    (("facebookresearch/dinov2", "dinov2_vitb14"), {"force_reload": True}),
                    (("facebookresearch/dinov2", "dinov2_vitb14"), {"source": "github"})]
        for args, kwargs in requests:
            with self.subTest(request=args, overrides=kwargs):
                with cached_dinov2_runtime(torch_module=self.torch):
                    with self.assertRaisesRegex(RuntimeError, "Unexpected"):
                        self.hub.load(*args, **kwargs)
        self.hub.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
