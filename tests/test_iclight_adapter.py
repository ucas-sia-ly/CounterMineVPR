"""CPU-only checks for the diagnostic IC-Light adapter; no model downloads."""

import dataclasses
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from countermine.probe.iclight_adapter import (
    ICLightAdapter,
    ICLightConfig,
    _apply_offset,
    _configure_unet,
    _numpy_to_tensor,
    _tensor_to_numpy,
)


class _FakeVAE:
    """Retain actual CPU tensor math while recording the official VAE passes."""

    device = torch.device("cpu")
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=0.25)

    def __init__(self):
        self.encoded = []
        self.decoded = []

    def encode(self, images):
        assert torch.is_inference_mode_enabled()
        self.encoded.append(images.clone())
        pooled = torch.nn.functional.adaptive_avg_pool2d(images, (64, 64))
        latents = torch.cat((pooled, pooled[:, :1]), dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latents))

    def decode(self, latents):
        assert torch.is_inference_mode_enabled()
        self.decoded.append(latents.clone())
        pixels = torch.nn.functional.interpolate(latents[:, :3], size=(512, 512))
        return SimpleNamespace(sample=pixels.clamp(-1, 1))


class _FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        assert torch.is_inference_mode_enabled()
        self.calls.append(kwargs)
        latents = torch.rand((1, 4, 64, 64), generator=kwargs["generator"])
        return SimpleNamespace(images=latents * 0.25)


class _FakeRMBG:
    def __init__(self, alpha=0.25):
        self.feeds = []
        self.alpha = alpha

    def __call__(self, feed):
        assert torch.is_inference_mode_enabled()
        self.feeds.append(feed.clone())
        return [[torch.full((1, 1, 1024, 1024), self.alpha)]]


class _TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.calls = []

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        self.calls.append((sample.clone(), kwargs))
        return sample


class ICLightAdapterTest(unittest.TestCase):
    def _adapter_with_cpu_components(self):
        adapter = ICLightAdapter(ICLightConfig(device="cpu"))
        adapter.device = torch.device("cpu")
        adapter.vae = _FakeVAE()
        adapter.unet = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
        adapter.t2i_pipe = _FakePipeline()
        adapter.i2i_pipe = _FakePipeline()
        adapter._ensure_models = mock.Mock()
        adapter._encode_prompt_pair = mock.Mock(return_value=(
            torch.zeros((1, 3, 4)), torch.ones((1, 3, 4)),
        ))
        return adapter

    def test_fixed_first_audit_defaults_are_serializable_and_frozen(self):
        config = ICLightConfig()
        self.assertEqual(config.width, 512)
        self.assertEqual(config.height, 512)
        self.assertEqual(config.num_samples, 1)
        self.assertEqual(config.seed, 12345)
        self.assertEqual(config.steps, 25)
        self.assertEqual(config.cfg, 2.0)
        self.assertEqual(config.highres_scale, 1.0)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.prompt, (
            "soft diffuse overcast daylight, uniform outdoor illumination, natural lighting"
        ))
        json.dumps(dataclasses.asdict(config))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.seed = 6

    def test_config_rejects_noncanonical_or_multiple_output_settings(self):
        for overrides in (
            {"width": 256}, {"height": 640}, {"num_samples": 2},
            {"highres_scale": 1.5}, {"seed": -1}, {"seed": True},
            {"steps": 0}, {"steps": 2.5}, {"cfg": float("nan")},
            {"highres_denoise": 0}, {"lowres_denoise": 1.5},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    ICLightConfig(**overrides)

    def test_config_rejects_prompt_changes_and_directional_backgrounds(self):
        for overrides in (
            {"prompt": "dramatic lighting"}, {"added_prompt": "best quality"},
            {"background": "Left Light"}, {"rmbg_sigma": 0.1},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    ICLightConfig(**overrides)

    def test_import_and_constructor_do_not_import_model_or_ui_modules(self):
        # A fresh interpreter distinguishes lazy imports from modules loaded by
        # other tests. The guard fails before any model or UI side effect occurs.
        script = """
import builtins
original_import = builtins.__import__
forbidden = {'gradio', 'gradio_demo', 'diffusers', 'transformers',
             'safetensors', 'briarmbg'}
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in forbidden:
        raise AssertionError('Eager dependency import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from countermine.probe.iclight_adapter import ICLightAdapter
adapter = ICLightAdapter()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_constructor_does_not_initialize_weights(self):
        with mock.patch.object(ICLightAdapter, "_ensure_models") as load_models, \
             mock.patch.object(ICLightAdapter, "_ensure_rmbg") as load_rmbg:
            ICLightAdapter()
        load_models.assert_not_called()
        load_rmbg.assert_not_called()

    def test_numpy_conversion_preserves_official_127_normalization(self):
        pixels = np.array([[[0, 127, 255]]], dtype=np.uint8)
        result = _numpy_to_tensor([pixels])
        self.assertEqual(tuple(result.shape), (1, 3, 1, 1))
        torch.testing.assert_close(
            result.flatten(), torch.tensor([-1.0, 0.0, 255.0 / 127.0 - 1.0]),
        )
        self.assertEqual(pixels.tolist(), [[[0, 127, 255]]])

    def test_unet_architecture_preserves_original_channels_and_concatenates_conditions(self):
        unet = _TinyUNet()
        original_weight = unet.conv_in.weight.detach().clone()
        original_bias = unet.conv_in.bias
        _configure_unet(unet)
        self.assertEqual(unet.conv_in.in_channels, 8)
        torch.testing.assert_close(unet.conv_in.weight[:, :4], original_weight)
        torch.testing.assert_close(unet.conv_in.weight[:, 4:], torch.zeros_like(original_weight))
        self.assertIs(unet.conv_in.bias, original_bias)
        sample = torch.zeros((4, 4, 2, 2))
        conditions = torch.stack((torch.ones((4, 2, 2)), torch.full((4, 2, 2), 2.0)))
        result = unet(
            sample, 0, torch.empty(0),
            cross_attention_kwargs={"concat_conds": conditions},
            return_dict=False,
        )
        self.assertEqual(tuple(result.shape), (4, 8, 2, 2))
        torch.testing.assert_close(result[:, :4], sample)
        torch.testing.assert_close(result[:, 4:], torch.cat((conditions, conditions)))
        self.assertEqual(unet.calls[0][1]["cross_attention_kwargs"], {})
        self.assertFalse(unet.calls[0][1]["return_dict"])

    def test_local_fc_checkpoint_is_added_to_original_unet_weights(self):
        from safetensors.torch import save_file

        unet = _configure_unet(_TinyUNet())
        original = {key: tensor.clone() for key, tensor in unet.state_dict().items()}
        offset = {key: torch.full_like(tensor, 0.25) for key, tensor in original.items()}
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "fc_offset.safetensors"
            save_file(offset, str(checkpoint))
            _apply_offset(unet, checkpoint)

        for key, tensor in unet.state_dict().items():
            with self.subTest(key=key):
                torch.testing.assert_close(tensor, original[key] + offset[key])

    def test_invalid_fc_checkpoint_keys_or_shapes_fail_before_changing_weights(self):
        from safetensors.torch import save_file

        unet = _configure_unet(_TinyUNet())
        original = {key: tensor.clone() for key, tensor in unet.state_dict().items()}
        valid = {key: torch.full_like(tensor, 0.25) for key, tensor in original.items()}
        missing_key = {"conv_in.weight": valid["conv_in.weight"]}
        wrong_shape = dict(valid, **{"conv_in.bias": torch.ones(3)})
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "invalid_offset.safetensors"
            for offset in (missing_key, wrong_shape):
                with self.subTest(keys=list(offset), shapes=[t.shape for t in offset.values()]):
                    save_file(offset, str(checkpoint))
                    with self.assertRaises(ValueError):
                        _apply_offset(unet, checkpoint)
                    for key, tensor in unet.state_dict().items():
                        torch.testing.assert_close(tensor, original[key])

    def test_official_rmbg_resizes_mask_and_composites_over_127_gray(self):
        adapter = ICLightAdapter(ICLightConfig(device="cpu"))
        adapter.device = torch.device("cpu")
        adapter.rmbg = _FakeRMBG()
        source = np.empty((512, 512, 3), dtype=np.uint8)
        source[:] = (0, 127, 255)
        original = source.copy()
        with mock.patch.object(adapter, "_ensure_rmbg") as load_rmbg, torch.inference_mode():
            result = adapter._run_rmbg(source)
        load_rmbg.assert_called_once()
        self.assertEqual(result.shape, (512, 512, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result[0, 0].tolist(), [95, 127, 159])
        self.assertTrue(np.array_equal(source, original))
        feed = adapter.rmbg.feeds[0]
        self.assertEqual(tuple(feed.shape), (1, 3, 1024, 1024))
        torch.testing.assert_close(
            feed[0, :, 0, 0], torch.tensor([-1.0, 0.0, 255.0 / 127.0 - 1.0]),
        )

    def test_official_rmbg_rejects_nonfinite_alpha_before_compositing(self):
        adapter = ICLightAdapter(ICLightConfig(device="cpu"))
        adapter.device = torch.device("cpu")
        source = np.full((512, 512, 3), 127, dtype=np.uint8)
        for bad_value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad_value=bad_value):
                adapter.rmbg = _FakeRMBG(bad_value)
                with mock.patch.object(adapter, "_ensure_rmbg"), torch.inference_mode():
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        adapter._run_rmbg(source)

    def test_tensor_conversion_rejects_nonfinite_before_uint8_quantization(self):
        for bad_value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises((ValueError, RuntimeError)):
                    _tensor_to_numpy(torch.full((1, 3, 2, 2), bad_value))

    def test_invalid_input_or_mode_fails_before_model_loading(self):
        adapter = ICLightAdapter()
        with mock.patch.object(adapter, "_ensure_models") as load_models:
            for image, mode in (
                (Image.new("RGB", (511, 512)), "full_scene"),
                (Image.new("RGB", (512, 513)), "official_rmbg"),
                (Image.new("L", (512, 512)), "full_scene"),
                (Image.new("RGBA", (512, 512)), "official_rmbg"),
                (Image.new("RGB", (512, 512)), "invalid_mode"),
            ):
                with self.subTest(size=image.size, image_mode=image.mode, mode=mode):
                    with self.assertRaises((AssertionError, ValueError, TypeError)):
                        adapter.relight(image, mode)
            load_models.assert_not_called()

    def test_full_scene_bypasses_rmbg_and_preserves_source(self):
        adapter = ICLightAdapter(ICLightConfig(device="cpu"))
        adapter.device = torch.device("cpu")
        source = Image.new("RGB", (512, 512), (14, 123, 217))
        original = source.tobytes()

        def process(foreground):
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertTrue(np.array_equal(foreground, np.asarray(source)))
            foreground[:] = 0  # An inference backend may mutate its own input.
            return [np.full((512, 512, 3), 120, dtype=np.uint8)]

        with mock.patch.object(adapter, "_ensure_models"), \
             mock.patch.object(adapter, "_ensure_rmbg") as load_rmbg, \
             mock.patch.object(adapter, "_run_rmbg") as run_rmbg, \
             mock.patch.object(adapter, "_process", side_effect=process), \
             mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")):
            result = adapter.relight(source, "full_scene")

        load_rmbg.assert_not_called()
        run_rmbg.assert_not_called()
        self.assertEqual(source.tobytes(), original)
        self.assertEqual(result.size, (512, 512))
        self.assertEqual(result.mode, "RGB")
        self.assertIsNot(source, result)
        self.assertFalse(torch.is_inference_mode_enabled())
        self.assertGreaterEqual(adapter.last_run_stats["elapsed_seconds"], 0.0)
        self.assertIsNone(adapter.last_run_stats["peak_cuda_memory_allocated_bytes"])
        self.assertIsNone(adapter.last_run_stats["peak_cuda_memory_reserved_bytes"])

    def test_official_mode_conditions_on_the_rmbg_result(self):
        adapter = ICLightAdapter(ICLightConfig(device="cpu"))
        adapter.device = torch.device("cpu")
        source = Image.new("RGB", (512, 512), (7, 40, 220))
        original = source.tobytes()
        foreground = np.full((512, 512, 3), 127, dtype=np.uint8)

        def remove_background(input_array):
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertTrue(np.array_equal(input_array, np.asarray(source)))
            input_array[:] = 255
            return foreground

        def process(actual_foreground):
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertTrue(np.array_equal(actual_foreground, foreground))
            return [foreground.copy()]

        with mock.patch.object(adapter, "_ensure_models"), \
             mock.patch.object(adapter, "_run_rmbg", side_effect=remove_background) as rmbg, \
             mock.patch.object(adapter, "_process", side_effect=process):
            result = adapter.relight(source, "official_rmbg")

        rmbg.assert_called_once()
        self.assertEqual(source.tobytes(), original)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (127, 127, 127))

    def test_output_must_be_one_finite_canonical_rgb_image(self):
        adapter = ICLightAdapter(ICLightConfig(device="cpu"))
        adapter.device = torch.device("cpu")
        source = Image.new("RGB", (512, 512))
        valid = np.zeros((512, 512, 3), dtype=np.uint8)
        invalid_outputs = (
            [], [valid, valid], [np.zeros((511, 512, 3), dtype=np.uint8)],
            [np.zeros((512, 512), dtype=np.uint8)],
            [np.full((512, 512, 3), float("nan"))],
        )
        for outputs in invalid_outputs:
            with self.subTest(shapes=[out.shape for out in outputs]):
                with mock.patch.object(adapter, "_ensure_models"), \
                     mock.patch.object(adapter, "_process", return_value=outputs):
                    with self.assertRaises((AssertionError, ValueError, RuntimeError)):
                        adapter.relight(source, "full_scene")

    def test_official_two_pass_schedule_and_seed_are_repeatable_on_cpu(self):
        adapter = self._adapter_with_cpu_components()
        source = Image.new("RGB", (512, 512), (127, 127, 127))
        original = source.tobytes()
        with mock.patch.object(adapter, "_ensure_rmbg") as load_rmbg:
            first = adapter.relight(source, "full_scene")
            second = adapter.relight(source, "full_scene")

        load_rmbg.assert_not_called()
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(source.tobytes(), original)
        self.assertEqual(len(adapter.t2i_pipe.calls), 2)
        self.assertEqual(len(adapter.i2i_pipe.calls), 2)
        self.assertEqual(len(adapter.vae.encoded), 6)
        self.assertEqual(len(adapter.vae.decoded), 4)
        initial = adapter.t2i_pipe.calls[0]
        refinement = adapter.i2i_pipe.calls[0]
        for call in (initial, refinement):
            self.assertEqual(call["width"], 512)
            self.assertEqual(call["height"], 512)
            self.assertEqual(call["num_images_per_prompt"], 1)
            self.assertEqual(call["guidance_scale"], 2.0)
            self.assertEqual(call["output_type"], "latent")
            self.assertEqual(call["generator"].initial_seed(), 12345)
            self.assertEqual(
                tuple(call["cross_attention_kwargs"]["concat_conds"].shape),
                (1, 4, 64, 64),
            )
        self.assertIs(initial["generator"], refinement["generator"])
        self.assertIsNot(initial["generator"], adapter.t2i_pipe.calls[1]["generator"])
        self.assertEqual(initial["num_inference_steps"], 25)
        self.assertEqual(refinement["strength"], adapter.config.highres_denoise)
        self.assertEqual(
            refinement["num_inference_steps"],
            round(25 / adapter.config.highres_denoise),
        )
        torch.testing.assert_close(adapter.vae.encoded[0], torch.zeros((1, 3, 512, 512)))


if __name__ == "__main__":
    unittest.main()
