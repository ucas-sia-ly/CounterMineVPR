"""Lazy, foreground-conditioned IC-Light inference for diagnostic probes.

Adapted from lllyasviel/IC-Light's ``gradio_demo.py`` (Apache-2.0; the
original license is retained in ``third_party/IC-Light/LICENSE``). Changes:
remove Gradio/global model initialization, restrict geometry to the canonical
probe space, add full-scene conditioning, and expose reproducible settings.
The original eight-channel UNet, additive FC checkpoint, sampler, normalization,
and two-stage inference are retained. Synthetic outputs are diagnostic only.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np
from PIL import Image


IC_LIGHT_PROMPT = (
    "soft diffuse overcast daylight, uniform outdoor illumination, natural lighting"
)
MODES = ("official_rmbg", "full_scene")
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ICLightConfig:
    """Complete inference settings for the fixed, mild 512-pixel smoke audit.

    ``background=None`` selects the official text-to-image first pass, without
    directional gradients. ``lowres_denoise`` is retained for configuration
    parity but is unused on that branch. The official refinement pass still
    runs when ``highres_scale=1.0``. Model assets use the Hugging Face cache,
    never the upstream checkout's ``models/`` directory.
    """

    width: int = 512
    height: int = 512
    num_samples: int = 1
    seed: int = 12345
    steps: int = 25
    cfg: float = 2.0
    highres_scale: float = 1.0
    highres_denoise: float = 0.5
    lowres_denoise: float = 0.9
    prompt: str = IC_LIGHT_PROMPT
    added_prompt: str = ""
    negative_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality"
    background: None = None
    rmbg_sigma: float = 0.0
    text_encoder_dtype: str = "float16"
    unet_dtype: str = "float16"
    vae_dtype: str = "bfloat16"
    rmbg_dtype: str = "float32"
    scheduler_num_train_timesteps: int = 1000
    scheduler_beta_start: float = 0.00085
    scheduler_beta_end: float = 0.012
    scheduler_algorithm_type: str = "sde-dpmsolver++"
    scheduler_use_karras_sigmas: bool = True
    scheduler_steps_offset: int = 1
    device: str = "cuda"
    base_model: str = "stablediffusionapi/realistic-vision-v51"
    base_revision: str | None = None
    offset_model: str = "lllyasviel/ic-light"
    offset_filename: str = "iclight_sd15_fc.safetensors"
    offset_revision: str | None = None
    rmbg_model: str = "briaai/RMBG-1.4"
    rmbg_revision: str | None = None
    iclight_root: str = "third_party/IC-Light"
    checkpoint_path: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False

    def __post_init__(self):
        if (self.width, self.height) != (512, 512):
            raise ValueError("The audit requires width=height=512")
        if self.num_samples != 1:
            raise ValueError("The audit requires num_samples=1")
        if self.highres_scale != 1.0:
            raise ValueError("The canonical audit requires highres_scale=1.0")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(self.cfg) or self.cfg < 1.0:
            raise ValueError("cfg must be finite and at least 1.0")
        for name in ("highres_denoise", "lowres_denoise"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.prompt != IC_LIGHT_PROMPT or self.added_prompt:
            raise ValueError("Step 2B uses the fixed mild prompt without added prompts")
        if self.background is not None or self.rmbg_sigma != 0.0:
            raise ValueError("The first audit uses background=None and rmbg_sigma=0")
        official_settings = {
            "text_encoder_dtype": "float16", "unet_dtype": "float16",
            "vae_dtype": "bfloat16", "rmbg_dtype": "float32",
            "scheduler_num_train_timesteps": 1000, "scheduler_beta_start": 0.00085,
            "scheduler_beta_end": 0.012, "scheduler_algorithm_type": "sde-dpmsolver++",
            "scheduler_use_karras_sigmas": True, "scheduler_steps_offset": 1,
        }
        for name, official_value in official_settings.items():
            if getattr(self, name) != official_value:
                raise ValueError(f"The audit preserves IC-Light's official {name}={official_value!r}")


def _numpy_to_tensor(images):
    """Official normalization: pixel 127 is exactly zero (not /127.5)."""
    import torch

    return (torch.from_numpy(np.stack(images, axis=0)).float() / 127.0 - 1.0).movedim(-1, 1)


def _tensor_to_numpy(images):
    import torch

    result = []
    for image in images:
        if not torch.isfinite(image).all().item():
            raise ValueError("IC-Light produced non-finite pixels")
        pixels = image.movedim(0, -1) * 127.5 + 127.5
        result.append(pixels.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8))
    return result


def _resize_without_crop(image, width, height):
    return np.array(Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS))


def _resize_and_center_crop(image, width, height):
    """The official inference resize; canonical inputs retain their geometry."""
    source = Image.fromarray(image)
    scale = max(width / source.width, height / source.height)
    resized_width = int(round(source.width * scale))
    resized_height = int(round(source.height * scale))
    resized = source.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    return np.array(resized.crop(((resized_width - width) / 2,
                                  (resized_height - height) / 2,
                                  (resized_width + width) / 2,
                                  (resized_height + height) / 2)))


def _configure_unet(unet):
    """Install the original 4+4-channel FC conditioning architecture and hook."""
    import torch

    original_conv = unet.conv_in
    if original_conv.in_channels != 4:
        raise ValueError("IC-Light requires the original four-channel SD1.5 UNet")
    with torch.no_grad():
        conv = torch.nn.Conv2d(8, original_conv.out_channels,
                               original_conv.kernel_size, original_conv.stride,
                               original_conv.padding,
                               device=original_conv.weight.device,
                               dtype=original_conv.weight.dtype)
        conv.weight.zero_()
        conv.weight[:, :4].copy_(original_conv.weight)
        conv.bias = original_conv.bias
        unet.conv_in = conv

    original_forward = unet.forward

    def conditioned_forward(sample, timestep, encoder_hidden_states, **kwargs):
        concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
        if concat.shape[0] == 0 or sample.shape[0] % concat.shape[0]:
            raise ValueError("Conditioning batch does not divide the UNet sample batch")
        concat = torch.cat([concat] * (sample.shape[0] // concat.shape[0]), dim=0)
        kwargs["cross_attention_kwargs"] = {}
        return original_forward(torch.cat([sample, concat], dim=1), timestep,
                                encoder_hidden_states, **kwargs)

    unet.forward = conditioned_forward
    return unet


def _apply_offset(unet, checkpoint_path):
    """Add the FC delta in-place to avoid a third full UNet copy in host RAM."""
    import torch
    from safetensors.torch import load_file

    offset = load_file(str(checkpoint_path), device="cpu")
    origin = unet.state_dict()
    if origin.keys() != offset.keys():
        raise ValueError("IC-Light FC checkpoint keys do not match the expanded UNet")
    for key in origin:
        if origin[key].shape != offset[key].shape:
            raise ValueError(f"IC-Light checkpoint shape mismatch for {key}")
    with torch.no_grad():
        for key in origin:
            origin[key].add_(offset[key])


def _read_only_bria_class(iclight_root):
    """Load upstream architecture source without creating third-party bytecode."""
    source = Path(iclight_root) / "briarmbg.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing IC-Light RMBG architecture: {source}")
    module_name = "countermine.probe._iclight_briarmbg"
    module = ModuleType(module_name)
    module.__file__ = str(source)
    module.__package__ = "countermine.probe"
    sys.modules[module_name] = module
    try:
        exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module.BriaRMBG


def _resolve_project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else _REPO_ROOT / path


def _reject_third_party_write(path):
    resolved = Path(path).expanduser().resolve()
    if resolved.is_relative_to((_REPO_ROOT / "third_party").resolve()):
        raise ValueError("Model caches must be outside read-only third_party/")
    return resolved


class ICLightAdapter:
    """Sequential IC-Light adapter; construction/import never initializes models.

    ``full_scene`` neither loads nor invokes RMBG. ``official_rmbg`` lazily loads
    the unchanged Bria architecture and follows ``process_relight``'s alpha
    blending onto gray 127. ``last_run_stats`` reports elapsed wall time and,
    on CUDA, peak allocated/reserved bytes for the completed call, including
    lazy loading on the first invocation.
    """

    def __init__(self, config: ICLightConfig = ICLightConfig()):
        self.config = config
        self.device = None
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.t2i_pipe = None
        self.i2i_pipe = None
        self.rmbg = None
        self.last_run_stats = {}

    def _hub_options(self, revision=None):
        from huggingface_hub import constants

        cache = self.config.cache_dir or constants.HF_HUB_CACHE
        cache = _reject_third_party_write(_resolve_project_path(cache))
        return {"cache_dir": str(cache), "local_files_only": self.config.local_files_only,
                "revision": revision}

    def _ensure_models(self):
        if self.t2i_pipe is not None and self.i2i_pipe is not None:
            return
        try:
            import torch
            from diffusers import (AutoencoderKL, UNet2DConditionModel,
                                   DPMSolverMultistepScheduler,
                                   StableDiffusionPipeline, StableDiffusionImg2ImgPipeline)
            from diffusers.models.attention_processor import AttnProcessor2_0
            from huggingface_hub import hf_hub_download
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as error:
            raise RuntimeError("IC-Light inference requires torch, diffusers, transformers, "
                               "safetensors, and huggingface_hub; install them in the "
                               "audit environment without modifying third_party/") from error

        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("IC-Light CUDA inference requested but CUDA is unavailable")
        options = self._hub_options(self.config.base_revision)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.config.base_model,
                                                       subfolder="tokenizer", **options)
        self.text_encoder = CLIPTextModel.from_pretrained(self.config.base_model,
                                                         subfolder="text_encoder", **options)
        self.vae = AutoencoderKL.from_pretrained(self.config.base_model,
                                                subfolder="vae", **options)
        self.unet = UNet2DConditionModel.from_pretrained(self.config.base_model,
                                                        subfolder="unet", **options)
        _configure_unet(self.unet)
        if self.config.checkpoint_path:
            checkpoint = _resolve_project_path(self.config.checkpoint_path)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"IC-Light FC checkpoint does not exist: {checkpoint}")
        else:
            checkpoint = hf_hub_download(repo_id=self.config.offset_model,
                                         filename=self.config.offset_filename,
                                         **self._hub_options(self.config.offset_revision))
        _apply_offset(self.unet, checkpoint)
        self.text_encoder = self.text_encoder.to(
            device=self.device, dtype=getattr(torch, self.config.text_encoder_dtype)).eval()
        self.vae = self.vae.to(device=self.device, dtype=getattr(torch, self.config.vae_dtype)).eval()
        self.unet = self.unet.to(device=self.device, dtype=getattr(torch, self.config.unet_dtype)).eval()
        self.unet.set_attn_processor(AttnProcessor2_0())
        self.vae.set_attn_processor(AttnProcessor2_0())
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.config.scheduler_num_train_timesteps,
            beta_start=self.config.scheduler_beta_start, beta_end=self.config.scheduler_beta_end,
            algorithm_type=self.config.scheduler_algorithm_type,
            use_karras_sigmas=self.config.scheduler_use_karras_sigmas,
            steps_offset=self.config.scheduler_steps_offset)
        # The official demo shares this scheduler; each pipeline call resets it.
        components = dict(vae=self.vae, text_encoder=self.text_encoder,
                          tokenizer=self.tokenizer, unet=self.unet, scheduler=scheduler,
                          safety_checker=None, requires_safety_checker=False,
                          feature_extractor=None, image_encoder=None)
        self.t2i_pipe = StableDiffusionPipeline(**components)
        self.i2i_pipe = StableDiffusionImg2ImgPipeline(**components)
        self.t2i_pipe.set_progress_bar_config(disable=True)
        self.i2i_pipe.set_progress_bar_config(disable=True)

    def _ensure_rmbg(self):
        if self.rmbg is not None:
            return
        import torch

        bria_class = _read_only_bria_class(_resolve_project_path(self.config.iclight_root))
        self.rmbg = bria_class.from_pretrained(
            self.config.rmbg_model, **self._hub_options(self.config.rmbg_revision)
        ).to(device=self.device, dtype=getattr(torch, self.config.rmbg_dtype)).eval()

    def _run_rmbg(self, image):
        import torch

        with torch.inference_mode():
            self._ensure_rmbg()
            height, width, channels = image.shape
            assert channels == 3
            scale = (256.0 / float(height * width)) ** 0.5
            feed = _resize_without_crop(image, int(64 * round(width * scale)),
                                       int(64 * round(height * scale)))
            feed = _numpy_to_tensor([feed]).to(
                device=self.device, dtype=getattr(torch, self.config.rmbg_dtype))
            alpha = self.rmbg(feed)[0][0]
            alpha = torch.nn.functional.interpolate(alpha, size=(height, width), mode="bilinear")
            alpha = alpha.movedim(1, -1)[0].detach().float().cpu().numpy()
            if not np.isfinite(alpha).all():
                raise ValueError("RMBG produced non-finite alpha values")
            alpha = alpha.clip(0, 1)
            foreground = 127 + (image.astype(np.float32) - 127 + self.config.rmbg_sigma) * alpha
            return foreground.clip(0, 255).astype(np.uint8)

    def _encode_prompt_inner(self, text):
        import torch

        max_length = self.tokenizer.model_max_length
        chunk_length = max_length - 2
        start = self.tokenizer.bos_token_id
        end = self.tokenizer.eos_token_id
        tokens = self.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[start] + tokens[i:i + chunk_length] + [end]
                  for i in range(0, len(tokens), chunk_length)]
        if not chunks:
            chunks = [[start, end]]
        chunks = [(chunk + [end] * max_length)[:max_length] for chunk in chunks]
        ids = torch.tensor(chunks, device=self.device, dtype=torch.int64)
        return self.text_encoder(ids).last_hidden_state

    def _encode_prompt_pair(self):
        import torch

        # An empty added prompt must not introduce an extra comma into the audit.
        positive = self.config.prompt
        if self.config.added_prompt:
            positive += ", " + self.config.added_prompt
        conditioned = self._encode_prompt_inner(positive)
        unconditioned = self._encode_prompt_inner(self.config.negative_prompt)
        count = max(len(conditioned), len(unconditioned))
        conditioned = torch.cat([conditioned] * math.ceil(count / len(conditioned)), dim=0)[:count]
        unconditioned = torch.cat([unconditioned] * math.ceil(count / len(unconditioned)), dim=0)[:count]
        return (torch.cat([chunk[None] for chunk in conditioned], dim=1),
                torch.cat([chunk[None] for chunk in unconditioned], dim=1))

    def _process(self, foreground):
        import torch

        config = self.config
        rng = torch.Generator(device=self.device).manual_seed(config.seed)
        fg = _resize_and_center_crop(foreground, config.width, config.height)
        concat = _numpy_to_tensor([fg]).to(device=self.vae.device, dtype=self.vae.dtype)
        concat = self.vae.encode(concat).latent_dist.mode() * self.vae.config.scaling_factor
        conditioned, unconditioned = self._encode_prompt_pair()
        common = dict(prompt_embeds=conditioned, negative_prompt_embeds=unconditioned,
                      num_images_per_prompt=config.num_samples, generator=rng,
                      output_type="latent", guidance_scale=config.cfg)
        latents = self.t2i_pipe(
            width=config.width, height=config.height, num_inference_steps=config.steps,
            cross_attention_kwargs={"concat_conds": concat}, **common,
        ).images.to(self.vae.dtype) / self.vae.config.scaling_factor
        pixels = _tensor_to_numpy(self.vae.decode(latents).sample)
        pixels = [_resize_without_crop(pixel,
                   int(round(config.width * config.highres_scale / 64.0) * 64),
                   int(round(config.height * config.highres_scale / 64.0) * 64))
                  for pixel in pixels]
        pixels = _numpy_to_tensor(pixels).to(device=self.vae.device, dtype=self.vae.dtype)
        latents = self.vae.encode(pixels).latent_dist.mode() * self.vae.config.scaling_factor
        latents = latents.to(device=self.unet.device, dtype=self.unet.dtype)
        height, width = latents.shape[2] * 8, latents.shape[3] * 8
        fg = _resize_and_center_crop(foreground, width, height)
        concat = _numpy_to_tensor([fg]).to(device=self.vae.device, dtype=self.vae.dtype)
        concat = self.vae.encode(concat).latent_dist.mode() * self.vae.config.scaling_factor
        latents = self.i2i_pipe(
            image=latents, strength=config.highres_denoise, width=width, height=height,
            num_inference_steps=int(round(config.steps / config.highres_denoise)),
            cross_attention_kwargs={"concat_conds": concat}, **common,
        ).images.to(self.vae.dtype) / self.vae.config.scaling_factor
        return _tensor_to_numpy(self.vae.decode(latents).sample)

    def relight(self, image: Image.Image, mode: str) -> Image.Image:
        """Return one canonical RGB probe without ever modifying ``image``."""
        if not isinstance(image, Image.Image):
            raise TypeError("ICLightAdapter requires a PIL image")
        assert image.size == (512, 512), "IC-Light requires an exactly 512x512 canonical input"
        assert image.mode == "RGB", "IC-Light requires an RGB canonical input"
        if mode not in MODES:
            raise ValueError(f"Unknown preprocessing mode {mode!r}; expected one of {MODES}")
        import torch

        self.last_run_stats = {}
        start = time.perf_counter()
        # This copies the pixels even if future conditioning code edits its array.
        foreground = np.array(image, dtype=np.uint8, copy=True)
        run_device = self.device or torch.device(self.config.device)
        cuda = run_device.type == "cuda" and torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize(run_device)
            torch.cuda.reset_peak_memory_stats(run_device)
        with torch.inference_mode():
            self._ensure_models()
            if mode == "official_rmbg":
                foreground = self._run_rmbg(foreground)
            outputs = self._process(foreground)
        if len(outputs) != 1:
            raise ValueError(f"IC-Light returned {len(outputs)} images; expected exactly one")
        pixels = outputs[0]
        if pixels.shape != (512, 512, 3) or not np.isfinite(pixels).all():
            raise ValueError("IC-Light output must be finite 512x512 RGB pixels")
        if pixels.dtype != np.uint8:
            raise ValueError("IC-Light output must use uint8 RGB pixels")
        output = Image.fromarray(pixels)
        assert output.size == (512, 512) and output.mode == "RGB"
        if cuda:
            torch.cuda.synchronize(run_device)
        self.last_run_stats = {
            "elapsed_seconds": time.perf_counter() - start,
            "peak_cuda_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(run_device)) if cuda else None),
            "peak_cuda_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(run_device)) if cuda else None),
        }
        return output
