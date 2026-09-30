"""Small inference wrapper around the unmodified SALAD submodule."""

from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T


REPO_ROOT = Path(__file__).resolve().parents[2]
SALAD_DIR = REPO_ROOT / "salad"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess_image(image: Image.Image, image_size: int = 322) -> torch.Tensor:
    """Resize to a square and apply the transform used by salad/eval.py."""
    transform = T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform(image.convert("RGB"))


class SaladEncoder:
    """Extract L2-normalized SALAD descriptors in bounded batches."""

    def __init__(self, image_size: int = 322, batch_size: int = 8) -> None:
        if image_size <= 0 or image_size % 14 != 0:
            raise ValueError("image_size must be a positive multiple of 14")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not (SALAD_DIR / "hubconf.py").is_file():
            raise FileNotFoundError(f"SALAD submodule is missing: {SALAD_DIR}")

        self.image_size = image_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(
            str(SALAD_DIR), "dinov2_salad", source="local", pretrained=True
        ).eval().to(self.device)

    def encode(self, images: Sequence[Image.Image | str | Path]) -> torch.Tensor:
        """Return CPU descriptors in input order, with shape [N, D]."""
        if not images:
            raise ValueError("images must contain at least one PIL image or path")

        descriptors = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                tensors = []
                for item in images[start : start + self.batch_size]:
                    if isinstance(item, Image.Image):
                        tensors.append(preprocess_image(item, self.image_size))
                    elif isinstance(item, (str, Path)):
                        with Image.open(item) as image:
                            tensors.append(preprocess_image(image, self.image_size))
                    else:
                        raise TypeError(f"Unsupported image type: {type(item).__name__}")

                batch = torch.stack(tensors).to(self.device)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with autocast:
                    output = self.model(batch)
                descriptors.append(F.normalize(output.float(), p=2, dim=1).cpu())
                del batch, output, tensors

        return torch.cat(descriptors, dim=0)
