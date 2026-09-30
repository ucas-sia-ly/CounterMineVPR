"""Small inference wrapper around the unmodified SALAD submodule."""

from contextlib import nullcontext
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T


REPO_ROOT = Path(__file__).resolve().parents[2]
SALAD_DIR = REPO_ROOT / "salad"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 定义图像预处理变换
# 包括调整图像大小、转换为张量、归一化
@lru_cache(maxsize=8)
def _preprocess_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )

# 定义图像预处理函数
# 对输入图像进行调整大小、转换为张量、归一化等变换
def preprocess_image(image: Image.Image, image_size: int = 322) -> torch.Tensor:
    """Resize to a square and apply the transform used by salad/eval.py."""
    return _preprocess_transform(image_size)(image.convert("RGB"))

# 定义SALAD编码器类
# 用于批量编码图像，返回归一化后的描述符
class SaladEncoder:
    """Extract L2-normalized SALAD descriptors in bounded batches."""

    MODEL_NAME = "dinov2_salad"

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
            str(SALAD_DIR), self.MODEL_NAME, source="local", pretrained=True
        ).eval().to(self.device)

# 定义迭代编码函数
# 对输入图像进行批量编码，返回每个批次的起始行索引和描述符
# 描述符为 CPU 上的张量，形状为 [N, D]，其中 N 为图像数量，D 为描述符维度
    def iter_encode(
        self, images: Iterable[Image.Image | str | Path]
    ) -> Iterator[tuple[int, torch.Tensor]]:
        """Yield ``(start_row_index, CPU descriptors)`` in input order."""
        source = iter(images)
        start = 0
        while items := list(islice(source, self.batch_size)):
            tensors = []
            for item in items:
                if isinstance(item, Image.Image):
                    tensors.append(preprocess_image(item, self.image_size))
                elif isinstance(item, (str, Path)):
                    with Image.open(item) as image:
                        tensors.append(preprocess_image(image, self.image_size))
                else:
                    raise TypeError(f"Unsupported image type: {type(item).__name__}")

            batch = torch.stack(tensors).to(self.device)
            del tensors
            with torch.no_grad():
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with autocast:
                    output = self.model(batch)
                descriptors = F.normalize(output.float(), p=2, dim=1).cpu()

            count = len(items)
            del batch, output, items
            yield start, descriptors
            start += count

# 定义编码函数
# 对输入图像进行批量编码，返回归一化后的描述符
    def encode(self, images: Iterable[Image.Image | str | Path]) -> torch.Tensor:
        """Return CPU descriptors in input order, with shape [N, D]."""
        batches = [descriptors for _, descriptors in self.iter_encode(images)]
        if not batches:
            raise ValueError("images must contain at least one PIL image or path")
        return torch.cat(batches, dim=0)
