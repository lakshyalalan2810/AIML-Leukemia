"""Single clean preprocessing module for ConvNeXtV2-Tiny and EfficientNet-B1."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def apply_clahe(img: Image.Image, clip_limit: float = 2.0, grid_size: tuple[int, int] = (8, 8)) -> Image.Image:
    """Optional CLAHE preprocessing condition.

    Operates in LAB color space on the L channel.
    Disabled by default in all primary runs.
    """
    try:
        import cv2
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)
    except ImportError:
        # If cv2 is not available, return original image unmodified
        return img


class PreprocessingPipeline:
    """Standardized preprocessing callable."""

    def __init__(
        self,
        input_size: int,
        use_clahe: bool = False,
        resize_strategy: str = "direct",
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        self.input_size = input_size
        self.use_clahe = use_clahe
        if resize_strategy not in {"direct", "fit_pad"}:
            raise ValueError("resize_strategy must be 'direct' or 'fit_pad'")
        self.resize_strategy = resize_strategy
        self.mean = mean
        self.std = std

        self.normalize = T.Normalize(mean=list(mean), std=list(std))

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1. RGB verification
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 2. Optional CLAHE (disabled in primary)
        if self.use_clahe:
            img = apply_clahe(img)

        # 3. Fixed deterministic geometry; C-NMC keeps the historical direct default.
        if self.resize_strategy == "direct":
            img = img.resize((self.input_size, self.input_size), resample=Image.BICUBIC)
        else:
            scale = min(self.input_size / img.width, self.input_size / img.height)
            resized = img.resize(
                (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                resample=Image.BICUBIC,
            )
            fill = tuple(round(value * 255) for value in self.mean)
            img = Image.new("RGB", (self.input_size, self.input_size), fill)
            img.paste(resized, ((self.input_size - resized.width) // 2, (self.input_size - resized.height) // 2))

        # 4. To tensor [0, 1]
        tensor = T.functional.to_tensor(img)

        # 5. ImageNet normalization
        tensor = self.normalize(tensor)

        return tensor


def get_transforms(
    input_size: int,
    use_clahe: bool = False,
    resize_strategy: str = "direct",
) -> Callable[[Image.Image], torch.Tensor]:
    """Factory for standard deterministic preprocessing pipeline."""
    return PreprocessingPipeline(
        input_size=input_size, use_clahe=use_clahe, resize_strategy=resize_strategy
    )


def get_dataset_transforms(
    dataset: str, input_size: int, use_clahe: bool = False
) -> Callable[[Image.Image], torch.Tensor]:
    """Preserve morphology for ALL-IDB; retain C-NMC's locked direct resize."""
    strategy = "fit_pad" if dataset.lower().replace("-", "") in {"allidb1", "allidb2"} else "direct"
    return get_transforms(input_size, use_clahe, strategy)
