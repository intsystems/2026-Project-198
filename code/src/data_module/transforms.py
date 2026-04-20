"""Plagiarism transformation classes used for evaluation.

Each transformation simulates a specific plagiarism operation.
"""

import random
from typing import Callable, Dict

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


PLAGIARISM_TRANSFORMS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}


def _register_transform(name: str):
    def decorator(fn: Callable):
        PLAGIARISM_TRANSFORMS[name] = fn
        return fn
    return decorator


@_register_transform("CJ")
def color_jitter(img: torch.Tensor) -> torch.Tensor:
    """ColorJitter plagiarism transformation."""
    pil = T.ToPILImage()(img)
    pil = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(pil)
    return T.ToTensor()(pil)


@_register_transform("GN")
def gaussian_noise(img: torch.Tensor) -> torch.Tensor:
    """Additive Gaussian noise."""
    return (img + 0.05 * torch.randn_like(img)).clamp(0, 1)


@_register_transform("GS")
def gaussian_blur(img: torch.Tensor) -> torch.Tensor:
    """Gaussian blur."""
    return T.GaussianBlur(kernel_size=5)(img)


@_register_transform("R90")
def rotate_90(img: torch.Tensor) -> torch.Tensor:
    """Rotation by 90 degrees."""
    return TF.rotate(img, 90, interpolation=TF.InterpolationMode.BILINEAR)


@_register_transform("R180")
def rotate_180(img: torch.Tensor) -> torch.Tensor:
    """Rotation by 180 degrees."""
    return TF.rotate(img, 180, interpolation=TF.InterpolationMode.BILINEAR)


@_register_transform("R270")
def rotate_270(img: torch.Tensor) -> torch.Tensor:
    """Rotation by 270 degrees."""
    return TF.rotate(img, 270, interpolation=TF.InterpolationMode.BILINEAR)


@_register_transform("WM")
def watermark(img: torch.Tensor) -> torch.Tensor:
    """Overlay a text watermark."""
    from PIL import ImageDraw

    pil = T.ToPILImage()(img)
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    draw.text((w // 10, h // 10), "\u00a9 COPY", fill=(255, 255, 255))
    return T.ToTensor()(pil)


@_register_transform("COMBO")
def combo_transform(img: torch.Tensor) -> torch.Tensor:
    """Random combination of 2-3 transformations."""
    single_names = ["CJ", "GN", "GS", "R90", "R180", "R270", "WM"]
    k = random.randint(2, 3)
    chosen = random.sample(single_names, k)
    result = img
    for name in chosen:
        result = PLAGIARISM_TRANSFORMS[name](result)
    return result


def apply_plagiarism_transform(
    img: torch.Tensor, name: str
) -> torch.Tensor:
    """Apply a named plagiarism transformation.

    Args:
        img: CHW float tensor in [0, 1].
        name: transformation class key.

    Returns:
        Transformed CHW tensor.
    """
    fn = PLAGIARISM_TRANSFORMS[name]
    return fn(img)
