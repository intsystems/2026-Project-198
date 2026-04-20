"""Block P (photometric) and Block G (geometric) augmentation pipelines.

Block P is applied to all encoders. Block G is applied only to ViT-L/16 (augmented).
"""

import logging
from typing import Any, Dict, List

import torchvision.transforms as T

logger = logging.getLogger(__name__)


def build_photometric_transform(cfg: Dict[str, Any]) -> T.Compose:
    """Build Block P: photometric augmentations applied to all encoders.

    Args:
        cfg: augmentation config section.

    Returns:
        Composed torchvision transform.
    """
    aug = cfg["augmentation"]
    transforms: List[T.Transform] = [
        T.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        T.ColorJitter(
            brightness=aug["color_jitter"]["brightness"],
            contrast=aug["color_jitter"]["contrast"],
            saturation=aug["color_jitter"]["saturation"],
            hue=aug["color_jitter"]["hue"],
        ),
        T.RandomGrayscale(p=aug["grayscale_prob"]),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=aug["gaussian_blur_kernel"])],
            p=aug["gaussian_blur_prob"],
        ),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    logger.info("Block P (photometric) augmentation built")
    return T.Compose(transforms)


def build_geometric_transform(cfg: Dict[str, Any]) -> T.Compose:
    """Build Block G: geometric augmentations for ViT-L/16 (augmented) only.

    Args:
        cfg: augmentation config section.

    Returns:
        Composed torchvision transform.
    """
    aug = cfg["augmentation"]
    transforms: List[T.Transform] = [
        T.RandomHorizontalFlip(p=aug["hflip_prob"]),
        T.RandomVerticalFlip(p=aug["vflip_prob"]),
        T.RandomChoice([T.RandomRotation((a, a)) for a in aug["rotation_choices"]]),
        T.RandomResizedCrop(
            cfg["data"]["image_size"],
            scale=tuple(aug["resized_crop_scale"]),
            ratio=tuple(aug["resized_crop_ratio"]),
        ),
    ]
    logger.info("Block G (geometric) augmentation built")
    return T.Compose(transforms)


def build_clean_transform(cfg: Dict[str, Any]) -> T.Compose:
    """Resize + ToTensor + ImageNet normalize (no augmentation)."""
    tfm = T.Compose(
        [
            T.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    logger.info("Clean transform (resize + normalize) built")
    return tfm


def build_train_transform(cfg: Dict[str, Any]) -> T.Compose:
    """Train transform: optional Block G, then Block P or clean (per flags)."""
    aug = cfg["augmentation"]
    if aug.get("skip_photometric_train", False):
        photo = build_clean_transform(cfg)
    else:
        photo = build_photometric_transform(cfg)
    if aug.get("skip_geometric", True):
        return photo
    geo = build_geometric_transform(cfg)
    return T.Compose(list(geo.transforms) + list(photo.transforms))


def build_val_transform(cfg: Dict[str, Any]) -> T.Compose:
    """Val transform: never Block G; Block P or clean per skip_photometric_val."""
    if cfg["augmentation"].get("skip_photometric_val", True):
        return build_clean_transform(cfg)
    return build_photometric_transform(cfg)
