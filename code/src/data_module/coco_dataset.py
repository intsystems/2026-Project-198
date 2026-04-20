"""COCO2017 pair dataset for siamese training.

Implements the cross-batch pairing strategy from Dorin et al. (2024):
a batch of B images yields B^2 pairs with labels y_ij = 1[i == j].
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


class COCOPairDataset(Dataset):
    """COCO2017 image dataset for siamese plagiarism detection training.

    Each __getitem__ returns a single image (two augmented views are generated
    at the collate/training level via cross-batch pairing).

    Args:
        root: Path to COCO2017 images directory (e.g. train2017/).
        transform: Torchvision transform applied to each image.
        max_images: Optional cap on number of images to load.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        max_images: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.image_paths = self._collect_images(max_images)
        logger.info("COCOPairDataset: %d images from %s", len(self.image_paths), root)

    def _collect_images(self, max_images: Optional[int]) -> List[Path]:
        """Collect image file paths from directory."""
        extensions = {".jpg", ".jpeg", ".png"}
        paths = sorted(
            p for p in self.root.iterdir()
            if p.suffix.lower() in extensions
        )
        if max_images is not None:
            paths = paths[:max_images]
        return paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and transform a single image.

        Args:
            idx: Image index.

        Returns:
            Transformed image tensor [C, H, W].
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def build_coco_dataloaders(
    cfg: Dict[str, Any],
    train_transform: Callable,
    val_transform: Callable,
) -> Dict[str, DataLoader]:
    """Build train/val dataloaders from COCO2017.

    Uses two datasets with separate transforms (train vs val) and the same
    index split as ``torch.utils.data.random_split`` for reproducibility.

    Args:
        cfg: Full config dict.
        train_transform: Transform for training subset.
        val_transform: Transform for validation subset.

    Returns:
        Dict with 'train' and 'val' DataLoader objects.
    """
    coco_root = cfg["data"]["coco_root"]
    train_dir = os.path.join(coco_root, "train2017")

    if not os.path.isdir(train_dir):
        train_dir = coco_root
        logger.warning("train2017/ not found, using coco_root directly: %s", coco_root)

    train_ds = COCOPairDataset(root=train_dir, transform=train_transform)
    val_ds = COCOPairDataset(root=train_dir, transform=val_transform)

    n = len(train_ds)
    val_n = int(n * cfg["data"]["val_size"])
    train_size = n - val_n

    generator = torch.Generator().manual_seed(cfg["seed"])
    indices = torch.randperm(n, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_n]

    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    logger.info(
        "COCO dataloaders: train=%d, val=%d (same split as random_split)",
        train_size,
        val_n,
    )
    return {"train": train_loader, "val": val_loader}
