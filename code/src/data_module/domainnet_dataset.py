"""DomainNet evaluation dataset for plagiarism detection.

Builds balanced positive/negative pairs per domain for FPR/Recall evaluation.
"""

import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from .transforms import PLAGIARISM_TRANSFORMS

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DomainNetEvalDataset(Dataset):
    """DomainNet balanced evaluation dataset.

    For each domain, generates `pairs_per_domain` positive pairs (source +
    transformed copy) and `pairs_per_domain` negative pairs (two independent
    images from the same domain).

    Args:
        root: Path to DomainNet root (contains domain subdirectories).
        domains: List of domain names to include.
        pairs_per_domain: Number of positive (and negative) pairs per domain.
        image_size: Target image size.
        seed: Random seed for pair construction.
    """

    def __init__(
        self,
        root: str,
        domains: List[str],
        pairs_per_domain: int = 100,
        image_size: int = 224,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.normalize = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.to_tensor = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        self.pairs: List[Dict[str, Any]] = []
        self._build_pairs(domains, pairs_per_domain, seed)
        logger.info(
            "DomainNetEvalDataset: %d pairs (%d domains x %d pos + %d neg)",
            len(self.pairs), len(domains), pairs_per_domain, pairs_per_domain,
        )

    def _collect_domain_images(self, domain: str) -> List[Path]:
        """Collect all image paths for a domain."""
        domain_dir = self.root / domain
        if not domain_dir.is_dir():
            logger.warning("Domain directory not found: %s", domain_dir)
            return []
        extensions = {".jpg", ".jpeg", ".png"}
        paths = sorted(
            p for p in domain_dir.rglob("*")
            if p.suffix.lower() in extensions
        )
        return paths

    def _build_pairs(
        self, domains: List[str], pairs_per_domain: int, seed: int
    ) -> None:
        """Build balanced positive and negative pairs."""
        rng = random.Random(seed)
        transform_names = list(PLAGIARISM_TRANSFORMS.keys())

        for domain in domains:
            images = self._collect_domain_images(domain)
            if len(images) < 2 * pairs_per_domain:
                logger.warning(
                    "Domain %s has %d images, need %d. Using all available.",
                    domain, len(images), 2 * pairs_per_domain,
                )

            rng.shuffle(images)

            # Positive pairs: source + transformed copy
            for i in range(min(pairs_per_domain, len(images))):
                tfm_name = rng.choice(transform_names)
                self.pairs.append({
                    "img1_path": str(images[i]),
                    "img2_path": str(images[i]),
                    "label": 1,
                    "transform": tfm_name,
                    "domain": domain,
                    "is_positive": True,
                })

            # Negative pairs: two different images from same domain
            neg_images = images[:2 * pairs_per_domain]
            for i in range(min(pairs_per_domain, len(neg_images) // 2)):
                idx_a = 2 * i
                idx_b = 2 * i + 1
                self.pairs.append({
                    "img1_path": str(neg_images[idx_a]),
                    "img2_path": str(neg_images[idx_b]),
                    "label": 0,
                    "transform": "none",
                    "domain": domain,
                    "is_positive": False,
                })

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a pair with metadata.

        Returns:
            Dict with keys: img1, img2, label, transform, domain.
        """
        pair = self.pairs[idx]

        img1 = Image.open(pair["img1_path"]).convert("RGB")
        img1_tensor = self.normalize(img1)

        if pair["is_positive"]:
            # Load raw, apply plagiarism transform, then normalize
            img2_raw = self.to_tensor(Image.open(pair["img2_path"]).convert("RGB"))
            img2_transformed = PLAGIARISM_TRANSFORMS[pair["transform"]](img2_raw)
            img2_tensor = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(
                img2_transformed
            )
        else:
            img2 = Image.open(pair["img2_path"]).convert("RGB")
            img2_tensor = self.normalize(img2)

        return {
            "img1": img1_tensor,
            "img2": img2_tensor,
            "label": torch.tensor(pair["label"], dtype=torch.float32),
            "transform": pair["transform"],
            "domain": pair["domain"],
        }
