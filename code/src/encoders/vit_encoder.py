"""Standard ViT-L/16 encoder (baseline and augmented variants).

No equivariance guarantee. Used with Block P only (baseline) or
Block P + Block G (augmented).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ViT_L_16_Weights, vit_l_16

from src.encoders import register_encoder

logger = logging.getLogger(__name__)


class _ViTBase(nn.Module):
    """Shared ViT-L/16 backbone logic for baseline and augmented variants.

    Args:
        freeze: If True, all backbone parameters are frozen.
        weights_path: Optional path to custom pretrained weights.
    """

    FEATURE_DIM: int = 1024

    def __init__(
        self,
        freeze: bool = True,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.feature_dim = self.FEATURE_DIM

        if weights_path:
            self.backbone = vit_l_16(weights=None)
            self.backbone.heads = nn.Identity()
            self._load_custom_weights(weights_path)
        else:
            weights = ViT_L_16_Weights.DEFAULT
            self.backbone = vit_l_16(weights=weights)
            self.backbone.heads = nn.Identity()

        self.preprocess = ViT_L_16_Weights.DEFAULT.transforms()

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("%s: backbone frozen", self.__class__.__name__)

    def _load_custom_weights(self, path: str) -> None:
        """Load weights with flexible key matching."""
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            for key in ("model", "state_dict", "teacher", "model_state_dict"):
                if key in raw:
                    raw = raw[key]
                    break
        cleaned = {}
        for k, v in raw.items():
            nk = k
            for prefix in ("backbone.", "module.", "encoder."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
        cleaned = {k: v for k, v in cleaned.items() if not k.startswith("head.")}

        model_state = self.backbone.state_dict()
        filtered = {
            k: v for k, v in cleaned.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        self.backbone.load_state_dict(filtered, strict=False)
        logger.info(
            "%s: loaded %d/%d weight keys from %s",
            self.__class__.__name__, len(filtered), len(model_state), path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract [CLS] token embedding.

        Args:
            x: Input tensor [B, 3, 224, 224].

        Returns:
            Embedding tensor [B, 1024].
        """
        return self.backbone(x)


@register_encoder("vit_baseline")
class ViTBaselineEncoder(_ViTBase):
    """ViT-L/16 baseline encoder. Trained with Block P only."""
    pass


@register_encoder("vit_augmented")
class ViTAugmentedEncoder(_ViTBase):
    """ViT-L/16 augmented encoder. Trained with Block P + Block G."""
    pass
