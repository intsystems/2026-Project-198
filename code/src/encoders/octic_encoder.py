"""Octic ViT encoder (D4-equivariant).

Wraps the octic_vits library (Nordstrom et al., 2025). Falls back to
standard ViT-L/16 with a warning if octic_vits is not installed.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from src.encoders import register_encoder

logger = logging.getLogger(__name__)

_OCTIC_AVAILABLE = False
try:
    from octic_vits import octic_deit3_large_patch16_224
    _OCTIC_AVAILABLE = True
except ImportError:
    logger.warning(
        "octic_vits not installed. OcticViTEncoder will fall back to ViT-L/16. "
        "Install via: pip install git+https://github.com/nordstrom/octic-vit.git"
    )


@register_encoder("octic_vit")
class OcticViTEncoder(nn.Module):
    """D4-equivariant ViT encoder using Octic weight sharing.

    By Schur's lemma, equivariant linear maps act independently on each
    irreducible component of D4. Invariant pooling projects onto the
    trivial component A1.

    Args:
        freeze: If True, all backbone parameters are frozen.
        weights_path: Path to D4 DINOv2 checkpoint.
    """

    FEATURE_DIM: int = 1024

    def __init__(
        self,
        freeze: bool = True,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.feature_dim = self.FEATURE_DIM

        if _OCTIC_AVAILABLE:
            self.backbone = octic_deit3_large_patch16_224()
            if weights_path:
                self._load_weights(weights_path)
        else:
            from torchvision.models import ViT_L_16_Weights, vit_l_16
            logger.warning("Using standard ViT-L/16 as Octic fallback")
            self.backbone = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
            self.backbone.heads = nn.Identity()

        from torchvision.models import ViT_L_16_Weights
        self.preprocess = ViT_L_16_Weights.DEFAULT.transforms()

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("OcticViTEncoder: backbone frozen")

    def _load_weights(self, path: str) -> None:
        """Load D4-equivariant pretrained weights."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model", "state_dict"):
                if key in state:
                    state = state[key]
                    break
        model_state = self.backbone.state_dict()
        filtered = {
            k: v for k, v in state.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        self.backbone.load_state_dict(filtered, strict=False)
        logger.info(
            "OcticViTEncoder: loaded %d/%d keys from %s",
            len(filtered), len(model_state), path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract D4-invariant embedding.

        Args:
            x: Input tensor [B, 3, 224, 224].

        Returns:
            Embedding tensor [B, 1024].
        """
        return self.backbone(x)
