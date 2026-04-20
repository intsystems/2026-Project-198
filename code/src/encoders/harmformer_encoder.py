"""Harmformer encoder (SE(2)-equivariant).

Achieves SO(2)-invariance via orbit-mean pooling (Reynolds operator):
the embedding is the mean of the backbone outputs over N_rots uniformly
spaced rotations in [0, 360).

Reference: Karella et al. (2024), "Harmformer".
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import ViT_L_16_Weights, vit_l_16

from src.encoders import register_encoder

logger = logging.getLogger(__name__)


@register_encoder("harmformer")
class HarmformerEncoder(nn.Module):
    """SE(2)-equivariant encoder via orbit averaging over rotations.

    For N uniformly spaced rotation angles {k * 360/N}_{k=0}^{N-1},
    the embedding is z = (1/N) sum_k backbone(rotate(x, angle_k)).

    Args:
        n_rots: Number of discrete rotations for orbit averaging.
        freeze: If True, freeze backbone weights.
        weights_path: Optional path to pretrained ViT-L/16 weights.
    """

    FEATURE_DIM: int = 1024

    def __init__(
        self,
        n_rots: int = 8,
        freeze: bool = True,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.feature_dim = self.FEATURE_DIM
        self.n_rots = n_rots

        if weights_path:
            self.backbone = vit_l_16(weights=None)
            self.backbone.heads = nn.Identity()
            self._load_weights(weights_path)
        else:
            weights = ViT_L_16_Weights.DEFAULT
            self.backbone = vit_l_16(weights=weights)
            self.backbone.heads = nn.Identity()

        self.preprocess = ViT_L_16_Weights.DEFAULT.transforms()

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("HarmformerEncoder: backbone frozen, N_rots=%d", n_rots)

    def _load_weights(self, path: str) -> None:
        """Load pretrained weights with flexible key matching."""
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
            "HarmformerEncoder: loaded %d/%d keys from %s",
            len(filtered), len(model_state), path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pool ViT-L/16 embeddings over N_rots equidistant rotations.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Embedding tensor [B, 1024].
        """
        embeddings = []
        for k in range(self.n_rots):
            angle = 360.0 * k / self.n_rots
            if angle == 0.0:
                x_rot = x
            else:
                x_rot = TF.rotate(
                    x, angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                )
            embeddings.append(self.backbone(x_rot))
        return torch.stack(embeddings, dim=1).mean(dim=1)
