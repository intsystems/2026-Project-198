"""Shift-Equivariant ViT encoder (Z^2-equivariant).

Implements exact cyclic shift-equivariance via four adaptive polyphase modules
from Rojas-Gomez et al. (CVPR 2024):
- A-token: per-sample polyphase offset selection
- A-WSA: adaptive window self-attention partition
- A-PMerge: polyphase downsampling
- A-RPE: circular relative position encoding

Falls back to a shift-averaging approximation over ViT-L/16 if the
shift-equivariant library is not available.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ViT_L_16_Weights, vit_l_16

from src.encoders import register_encoder

logger = logging.getLogger(__name__)

_SHIFT_EQ_AVAILABLE = False
try:
    from shift_eq_vit import shift_equivariant_vit_large
    _SHIFT_EQ_AVAILABLE = True
except ImportError:
    logger.warning(
        "shift_eq_vit not installed. ShiftEquivariantEncoder will use "
        "shift-averaging approximation over ViT-L/16."
    )


@register_encoder("shift_eq_vit")
class ShiftEquivariantEncoder(nn.Module):
    """Z^2-equivariant encoder using adaptive polyphase ViT modules.

    Exact cyclic shift-equivariance: the embedding is invariant to
    integer pixel shifts (t_{a,b} . x)[i,j,c] = x[(i+a) mod H, (j+b) mod W, c].

    Args:
        freeze: If True, freeze backbone weights.
        n_shifts: Number of cyclic shifts for averaging fallback.
        weights_path: Optional path to pretrained weights.
    """

    FEATURE_DIM: int = 1024

    def __init__(
        self,
        freeze: bool = False,
        n_shifts: int = 4,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.feature_dim = self.FEATURE_DIM
        self.n_shifts = n_shifts
        self._use_native = _SHIFT_EQ_AVAILABLE

        if _SHIFT_EQ_AVAILABLE:
            self.backbone = shift_equivariant_vit_large()
            if weights_path:
                self._load_weights(weights_path)
        else:
            weights = ViT_L_16_Weights.DEFAULT
            self.backbone = vit_l_16(weights=weights)
            self.backbone.heads = nn.Identity()

        self.preprocess = ViT_L_16_Weights.DEFAULT.transforms()

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("ShiftEquivariantEncoder: backbone frozen")

    def _load_weights(self, path: str) -> None:
        """Load pretrained weights."""
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
            "ShiftEquivariantEncoder: loaded %d/%d keys from %s",
            len(filtered), len(model_state), path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract Z^2-invariant embedding.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Embedding tensor [B, 1024].
        """
        if self._use_native:
            return self.backbone(x)

        # Fallback: average over cyclic shifts
        h, w = x.shape[-2], x.shape[-1]
        step_h = h // self.n_shifts
        step_w = w // self.n_shifts
        embeddings = []
        for i in range(self.n_shifts):
            shifted = torch.roll(x, shifts=(i * step_h, i * step_w), dims=(-2, -1))
            embeddings.append(self.backbone(shifted))
        return torch.stack(embeddings, dim=1).mean(dim=1)
