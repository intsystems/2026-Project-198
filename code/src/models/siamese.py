"""Siamese network for plagiarism detection.

f(x1, x2) = sigmoid(h(phi(x1), phi(x2)))

where phi is the shared encoder and h is the symmetric fusion module.
"""

import torch
import torch.nn as nn

from .fusion import FusionHead


class SiameseNet(nn.Module):
    """Siamese plagiarism detector.

    Two branches share one encoder and process query/candidate images in
    parallel. A fusion module combines embeddings into a plagiarism score.

    Args:
        encoder: Shared encoder module (must have .feature_dim attribute).
        hidden_dim: Fusion head hidden dimension.
        dropout: Fusion head dropout.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = FusionHead(
            input_dim=encoder.feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> dict:
        """Forward pass through siamese network.

        Args:
            x1: Query image [B, C, H, W].
            x2: Candidate image [B, C, H, W].

        Returns:
            Dict with keys: logits [B, 1], z1 [B, D], z2 [B, D].
        """
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        logits = self.head(z1, z2)
        return {"logits": logits, "z1": z1, "z2": z2}

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings without fusion (for t-SNE visualization).

        Args:
            x: Image tensor [B, C, H, W].

        Returns:
            Embedding tensor [B, D].
        """
        return self.encoder(x)
