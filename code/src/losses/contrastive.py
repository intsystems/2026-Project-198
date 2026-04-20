"""L2 contrastive regularizer (Hadsell, Chopra, LeCun, CVPR 2006).

L(z1, z2, y) = y * ||z1 - z2||^2 + (1 - y) * max(0, m - ||z1 - z2||)^2

Proposition: L is invariant under orthogonal transformations of the
embedding space, since ||Uz1 - Uz2|| = ||z1 - z2|| for orthogonal U.
"""

import torch
import torch.nn as nn


class L2ContrastiveLoss(nn.Module):
    """Hadsell-Chopra-LeCun L2 contrastive loss.

    Args:
        margin: Margin parameter m for negative pairs.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            z1: First embedding [B, D].
            z2: Second embedding [B, D].
            y: Labels [B], 1 for genuine (plagiarism), 0 for impostor.

        Returns:
            Scalar loss.
        """
        y_pos = y.float().clamp(0.0, 1.0)
        dist = torch.norm(z1 - z2, p=2, dim=-1)
        pos_loss = y_pos * dist.pow(2)
        neg_loss = (1.0 - y_pos) * torch.clamp(self.margin - dist, min=0.0).pow(2)
        return (pos_loss + neg_loss).mean()
