"""Fusion module for siamese plagiarism detector.

Combines two embeddings z1, z2 into a scalar via a two-layer MLP
applied to element-wise features:

  h(z1, z2) = W2 ReLU(W1 [|z1 - z2| || z1 * z2] + b1) + b2

Proposition: h(z1, z2) = h(z2, z1) by symmetry of |.| and Hadamard product.
"""

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """Symmetric fusion module for comparing two embeddings.

    Args:
        input_dim: Dimension of each embedding vector.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Input: [|z1-z2| || z1*z2] has dimension 2 * input_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute symmetric fusion score.

        Args:
            z1: First embedding [B, D].
            z2: Second embedding [B, D].

        Returns:
            Logit tensor [B, 1].
        """
        abs_diff = torch.abs(z1 - z2)
        hadamard = z1 * z2
        features = torch.cat([abs_diff, hadamard], dim=-1)
        return self.mlp(features)
