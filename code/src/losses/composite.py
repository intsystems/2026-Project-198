"""Composite loss: weighted BCE + L2 contrastive regularizer.

L = L_BCE + lambda * L_contr^{L2}

Setting lambda = 0 recovers the variant without contrastive regularization.
"""

import torch
import torch.nn as nn

from .contrastive import L2ContrastiveLoss


class CompositeLoss(nn.Module):
    """Composite training loss for siamese plagiarism detection.

    Args:
        bce_weight_pos: BCE weight for positive class.
        bce_weight_neg: BCE weight for negative class.
        contrastive_margin: L2 contrastive margin m.
        lambda_reg: Contrastive regularizer weight. 0.0 disables it.
    """

    def __init__(
        self,
        bce_weight_pos: float = 0.3,
        bce_weight_neg: float = 0.7,
        contrastive_margin: float = 1.0,
        lambda_reg: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg

        # Weighted BCE: weight tensor applied per-sample based on label
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.w_pos = bce_weight_pos
        self.w_neg = bce_weight_neg

        if lambda_reg > 0:
            self.contrastive = L2ContrastiveLoss(margin=contrastive_margin)
        else:
            self.contrastive = None

    def forward(
        self,
        logits: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Compute composite loss.

        Args:
            logits: Raw scores from fusion head [B, 1].
            z1: First embedding [B, D].
            z2: Second embedding [B, D].
            labels: Binary labels [B].

        Returns:
            Dict with keys: total, bce, contrastive.
        """
        labels_flat = labels.view(-1)
        logits_flat = logits.view(-1)

        # Per-sample weighted BCE
        bce_raw = self.bce(logits_flat, labels_flat)
        weights = torch.where(
            labels_flat == 1.0,
            torch.tensor(self.w_pos, device=labels.device),
            torch.tensor(self.w_neg, device=labels.device),
        )
        bce_loss = (bce_raw * weights).mean()

        result = {"bce": bce_loss, "contrastive": torch.tensor(0.0, device=labels.device)}

        if self.contrastive is not None and self.lambda_reg > 0:
            contr_loss = self.contrastive(z1, z2, labels_flat)
            result["contrastive"] = contr_loss
            result["total"] = bce_loss + self.lambda_reg * contr_loss
        else:
            result["total"] = bce_loss

        return result
