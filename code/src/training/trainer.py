"""Training loop with cross-batch pairing strategy.

A batch of B images is augmented to form two views per image, and all B^2
pairwise combinations are formed with labels y_ij = 1[i == j].
"""

import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.losses.composite import CompositeLoss
from src.models.siamese import SiameseNet

logger = logging.getLogger(__name__)


class Trainer:
    """Siamese network trainer with cross-batch pairing.

    Args:
        model: SiameseNet instance.
        criterion: CompositeLoss instance.
        optimizer: PyTorch optimizer.
        scheduler: Optional LR scheduler.
        device: Target device.
        checkpoint_dir: Directory for saving checkpoints.
    """

    def __init__(
        self,
        model: SiameseNet,
        criterion: CompositeLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: str = "outputs/checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _cross_batch_pairs(
        self, images: torch.Tensor
    ) -> tuple:
        """Generate B^2 cross-batch pairs from a batch of B images.

        Each image is paired with itself (positive, label=1) and with
        all other images (negative, label=0).

        Args:
            images: Batch of images [B, C, H, W].

        Returns:
            Tuple of (x1, x2, labels) where x1, x2 are [B^2, C, H, W]
            and labels is [B^2].
        """
        b = images.shape[0]
        idx_i = torch.arange(b).unsqueeze(1).expand(b, b).reshape(-1)
        idx_j = torch.arange(b).unsqueeze(0).expand(b, b).reshape(-1)

        x1 = images[idx_i]
        x2 = images[idx_j]
        labels = (idx_i == idx_j).float().to(self.device)

        return x1, x2, labels

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader (yields single images).

        Returns:
            Dict with epoch-level loss statistics.
        """
        self.model.train()
        self.model.encoder.eval()
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        total_loss = 0.0
        total_bce = 0.0
        total_contr = 0.0
        n_batches = 0

        for images in dataloader:
            images = images.to(self.device)
            x1, x2, labels = self._cross_batch_pairs(images)

            self.optimizer.zero_grad()
            output = self.model(x1, x2)
            losses = self.criterion(
                output["logits"], output["z1"], output["z2"], labels,
            )
            losses["total"].backward()
            self.optimizer.step()

            total_loss += losses["total"].item()
            total_bce += losses["bce"].item()
            total_contr += losses["contrastive"].item()
            n_batches += 1

        if n_batches == 0:
            return {"loss": 0.0, "bce": 0.0, "contrastive": 0.0}

        return {
            "loss": total_loss / n_batches,
            "bce": total_bce / n_batches,
            "contrastive": total_contr / n_batches,
        }

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dict with validation loss statistics.
        """
        self.model.eval()
        total_loss = 0.0
        total_bce = 0.0
        total_contr = 0.0
        n_batches = 0

        for images in dataloader:
            images = images.to(self.device)
            x1, x2, labels = self._cross_batch_pairs(images)

            output = self.model(x1, x2)
            losses = self.criterion(
                output["logits"], output["z1"], output["z2"], labels,
            )

            total_loss += losses["total"].item()
            total_bce += losses["bce"].item()
            total_contr += losses["contrastive"].item()
            n_batches += 1

        if n_batches == 0:
            return {"loss": 0.0, "bce": 0.0, "contrastive": 0.0}

        return {
            "loss": total_loss / n_batches,
            "bce": total_bce / n_batches,
            "contrastive": total_contr / n_batches,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 15,
        run_name: str = "run",
    ) -> List[Dict[str, Any]]:
        """Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of training epochs.
            run_name: Name prefix for checkpoint files.

        Returns:
            List of per-epoch history dicts.
        """
        history: List[Dict[str, Any]] = []
        best_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | lr=%.6f",
                epoch, num_epochs, train_metrics["loss"], val_metrics["loss"], lr,
            )

            epoch_record = {
                "epoch": epoch,
                "lr": lr,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(epoch_record)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                path = os.path.join(self.checkpoint_dir, f"{run_name}_best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": best_val_loss,
                }, path)
                logger.info("  Best model saved -> %s", path)

        path = os.path.join(self.checkpoint_dir, f"{run_name}_final.pt")
        torch.save({
            "epoch": num_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": history,
        }, path)
        logger.info("Final checkpoint saved -> %s", path)

        return history
