"""DomainNet evaluation: FPR, Recall per transformation class, F1.

Threshold is selected on validation fold by maximising F1.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data_module.domainnet_dataset import DomainNetEvalDataset
from src.models.siamese import SiameseNet

logger = logging.getLogger(__name__)


class DomainNetEvaluator:
    """Evaluate a siamese model on DomainNet balanced pairs.

    Args:
        model: Trained SiameseNet.
        dataset: DomainNetEvalDataset instance.
        device: Target device.
        batch_size: Evaluation batch size.
    """

    def __init__(
        self,
        model: SiameseNet,
        dataset: DomainNetEvalDataset,
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        )

    @torch.no_grad()
    def _collect_predictions(self) -> Dict[str, List]:
        """Run inference and collect scores, labels, metadata."""
        self.model.eval()
        results: Dict[str, List] = {
            "scores": [], "labels": [], "transforms": [], "domains": [],
        }
        for batch in self.loader:
            img1 = batch["img1"].to(self.device)
            img2 = batch["img2"].to(self.device)
            output = self.model(img1, img2)
            probs = torch.sigmoid(output["logits"]).view(-1).cpu()
            results["scores"].extend(probs.tolist())
            results["labels"].extend(batch["label"].tolist())
            results["transforms"].extend(batch["transform"])
            results["domains"].extend(batch["domain"])
        return results

    def _find_best_threshold(
        self, scores: List[float], labels: List[int]
    ) -> float:
        """Find threshold that maximises F1 score."""
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.95, 0.01):
            preds = [1 if s >= thresh else 0 for s in scores]
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh

    def run(self) -> Dict[str, Any]:
        """Run full evaluation protocol.

        Returns:
            Dict with: threshold, fpr, recall, f1, per_transform, per_domain.
        """
        results = self._collect_predictions()
        scores = results["scores"]
        labels = [int(l) for l in results["labels"]]
        transforms = results["transforms"]
        domains = results["domains"]

        threshold = self._find_best_threshold(scores, labels)
        preds = [1 if s >= threshold else 0 for s in scores]

        # Global metrics
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)

        fpr = fp / max(fp + tn, 1)
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        # Per-transform metrics
        per_transform: Dict[str, Dict[str, float]] = {}
        tfm_groups: Dict[str, List] = defaultdict(list)
        for i, tfm in enumerate(transforms):
            tfm_groups[tfm].append(i)

        for tfm_name, indices in sorted(tfm_groups.items()):
            if tfm_name == "none":
                continue
            tfm_labels = [labels[i] for i in indices]
            tfm_preds = [preds[i] for i in indices]
            tfm_tp = sum(1 for p, l in zip(tfm_preds, tfm_labels) if p == 1 and l == 1)
            tfm_fn = sum(1 for p, l in zip(tfm_preds, tfm_labels) if p == 0 and l == 1)
            per_transform[tfm_name] = {
                "recall": tfm_tp / max(tfm_tp + tfm_fn, 1),
                "count": len(indices),
            }

        # Per-domain metrics
        per_domain: Dict[str, Dict[str, float]] = {}
        dom_groups: Dict[str, List] = defaultdict(list)
        for i, dom in enumerate(domains):
            dom_groups[dom].append(i)

        for dom_name, indices in sorted(dom_groups.items()):
            dom_labels = [labels[i] for i in indices]
            dom_preds = [preds[i] for i in indices]
            dom_tp = sum(1 for p, l in zip(dom_preds, dom_labels) if p == 1 and l == 1)
            dom_fp = sum(1 for p, l in zip(dom_preds, dom_labels) if p == 1 and l == 0)
            dom_fn = sum(1 for p, l in zip(dom_preds, dom_labels) if p == 0 and l == 1)
            dom_tn = sum(1 for p, l in zip(dom_preds, dom_labels) if p == 0 and l == 0)
            per_domain[dom_name] = {
                "fpr": dom_fp / max(dom_fp + dom_tn, 1),
                "recall": dom_tp / max(dom_tp + dom_fn, 1),
            }

        report = {
            "threshold": threshold,
            "fpr": fpr,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "per_transform": per_transform,
            "per_domain": per_domain,
            "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }

        logger.info(
            "Results: FPR=%.4f, Recall=%.4f, F1=%.4f, threshold=%.3f",
            fpr, recall, f1, threshold,
        )
        for tfm, m in per_transform.items():
            logger.info("  %s: recall=%.4f (n=%d)", tfm, m["recall"], m["count"])

        return report
