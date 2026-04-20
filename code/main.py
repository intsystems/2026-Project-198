"""Single entry point for training and evaluating siamese plagiarism detectors.

Usage:
    python main.py --config configs/default.yaml --run-config configs/runs/vit_baseline_no_reg.yaml
    python main.py --config configs/default.yaml --run-config configs/runs/octic_vit_reg.yaml
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# Add code/ to path so src.* imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.seed import set_seed
from src.data_module.augmentations import build_train_transform, build_val_transform
from src.data_module.coco_dataset import build_coco_dataloaders
from src.data_module.domainnet_dataset import DomainNetEvalDataset
from src.encoders import EncoderFactory
from src.losses.composite import CompositeLoss
from src.models.siamese import SiameseNet
from src.training.trainer import Trainer
from src.validation.evaluator import DomainNetEvaluator
from src.validation.tsne import compute_tsne_embeddings, plot_tsne

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(base_path: str, run_path: str) -> Dict[str, Any]:
    """Load and merge base config with run-specific overrides.

    Args:
        base_path: Path to default.yaml.
        run_path: Path to run-specific YAML.

    Returns:
        Merged config dict.
    """
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    with open(run_path) as f:
        overrides = yaml.safe_load(f)

    # Remove Hydra-style defaults key
    overrides.pop("defaults", None)

    # Deep merge overrides into base config
    def _merge(base: dict, over: dict) -> dict:
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                _merge(base[k], v)
            else:
                base[k] = v
        return base

    return _merge(cfg, overrides)


def run_experiment(cfg: Dict[str, Any], run_name: str) -> None:
    """Execute a single training + evaluation run.

    Args:
        cfg: Merged config dict.
        run_name: Identifier for this run.
    """
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("Run: %s", run_name)
    logger.info("Encoder: %s, L2_reg lambda=%.2f, skip_geo=%s",
                cfg["encoder"]["name"],
                cfg["contrastive"]["lambda_reg"],
                cfg["augmentation"]["skip_geometric"])

    # --- Build encoder ---
    encoder_kwargs = {
        "freeze": cfg["encoder"]["freeze"],
    }
    if cfg["encoder"].get("weights_path"):
        encoder_kwargs["weights_path"] = cfg["encoder"]["weights_path"]
    if cfg["encoder"]["name"] == "harmformer":
        encoder_kwargs["n_rots"] = 8

    encoder = EncoderFactory(cfg["encoder"]["name"], **encoder_kwargs)
    logger.info("Encoder %s loaded (feature_dim=%d)", cfg["encoder"]["name"], encoder.feature_dim)

    # --- Build model ---
    model = SiameseNet(encoder, hidden_dim=512, dropout=0.1)

    # --- Build loss ---
    criterion = CompositeLoss(
        bce_weight_pos=cfg["training"]["bce_weight_pos"],
        bce_weight_neg=cfg["training"]["bce_weight_neg"],
        contrastive_margin=cfg["contrastive"]["margin"],
        lambda_reg=cfg["contrastive"]["lambda_reg"],
    )

    # --- Build optimizer & scheduler ---
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["training"]["milestones"],
        gamma=cfg["training"]["gamma"],
    )

    # --- Build data ---
    train_tfm = build_train_transform(cfg)
    val_tfm = build_val_transform(cfg)
    dataloaders = build_coco_dataloaders(cfg, train_tfm, val_tfm)

    # --- Train ---
    checkpoint_dir = os.path.join(cfg["output"]["checkpoints"], run_name)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    history = trainer.fit(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        num_epochs=cfg["training"]["num_epochs"],
        run_name=run_name,
    )

    # --- Evaluate on DomainNet ---
    logger.info("Starting DomainNet evaluation...")
    domainnet_ds = DomainNetEvalDataset(
        root=cfg["data"]["domainnet_root"],
        domains=cfg["data"]["domainnet_domains"],
        pairs_per_domain=cfg["data"]["pairs_per_domain"],
        image_size=cfg["data"]["image_size"],
        seed=cfg["seed"],
    )
    evaluator = DomainNetEvaluator(
        model=model,
        dataset=domainnet_ds,
        device=device,
        batch_size=cfg["training"]["batch_size"],
    )
    report = evaluator.run()

    # --- t-SNE ---
    logger.info("Computing t-SNE embeddings...")
    from torch.utils.data import DataLoader
    tsne_loader = DataLoader(domainnet_ds, batch_size=32, shuffle=False, num_workers=2)
    coords, domains = compute_tsne_embeddings(
        model, tsne_loader, device=device,
        perplexity=cfg["evaluation"]["tsne_perplexity"],
    )
    tsne_path = os.path.join(cfg["output"]["figures"], f"{run_name}_tsne.pdf")
    plot_tsne(coords, domains, title=f"t-SNE: {run_name}", save_path=tsne_path)

    # --- Save report ---
    report_path = os.path.join(cfg["output"]["logs"], f"{run_name}_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved -> %s", report_path)

    # --- Save history ---
    history_path = os.path.join(cfg["output"]["logs"], f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info("History saved -> %s", history_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image Plagiarism Detection with Equivariant Encoders",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--run-config", type=str, required=True,
        help="Path to run-specific config YAML (e.g. configs/runs/vit_baseline_no_reg.yaml)",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Override run name (defaults to run config filename stem)",
    )
    args = parser.parse_args()

    run_name = args.run_name or Path(args.run_config).stem
    cfg = load_config(args.config, args.run_config)
    run_experiment(cfg, run_name)


if __name__ == "__main__":
    main()
