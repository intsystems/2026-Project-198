"""t-SNE visualization of encoder embeddings on DomainNet.

Produces scatter plots with six DomainNet domains as distinct point colours.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

DOMAIN_COLORS = {
    "real": "#e41a1c",
    "painting": "#377eb8",
    "clipart": "#4daf4a",
    "quickdraw": "#984ea3",
    "infograph": "#ff7f00",
    "sketch": "#a65628",
}


@torch.no_grad()
def compute_tsne_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    perplexity: int = 30,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """Compute t-SNE 2D projections from encoder embeddings.

    Args:
        model: SiameseNet (uses .get_embeddings method).
        dataloader: DomainNetEvalDataset loader.
        device: Target device.
        perplexity: t-SNE perplexity parameter.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (tsne_coords [N, 2], domain_labels [N]).
    """
    model.eval()
    all_embeddings = []
    all_domains: List[str] = []

    for batch in dataloader:
        img1 = batch["img1"].to(device)
        z = model.get_embeddings(img1)
        all_embeddings.append(z.cpu().numpy())
        all_domains.extend(batch["domain"])

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info("Computing t-SNE on %d embeddings (dim=%d)", *embeddings.shape)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(embeddings)
    return coords, all_domains


def plot_tsne(
    coords: np.ndarray,
    domains: List[str],
    title: str = "t-SNE Encoder Embeddings",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot t-SNE scatter colored by DomainNet domain.

    Args:
        coords: t-SNE coordinates [N, 2].
        domains: Domain labels [N].
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    unique_domains = sorted(set(domains))
    for domain in unique_domains:
        mask = [d == domain for d in domains]
        mask_idx = np.array(mask)
        color = DOMAIN_COLORS.get(domain, "#333333")
        ax.scatter(
            coords[mask_idx, 0],
            coords[mask_idx, 1],
            c=color,
            label=domain,
            alpha=0.6,
            s=15,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("t-SNE figure saved -> %s", save_path)

    return fig
