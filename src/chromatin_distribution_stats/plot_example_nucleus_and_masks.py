from skimage.measure import label as cc_label
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tifffile import imread
from chromatin_distribution_stats import config
from utils import ensure_labeled_nuclei
import os

def plot_raw_and_het(
    img: np.ndarray,
    nuc_labels: np.ndarray,
    het_mask: np.ndarray,
    nucleus_id: int | None = None,
    show_all: bool = True,
    labels_global: np.ndarray | None = None,
) -> plt.Figure:
    """
    (A) Raw EM + nuclear outline(s)
    (B) Raw EM + heterochromatin overlay (NO nuclear outlines)
    (C) Optional k-means label heatmap (if `labels_global` provided)
    """
    img = np.asarray(img)
    nuc_labels = np.asarray(nuc_labels)
    het_mask = (np.asarray(het_mask) > 0)

    if img.ndim != 2:
        img = img[..., 0]
    if nuc_labels.ndim != 2:
        nuc_labels = nuc_labels[..., 0]
    if het_mask.ndim != 2:
        het_mask = het_mask[..., 0].astype(bool)

    if not show_all:
        if nucleus_id is None:
            raise ValueError("When show_all=False, you must provide nucleus_id.")
        mask_L = (nuc_labels == int(nucleus_id))
        if not mask_L.any():
            raise ValueError(f"Nucleus {nucleus_id} not found.")
    else:
        mask_L = (nuc_labels > 0)

    # Layout: 1x2 or 1x3
    ncols = 3 if labels_global is not None else 2
    fig = plt.figure(figsize=(6 * ncols, 5))
    gs = fig.add_gridspec(1, ncols, wspace=0.25, hspace=0.0)

    # (A) Raw + outlines
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title("Raw EM + nuclear outline")
    ax0.imshow(img, cmap="gray")
    if show_all:
        ax0.contour(nuc_labels > 0, levels=[0.5], linewidths=0.7)
    else:
        ax0.contour(mask_L, levels=[0.5], linewidths=1.0)
    ax0.set_xticks([]); ax0.set_yticks([])

    # (B) Raw + het overlay (NO outlines here)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("Raw EM + heterochromatin overlay" + (" (all nuclei)" if show_all else " (this nucleus)"))
    ax1.imshow(img, cmap="gray")
    if show_all:
        ov = np.ma.masked_where(~het_mask, het_mask)
    else:
        ov = np.ma.masked_where(~(het_mask & mask_L), het_mask & mask_L)
    ax1.imshow(ov, alpha=0.35)
    ax1.set_xticks([]); ax1.set_yticks([])

    # (C) Optional k-means labels
    if labels_global is not None:
        labels_global = np.asarray(labels_global)
        if labels_global.ndim != 2:
            labels_global = labels_global[..., 0]
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title("k-means labels" + (" (all nuclei)" if show_all else " (this nucleus)"))
        show = np.full_like(labels_global, fill_value=np.nan, dtype=float)
        if show_all:
            m = (labels_global >= 0) & (nuc_labels > 0)
            show[m] = labels_global[m]
        else:
            show[~mask_L] = np.nan
            show[mask_L] = labels_global[mask_L]
        im = ax2.imshow(show, interpolation="nearest")
        ax2.set_xticks([]); ax2.set_yticks([])
        cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cb.set_label("Cluster ID")

    fig.suptitle("Chromatin distribution — raw & overlays", y=0.98, fontsize=14)
    fig.tight_layout()
    return fig


def main():
    # Load required inputs
    img = imread(config.plot_nuc_im_path)
    nuc_mask_raw = imread(config.plot_nuc_mask_path)
    het_mask = imread(config.plot_nuc_het_mask_path)

    # Ensure labeled nuclei (0=bg, 1..N)
    nuc_labels = ensure_labeled_nuclei(nuc_mask_raw, connectivity=2)

    # Optional: k-means labels if provided in config
    labels_global = None
    labels_path = getattr(config, "plot_labels_path", None)
    if labels_path and os.path.exists(labels_path):
        labels_global = imread(labels_path)

    fig = plot_raw_and_het(
        img=img,
        nuc_labels=nuc_labels,
        het_mask=het_mask,
        nucleus_id=getattr(config, "plot_nucleus_id", 1),  # choose nucleus when show_all=False
        show_all=getattr(config, "plot_show_all", True),
        labels_global=labels_global,
    )

    # Where to save
    out_path = getattr(config, "plot_raw_het_path", None) or getattr(config, "plot_profiles_stacked_path", "raw_het_overlay.png")
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
