import numpy as np
import pytest
from tifffile import imwrite
from skimage.measure import label as cc_label

from chromatin_distribution_stats.calculate_heterochromatin_distribution_metrics import compute_metrics_all

def test_compute_metrics_all_from_paths(tmp_path, toy_nucleus):
    img, nuc, het_outer, _ = toy_nucleus
    # Save masks
    het_path = tmp_path / "het.tif"
    nuc_path = tmp_path / "nuc.tif"
    imwrite(het_path, het_outer.astype(np.uint8))
    imwrite(nuc_path, nuc.astype(np.uint8))

    df, prof_stack, bin_centers = compute_metrics_all(
        het_mask_path=str(het_path),
        nuc_mask_path=str(nuc_path),
        n_bins=12,
        outer_width=0.2,
        inner_width=0.2,
    )

    # Expect exactly one nucleus (single connected component)
    assert len(df) == 1
    assert prof_stack.shape == (1, 12)
    assert bin_centers.shape == (12,)
    row = df.iloc[0]
    assert row["area_px"] == int(nuc.sum())
    assert row["odds_ratio"] > 1.0
