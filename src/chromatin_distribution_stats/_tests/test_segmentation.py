import numpy as np
import pytest

from chromatin_distribution_stats.k_means_heterochromatin import _kmeans_heterochromatin 

def test_kmeans_basic_shapes(toy_nucleus):
    img, nuc, het_outer, _ = toy_nucleus
    het_mask, labels_full, model = _kmeans_heterochromatin(
        img=img, nuc_mask=nuc, K=2, include_distance=True, sample_frac=1.0, random_state=0
    )
    assert het_mask.shape == nuc.shape
    assert labels_full.shape == nuc.shape
    # outside nucleus is -1
    assert np.all(labels_full[~nuc] == -1)
    # inside nucleus assigned to {0..K-1}
    inside_vals = labels_full[nuc]
    assert np.all((inside_vals >= 0) & (inside_vals < 2))

def test_kmeans_prefers_darker_rim(toy_nucleus):
    """On our synthetic image with darker rim, the chosen het mask should be enriched at the periphery."""
    img, nuc, het_outer, _ = toy_nucleus
    het_mask, labels_full, model = _kmeans_heterochromatin(
        img=img, nuc_mask=nuc, K=2, include_distance=False, sample_frac=1.0, random_state=0
    )
    # Compare fractions in outer vs inner 20% shells, using same logic as metrics
    from chromatin_distribution_stats.utils import _normalized_radius
    r = _normalized_radius(nuc)
    outer = (r <= 0.2) & nuc
    inner = (r >= 0.8) & nuc
    p_outer = het_mask[outer].mean()
    p_inner = het_mask[inner].mean()
    assert p_outer >= p_inner  # at least not worse than interior, on this toy

