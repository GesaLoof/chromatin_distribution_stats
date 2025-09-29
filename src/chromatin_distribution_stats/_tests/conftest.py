import os
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # headless backend for tests

@pytest.fixture
def rng():
    return np.random.default_rng(0)

def make_disk_mask(h, w, cy, cx, r):
    Y, X = np.ogrid[:h, :w]
    return (Y - cy)**2 + (X - cx)**2 <= r**2

@pytest.fixture
def toy_nucleus():
    """Single ~circular nucleus (mask), with 'outer rim' ring for ground-truth het."""
    h, w = 128, 128
    cy, cx, r = 64, 64, 45
    nuc = make_disk_mask(h, w, cy, cx, r)
    # Outer ring ~8 px thick
    rim = nuc & (~make_disk_mask(h, w, cy, cx, r-8))
    het_outer = rim.copy()
    het_inner = make_disk_mask(h, w, cy, cx, 10)  # central blob (unused by default)
    # Synthetic EM image: darker at rim, brighter inside (helps k-means)
    img = np.full((h, w), 0.6, dtype=np.float32)
    img[het_outer] = 0.2
    img[~nuc] = 0.8
    return img, nuc, het_outer, het_inner
