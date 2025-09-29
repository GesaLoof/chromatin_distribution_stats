import numpy as np
import pytest

from chromatin_distribution_stats.utils import _normalized_radius, _distance_from_rim

def test_normalized_radius_basic(toy_nucleus):
    _, nuc, _, _ = toy_nucleus
    print(f"{nuc=}")
    print(f"{np.unique(nuc)=}")

    r = _normalized_radius(nuc)
    assert r.shape == nuc.shape
    assert np.all(r[~nuc] == 0)  # make sure that outside of nuclei distances are 0
    assert np.isclose(r[nuc].min(), 0.0, atol=1e-6)
    assert np.isclose(r[nuc].max(), 1.0, atol=1e-6)
    assert (nuc.dtype == bool) or set(np.unique(nuc)).issubset({0,1}), "Mask must be boolean/binary."
    assert (nuc >  0).any(), "Empty nucleus mask."
    assert (nuc == 0).any(), "Mask has no zero values --> missing background."
    assert nuc.dtype == bool or set(np.unique(nuc)).issubset({0,1})
    assert (nuc > 0).any() and (~nuc).any()  # has inside and outside

def test_distance_from_rim_units(toy_nucleus):
    _, nuc, _, _ = toy_nucleus
    r_px, rmax_px, unit_px = _distance_from_rim(nuc, pixel_size=None)
    assert unit_px == "px"
    assert np.isclose(r_px[nuc].max(), rmax_px)

    um_size = 0.01  # 10 nm/px as example
    r_um, rmax_um, unit_um = _distance_from_rim(nuc, pixel_size=um_size)
    assert unit_um == "um"
    # same geometry, different units
    assert np.isclose(r_um[nuc].max(), rmax_px * um_size)
