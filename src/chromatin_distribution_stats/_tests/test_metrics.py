import numpy as np
import pytest

from chromatin_distribution_stats.calculate_heterochromatin_distribution_metrics import hetero_distribution_metrics

def test_metrics_periphery_enrichment(toy_nucleus):
    img, nuc, het_outer, _ = toy_nucleus
    # Analyze with outer ring as heterochromatin
    m = hetero_distribution_metrics(
        het_mask=het_outer,
        nuc_mask=nuc,
        outer_width=0.20,
        inner_width=0.20,
        n_bins=10,
    )

    # sanity: keys present
    for k in [
        "p_het_outer","p_het_inner","enrichment_ratio",
        "risk_ratio","risk_ratio_CI95_low","risk_ratio_CI95_high",
        "odds_ratio","odds_ratio_CI95_low","odds_ratio_CI95_high",
        "mean_r_het","mean_r_eu","delta_mean_r_het_vs_eu",
        "ks_D","emd","radial_profile","radial_bin_centers",
        "radial_auc","radial_slope"
    ]:
        assert k in m

    # expected direction: periphery enriched
    assert m["p_het_outer"] > m["p_het_inner"]
    assert m["odds_ratio"] > 1.0
    assert m["risk_ratio"] > 1.0
    assert m["mean_r_het"] < m["mean_r_eu"]  # het closer to rim (smaller r)

    # radial profile length and centers
    rp = m["radial_profile"]; bc = m["radial_bin_centers"]
    assert isinstance(rp, np.ndarray) and isinstance(bc, np.ndarray)
    assert rp.shape == bc.shape == (10,)
