import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from chromatin_distribution_stats.plot_distribution_stats_by_condition import (
    plot_summary_by_condition,
    plot_profiles_stacked_by_condition,
    plot_rp_summary_by_condition,
    plot_rp_histograms_stacked_by_condition,
)
from chromatin_distribution_stats.plot_distribution_stats_by_condition import plot_rp_forest_by_condition

def _toy_df_for_plots():
    # Two conditions, three nuclei each
    data = []
    for cond, shift in [("A", +0.3), ("B", -0.2)]:
        for i in range(3):
            # fake profile & centers
            centers = np.linspace(0.025, 0.975, 20)
            prof = np.clip(0.3 + shift*(1-centers) + 0.02*np.random.randn(20), 0, 1)
            data.append({
                "condition": cond,
                "nucleus_id": i+1,
                "area_px": 5000 + 200*i,
                "delta_mean_r_het_vs_eu": -0.05 + 0.02*np.random.randn(),
                "ks_D": abs(np.random.randn())*0.2,
                "odds_ratio": np.exp(np.random.randn()*0.1 + (0.3 if cond=="A" else -0.1)),
                "risk_ratio": np.exp(np.random.randn()*0.1 + (0.25 if cond=="A" else -0.05)),
                "risk_ratio_CI95_low": 0.8, "risk_ratio_CI95_high": 1.7,
                "radial_profile": prof, "radial_bin_centers": centers,
                "path_to_image": f"/tmp/{cond}_{i}.tif",
            })
    return pd.DataFrame(data)

def test_plot_summary_and_stacked_profiles_smoke():
    df = _toy_df_for_plots()
    # Summary (no histogram in your current version)
    fig1, stats = plot_summary_by_condition(
        df, cond_col="condition", group_order=["A","B"], metric_col="odds_ratio",
        title="Summary", colors=["tab:blue","tab:orange"]
    )
    assert isinstance(fig1, plt.Figure)
    assert "A" in stats and "B" in stats
    plt.close(fig1)

    # Stacked radial profiles
    fig2 = plot_profiles_stacked_by_condition(
        df, cond_col="condition", group_order=["A","B"], colors=["tab:blue","tab:orange"],
        title="Stacked profiles", show_individual=True
    )
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

def test_plot_rp_helpers_and_forest_smoke():
    df = _toy_df_for_plots()
    # RP summary
    fig3, _ = plot_rp_summary_by_condition(
        df, cond_col="condition", group_order=["A","B"], rp_col="risk_ratio",
        colors=["tab:blue","tab:orange"]
    )
    assert isinstance(fig3, plt.Figure)
    plt.close(fig3)

    # RP hist stacked
    fig4 = plot_rp_histograms_stacked_by_condition(
        df, cond_col="condition", group_order=["A","B"], rp_col="risk_ratio",
        colors=["tab:blue","tab:orange"]
    )
    assert isinstance(fig4, plt.Figure)
    plt.close(fig4)

    # Forest plot with de-dup/filter knobs
    fig5 = plot_rp_forest_by_condition(
        df.assign(shell_mode="normalized"),
        cond_col="condition",
        group_order=["A","B"],
        rp_col="risk_ratio",
        lo_col="risk_ratio_CI95_low",
        hi_col="risk_ratio_CI95_high",
        id_cols=["path_to_image","nucleus_id"],
        colors=["tab:blue","tab:orange"],
        filter_query='shell_mode == "normalized"',
        dedupe_by=["path_to_image","nucleus_id"],
        summary_label="pooled",
    )
    assert isinstance(fig5, plt.Figure)
    plt.close(fig5)
