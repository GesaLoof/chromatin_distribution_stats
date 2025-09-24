from skimage.measure import label as cc_label
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import ks_2samp, wasserstein_distance
from skimage.measure import label as cc_label, regionprops
from tifffile import imread, imwrite
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter, uniform_filter, distance_transform_edt
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
from chromatin_distribution_stats import config
# TODO check imports



# Visualization utilities for a single example nucleus:
#   - raw EM image
#   - heterochromatin segmentation
#   - k-means label map inside the nucleus
#   - radial heterochromatin profile
#   - odds ratio with 95% CI
#   - textual summary of key metrics



def ensure_labeled_nuclei(nuc_labels, connectivity=2):
    """
    Return an integer-labeled nuclei image (1..N).
    - If `nuc_labels` is boolean or binary, label it.
    - Otherwise assume it's already per-nucleus labeled and return as int.
    """
    arr = np.asarray(nuc_labels)
    if arr.dtype == bool:
        print("labelling boolean mask")
        return cc_label(arr, connectivity=connectivity)
    vals = np.unique(arr)
    if vals.size <= 2 and vals.min() == 0 and vals.max() == 1:
        print("labelling binary mask")
        return cc_label(arr.astype(bool), connectivity=connectivity)
    return arr.astype(int, copy=False)


nuc_labels = ensure_labeled_nuclei(nuc_labels)




def plot_example_nucleus(
    img, nuc_labels, het_global, labels_global, nucleus_id,
    metrics_row=None, show_all=True
):
    """
    If show_all=False (default): highlight only the chosen nucleus (old behavior).
    If show_all=True: show overlays/labels for ALL nuclei in the first row.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    mask_L = (nuc_labels == nucleus_id)
    if not mask_L.any():
        raise ValueError(f"Nucleus {nucleus_id} not found.")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, wspace=0.25, hspace=0.3)

    # --- Raw EM ---
    ax0 = fig.add_subplot(gs[0,0])
    ax0.set_title("Raw EM")
    ax0.imshow(img, cmap="gray")
    # Outline: one nucleus (show_all=False) vs all nuclei (show_all=True)
    if show_all:
        ax0.contour(nuc_labels > 0, levels=[0.5], linewidths=0.6)
    else:
        ax0.contour(mask_L, levels=[0.5], linewidths=1.0)
    ax0.set_xticks([]); ax0.set_yticks([])

    # --- Heterochromatin overlay ---
    ax1 = fig.add_subplot(gs[0,1])
    ax1.set_title("Heterochromatin overlay" + (" (ALL nuclei)" if show_all else ""))
    ax1.imshow(img, cmap="gray")
    if show_all:
        ov = np.ma.masked_where(~het_global, het_global)       # all nuclei
    else:
        ov = np.ma.masked_where(~(het_global & mask_L), het_global & mask_L)  # one nucleus
    ax1.imshow(ov, alpha=0.35)
    ax1.set_xticks([]); ax1.set_yticks([])

    # --- k-means labels heatmap ---
    ax2 = fig.add_subplot(gs[0,2])
    ax2.set_title("k-means labels" + (" (ALL nuclei)" if show_all else " (this nucleus)"))
    show = np.full_like(labels_global, fill_value=np.nan, dtype=float)
    if show_all:
        # show all labels; keep outside nuclei as NaN (labels_global == -1)
        show[(labels_global >= 0) & (nuc_labels > 0)] = labels_global[(labels_global >= 0) & (nuc_labels > 0)]
    else:
        show[mask_L] = labels_global[mask_L]
    im = ax2.imshow(show, interpolation="nearest")
    ax2.set_xticks([]); ax2.set_yticks([])
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("Cluster ID")

    # --- metrics (same as before) ---
    if metrics_row is None:
        from pandas import Series
        from math import isnan
        # recompute metrics just for the chosen nucleus
        m = hetero_distribution_metrics(het_global & mask_L, mask_L)
        metrics_row = Series(m)
        metrics_row["area_px"] = int(mask_L.sum())
        metrics_row.name = nucleus_id

    # radial profile
    ax3 = fig.add_subplot(gs[1,0])
    ax3.set_title("Radial heterochromatin fraction")
    x = metrics_row["radial_bin_centers"]; y = metrics_row["radial_profile"]
    ax3.plot(x, y, marker="o"); ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Normalized radius (0 = rim, 1 = center)"); ax3.set_ylabel("Heterochromatin fraction")

    # OR panel
    ax4 = fig.add_subplot(gs[1,1])
    ax4.set_title("Periphery enrichment (Odds Ratio)")
    OR = float(metrics_row["odds_ratio"]); lo = float(metrics_row["odds_ratio_CI95_low"]); hi = float(metrics_row["odds_ratio_CI95_high"])
    ax4.errorbar([0], [OR], yerr=[[OR-lo],[hi-OR]], fmt="o", capsize=5)
    ax4.axhline(1.0, linestyle="--"); ax4.set_xticks([]); ax4.set_ylabel("OR (outer vs inner)")
    ax4.set_yscale("log"); ax4.grid(True, which="both", alpha=0.3)

    # text panel
    ax5 = fig.add_subplot(gs[1,2]); ax5.axis("off")
    lines = [
        f"Nucleus id: {int(getattr(metrics_row, 'name', nucleus_id))}",
        f"Area (px): {int(metrics_row.get('area_px', int(mask_L.sum())))}",
        f"p_het_outer: {metrics_row['p_het_outer']:.3f}",
        f"p_het_inner: {metrics_row['p_het_inner']:.3f}",
        f"OR [95% CI]: {OR:.2f} [{lo:.2f}, {hi:.2f}]",
        f"Δ mean r (het - eu): {metrics_row['delta_mean_r_het_vs_eu']:.3f}",
        f"KS D (p): {metrics_row['ks_D']:.3f} ({metrics_row['ks_p']:.2g})",
        f"Radial slope: {metrics_row['radial_slope']:.3f}",
    ]
    ax5.text(0, 1, "\n".join(lines), va="top", family="monospace")

    fig.suptitle("Chromatin distribution — example nucleus", y=0.98, fontsize=14)
    fig.tight_layout()
    return fig


# Population-level summaries across all nuclei:
#   - histogram of log(OR)
#   - mean radial profile with a 95% band
#   - scatter of Δ mean radius vs nucleus size
#   - scatter of KS statistic vs OR (consistency view)

def plot_summary(
    df_metrics: pd.DataFrame,
    prof_stack: np.ndarray,
    bin_centers: np.ndarray,
) -> plt.Figure:
    """
    Create a 2x2 summary figure aggregating heterochromatin distribution across nuclei.

    Parameters
    ----------
    df_metrics : pandas.DataFrame
        Output from `compute_metrics_all`, with one row per nucleus.
        Must contain columns: 'odds_ratio', 'area_px', 'delta_mean_r_het_vs_eu',
        'ks_D', and 'radial_profile' (the latter only used to build prof_stack upstream).
    prof_stack : np.ndarray, shape (N, n_bins)
        Stacked radial profiles across nuclei (N = #nuclei).
    bin_centers : np.ndarray, shape (n_bins,)
        Normalized radius values for the centers of the radial bins.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The constructed figure.

    Notes
    -----
    - The log(OR) histogram shows periphery enrichment to the right of 0.
    - The radial profile plot shows mean ± 95% interval (2.5th–97.5th percentiles).
    """
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.35)

    # (1) Histogram of log(OR): >0 indicates periphery enrichment on average.
    ax0 = fig.add_subplot(gs[0, 0])
    logOR = np.log(df_metrics["odds_ratio"].values)
    ax0.hist(logOR, bins=20)
    ax0.axvline(0, linestyle="--")
    ax0.set_title("log(Odds Ratio) across nuclei")
    ax0.set_xlabel("log(OR)  [>0 → periphery-enriched]")
    ax0.set_ylabel("# nuclei")

    # (2) Mean radial profile with a 95% band across nuclei.
    ax1 = fig.add_subplot(gs[0, 1])
    if prof_stack.size:
        mean_prof = np.nanmean(prof_stack, axis=0)
        lo = np.nanpercentile(prof_stack, 2.5, axis=0)
        hi = np.nanpercentile(prof_stack, 97.5, axis=0)
        ax1.plot(bin_centers, mean_prof, marker="o")
        ax1.fill_between(bin_centers, lo, hi, alpha=0.2, linewidth=0)
    ax1.set_title("Radial heterochromatin fraction (mean ± 95% range)")
    ax1.set_xlabel("Normalized radius (0 = rim, 1 = center)")
    ax1.set_ylabel("Heterochromatin fraction")
    ax1.grid(True, alpha=0.3)

    # (3) Scatter: Δ mean radius vs nucleus area (effect vs size).
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(df_metrics["area_px"], df_metrics["delta_mean_r_het_vs_eu"])
    ax2.axhline(0, linestyle="--")
    ax2.set_xlabel("Nucleus area (pixels)")
    ax2.set_ylabel("Δ mean r (het - eu)  [<0 → periphery]")
    ax2.set_title("Per-nucleus effect vs size")
    ax2.grid(True, alpha=0.3)

    # (4) Agreement between metrics: KS statistic vs OR.
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(df_metrics["odds_ratio"], df_metrics["ks_D"])
    ax3.set_xscale("log")
    ax3.axvline(1.0, linestyle="--")
    ax3.set_xlabel("Odds Ratio (outer vs inner)")
    ax3.set_ylabel("KS D (het vs eu radii)")
    ax3.set_title("Agreement between metrics")
    ax3.grid(True, which="both", alpha=0.3)

    fig.suptitle("Chromatin distribution — cohort summary", y=0.98, fontsize=14)
    fig.tight_layout()
    return fig


def plot_summary_by_condition(
    df: pd.DataFrame,
    cond_col: str = "condition",
    group_order: list[str] | None = None,
    metric_col: str = "odds_ratio",
    title: str = "Chromatin distribution — condition summary",
) -> tuple[plt.Figure, dict]:
    """
    2×2 cohort summary, stratified by condition.

    Panels:
      (A) Overlaid histograms of log(metric_col) per nucleus, per condition
      (B) Mean radial profile ± 95% band per condition
      (C) Scatter: Δ mean radius (het−eu) vs nucleus area, colored by condition
      (D) Scatter: KS D vs metric_col, colored by condition

    Parameters
    ----------
    df : pandas.DataFrame
        Per-nucleus metrics (stack of all images/conditions). Expected columns:
          - cond_col (default 'condition')
          - 'odds_ratio' (or other metric if metric_col is set)
          - 'delta_mean_r_het_vs_eu', 'ks_D', 'area_px'
          - 'radial_profile' (array-like), 'radial_bin_centers' (array-like)
    cond_col : str
        Column name containing condition labels.
    group_order : list[str] | None
        Optional explicit condition order. If None, uses sorted unique.
    metric_col : str
        Metric shown in panels A and D (x-axis in D). Defaults to 'odds_ratio'.
        If you switched to stabilized RP, you can set this to 'risk_ratio'.
    title : str
        Figure suptitle.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The constructed figure.
    stats : dict
        Summary stats by condition:
          - 'N' (nuclei)
          - 'mean_log_metric' with bootstrap CI (quick, heuristic)
          - If exactly 2 groups: Mann–Whitney U test on log(metric)
    """
    # --- sanity & setup ---
    assert cond_col in df.columns, f"'{cond_col}' column not found in df."
    assert metric_col in df.columns, f"'{metric_col}' column not found in df."

    groups = group_order or sorted(df[cond_col].dropna().unique().tolist())
    assert len(groups) >= 2, "Need at least two conditions."

    # Prepare log(metric) per condition (drop non-positive / non-finite)
    data_log = {}
    for g in groups:
        v = pd.to_numeric(df.loc[df[cond_col] == g, metric_col], errors="coerce").values
        v = v[np.isfinite(v) & (v > 0)]
        data_log[g] = np.log(v)

    # Helper: bootstrap mean CI on log metric (lightweight)
    def _boot_mean_ci(x: np.ndarray, n=3000, alpha=0.05, seed=0):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan, (np.nan, np.nan)
        rng = np.random.default_rng(seed)
        boots = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(n)])
        return float(x.mean()), (float(np.percentile(boots, 100*alpha/2)), float(np.percentile(boots, 100*(1-alpha/2))))

    # --- figure layout ---
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    stats = {}

    # (A) Overlaid histograms of log(metric)
    ax0.set_title(f"log({metric_col}) per nucleus")
    for g in groups:
        x = data_log[g]
        if x.size == 0:
            continue
        ax0.hist(x, bins=20, histtype="step", label=str(g))  # default color cycle
    ax0.axvline(0.0, linestyle="--")
    ax0.set_xlabel(f"log({metric_col})  [>0 → periphery-enriched]")
    ax0.set_ylabel("# nuclei")
    ax0.legend()

    # (B) Mean radial profile ± 95% band per condition
    ax1.set_title("Radial heterochromatin fraction (mean ± 95% band)")
    bin_centers_global = None
    for g in groups:
        sub = df.loc[df[cond_col] == g]
        # Collect profiles (skip rows without arrays)
        profs = []
        bc_ref = None
        for prof, bc in zip(sub["radial_profile"].values, sub["radial_bin_centers"].values):
            if isinstance(prof, (list, np.ndarray)) and isinstance(bc, (list, np.ndarray)):
                profs.append(np.asarray(prof, dtype=float))
                if bin_centers_global is None and bc_ref is None:
                    bc_ref = np.asarray(bc, dtype=float)
        if not profs:
            continue
        prof_stack = np.vstack(profs)
        if bin_centers_global is None:
            bin_centers_global = bc_ref
        mean_prof = np.nanmean(prof_stack, axis=0)
        lo = np.nanpercentile(prof_stack, 2.5, axis=0)
        hi = np.nanpercentile(prof_stack, 97.5, axis=0)
        ax1.plot(bin_centers_global, mean_prof, marker="o", label=str(g))
        ax1.fill_between(bin_centers_global, lo, hi, alpha=0.2, linewidth=0)

    ax1.set_xlabel("Normalized radius (0 = rim, 1 = center)")
    ax1.set_ylabel("Heterochromatin fraction")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # (C) Δ mean radius vs nucleus area (colored by condition)
    ax2.set_title("Per-nucleus effect vs size")
    for g in groups:
        sub = df.loc[df[cond_col] == g]
        x = pd.to_numeric(sub["area_px"], errors="coerce").values
        y = pd.to_numeric(sub["delta_mean_r_het_vs_eu"], errors="coerce").values
        mask = np.isfinite(x) & np.isfinite(y)
        ax2.scatter(x[mask], y[mask], label=str(g), alpha=0.7, s=12)
    ax2.axhline(0.0, linestyle="--")
    ax2.set_xlabel("Nucleus area (pixels)")
    ax2.set_ylabel("Δ mean r (het − eu)  [<0 → periphery]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # (D) KS D vs metric (colored by condition)
    ax3.set_title("Agreement between metrics")
    for g in groups:
        sub = df.loc[df[cond_col] == g]
        x = pd.to_numeric(sub[metric_col], errors="coerce").values
        y = pd.to_numeric(sub["ks_D"], errors="coerce").values
        mask = np.isfinite(x) & (x > 0) & np.isfinite(y)
        ax3.scatter(x[mask], y[mask], label=str(g), alpha=0.7, s=12)
    ax3.set_xscale("log")
    ax3.axvline(1.0, linestyle="--")
    ax3.set_xlabel(f"{metric_col} (outer vs inner)")
    ax3.set_ylabel("KS D (het vs eu radii)")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()

    # Figure title
    fig.suptitle(title, y=0.98, fontsize=14)
    fig.tight_layout()

    # --- stats summary by condition (mean log metric with bootstrap CI) ---
    for g in groups:
        m, (lo, hi) = _boot_mean_ci(data_log[g], n=3000, alpha=0.05, seed=0)
        stats[g] = {"N": int((df[cond_col] == g).sum()),
                    "mean_log_metric": m, "CI95_log_metric": (lo, hi)}

    # If exactly two groups, add a Mann–Whitney U on log(metric)
    if len(groups) == 2 and data_log[groups[0]].size and data_log[groups[1]].size:
        from scipy.stats import mannwhitneyu
        U, p = mannwhitneyu(data_log[groups[0]], data_log[groups[1]], alternative="two-sided")
        stats["mannwhitneyu_log_metric"] = {"groups": groups, "U": float(U), "p": float(p)}

    return fig, stats

def main():
    # load metrics dataframe
    metrics_dataframe = pd.read_csv(config.path_to_stats_dataframe)
    # Example: compare control vs treated (or more groups)
    fig, stats = plot_summary_by_condition(
        metrics_dataframe,               # combined per-nucleus DataFrame from all folders
        cond_col=config.cond_col,            # default column name
        group_order=config.group_order,  # optional explicit order
        metric_col=config.metric_col,         # or "risk_ratio" if you prefer RP
        title=config.title
    )
    fig.savefig(config.plot_metrics_path, dpi=300)
    print("Saved summary figure to", config.plot_metrics_path)
    print("Condition summary stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    plt.close(fig)


if __name__ == "__main__":
    main()  # only run when executed as a script
