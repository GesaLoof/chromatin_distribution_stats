from skimage.measure import label as cc_label
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label as cc_label, regionprops
from tifffile import imread, imwrite
from sklearn.cluster import KMeans
from chromatin_distribution_stats import config
from utils import ensure_labeled_nuclei, coerce_profiles_from_csv
import os
import ast, re
from typing import Dict, List, Sequence
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# TODO check imports



def plot_summary_by_condition(
    df: pd.DataFrame,
    cond_col: str = "condition",
    group_order: list[str] | None = None,
    metric_col: str = "odds_ratio",
    title: str = "Chromatin distribution — condition summary",
    colors=None,              
) -> tuple[plt.Figure, dict]:
    """
    Cohort summary, stratified by condition (no radial-profile panel).

    Panels:
      (A) Overlaid histograms of log(metric_col) per nucleus, per condition
      (B) Scatter: Δ mean radius (het−eu) vs nucleus area, colored by condition
      (C) Scatter: KS D vs metric_col, colored by condition
    """
    # --- sanity & setup ---
    assert cond_col in df.columns, f"'{cond_col}' column not found in df."
    assert metric_col in df.columns, f"'{metric_col}' column not found in df."
    groups = group_order or sorted(df[cond_col].dropna().unique().tolist())
    assert len(groups) >= 2, "Need at least two conditions."

    # Resolve colors consistently with your stacked plot
    color_map = _resolve_colors(groups, colors)

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
        return float(x.mean()), (
            float(np.percentile(boots, 100*alpha/2)),
            float(np.percentile(boots, 100*(1-alpha/2))),
        )

    # --- figure layout: 1×3 wide ---
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, wspace=0.28, hspace=0.0)
    ax0 = fig.add_subplot(gs[0, 0])  # histogram
    ax2 = fig.add_subplot(gs[0, 1])  # Δ mean r vs area
    ax3 = fig.add_subplot(gs[0, 2])  # KS D vs metric

    stats = {}

    # (A) Overlaid histograms of log(metric)
    ax0.set_title(f"log({metric_col}) per nucleus")
    for g in groups:
        x = data_log[g]
        if x.size == 0:
            continue
        ax0.hist(x, bins=20, histtype="step", label=str(g), color=color_map[g])
    ax0.axvline(0.0, linestyle="--", color="0.5")
    ax0.set_xlabel(f"log({metric_col})  [>0 → periphery-enriched]")
    ax0.set_ylabel("# nuclei")
    ax0.legend(frameon=False)
    ax0.grid(True, alpha=0.25)

    # (B) Δ mean radius vs nucleus area (colored by condition)
    ax2.set_title("Per-nucleus effect vs size")
    for g in groups:
        sub = df.loc[df[cond_col] == g]
        x = pd.to_numeric(sub["area_px"], errors="coerce").values
        y = pd.to_numeric(sub["delta_mean_r_het_vs_eu"], errors="coerce").values
        m = np.isfinite(x) & np.isfinite(y)
        ax2.scatter(x[m], y[m], label=str(g), alpha=0.75, s=14, color=color_map[g])
    ax2.axhline(0.0, linestyle="--", color="0.5")
    ax2.set_xlabel("Nucleus area (pixels)")
    ax2.set_ylabel("Δ mean r (het − eu)  [<0 → periphery]")
    ax2.grid(True, alpha=0.25)

    # (C) KS D vs metric (colored by condition)
    ax3.set_title("Agreement between metrics")
    for g in groups:
        sub = df.loc[df[cond_col] == g]
        x = pd.to_numeric(sub[metric_col], errors="coerce").values
        y = pd.to_numeric(sub["ks_D"], errors="coerce").values
        m = np.isfinite(x) & (x > 0) & np.isfinite(y)
        ax3.scatter(x[m], y[m], label=str(g), alpha=0.75, s=14, color=color_map[g])
    ax3.set_xscale("log")
    ax3.axvline(1.0, linestyle="--", color="0.5")
    ax3.set_xlabel(f"{metric_col} (outer vs inner)")
    ax3.set_ylabel("KS D (het vs eu radii)")
    ax3.grid(True, which="both", alpha=0.25)

    fig.suptitle(title, y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # --- stats summary by condition (mean log metric with bootstrap CI) ---
    for g in groups:
        m, (lo, hi) = _boot_mean_ci(data_log[g], n=3000, alpha=0.05, seed=0)
        stats[g] = {
            "N": int((df[cond_col] == g).sum()),
            "mean_log_metric": m,
            "CI95_log_metric": (lo, hi),
        }

    # If exactly two groups, add a Mann–Whitney U on log(metric)
    if len(groups) == 2 and data_log[groups[0]].size and data_log[groups[1]].size:
        from scipy.stats import mannwhitneyu
        U, p = mannwhitneyu(data_log[groups[0]], data_log[groups[1]], alternative="two-sided")
        stats["mannwhitneyu_log_metric"] = {"groups": groups, "U": float(U), "p": float(p)}

    return fig, stats



def _parse_array_like(x):
    if isinstance(x, np.ndarray): return x.astype(float)
    if isinstance(x, list):       return np.asarray(x, dtype=float)
    if pd.isna(x):                return np.array([], dtype=float)
    try:
        v = ast.literal_eval(str(x))
        return np.asarray(v, dtype=float)
    except Exception:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
        return np.asarray([float(t) for t in nums], dtype=float)

def _resolve_colors(groups: Sequence[str], colors=None) -> Dict[str, str | None]:
    if colors is None:
        return {g: None for g in groups}
    if isinstance(colors, dict):
        return {g: colors.get(g, None) for g in groups}
    if isinstance(colors, (list, tuple)):
        if len(colors) < len(groups):
            raise ValueError("Not enough colors for groups.")
        return {g: colors[i] for i, g in enumerate(groups)}
    raise TypeError("colors must be None, dict, or list/tuple")

def plot_profiles_stacked_by_condition(
    df: pd.DataFrame,
    cond_col: str = "condition",
    group_order: List[str] | None = None,
    profile_col: str = "radial_profile",
    centers_col: str = "radial_bin_centers",
    *,
    show_individual: bool = True,
    ci: float = 95.0,
    share_y: bool = True,
    height_per_row: float = 2.2,
    width: float = 8.0,
    title: str = "Radial heterochromatin fraction by condition (stacked)",
    colors=None,
    legend: bool = True,
    legend_position: str = "top",  # "top" or "bottom"
    legend_ncols: int = 3,
    legend_alpha: float = 0.25,
) -> plt.Figure:
    """
    Stack one radial-profile subplot per condition (normalized radius on x),
    with a clear, non-overlapping figure legend and suptitle.

    Uses constrained_layout; do NOT call tight_layout on the returned figure.
    """
    assert cond_col in df.columns
    assert profile_col in df.columns
    assert centers_col in df.columns

    groups = group_order or sorted(df[cond_col].dropna().unique().tolist())
    color_map = _resolve_colors(groups, colors)

    # --- Use constrained layout to manage space between title / legend / axes
    fig, axes = plt.subplots(
        len(groups), 1,
        figsize=(width, max(height_per_row * len(groups), 2.0)),
        sharex=True, sharey=share_y,
        constrained_layout=True,
    )
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, hspace=0.04)  # gentle spacing

    global_ymin, global_ymax = +np.inf, -np.inf
    ci_lo = (100 - ci) / 2.0
    ci_hi = 100 - ci_lo

    for ax, g in zip(axes, groups):
        sub = df.loc[df[cond_col] == g]
        profs, centers = [], None
        for p_raw, c_raw in zip(sub[profile_col].values, sub[centers_col].values):
            p = _parse_array_like(p_raw)
            if p.size == 0:
                continue
            if centers is None:
                c = _parse_array_like(c_raw)
                centers = c if c.size == p.size else (np.arange(p.size) + 0.5) / p.size
            profs.append(p.astype(float))

        if not profs:
            ax.text(0.5, 0.5, f"{g}: no valid profiles", ha="center", va="center")
            ax.set_facecolor("0.95")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            continue

        prof_stack = np.vstack(profs)
        col = color_map[g]

        if show_individual:
            for p in prof_stack:
                ax.plot(centers, p, linewidth=0.6, alpha=0.25, color=col)

        mean_prof = np.nanmean(prof_stack, axis=0)
        lo = np.nanpercentile(prof_stack, ci_lo, axis=0)
        hi = np.nanpercentile(prof_stack, ci_hi, axis=0)

        ax.plot(centers, mean_prof, marker="o", linewidth=1.8, color=col, label=f"{g} mean")
        ax.fill_between(centers, lo, hi, alpha=legend_alpha, linewidth=0, color=col, label=f"{g} {int(ci)}% band")

        N = prof_stack.shape[0]
        ax.set_ylabel("Het frac")
        ax.set_title(f"{g}  (N={N})", loc="left", fontsize=11)
        ax.grid(True, alpha=0.3)

        y_valid = np.r_[lo, hi, mean_prof]
        y_valid = y_valid[np.isfinite(y_valid)]
        if y_valid.size:
            global_ymin = min(global_ymin, float(np.nanmin(y_valid)))
            global_ymax = max(global_ymax, float(np.nanmax(y_valid)))

    axes[-1].set_xlabel("Normalized radius (0 = rim, 1 = center)")
    for ax in axes:
        ax.set_xlim(0, 1)
    if share_y and np.isfinite(global_ymin) and np.isfinite(global_ymax):
        pad = 0.02 * (global_ymax - global_ymin + 1e-8)
        ymin = max(0.0, global_ymin - pad)
        ymax = min(1.0, global_ymax + pad) if global_ymax <= 1.0 else (global_ymax + pad)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    # -------- Figure-level legend explaining glyphs --------
    if legend:
        handles = []
        labels = []
        if show_individual:
            handles.append(Line2D([0], [0], lw=0.8, alpha=0.25, color="k"))
            labels.append("Individual nuclei")
        handles.append(Line2D([0], [0], lw=1.8, marker="o", color="k"))
        labels.append("Mean profile")
        handles.append(Patch(facecolor="k", alpha=legend_alpha))
        labels.append(f"{int(ci)}% CI")

        if legend_position == "top":
            # Put legend above the axes; move suptitle a bit higher to avoid overlap
            fig.suptitle(title, fontsize=14, y=1.06)
            fig.legend(handles, labels, ncol=legend_ncols, frameon=False,
                       loc="upper center", bbox_to_anchor=(0.5, 1.015))
            # Give a touch more headroom above the legend
            fig.set_constrained_layout_pads(h_pad=0.06)
        elif legend_position == "bottom":
            fig.suptitle(title, fontsize=14)
            fig.legend(handles, labels, ncol=legend_ncols, frameon=False,
                       loc="lower center", bbox_to_anchor=(0.5, -0.02))
            fig.set_constrained_layout_pads(h_pad=0.2 )
        else:
            raise ValueError("legend_position must be 'top' or 'bottom'")
    else:
        fig.suptitle(title, fontsize=14)

    return fig



def main():
    # load metrics dataframe
    metrics_dataframe = pd.read_csv(config.path_to_stats_dataframe)
    # make sure radial profiles are arrays
    metrics_dataframe = coerce_profiles_from_csv(metrics_dataframe)

    # Example: compare control vs treated (or more groups)
    fig, stats = plot_summary_by_condition(
        metrics_dataframe,               # combined per-nucleus DataFrame from all folders
        cond_col=config.cond_col,        # default column name thatindicates condition of each nucleus
        group_order=config.group_order,  # optional explicit order of conditions
        metric_col=config.metric_col,    # or "risk_ratio" if you prefer RP
        title=config.title,
        colors=config.group_colors, 
    )
    if config.plot_metrics_path:
        out_dir = os.path.dirname(config.plot_metrics_path)
    if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    fig.savefig(config.plot_metrics_path, dpi=300)
    print("Saved summary figure to", config.plot_metrics_path)
    print("Condition summary stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    plt.close(fig)


    # If your df came from CSV, parse arrays first (or switch to pickle/parquet)
    # metrics_dataframe = coerce_profiles_from_csv(metrics_dataframe)  # from previous message

    fig = plot_profiles_stacked_by_condition(
        metrics_dataframe,
        cond_col=config.cond_col,
        group_order=config.group_order,  # or None
        colors=config.group_colors,      # optional: dict or list aligned with group_order
        show_individual=True,            # set False for cleaner mean+band only
        ci=95.0,
        share_y=True,
        height_per_row=2.0,
        width=7.5,
        title="Radial heterochromatin fraction (stacked by condition)",
        legend = True,
        legend_position = "bottom",  # "top" or "bottom"
        legend_ncols = 3,
        legend_alpha = 0.25,


    )
    fig.savefig(config.plot_profiles_stacked_path, dpi=300)
    plt.close(fig)



if __name__ == "__main__":
    main()  # only run when executed as a script
