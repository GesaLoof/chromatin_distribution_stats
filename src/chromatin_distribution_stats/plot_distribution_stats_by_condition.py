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
from matplotlib.patches import Patch, Polygon
from typing import Tuple, List, Optional
# TODO check imports



def plot_summary_by_condition(
    df: pd.DataFrame,
    cond_col: str = "condition",
    group_order: list[str] | None = None,
    metric_col: str = "odds_ratio",
    title: str = "Chromatin distribution — condition summary",
    *,
    colors=None,   # dict or list aligned with group_order
) -> tuple[plt.Figure, dict]:
    """
    Cohort summary, stratified by condition (no histogram panel).

    Panels:
      (A) Δ mean radius (het−eu) vs nucleus area, colored by condition
      (B) KS D vs metric_col, colored by condition
    """
    assert cond_col in df.columns, f"'{cond_col}' column not found in df."
    assert metric_col in df.columns, f"'{metric_col}' column not found in df."
    groups = group_order or sorted(df[cond_col].dropna().unique().tolist())
    assert len(groups) >= 2, "Need at least two conditions."

    color_map = _resolve_colors(groups, colors)

    # Prepare log(metric) per condition for stats (drop non-positive / non-finite)
    data_log = {}
    for g in groups:
        v = pd.to_numeric(df.loc[df[cond_col] == g, metric_col], errors="coerce").values
        v = v[np.isfinite(v) & (v > 0)]
        data_log[g] = np.log(v)

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

    # --- figure: 1×2 ---
    fig = plt.figure(figsize=(12, 4.8))
    gs = fig.add_gridspec(1, 2, wspace=0.28)
    axA = fig.add_subplot(gs[0, 0])  # Δ mean r vs area
    axB = fig.add_subplot(gs[0, 1])  # KS D vs metric

    # (A) Δ mean radius vs nucleus area
    axA.set_title("Per-nucleus effect vs size")
    for g in groups:
        sub = df.loc[df[cond_col] == g]
        x = pd.to_numeric(sub["area_px"], errors="coerce").values
        y = pd.to_numeric(sub["delta_mean_r_het_vs_eu"], errors="coerce").values
        m = np.isfinite(x) & np.isfinite(y)
        axA.scatter(x[m], y[m], label=str(g), alpha=0.75, s=14, color=color_map[g])
    axA.axhline(0.0, linestyle="--", color="0.5")
    axA.set_xlabel("Nucleus area (pixels)")
    axA.set_ylabel("Δ mean r (het − eu)  [<0 → periphery]")
    axA.grid(True, alpha=0.25)
    axA.legend(frameon=False)

    # (B) KS D vs metric
    axB.set_title("Agreement between metrics")
    for g in groups:
        sub = df.loc[df[cond_col] == g]
        x = pd.to_numeric(sub[metric_col], errors="coerce").values
        y = pd.to_numeric(sub["ks_D"], errors="coerce").values
        m = np.isfinite(x) & (x > 0) & np.isfinite(y)
        axB.scatter(x[m], y[m], label=str(g), alpha=0.75, s=14, color=color_map[g])
    axB.set_xscale("log")
    axB.axvline(1.0, linestyle="--", color="0.5")
    axB.set_xlabel(f"{metric_col} (outer vs inner)")
    axB.set_ylabel("KS D (het vs eu radii)")
    axB.grid(True, which="both", alpha=0.25)
    axB.legend(frameon=False)

    fig.suptitle(title, y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # --- stats summary on log(metric) ---
    stats = {}
    for g in groups:
        m, (lo, hi) = _boot_mean_ci(data_log[g], n=3000, alpha=0.05, seed=0)
        stats[g] = {"N": int((df[cond_col] == g).sum()),
                    "mean_log_metric": m, "CI95_log_metric": (lo, hi)}

    if len(groups) == 2 and data_log[groups[0]].size and data_log[groups[1]].size:
        from scipy.stats import mannwhitneyu
        U, p = mannwhitneyu(data_log[groups[0]], data_log[groups[1]], alternative="two-sided")
        stats["mannwhitneyu_log_metric"] = {"groups": groups, "U": float(U), "p": float(p)}

    return fig, stats


def plot_histograms_stacked_by_condition(
    df: pd.DataFrame,
    cond_col: str = "condition",
    group_order: list[str] | None = None,
    metric_col: str = "odds_ratio",
    *,
    colors=None,                # dict or list aligned with group_order
    log_metric: bool = True,    # plot log(metric) for OR/RR
    bins: int | str = 30,       # int or 'auto'
    density: bool = True,       # normalize to probability density
    height_per_row: float = 1.6,
    width: float = 8.0,
    title: str = "Distribution of per-nucleus metric by condition",
) -> plt.Figure:
    """
    One histogram panel per condition stacked vertically for legibility.

    - Uses shared bin edges across conditions (on log-scale if log_metric=True).
    - Shows N in each row title.
    """
    assert cond_col in df.columns, f"'{cond_col}' column not found in df."
    assert metric_col in df.columns, f"'{metric_col}' column not found in df."
    groups = group_order or sorted(df[cond_col].dropna().unique().tolist())
    color_map = _resolve_colors(groups, colors)

    # Collect per-group vectors (optionally log-transformed)
    vals = {}
    for g in groups:
        x = pd.to_numeric(df.loc[df[cond_col] == g, metric_col], errors="coerce").values
        if log_metric:
            x = x[np.isfinite(x) & (x > 0)]
            x = np.log(x)
        else:
            x = x[np.isfinite(x)]
        vals[g] = x

    # Shared bin edges across all groups
    allx = np.concatenate([v for v in vals.values() if v.size > 0]) if any(v.size for v in vals.values()) else np.array([])
    if allx.size == 0:
        raise ValueError("No valid metric values to plot.")
    edges = np.histogram_bin_edges(allx, bins=bins)

    # Figure with one row per condition
    n = len(groups)
    fig, axes = plt.subplots(
        n, 1, figsize=(width, max(n * height_per_row, 2.0)),
        sharex=True, sharey=True, constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    # Draw each histogram
    for ax, g in zip(axes, groups):
        x = vals[g]
        if x.size == 0:
            ax.text(0.5, 0.5, f"{g}: no data", ha="center", va="center")
            ax.set_facecolor("0.95")
        else:
            ax.hist(x, bins=edges, density=density, color=color_map[g], alpha=0.7, edgecolor="none")
        ax.grid(True, alpha=0.25)
        N = int((df[cond_col] == g).sum())
        ax.set_title(f"{g}  (N={N})", loc="left", fontsize=10)
        # reference lines (0 on log scale ≡ metric=1)
        if log_metric:
            ax.axvline(0.0, linestyle="--", color="0.5", linewidth=1)

    # Axis labels & title
    if log_metric:
        axes[-1].set_xlabel(f"log({metric_col})  [> 0 → periphery enriched]")
    else:
        axes[-1].set_xlabel(metric_col)
    axes[0].set_ylabel("Density" if density else "Count")
    fig.suptitle(title, y=0.995, fontsize=14)
    return fig



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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse your color helper
# def _resolve_colors(groups, colors): ...

def plot_rp_summary_by_condition(
    df: pd.DataFrame,
    cond_col: str = "condition",
    group_order: list[str] | None = None,
    rp_col: str = "risk_ratio",   # or "enrichment_ratio"
    title: str = "Chromatin distribution — RP (condition summary)",
    *,
    colors=None,                  # dict or list aligned with group_order
) -> tuple[plt.Figure, dict]:
    """
    Cohort summary using Ratio of Proportions (RP) instead of OR.

    Panels:
      (A) Δ mean radius (het−eu) vs nucleus area, colored by condition
      (B) KS D vs RP, colored by condition

    Notes
    -----
    - RP > 1: periphery enriched; RP < 1: interior enriched.
    - We also report mean(log RP) with a bootstrap 95% CI per condition.
    """
    assert cond_col in df.columns, f"'{cond_col}' not found in df."
    assert rp_col in df.columns, f"'{rp_col}' not found in df."
    groups = group_order or sorted(df[cond_col].dropna().unique().tolist())
    assert len(groups) >= 2, "Need at least two conditions."

    color_map = _resolve_colors(groups, colors)

    # Prep log(RP) for stats
    data_log = {}
    for g in groups:
        v = pd.to_numeric(df.loc[df[cond_col] == g, rp_col], errors="coerce").values
        v = v[np.isfinite(v) & (v > 0)]
        data_log[g] = np.log(v)

    def _boot_mean_ci(x: np.ndarray, n=3000, alpha=0.05, seed=0):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan, (np.nan, np.nan)
        rng = np.random.default_rng(seed)
        boots = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(n)])
        return float(x.mean()), (float(np.percentile(boots, 100*alpha/2)),
                                 float(np.percentile(boots, 100*(1-alpha/2))))

    # --- figure: 1×2
    fig = plt.figure(figsize=(12, 4.8))
    gs = fig.add_gridspec(1, 2, wspace=0.28)
    axA = fig.add_subplot(gs[0, 0])  # Δ mean r vs area
    axB = fig.add_subplot(gs[0, 1])  # KS D vs RP

    # (A) Δ mean radius vs area
    axA.set_title("Per-nucleus effect vs size")
    for g in groups:
        sub = df.loc[df[cond_col] == g]
        x = pd.to_numeric(sub["area_px"], errors="coerce").values
        y = pd.to_numeric(sub["delta_mean_r_het_vs_eu"], errors="coerce").values
        m = np.isfinite(x) & np.isfinite(y)
        axA.scatter(x[m], y[m], label=str(g), alpha=0.75, s=14, color=color_map[g])
    axA.axhline(0.0, linestyle="--", color="0.5")
    axA.set_xlabel("Nucleus area (pixels)")
    axA.set_ylabel("Δ mean r (het − eu)  [<0 → periphery]")
    axA.grid(True, alpha=0.25)
    axA.legend(frameon=False)

    # (B) KS D vs RP
    axB.set_title("Agreement between metrics")
    for g in groups:
        sub = df.loc[df[cond_col] == g]
        x = pd.to_numeric(sub[rp_col], errors="coerce").values
        y = pd.to_numeric(sub["ks_D"], errors="coerce").values
        m = np.isfinite(x) & (x > 0) & np.isfinite(y)
        axB.scatter(x[m], y[m], label=str(g), alpha=0.75, s=14, color=color_map[g])
    axB.set_xscale("log")
    axB.axvline(1.0, linestyle="--", color="0.5")  # RP=1 reference
    axB.set_xlabel(f"{rp_col} (outer vs inner)")
    axB.set_ylabel("KS D (het vs eu radii)")
    axB.grid(True, which="both", alpha=0.25)
    axB.legend(frameon=False)

    fig.suptitle(title, y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # stats: mean log(RP) + bootstrap CI
    stats = {}
    for g in groups:
        m, (lo, hi) = _boot_mean_ci(data_log[g], n=3000, alpha=0.05, seed=0)
        stats[g] = {"N": int((df[cond_col] == g).sum()),
                    "mean_log_RP": m, "CI95_log_RP": (lo, hi)}
    if len(groups) == 2 and data_log[groups[0]].size and data_log[groups[1]].size:
        from scipy.stats import mannwhitneyu
        U, p = mannwhitneyu(data_log[groups[0]], data_log[groups[1]], alternative="two-sided")
        stats["mannwhitneyu_log_RP"] = {"groups": groups, "U": float(U), "p": float(p)}

    return fig, stats


def plot_rp_histograms_stacked_by_condition(
    df: pd.DataFrame,
    cond_col: str = "condition",
    group_order: list[str] | None = None,
    rp_col: str = "risk_ratio",   # or "enrichment_ratio"
    *,
    colors=None,                  # dict or list aligned with group_order
    bins: int | str = 30,
    density: bool = True,
    height_per_row: float = 1.6,
    width: float = 8.0,
    title: str = "Distribution of RP (stacked by condition)",
) -> plt.Figure:
    """
    One histogram per condition stacked vertically for readability.
    Plots log(RP) so that 0 means 'no enrichment' (RP=1).
    """
    assert cond_col in df.columns, f"'{cond_col}' column not found in df."
    assert rp_col in df.columns, f"'{rp_col}' column not found in df."
    groups = group_order or sorted(df[cond_col].dropna().unique().tolist())
    color_map = _resolve_colors(groups, colors)

    vals = {}
    for g in groups:
        x = pd.to_numeric(df.loc[df[cond_col] == g, rp_col], errors="coerce").values
        x = x[np.isfinite(x) & (x > 0)]
        vals[g] = np.log(x)  # log(RP)

    allx = np.concatenate([v for v in vals.values() if v.size > 0]) if any(v.size for v in vals.values()) else np.array([])
    if allx.size == 0:
        raise ValueError("No valid RP values to plot.")
    edges = np.histogram_bin_edges(allx, bins=bins)

    n = len(groups)
    fig, axes = plt.subplots(
        n, 1, figsize=(width, max(n * height_per_row, 2.0)),
        sharex=True, sharey=True, constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    for ax, g in zip(axes, groups):
        x = vals[g]
        if x.size == 0:
            ax.text(0.5, 0.5, f"{g}: no data", ha="center", va="center")
            ax.set_facecolor("0.95")
        else:
            ax.hist(x, bins=edges, density=density, color=color_map[g], alpha=0.7, edgecolor="none")
        ax.grid(True, alpha=0.25)
        N = int((df[cond_col] == g).sum())
        ax.set_title(f"{g}  (N={N})", loc="left", fontsize=10)
        ax.axvline(0.0, linestyle="--", color="0.5", linewidth=1)  # log(RP)=0 ⇒ RP=1

    axes[-1].set_xlabel("log(RP)  [0 → RP=1]")
    axes[0].set_ylabel("Density" if density else "Count")
    fig.suptitle(title, y=0.995, fontsize=14)
    return fig


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

    axes[-1].set_xlabel("Normalized radius (0 = periphery, 1 = center)")
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


def _pooled_fixed_effect_from_ci(rr: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[float, float, float]:
    """
    Fixed-effect pooled estimate on log scale using CIs to back out SE.
    Returns (pooled_rr, pooled_lo, pooled_hi). Drops rows with non-finite inputs.
    """
    rr, lo, hi = np.asarray(rr, float), np.asarray(lo, float), np.asarray(hi, float)
    mask = np.isfinite(rr) & np.isfinite(lo) & np.isfinite(hi) & (rr > 0) & (lo > 0) & (hi > 0)
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan
    lrr = np.log(rr[mask])
    llo = np.log(lo[mask])
    lhi = np.log(hi[mask])
    # approximate SE from CI width on log scale
    se = (lhi - llo) / (2 * 1.96)
    # guard tiny/zero se
    se = np.where(se <= 0, np.nan, se)
    ok = np.isfinite(se) & (se > 0)
    if ok.sum() == 0:
        return np.nan, np.nan, np.nan
    w = 1.0 / (se[ok] ** 2)
    m = np.sum(w * lrr[ok]) / np.sum(w)                  # pooled log-RR
    se_m = np.sqrt(1.0 / np.sum(w))                      # SE of pooled
    lo_p = np.exp(m - 1.96 * se_m)
    hi_p = np.exp(m + 1.96 * se_m)
    return float(np.exp(m)), float(lo_p), float(hi_p)

def plot_rp_forest_by_condition(
    df: pd.DataFrame,
    cond_col: str = "condition",
    group_order: Optional[List[str]] = None,
    *,
    rp_col: str = "risk_ratio",                # or "enrichment_ratio"
    lo_col: str = "risk_ratio_CI95_low",
    hi_col: str = "risk_ratio_CI95_high",
    id_cols: Optional[List[str]] = None,       # columns to build left-hand labels (e.g., ["image_id","nucleus_id"])
    colors=None,                               # dict or list aligned with group_order
    max_rows: int = 60,                        # truncate per condition to avoid ultra-tall plots
    sort_by: str = "abs_log",                  # "value" | "abs_log" | "none"
    figsize_per_row: float = 0.22,             # height per row (inches)
    width: float = 7.0,
    title: str = "Forest plot of Ratio of Proportions (RP) by condition",
) -> plt.Figure:
    """
    Forest plot of per-nucleus RP with 95% CIs, one subplot per condition, plus a pooled summary diamond.

    - x-axis is logarithmic; vertical reference at RP=1.
    - Pooled (fixed-effect) summary is computed from reported CIs (backing out SE on log scale).
    """
    assert cond_col in df.columns, f"'{cond_col}' not found"
    for c in (rp_col, lo_col, hi_col):
        assert c in df.columns, f"'{c}' not found"

    groups = group_order or sorted(df[cond_col].dropna().unique().tolist())
    cmap = _resolve_colors(groups, colors)

    # build labels once
    def _build_label(row) -> str:
        if not id_cols:
            # default: try path or nucleus_id if present
            if "path_to_image" in row and "nucleus_id" in row:
                return f"{os.path.basename(str(row['path_to_image']))}  (id {int(row['nucleus_id'])})"
            if "nucleus_id" in row:
                return f"nucleus {int(row['nucleus_id'])}"
            return ""
        parts = []
        for c in id_cols:
            if c in row and pd.notna(row[c]):
                v = row[c]
                try:
                    v = int(v)
                except Exception:
                    pass
                parts.append(str(v))
        return " · ".join(parts)

    # determine global x-limits from all groups (robust range)
    all_lo = pd.to_numeric(df[lo_col], errors="coerce").values
    all_hi = pd.to_numeric(df[hi_col], errors="coerce").values
    all_lo = all_lo[np.isfinite(all_lo) & (all_lo > 0)]
    all_hi = all_hi[np.isfinite(all_hi) & (all_hi > 0)]
    if all_lo.size == 0 or all_hi.size == 0:
        xmin, xmax = 0.5, 2.0
    else:
        xmin, xmax = float(np.nanpercentile(all_lo, 2)), float(np.nanpercentile(all_hi, 98))
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin <= 0 or xmin >= xmax:
            xmin, xmax = 0.5, 2.0

    # build figure
    import os
    nrows_total = 0
    panels = []
    per_panel_axes = []
    # we’ll precompute each panel height
    for g in groups:
        sub = df.loc[df[cond_col] == g].copy()
        # clean
        sub[rp_col] = pd.to_numeric(sub[rp_col], errors="coerce")
        sub[lo_col] = pd.to_numeric(sub[lo_col], errors="coerce")
        sub[hi_col] = pd.to_numeric(sub[hi_col], errors="coerce")
        sub = sub[np.isfinite(sub[rp_col]) & np.isfinite(sub[lo_col]) & np.isfinite(sub[hi_col])]
        sub = sub[(sub[rp_col] > 0) & (sub[lo_col] > 0) & (sub[hi_col] > 0)]
        if sub.empty:
            panels.append((g, sub, 0))
            continue
        # sorting
        if sort_by == "value":
            sub = sub.sort_values(rp_col)
        elif sort_by == "abs_log":
            sub = sub.assign(_abslog=np.abs(np.log(sub[rp_col]))).sort_values("_abslog", ascending=False).drop(columns=["_abslog"])
        # truncate if too long
        if max_rows is not None and len(sub) > max_rows:
            sub = sub.iloc[:max_rows].copy()
        panels.append((g, sub, len(sub)))
        nrows_total += len(sub) + 2  # +2 for a bit of breathing room and summary row

    # compute figure height
    fig_h = max(2.5, nrows_total * figsize_per_row)
    fig, axes = plt.subplots(len(groups), 1, figsize=(width, fig_h), constrained_layout=True, sharex=True)
    if len(groups) == 1:
        axes = [axes]

    for ax, (g, sub, n) in zip(axes, panels):
        ax.set_xscale("log")
        ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1)

        if n == 0:
            ax.text(0.5, 0.5, f"{g}: no RP with CI", ha="center", va="center", transform=ax.transAxes)
            ax.set_yticks([])
            ax.grid(True, which="both", alpha=0.2)
            ax.set_xlim(xmin, xmax)
            ax.set_title(f"{g} (N=0)", loc="left")
            continue

        # y positions top->bottom
        y = np.arange(n)[::-1]
        # draw CIs as horizontal whiskers + point
        col = cmap[g]
        rr = sub[rp_col].values
        lo = sub[lo_col].values
        hi = sub[hi_col].values
        ax.hlines(y, lo, hi, color=col, lw=1.6, alpha=0.9)
        ax.plot(rr, y, "o", color=col, ms=4)

        # left labels
        labels = [ _build_label(row) for _, row in sub.iterrows() ]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.grid(True, which="both", axis="x", alpha=0.2)

        # pooled summary diamond
        pooled_rr, pooled_lo, pooled_hi = _pooled_fixed_effect_from_ci(rr, lo, hi)
        if np.isfinite(pooled_rr):
            ysum = -1.2  # a bit below the last row
            xL, xC, xR = pooled_lo, pooled_rr, pooled_hi
            # diamond polygon
            diamond = Polygon(
                [[xL, ysum], [xC, ysum + 0.4], [xR, ysum], [xC, ysum - 0.4]],
                closed=True, facecolor=col if col is not None else "C0", alpha=0.5, edgecolor="none"
            )
            ax.add_patch(diamond)
            ax.text(xR*1.02, ysum, "summary", va="center", fontsize=8)

        ax.set_ylim(-2.0, n - 0.5)
        ax.set_xlim(xmin, xmax)
        ax.set_title(f"{g}  (N={n})", loc="left")

    axes[-1].set_xlabel("Ratio of Proportions (RP).  log-scale; RP=1 is no enrichment")
    fig.suptitle(title, fontsize=14, y=0.995)
    return fig


def main():
    # load metrics dataframe
    metrics_dataframe = pd.read_csv(config.path_to_stats_dataframe)
    # If your df came from CSV, parse arrays first
    # make sure radial profiles are arrays
    metrics_dataframe = coerce_profiles_from_csv(metrics_dataframe)

    # Summary (2-panel) figure, saved to your configured path
    if config.plot_metrics_path is not None:
        fig_sum, stats = plot_summary_by_condition(
            metrics_dataframe,
            cond_col=config.cond_col,
            group_order=config.group_order,
            metric_col=config.metric_col,      # e.g. "odds_ratio" or "risk_ratio"
            colors=config.group_colors,
            title="Chromatin distribution per-nucleus",
        )
        fig_sum.savefig(config.plot_metrics_path, dpi=300)
        plt.close(fig_sum)

    # Stacked histograms for legibility
    if config.plot_histograms_stacked_path is not None:
        fig_hist = plot_histograms_stacked_by_condition(
            metrics_dataframe,
            cond_col=config.cond_col,
            group_order=config.group_order,
            metric_col="odds_ratio",
            colors=config.group_colors,
            log_metric=True,                   # recommended for ratios (OR/RR)
            bins=30,
            density=True,
            height_per_row=1.6,
            width=8.0,
            title="log(odds ratio) per nucleus",
        )
        fig_hist.savefig(config.plot_histograms_stacked_path, dpi=300)
        plt.close(fig_hist)

    # Summary using RP (recommended: risk_ratio)
    if config.plot_rp_summary_path is not None:
        fig_rp, stats_rp = plot_rp_summary_by_condition(
            metrics_dataframe,
            cond_col=config.cond_col,
            group_order=config.group_order,
            rp_col="risk_ratio",                # or "enrichment_ratio"
            colors=config.group_colors,
            title="Chromatin distribution — RP (condition summary)",
        )
        fig_rp.savefig(config.plot_rp_summary_path, dpi=300); plt.close(fig_rp)

    # Stacked histograms of log(RP) by condition
    if config.plot_rp_histograms_path is not None:
        fig_hist_rp = plot_rp_histograms_stacked_by_condition(
            metrics_dataframe,
            cond_col=config.cond_col,
            group_order=config.group_order,
            rp_col="risk_ratio",
            colors=config.group_colors,
            bins=30,
            density=True,
            height_per_row=1.6,
            width=8.0,
            title="Per-nucleus log(risk ratio)",
        )
        fig_hist_rp.savefig(config.plot_rp_histograms_path, dpi=300); plt.close(fig_hist_rp)
        plt.close(fig)

    if config.plot_profiles_stacked_path is not None:
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

    if config.plot_rp_forest_path is not None:
        fig = plot_rp_forest_by_condition(
            metrics_dataframe,
            cond_col=config.cond_col,
            group_order=config.group_order,         # e.g., ["control","treated"]
            rp_col="risk_ratio",                    # or "enrichment_ratio"
            lo_col="risk_ratio_CI95_low",
            hi_col="risk_ratio_CI95_high",
            id_cols=["path_to_image", "nucleus_id"],# labels on the left (customize)
            colors=config.group_colors,             # optional color control
            max_rows=60,                            # clip ultra-long lists
            sort_by="abs_log",                      # show most shifted nuclei on top
            width=8.0,
            figsize_per_row=0.22,
            title="Forest plot — Ratio of Proportions (RP) by condition",
        )
        fig.savefig(config.plot_rp_forest_path, dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    main()  # only run when executed as a script
