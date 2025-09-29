import numpy as np
import pandas as pd
# from scipy.ndimage import distance_transform_edt
from scipy.stats import ks_2samp, wasserstein_distance
from skimage.measure import label as cc_label, regionprops
from tifffile import imread
from sklearn.cluster import KMeans
# from scipy.ndimage import gaussian_filter, uniform_filter, distance_transform_edt
# from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
from chromatin_distribution_stats import config
from typing import Tuple
from skimage.measure import label as cc_label, regionprops
import os
from chromatin_distribution_stats.utils import _normalized_radius, _distance_from_rim
# TODO check imports


def hetero_distribution_metrics(
    het_mask: np.ndarray,
    nuc_mask: np.ndarray,
    outer_width: float = 0.20,
    inner_width: float = 0.20,
    n_bins: int = 20,
    eps: float = 1e-8,
    *,
    shell_mode: str = "normalized",        # "normalized" or "distance"
    pixel_size: float | None = None,        # add µm/px if you want microns in distance mode
    condition = config.condition) -> dict:
    """
    Compute metrics describing heterochromatin distribution within ONE nucleus.

    Parameters
    ----------
    het_mask : np.ndarray (bool)
        Heterochromatin mask for the entire image. The function will clip it to 'nuc_mask'.
    nuc_mask : np.ndarray (bool)
        Mask for a single nucleus (True inside, False outside).
    outer_width : float, optional (default: 0.20)
        Thickness of the peripheral shell as a fraction of radius (0 at rim → 1 at center).
        Example: 0.20 means use r ∈ [0.00, 0.20] for the periphery.
    inner_width : float, optional (default: 0.20)
        Thickness of the core shell as a fraction of radius.
        Example: 0.20 means use r ∈ [0.80, 1.00] for the interior.
    n_bins : int, optional (default: 20)
        Number of concentric shells for the radial heterochromatin fraction profile.
    eps : float, optional (default: 1e-8)
        Small constant to avoid division by zero in ratio computations.

   Design (hybrid):
    - Radial profiles, KS/EMD, mean radii are computed on the **normalized radius** r_norm ∈ [0,1]
      so profiles remain comparable across nuclei.
    - The **inner/outer shells** used for OR / RP / p_het_* are chosen by:
        * shell_mode="normalized": outer_width & inner_width are FRACTIONS of radius.
        * shell_mode="distance"  : outer_width & inner_width are ABSOLUTE THICKNESSES (px or µm).

    Returns
    -------
    metrics : dict
        A dictionary with the following keys:
        - "p_het_outer": fraction of heterochromatin pixels in the outer shell.
        - "p_het_inner": fraction of heterochromatin pixels in the inner shell.
        - "enrichment_ratio": naive ratio of proportions p_het_outer / p_het_inner (RP).
        - "risk_ratio": stabilized ratio of proportions (RP) using the same 0.5 pseudocounts as OR.
        - "risk_ratio_CI95_low", "risk_ratio_CI95_high": 95% CI for risk_ratio via log-normal approx.
        - "odds_ratio": odds of hetero in outer vs inner; computed as (a*d)/(b*c) with 0.5 pseudocounts.
        - "odds_ratio_CI95_low", "odds_ratio_CI95_high": 95% CI for the OR using a log-scale approximation.
        - "mean_r_het": mean normalized radius for heterochromatin pixels (lower = more peripheral).
        - "mean_r_eu": mean normalized radius for non-heterochromatin (euchromatin) pixels.
        - "mean_r_all": mean normalized radius across all nuclear pixels.
        - "delta_mean_r_het_vs_eu": mean_r_het - mean_r_eu (negative suggests periphery enrichment).
        - "delta_mean_r_het_vs_all": mean_r_het - mean_r_all.
        - "ks_D", "ks_p": Kolmogorov–Smirnov statistic and p-value comparing r-distributions of het vs eu.
        - "emd": 1D Wasserstein distance between r(het) and r(eu).
        - "radial_profile": array of length n_bins with heterochromatin fraction per shell.
        - "radial_bin_centers": array of length n_bins with the center radius of each shell.
        - "radial_auc": area under the radial profile curve.
        - "radial_slope": simple least-squares slope of the profile vs normalized radius.
        - "r_max": float (1.0 for normalized; max distance for distance mode)
        - shell_mode, shell_unit, shell_r_max  (metadata about shell definition)

    Notes
    -----
    - The function first clips het_mask to the nucleus: `het = het_mask & nuc_mask`.
    - The OR and the stabilized RR use Haldane–Anscombe 0.5 pseudocounts (helps when a cell is 0).
    - The normalized radius r has 0 at the nuclear envelope and 1 at the center.
    """
     # Clip to nucleus
    het = (het_mask & nuc_mask).astype(bool)
    nuc = nuc_mask.astype(bool)
    eu = nuc & (~het)

    # ---- Profiles & distribution metrics on NORMALIZED radius ----
    #nuc = np.asarray(nuc_mask)           # the object your test passes
    print("dtype:", nuc.dtype)
    print("inside sum:", int((nuc > 0).sum()), "outside sum:", int((nuc == 0).sum()))
    print("unique values:", np.unique(nuc))

    # Is polarity correct (inside==True)?
    assert (nuc.dtype == bool) or set(np.unique(nuc)).issubset({0,1}), "Mask must be boolean/binary."
    assert (nuc >  0).any(), "Empty nucleus mask."
    assert (nuc == 0).any(), "Mask is full—polarity might be flipped."

    r_norm = _normalized_radius(nuc)  # 0 at rim, 1 at center

    # Bins for normalized profile (shared semantics across nuclei)
    bins_norm = np.linspace(0.0, 1.0, n_bins + 1)
    shell_idx = np.digitize(r_norm[nuc], bins_norm, right=True) - 1
    het_in_nuc = het[nuc].astype(np.float32)

    prof = np.array([
        het_in_nuc[shell_idx == i].mean() if np.any(shell_idx == i) else np.nan
        for i in range(n_bins)
    ])
    bin_centers = 0.5 * (bins_norm[:-1] + bins_norm[1:])

    # AUC & slope on normalized profile
    if np.all(np.isnan(prof)):
        prof_valid = np.zeros_like(prof)
        print("Warning: radial profile is all NaNs (empty nucleus?)")
    else:
        prof_valid = np.where(np.isnan(prof), np.nanmean(prof), prof)
    auc = np.trapezoid(prof_valid, bin_centers)

    mask_ok = ~np.isnan(prof)
    if mask_ok.sum() >= 2:
        x = bin_centers[mask_ok]
        y = prof[mask_ok]
        x = (x - x.mean()) / (x.std() + 1e-8)
        slope = float(np.dot(x, y) / (np.dot(x, x) + 1e-8))
    else:
        slope = np.nan

    # Mean radii (normalized)
    mean_r_het = r_norm[het].mean() if het.any() else np.nan
    mean_r_eu  = r_norm[eu].mean()  if eu.any()  else np.nan
    mean_r_all = r_norm[nuc].mean() if nuc.any() else np.nan
    delta_mean_r_het_vs_eu  = (mean_r_het - mean_r_eu) if np.isfinite(mean_r_het) and np.isfinite(mean_r_eu) else np.nan
    delta_mean_r_het_vs_all = (mean_r_het - mean_r_all) if np.isfinite(mean_r_het) else np.nan

    # KS/EMD comparing r_norm distributions
    ks_D, ks_p = (np.nan, np.nan)
    emd = np.nan
    if het.any() and eu.any():
        ks_D, ks_p = ks_2samp(r_norm[het], r_norm[eu], alternative="two-sided", mode="auto")
        emd = wasserstein_distance(r_norm[het], r_norm[eu])

    # ---- Shells for OR/RP: pick by normalized or distance semantics ----
    if shell_mode == "normalized":
        outer = (r_norm <= outer_width) & nuc
        inner = (r_norm >= (1.0 - inner_width)) & nuc
        shell_unit = f"outer_{outer_width}_inner_{inner_width}"
        shell_r_max = 1.0
    elif shell_mode == "distance":
        # distance mode: absolute thickness from rim/center (px or µm)
        r_dist, rmax_dist, shell_unit = _distance_from_rim(nuc, pixel_size=pixel_size)
        ow = min(outer_width, rmax_dist/2)  # at most half-radius
        iw = min(inner_width, rmax_dist/2)
        outer = (r_dist <= ow) & nuc
        inner = (r_dist >= (rmax_dist - iw)) & nuc
        shell_r_max = float(rmax_dist)
        shell_unit = f"outer_{outer_width}_inner_{inner_width}_px:_{pixel_size if pixel_size is not None else 'px'}"

    p_het_outer = het[outer].mean() if np.any(outer) else np.nan
    p_het_inner = het[inner].mean() if np.any(inner) else np.nan
    enrichment_ratio = p_het_outer / (p_het_inner + eps)

    # 2×2 with 0.5 pseudocounts
    a = (het & outer).sum() + 0.5
    b = (eu  & outer).sum() + 0.5
    c = (het & inner).sum() + 0.5
    d = (eu  & inner).sum() + 0.5

    OR = (a * d) / (b * c)
    se_logOR = np.sqrt(1/a + 1/b + 1/c + 1/d)
    logOR = np.log(OR)
    OR_CI95 = (np.exp(logOR - 1.96 * se_logOR), np.exp(logOR + 1.96 * se_logOR))

    n_outer = a + b
    n_inner = c + d
    p_outer_stab = a / n_outer
    p_inner_stab = c / n_inner
    risk_ratio = p_outer_stab / (p_inner_stab + eps)
    se_logRR = np.sqrt(max(0.0, (1/a - 1/n_outer) + (1/c - 1/n_inner)))
    logRR = np.log(risk_ratio)
    RR_CI95 = (np.exp(logRR - 1.96 * se_logRR), np.exp(logRR + 1.96 * se_logRR))

    return {
        # proportions / ratios based on the chosen shells
        "p_het_outer": float(p_het_outer),
        "p_het_inner": float(p_het_inner),
        "enrichment_ratio": float(enrichment_ratio),
        "risk_ratio": float(risk_ratio),
        "risk_ratio_CI95_low": float(RR_CI95[0]),
        "risk_ratio_CI95_high": float(RR_CI95[1]),
        "odds_ratio": float(OR),
        "odds_ratio_CI95_low": float(OR_CI95[0]),
        "odds_ratio_CI95_high": float(OR_CI95[1]),

        # normalized-radius summaries (comparable across nuclei)
        "mean_r_het": float(mean_r_het) if np.isfinite(mean_r_het) else np.nan,
        "mean_r_eu":  float(mean_r_eu)  if np.isfinite(mean_r_eu)  else np.nan,
        "mean_r_all": float(mean_r_all),
        "delta_mean_r_het_vs_eu":  float(delta_mean_r_het_vs_eu)  if np.isfinite(delta_mean_r_het_vs_eu)  else np.nan,
        "delta_mean_r_het_vs_all": float(delta_mean_r_het_vs_all) if np.isfinite(delta_mean_r_het_vs_all) else np.nan,
        "ks_D": float(ks_D) if np.isfinite(ks_D) else np.nan,
        "ks_p": float(ks_p) if np.isfinite(ks_p) else np.nan,
        "emd": float(emd)   if np.isfinite(emd)   else np.nan,

        # normalized radial profile
        "radial_profile": prof,
        "radial_bin_centers": bin_centers,
        "radial_auc": float(auc),
        "radial_slope": float(slope),

        # metadata about shell definition
        "shell_mode": shell_mode,
        "shell_unit": shell_unit,
        "shell_r_max": float(shell_r_max),

        "condition": condition,
    }


def compute_metrics_all(
    het_mask_path: str,
    nuc_mask_path: str,
    n_bins: int = 20,
    outer_width: float = 0.20,
    inner_width: float = 0.20,
    prev_dataframe: pd.DataFrame = None,
    save_df: bool = False,
    shell_mode: str = "normalized",
    pixel_size: float | None = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compute per-nucleus heterochromatin distribution metrics across the whole image.

    Parameters
    ----------
    het_global : np.ndarray (bool)
        Heterochromatin mask for the entire image (all nuclei).
    nuc_labels : np.ndarray (int or bool)
        Nucleus mask. If boolean/0-1, nuclei will be connected-component labeled to 1..N.
        If already integer-labeled, values should be 0=background, 1..N=nuclei.
    n_bins : int, optional (default: 20)
        Number of concentric shells for the radial profiles.
    outer_width : float, optional (default: 0.20)
        Peripheral shell thickness as fraction of radius (r ∈ [0, outer_width]).
    inner_width : float, optional (default: 0.20)
        Core shell thickness as fraction of radius (r ∈ [1 - inner_width, 1]).

    Radius coordinate
    -----------------
    - shell_mode="normalized": r ∈ [0,1], 0 at rim, 1 at center.
      * outer_width, inner_width are FRACTIONS of radius thickness.
    - shell_mode="distance": r is distance from rim in px (or µm if pixel_size is given).
      * outer_width, inner_width are DISTANCES (same unit as r).
      * Inner shell is the band of thickness `inner_width` closest to the center:
        r >= (r_max - inner_width).

    Returns
    -------
    df : pandas.DataFrame
        One row per nucleus with all keys returned by `hetero_distribution_metrics`, plus:
          - 'nucleus_id' : int (the label id)
          - 'area_px'    : int (pixel count of the nucleus)
        The DataFrame index is set to 'nucleus_id'.
    prof_stack : np.ndarray, shape (N, n_bins)
        Stacked radial profiles (one row per nucleus). Useful for cohort summaries.
    bin_centers : np.ndarray, shape (n_bins,)
        The bin-center radii used for the profiles.

    Notes
    -----
    - This function does not (re)segment; it just aggregates metrics across labeled nuclei.
    """
    het_global = imread(het_mask_path).astype(bool)
    nuc_labels = imread(nuc_mask_path).astype(bool)
    nuc_labels = cc_label(nuc_labels, connectivity=2)

    rows, profiles, bin_centers = [], [], None

    for prop in regionprops(nuc_labels):
        L = prop.label
        mask_L = (nuc_labels == L)

        m = hetero_distribution_metrics(
            het_mask=het_global,
            nuc_mask=mask_L,
            outer_width=outer_width,
            inner_width=inner_width,
            n_bins=n_bins,
            shell_mode=shell_mode,          
            pixel_size=pixel_size,            
        )

        m["nucleus_id"] = L
        m["area_px"] = int(mask_L.sum())
        m["path_to_image"] = het_mask_path

        rows.append(m)
        profiles.append(m["radial_profile"])
        bin_centers = m["radial_bin_centers"]

    # assemble df (same as before)
    df = pd.DataFrame(rows) if prev_dataframe is None or config.overwrite_df else pd.concat([prev_dataframe, pd.DataFrame(rows)], ignore_index=True)
    prof_stack = np.vstack(profiles) if profiles else np.empty((0, n_bins))

    if config.output_metrics_df_path:
        out_dir = os.path.dirname(config.output_metrics_df_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        df.to_csv(config.output_metrics_df_path, index=False)
        print(f"Saved metrics dataframe to {config.output_metrics_df_path}")

    return df, prof_stack, bin_centers



def main():
    _, prof_stack, bin_centers = compute_metrics_all(config.input_het_mask_path,  # heterochromatin mask path
                                                    config.input_mask_path,      # nuclei mask path
                                                    n_bins=20,
                                                    outer_width=0.20,
                                                    inner_width=0.20,
                                                    radius_mode=config.radius_mode,
                                                    pixel_size=config.pixel_size,
                                                
                                                      )
    
    if config.save_profiles:
        np.save(config.output_profiles_path, prof_stack)
    if config.save_bin_centers:
        np.save(config.output_bin_centers_path, bin_centers)


if __name__ == "__main__":
    main()
                                                                                                                  