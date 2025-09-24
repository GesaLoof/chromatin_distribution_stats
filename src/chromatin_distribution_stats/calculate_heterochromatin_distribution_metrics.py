import numpy as np
from scipy.ndimage import distance_transform_edt
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
from typing import Tuple
import numpy as np
import pandas as pd
from skimage.measure import label as cc_label, regionprops
import os
# TODO check imports

def _normalized_radius(nuc_mask: np.ndarray) -> np.ndarray:
    """
    Compute a per-pixel normalized radial coordinate inside ONE nucleus.

    Definition
    ----------
    - For each pixel inside the nucleus, compute the Euclidean distance to the nuclear envelope
      (i.e., to the nearest background pixel) via a distance transform.
    - Normalize by the maximum interior distance so values fall in [0, 1]:
        0.0 = nuclear envelope (periphery), 1.0 = deepest interior.

    Parameters
    ----------
    nuc_mask : np.ndarray
        Binary mask for a single nucleus (True/1 inside, False/0 outside).
        If multiple nuclei are present, run this per nucleus.

    Returns
    -------
    r : np.ndarray (float32)
        Same shape as `nuc_mask`. Values in [0, 1] inside the nucleus; 0 outside.
        (Outside pixels are set to 0 just as a placeholder—ignore them downstream.)
    """
    nuc = nuc_mask.astype(bool)
    dist_in = distance_transform_edt(nuc)

    r = np.zeros_like(dist_in, dtype=np.float32)
    if nuc.any():
        maxd = dist_in[nuc].max()
        r[nuc] = dist_in[nuc] / (maxd + 1e-8)
    return r


def hetero_distribution_metrics(
    het_mask: np.ndarray,
    nuc_mask: np.ndarray,
    outer_width: float = 0.20,
    inner_width: float = 0.20,
    n_bins: int = 20,
    eps: float = 1e-8,
) -> dict:
    """
    Compute robust, scale-invariant metrics describing how heterochromatin is distributed
    within a single nucleus (periphery vs interior), given a heterochromatin mask.

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

    Notes
    -----
    - The function first clips het_mask to the nucleus: `het = het_mask & nuc_mask`.
    - The OR and the stabilized RR use Haldane–Anscombe 0.5 pseudocounts (helps when a cell is 0).
    - The normalized radius r has 0 at the nuclear envelope and 1 at the center.
    """
    # Ensure we only analyze pixels inside this nucleus; everything else is ignored.
    het = (het_mask & nuc_mask).astype(bool)
    print(f"Nucleus mask shape: {nuc_mask.shape}")
    nuc = nuc_mask.astype(bool)
    eu = nuc & (~het)  # euchromatin = nuclear pixels that are not heterochromatin

    # Normalized radius map inside this nucleus.
    r = _normalized_radius(nuc)

    # Define periphery (outer) and core (inner) shells by thresholding r.
    outer = (r <= outer_width) & nuc
    inner = (r >= (1.0 - inner_width)) & nuc

    # Fractions of heterochromatin in each shell (naive proportions).
    p_het_outer = het[outer].mean() if np.any(outer) else np.nan
    p_het_inner = het[inner].mean() if np.any(inner) else np.nan

    # Naive ratio of proportions (a.k.a. enrichment ratio / RP).
    enrichment_ratio = p_het_outer / (p_het_inner + eps)

    # 2x2 table counts with 0.5 pseudocounts: (a b; c d)
    # a = hetero in outer, b = non-hetero in outer, c = hetero in inner, d = non-hetero in inner
    a = (het & outer).sum() + 0.5
    b = (eu  & outer).sum() + 0.5
    c = (het & inner).sum() + 0.5
    d = (eu  & inner).sum() + 0.5

    # --- Odds Ratio (outer vs inner, event=heterochromatin) ---
    OR = (a * d) / (b * c)
    se_logOR = np.sqrt(1/a + 1/b + 1/c + 1/d)
    logOR = np.log(OR)
    OR_CI95 = (np.exp(logOR - 1.96 * se_logOR), np.exp(logOR + 1.96 * se_logOR))

    # --- Risk Ratio / Ratio of Proportions (stabilized with pseudocounts) ---
    # Proportions with the same 0.5 pseudocounts:
    n_outer = a + b
    n_inner = c + d
    p_outer_stab = a / n_outer
    p_inner_stab = c / n_inner
    risk_ratio = p_outer_stab / (p_inner_stab + eps)

    # Log-normal CI for RR: Var[log(RR)] ≈ (1/a - 1/n_outer) + (1/c - 1/n_inner)
    se_logRR = np.sqrt(max(0.0, (1/a - 1/n_outer) + (1/c - 1/n_inner)))
    logRR = np.log(risk_ratio)
    RR_CI95 = (np.exp(logRR - 1.96 * se_logRR), np.exp(logRR + 1.96 * se_logRR))

    # Mean radii (lower mean for heterochromatin suggests peripheral localization).
    mean_r_het = r[het].mean() if het.any() else np.nan
    mean_r_eu  = r[eu].mean()  if eu.any()  else np.nan
    mean_r_all = r[nuc].mean() if nuc.any() else np.nan
    delta_mean_r_het_vs_eu  = (mean_r_het - mean_r_eu) if np.isfinite(mean_r_het) and np.isfinite(mean_r_eu) else np.nan
    delta_mean_r_het_vs_all = (mean_r_het - mean_r_all) if np.isfinite(mean_r_het) else np.nan

    # Distributional comparisons between r(het) and r(eu): KS statistic and Earth Mover's Distance.
    ks_D, ks_p = (np.nan, np.nan)
    emd = np.nan
    if het.any() and eu.any():
        ks_D, ks_p = ks_2samp(r[het], r[eu], alternative="two-sided", mode="auto")
        emd = wasserstein_distance(r[het], r[eu])

    # Radial profile by binning nuclear pixels into concentric shells.
    print(f"Nucleus size (number of pixels): {nuc.sum()}")
    print(f"Maximum distance inside nucleus: {r.max()}")
    print(f"Normalized radius values: {r[nuc]}")
    bins = np.linspace(0, 1, n_bins + 1)
    print(f"{bins=}]")
    shell_idx = np.digitize(r[nuc], bins, right = True) - 1
    print(f"{shell_idx=}]")
    het_in_nuc = het[nuc].astype(np.float32)
    print(f"{het_in_nuc=}]")
    prof = np.array([
        het_in_nuc[shell_idx == i].mean() if np.any(shell_idx == i) else np.nan
        for i in range(n_bins)
    ])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Replace NaNs (empty shells) with the global mean to allow AUC computation.
    if np.all(np.isnan(prof)):
        prof_valid = np.zeros_like(prof)  # Replace all NaNs with 0 if the entire array is NaN
        print("Warning: radial profile is all NaNs (empty nucleus?)")
    else:
        prof_valid = np.where(np.isnan(prof), np.nanmean(prof), prof)
    auc = np.trapz(prof_valid, bin_centers)

    # Simple normalized least-squares slope of profile vs radius.
    mask_ok = ~np.isnan(prof)
    if mask_ok.sum() >= 2:
        x = bin_centers[mask_ok]
        y = prof[mask_ok]
        x = (x - x.mean()) / (x.std() + 1e-8)  # z-score x
        slope = float(np.dot(x, y) / (np.dot(x, x) + 1e-8))
    else:
        slope = np.nan

    return {
        "p_het_outer": float(p_het_outer),
        "p_het_inner": float(p_het_inner),
        "enrichment_ratio": float(enrichment_ratio),     # naive RP (no pseudocounts)
        "risk_ratio": float(risk_ratio),                 # stabilized RP (with pseudocounts)
        "risk_ratio_CI95_low": float(RR_CI95[0]),
        "risk_ratio_CI95_high": float(RR_CI95[1]),
        "odds_ratio": float(OR),
        "odds_ratio_CI95_low": float(OR_CI95[0]),
        "odds_ratio_CI95_high": float(OR_CI95[1]),
        "mean_r_het": float(mean_r_het) if np.isfinite(mean_r_het) else np.nan,
        "mean_r_eu": float(mean_r_eu) if np.isfinite(mean_r_eu) else np.nan,
        "mean_r_all": float(mean_r_all),
        "delta_mean_r_het_vs_eu": float(delta_mean_r_het_vs_eu) if np.isfinite(delta_mean_r_het_vs_eu) else np.nan,
        "delta_mean_r_het_vs_all": float(delta_mean_r_het_vs_all) if np.isfinite(delta_mean_r_het_vs_all) else np.nan,
        "ks_D": float(ks_D) if np.isfinite(ks_D) else np.nan,
        "ks_p": float(ks_p) if np.isfinite(ks_p) else np.nan,
        "emd": float(emd) if np.isfinite(emd) else np.nan,
        "radial_profile": prof,                # length n_bins
        "radial_bin_centers": bin_centers,     # length n_bins
        "radial_auc": float(auc),
        "radial_slope": float(slope),
    }


def compute_metrics_all(
    het_mask_path: str,
    nuc_mask_path: str,
    # TODO determine whether we want to bin nuclei or create equally sized groups?
    n_bins: int = 20,
    outer_width: float = 0.20,
    inner_width: float = 0.20,
    prev_dataframe: pd.DataFrame = None,
    save_df: bool = False,
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
    # load images
    het_global =imread(het_mask_path).astype(bool)
    nuc_labels = imread(nuc_mask_path).astype(bool)

    # Ensure integer-labeled nuclei: 0=background, 1..N=nuclei
    nuc_labels = cc_label(nuc_labels, connectivity=2)

    rows = []
    profiles = []
    bin_centers = None

    # Iterate over each labeled nucleus
    for prop in regionprops(nuc_labels):
        L = prop.label
        mask_L = (nuc_labels == L)

        # Compute metrics for this nucleus (the function will clip het to mask_L)
        m = hetero_distribution_metrics(
            het_mask=het_global,
            nuc_mask=mask_L,
            outer_width=outer_width,
            inner_width=inner_width,
            n_bins=n_bins,
        )

        # Attach nucleus id and size and source image path
        m["nucleus_id"] = L
        m["area_px"] = int(mask_L.sum())
        m["path_to_image"] = het_mask_path

        rows.append(m)
        profiles.append(m["radial_profile"])
        bin_centers = m["radial_bin_centers"]  # same for all nuclei

    # Assemble outputs
    if prev_dataframe is None: # create new df
        df = pd.DataFrame(rows)
        prof_stack = np.vstack(profiles) if profiles else np.empty((0, n_bins))
    elif prev_dataframe is not None and not config.overwrite_df: # append to previous df
        df = pd.DataFrame(rows)
        #append new df to previous df
        df = pd.concat([prev_dataframe, df], ignore_index=True)
        prof_stack = np.vstack(profiles) if profiles else np.empty((0, n_bins))
    else: # overwrite previous df
        df = pd.DataFrame(rows)
        prof_stack = np.vstack(profiles) if profiles else np.empty((0, n_bins))
    
    # add information about image paths
    
    if config.output_metrics_df_path:
        out_dir = os.path.dirname(config.output_metrics_df_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        df.to_csv(config.output_metrics_df_path, index=False)
        print(f"Saved metrics dataframe to {config.output_metrics_df_path}")

    return df, prof_stack, bin_centers


def main():
    _, prof_stack, bin_centers = compute_metrics_all(config.input_het_mask_path, # heterochromatin mask
                                                      config.input_mask_path, # nuclei mask
                                                      20, # n_bins: int
                                                      0.20, # outer_width: float
                                                      0.20 # inner_width: float
                                                      )
    
    if config.save_profiles:
        np.save(config.output_profiles_path, prof_stack)
    if config.save_bin_centers:
        np.save(config.output_bin_centers_path, bin_centers)


if __name__ == "__main__":
    main()
                                                                                                                  