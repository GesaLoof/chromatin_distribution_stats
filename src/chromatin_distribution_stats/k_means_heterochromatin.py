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
import os
from pathlib import Path
# TODO check imports

def _local_variance(x, size=7):
    """Fast local variance via box filters (works for float)."""
    m = uniform_filter(x, size=size)
    m2 = uniform_filter(x*x, size=size)
    v = m2 - m*m
    return np.clip(v, 0, None)

def _kmeans_heterochromatin(
    img, nuc_mask,
    K=2,
    include_distance=False,
    smooth_sigma=1.5,
    var_size=7,
    sample_frac=0.1,
    random_state=0,
    clean_iters=2
):
    """
    Segment heterochromatin inside a single nucleus using k-means on simple features.

    Parameters
    ----------
    img : 2D array (float or uint)
        EM image.
    nuc_mask : 2D bool array
        True for pixels inside the nucleus of interest.
    K : int
        Number of clusters (2 or 3 are typical).
    include_distance : bool
        If True, adds normalized distance-to-envelope as an extra feature (soft prior).
    smooth_sigma : float
        Sigma for Gaussian smoothing feature.
    var_size : int
        Window size for local variance feature (odd number, ~5–11).
    sample_frac : float in (0,1]
        Fraction of in-mask pixels used to fit k-means (speed-up). Always predicts on all.
    random_state : int
        Reproducibility.
    clean_iters : int
        Number of opening+closing passes for speckle cleanup (0 to disable).

    Returns
    -------
    het_mask : 2D bool array
        Predicted heterochromatin mask inside nuc_mask.
    labels_full : 2D int array
        Cluster id per pixel (outside nucleus = -1).
    model : fitted KMeans object
    """

    img = img.astype(np.float32, copy=False)
    inside = nuc_mask.astype(bool)

    # Per-nucleus z-normalization (critical)
    mu = img[inside].mean()
    sd = img[inside].std() + 1e-8
    z = (img - mu) / sd

    # Features: intensity, smoothed intensity, local variance
    g = gaussian_filter(z, smooth_sigma)
    var = _local_variance(z, size=var_size)

    feats = [z, g, var]
    if include_distance:
        dist = distance_transform_edt(inside)
        dist = dist / (dist.max() + 1e-8)
        feats.append(dist)

    F = np.stack(feats, axis=-1)
    F_in = F[inside]            # only cluster inside the nucleus

    # Optional subsample for fitting to speed up on big nuclei
    n = F_in.shape[0]
    if sample_frac < 1.0:
        rs = np.random.RandomState(random_state)
        m = max(20000, int(sample_frac * n))   # keep enough pixels for stable fit
        idx = rs.choice(n, size=min(m, n), replace=False)
        fit_data = F_in[idx]
    else:
        fit_data = F_in

    # Fit and predict
    km = KMeans(n_clusters=K, n_init=10, random_state=random_state)
    km.fit(fit_data)
    labels_in = km.predict(F_in)

    # Decide which cluster is heterochromatin.
    # Use the cluster with the lowest mean *raw-intensity feature* (index 0 in feats).
    centers = km.cluster_centers_
    order_by_dark = np.argsort(centers[:, 0])  # 0th feature is z-scored raw intensity

    if K == 3:
        # Simple heuristic: if the darkest cluster is tiny (<3%), treat it as nucleoli
        # and pick the 2nd darkest as heterochromatin.
        counts = np.bincount(labels_in, minlength=K).astype(np.float32)
        frac = counts / counts.sum()
        darkest = order_by_dark[0]
        if frac[darkest] < 0.03:
            het_label = order_by_dark[1]
        else:
            het_label = darkest
    else:
        het_label = order_by_dark[0]

    # Build masks back to image shape
    het_mask = np.zeros_like(inside, dtype=bool)
    het_mask[inside] = (labels_in == het_label)

    # Light cleanup (remove specks / fill pinholes)
    if clean_iters > 0:
        structure = generate_binary_structure(2, 1)
        for _ in range(clean_iters):
            het_mask = binary_opening(het_mask, structure)
            het_mask = binary_closing(het_mask, structure)

    # Full label map (outside nucleus = -1)
    labels_full = np.full_like(img, fill_value=-1, dtype=int)
    labels_full[inside] = labels_in

    return het_mask, labels_full, km

#process all nuclei in one image at once but independently

def kmeans_heterochromatin_all(
    img,
    nuclei_mask,                 # bool (many nuclei) or int-labeled mask
    K=2,
    include_distance=False,      # False mean distance to lamina will not be added as a feature (True means heterochromtin is biased to the lamina)
    smooth_sigma=1.5,
    var_size=7,
    sample_frac=0.1,
    random_state=0,
    clean_iters=2,
    unique_labels=False,         # if True, make global labels unique per nucleus (e.g., nuc_id*K + local_label)
    return_semantic=False        # if True, also return a simple semantic map (0=bg, 1=hetero, 2=non-hetero)
):
    """
    Apply k-means heterochromatin segmentation nucleus-by-nucleus and stitch results.

    Returns
    -------
    het_global : 2D bool
        Union of per-nucleus heterochromatin masks.
    labels_global : 2D int
        Cluster id per pixel; -1 outside nuclei.
        - If unique_labels=False: ids are 0..K-1 inside each nucleus (reused across nuclei).
        - If unique_labels=True: ids are unique across image (nuc_id*K + local_label).
    models : list of dict
        One entry per nucleus with {'nucleus_id', 'kmeans', 'centers', 'counts'}.
    semantic_global : 2D uint8 (optional)
        If return_semantic=True: 0=background, 1=heterochromatin, 2=non-heterochromatin.
    """
    # Normalize nucleus labeling
    #TODO: also handle binary masks
    if nuclei_mask.dtype == bool:
        nuc_labels = cc_label(nuclei_mask, connectivity=2)
    else:
        nuc_labels = nuclei_mask.astype(int, copy=False)

    H, W = nuc_labels.shape
    print(f"Found {nuc_labels.max()} nuclei in image of size {H}x{W}")
    het_global    = np.zeros((H, W), dtype=bool)
    labels_global = np.full((H, W), fill_value=-1, dtype=int)
    models = []

    # Iterate over each connected nucleus
    for prop in regionprops(nuc_labels):
        L = prop.label
        mask_L = (nuc_labels == L)

        # Run existing per-nucleus function
        het_L, labels_L, km = _kmeans_heterochromatin(
            img, mask_L,
            K=K,
            include_distance=include_distance,
            smooth_sigma=smooth_sigma,
            var_size=var_size,
            sample_frac=sample_frac,
            random_state=random_state,
            clean_iters=clean_iters
        )

        # Safety: ensure het_L doesn't spill outside due to closing
        het_L &= mask_L

        # Stitch heterochromatin
        het_global[mask_L] = het_L[mask_L]

        # Stitch labels:
        if unique_labels:
            # Reindex local labels 0..K-1 into unique ids per nucleus
            # (Keep -1 outside; already set in labels_L)
            for local_id in range(K):
                sel = mask_L & (labels_L == local_id)
                labels_global[sel] = L * K + local_id
        else:
            labels_global[mask_L] = labels_L[mask_L]

        # Collect per-nucleus info (useful for QC)
        labels_in = labels_L[mask_L]
        counts = np.bincount(labels_in[labels_in >= 0], minlength=K)
        models.append({
            'nucleus_id': L,
            'kmeans': km,
            'centers': km.cluster_centers_.copy(),
            'counts': counts
        })

    if return_semantic:
        # 0=background, 1=heterochromatin, 2=non-heterochromatin
        sem = np.zeros((H, W), dtype=np.uint8)
        sem[het_global] = 1
        sem[(nuc_labels > 0) & (~het_global)] = 2
        return het_global, labels_global, models, sem

    return het_global, labels_global, models


def main():
    # Example usage
    img = imread(config.k_input_im_path)
    nuc_mask = imread(config.k_input_nuc_mask_path).astype(bool)

    het_mask, labels_full, model = kmeans_heterochromatin_all(
        img, nuc_mask,
        K=config.K,
        include_distance=config.include_distance,
        smooth_sigma=config.smooth_sigma,
        var_size=config.var_size,
        sample_frac=config.sample_frac,
        random_state=config.random_state,
        clean_iters=config.clean_iters,
        return_semantic=config.return_semantic
    )

    #save results
    #check if output folder exists, otherwise create it
    if not os.path.exists(config.output_dir_het_mask_path):
        os.makedirs(config.output_dir_het_mask_path)
    #derive name from input image name
    im_name_stem = Path(config.k_input_im_path).stem
    print(im_name_stem)
    imwrite(os.path.join(config.output_dir_het_mask_path, f"{im_name_stem}_het_mask.tif"), het_mask.astype(np.uint8))
    if config.output_labels_path:
        imwrite(config.output_labels_path, labels_full.astype(np.int16))
    if config.output_model_path:
        import pickle
        with open(config.output_model_path, 'wb') as f:
            pickle.dump(model, f)


    return het_mask, labels_full, model

if __name__ == "__main__":
    main()