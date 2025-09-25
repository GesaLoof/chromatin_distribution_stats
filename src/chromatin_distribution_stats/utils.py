from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from skimage.measure import label as cc_label
import ast
import re
from scipy.ndimage import distance_transform_edt
#TODO add edt to imports


def _distance_from_rim(
    nuc_mask: np.ndarray,
    pixel_size: float | None = None,
) -> tuple[np.ndarray, float, str]:
    """
    Distance-to-envelope inside ONE nucleus.

    Parameters
    ----------
    nuc_mask : np.ndarray
        Single-nucleus mask (True/1 inside).
    pixel_size : float or None
        Microns per pixel. If None, output is in pixels.

    Returns
    -------
    r_dist : np.ndarray (float32)
        Distance from rim for each pixel (0 at rim). Zero outside the nucleus.
    rmax_dist : float
        Max distance inside the nucleus (same unit as r_dist).
    unit : str
        'um' if pixel_size is provided, else 'px'.
    """
    nuc = nuc_mask.astype(bool)
    dist_px = distance_transform_edt(nuc).astype(np.float32)
    if pixel_size is None:
        r_dist = dist_px
        unit = "px"
    else:
        r_dist = dist_px * float(pixel_size)
        unit = "um"
    rmax_dist = float(r_dist[nuc].max()) if nuc.any() else 0.0
    # keep zeros outside nucleus for convenience
    return r_dist, rmax_dist, unit


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


def _index_by_stem(paths: List[Path], drop_suffix: str = "") -> Dict[str, Path]:
    """Key file paths by (optionally trimmed) filename stem."""
    idx = {}
    for p in paths:
        stem = p.stem
        if drop_suffix and stem.endswith(drop_suffix):
            stem = stem[: -len(drop_suffix)]
        idx[stem] = p
    return idx


def _pair_images_and_masks(
    im_dir: Path,
    mask_dir: Path,
    image_glob: str = "*.tif",
    mask_glob: str = "*.tif",
    #TODO make suffix configurable
    primary_drop_suffix = None, # add suffix here if first input has to be trimmed as well
    pair_drop_suffix: str = "_mask",
    ) -> List[Tuple[Path, Path]]:
    
    """Pair images in im_dir with masks in mask_dir by matching (trimmed) stems."""
    
    primary_files = sorted(Path(im_dir).glob(image_glob))
    secondary_files = sorted(Path(mask_dir).glob(mask_glob))
    if primary_drop_suffix:
        one = _index_by_stem(primary_files, drop_suffix=primary_drop_suffix)
    else:
        one = _index_by_stem(primary_files)
    two = _index_by_stem(secondary_files, drop_suffix=pair_drop_suffix)
    common = sorted(set(one.keys()) & set(two.keys()))
    print(f"Found {len(common)} matching image/mask pairs in {im_dir} and {mask_dir}")
    print(f"Examples: {common[:5]}")
    if len(common) == 0:
        raise ValueError(f"No matching image/mask stems in {im_dir} and {mask_dir}")
    return [(one[k], two[k]) for k in common]


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


def _parse_array_literal(s):
    """Parse a Python/JSON-like list string to np.ndarray; robust fallback to regex."""
    if isinstance(s, (list, np.ndarray)):  # already ok
        return np.asarray(s, dtype=float)
    if pd.isna(s):
        return np.array([], dtype=float)
    # Try Python literal (e.g., "[0.1, 0.2, 0.3]")
    try:
        v = ast.literal_eval(s)
        return np.asarray(v, dtype=float)
    except Exception:
        pass
    # Fallback: extract all numbers from a string like "[0.1 0.2 0.3]"
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    return np.asarray([float(x) for x in nums], dtype=float)

def coerce_profiles_from_csv(df: pd.DataFrame) -> pd.DataFrame:
    if "radial_profile" in df.columns:
        df["radial_profile"] = df["radial_profile"].apply(_parse_array_literal)
    if "radial_bin_centers" in df.columns:
        df["radial_bin_centers"] = df["radial_bin_centers"].apply(_parse_array_literal)
    return df