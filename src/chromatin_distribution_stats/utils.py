from pathlib import Path
from typing import List, Tuple, Dict


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
