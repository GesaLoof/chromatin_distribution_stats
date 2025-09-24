
"""
Batch heterochromatin segmentation with k-means.

Given:
  - a folder of EM images (2D grayscale),
  - a folder of nucleus masks (same stems; may have a suffix like '_mask'),

This script:
  - runs per-nucleus k-means on simple features (intensity, smoothed intensity,
    local variance, and optional distance-to-envelope),
  - decides which cluster is heterochromatin (darkest centroid; with a tiny-cluster guard for K=3),
  - saves a binary heterochromatin mask per image into an output folder.

Outputs:
  <out_dir>/<image_stem>_het_mask.tif      (uint8; 0=background, 1=heterochromatin)
  <out_dir>/<image_stem>_labels.tif   (int32; -1=outside nuclei, 0..K-1 cluster ids)  [optional via --save-labels]

Notes:
  - Images are processed independently; within each nucleus, intensities are z-scored
    (mean/std computed from that nucleus only) to normalize contrast across cells/images.
"""


import numpy as np
from k_means_heterochromatin import kmeans_heterochromatin_all
from chromatin_distribution_stats import config
import os
from tifffile import imread, imwrite
from utils import _pair_images_and_masks


def main():
    #get list of image/mask pairs
    image_mask_pair_list = _pair_images_and_masks(config.k_batch_input_image_dir, config.k_batch_input_nuc_mask_dir)
    #check if output folder exists, otherwise create it
    if not os.path.exists(config.output_dir_het_mask_path):
      os.makedirs(config.output_dir_het_mask_path)
    # loop through images
    for pair in image_mask_pair_list:
        im = imread(pair[0])
        mask = imread(pair[1]).astype(bool)
        het_mask, labels_full, model = kmeans_heterochromatin_all(
            im, 
            mask,
            K=config.K,
            include_distance=config.include_distance,
            smooth_sigma=config.smooth_sigma,
            var_size=config.var_size,
            sample_frac=config.sample_frac,
            random_state=config.random_state,
            clean_iters=config.clean_iters,
            return_semantic=config.return_semantic)
        
        #save results
        #derive name from input image name
        im_name_stem = pair[0].stem
        imwrite(os.path.join(config.output_dir_het_mask_path, f"{im_name_stem}_het_mask.tif"), het_mask.astype(np.uint8))
        if config.output_labels_path:
            imwrite(config.output_labels_path, labels_full.astype(np.int16))
        if config.output_model_path:
            import pickle
            with open(config.output_model_path, 'wb') as f:
                pickle.dump(model, f)



if __name__ == "__main__":
    main()