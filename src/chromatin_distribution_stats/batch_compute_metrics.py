# input pairs of images
# load and append to existing dataframe if provided
# compute metrics
# save dataframe if desired
# save radial profiles if desired
from utils import _pair_images_and_masks
from chromatin_distribution_stats import config
import os
import pandas as pd
import numpy as np
from tifffile import imread, imwrite
from calculate_heterochromatin_distribution_metrics import compute_metrics_all

def main():
    # get list of image/mask pairs
    image_mask_pair_list = _pair_images_and_masks(config.k_batch_input_nuc_mask_dir, 
                                                  config.batch_input_het_mask_dir, 
                                                  primary_drop_suffix="_mask", 
                                                  pair_drop_suffix="_het_mask")
    #load df
    if os.path.exists(config.output_metrics_df_path):
        df = pd.read_csv(config.output_metrics_df_path)
    else:
        print("No previous dataframe found, starting new one")
        df = None
    # loop through nuc/ het mask pairs
    for pair in image_mask_pair_list:
        print(f"Processing pair: {pair[0]} and {pair[1]}")
        df, _, _ = compute_metrics_all(pair[1],
                                        pair[0], 
                                        prev_dataframe = df,
                                        save_df = True)

if __name__ == "__main__":
    main()