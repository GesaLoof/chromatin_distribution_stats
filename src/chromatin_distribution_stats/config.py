# k-means parameters
K = 2
include_distance = False
smooth_sigma = 1.5
var_size = 7
sample_frac = 0.1
random_state = 0
clean_iters = 2  # number of opening+closing passes for speckle cleanup  (0 to disable)
return_semantic = False  # If True returns a semantic mask 0=background, 1=heterochromatin, 2=non-heterochromatin
# input_im_path = "data/raw/em_image.tif"
# input_mask_path = "data/processed/nucleus_mask.tif"
# output_het_mask_path = "data/processed/het_mask.tif"
# output_labels_path = "data/processed/cluster_labels.tif"
# output_model_path = "data/processed/kmeans_model.pkl"


#batch process k means parameters
input_image_dir = "data/raw/"
output_mask_dir = "data/processed/"



# calculate distribution metrics parameters
input_mask_dir = "data/processed/"
input_het_mask_dir = "data/processed/"
# optional previous dataframe to append to
# prev_dataframe = pd.read_csv("data/processed/het_metrics.csv")
# save_df = True  # whether to save the dataframe to output_metrics_path
# output paths
n_bins = 20         # bins for radial profiles (0-1 normalized radius)
outer_width = 0.20  # fraction of radius for outer shell (for radial profile)
inner_width = 0.20  # fraction of radius for inner shell (for radial profile)
output_metrics_path = "data/processed/het_metrics.csv"
save_profiles = False
save_bin_centers = False
output_profiles_path = "data/processed/radial_profiles.npy"
output_bin_centers_path = "data/processed/radial_profile_bin_centers.npy"


# plot distribution stats parameters
path_to_stats_dataframe = "data/processed/het_metrics.csv"
#path_to_radial_profiles = "data/processed/radial_profiles.npy"
#path_to_radial_profile_bin_centers = "data/processed/radial_profile_bin_centers.npy"
cond_col="condition",            # default column name
group_order=["cond_a","cond_b", "cond_c"],  # optional explicit order
metric_col="odds_ratio",         # or "risk_ratio" if you prefer RP
title="Chromatin distribution — Control vs Treated"
plot_metrics_path = "data/figures/het_metrics.png"
plot_profiles_path = "data/figures/radial_profiles.png" # png or pdf