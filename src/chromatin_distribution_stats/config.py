#### k-means parameters ####
K = 2
include_distance = False
smooth_sigma = 1.5
var_size = 7
sample_frac = 0.1
random_state = 0
clean_iters = 2  # number of opening+closing passes for speckle cleanup  (0 to disable)
return_semantic = False  # If True returns a semantic mask 0=background, 1=heterochromatin, 2=non-heterochromatin
k_input_im_path = ""
k_input_nuc_mask_path = ""
output_dir_het_mask_path = "" #only folder, the rest will be handled automatically
# optional paths to save outputs (just set to None to skip saving or add path to save)
output_labels_path = None #"data/processed/cluster_labels.tif"
output_model_path = None #"data/processed/kmeans_model.pkl"


#### batch process k means parameters ####
k_batch_input_image_dir = ""
k_batch_input_nuc_mask_dir = ""
k_batch_output_dir_het_mask_path = "" #only folder, the rest will be handled automatically
k_batch_output_labels_path = None #"data/processed/cluster_labels.tif" optional paths to save outputs (just set to None to skip saving or add path to save)
k_batch_output_model_path = None #"data/processed/kmeans_model.pkl" optional paths to save outputs (just set to None to skip saving or add path to save)


#### calculate distribution metrics parameters ####
input_mask_path = "/"
input_het_mask_path = ""
# optional previous dataframe to append to
# prev_dataframe = pd.read_csv("data/processed/het_metrics.csv")
# save_df = True  # whether to save the dataframe to output_metrics_paths
n_bins = 20         # bins for radial profiles (0-1 normalized radius)
outer_width = 0.2  # fraction of radius for outer shell (for radial profile) if mode is normalized or distance in pixels/ microns if radius_mode="distance"
inner_width = 0.2  # fraction of radius for inner shell (for radial profile) if mode is normalized or distance in pixels/ microns if radius_mode="distance"
shell_mode = "normalized"  # "normalized" or "distance"
pixel_size = None  # e.g. 0.065 for 65nm pixels; set to None to use pixels as units
output_metrics_df_path = "data/processed/het_metrics.csv" # set to None if you don't want to save
save_profiles = False
save_bin_centers = False
output_profiles_path = "data/processed/radial_profiles.npy"
output_bin_centers_path = "data/processed/radial_profile_bin_centers.npy"
condition = "cond_c"  # condition name to add to dataframe (e.g. "control", "treated", etc.)

#### batch process calculate distribution metrics parameters ####
batch_input_nuc_mask_dir = ""
batch_input_het_mask_dir = ""

#### plot example nucleus image and masks ####
plot_nuc_im_path =  ""
plot_nuc_mask_path = ""
plot_nuc_het_mask_path = ""
plot_labels_path = None # path to k-means labels if you want to plot them as well
plot_show_all = True  # whether to show all nuclei or just one
plot_raw_het_path = "data/figures/example_nucleus_het_mask.png"

#### plot distribution stats parameters ####
path_to_stats_dataframe = "data/processed/het_metrics.csv"
overwrite_df = False  # whether to create a new dataframe even if it already exists (set to False when batch processing, otherwise it will overwrite previous images' data)
#path_to_radial_profiles = "data/processed/radial_profiles.npy"
#path_to_radial_profile_bin_centers = "data/processed/radial_profile_bin_centers.npy"
cond_col="condition"           # default column name
group_order=["cond_a","cond_b", "cond_c"]  # optional explicit order
group_colors = ["#e923c4",  "#06aa55", "#5362eb"]  # optional explicit colors for conditions
plot_metrics_path = "data/figures/het_metrics.png" # set to None if you don't want to plot this metric
plot_profiles_stacked_path = "data/figures/radial_profiles_stacked.png" # set to None if you don't want to plot this metric
plot_rp_summary_path = "data/figures/radial_profile_summary.png" # set to None if you don't want to plot this metric
plot_rp_histograms_path = "data/figures/radial_histograms.png" # set to None if you don't want to plot this metric
plot_histograms_stacked_path = "data/figures/radial_histograms_stacked.png" # set to None if you don't want to plot this metric
plot_profiles_path = "data/figures/radial_profiles.png" # set to None if you don't want to plot this metric
plot_rp_forest_path = "data/figures/radial_profile_forest.png" # set to None if you don't want to plot this metric