import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap

INTERPOLATION_MODE = "none"


# Default colors:
# Peak: (255, 32, 32)
# Homogeneous, in peak: (255, 32, 128)
# Homogeneous, out of peak: (128, 32, 255)
# Mask: (128, 128, 255)
# High snippet: (32, 192, 32)
# Low snippet: (32, 192, 32)
peak_color = np.array((255, 32, 32)) / 255.
hom_peak_color = np.array((255, 32, 128)) / 255.
hom_nonpeak_color = np.array((128, 32, 255)) / 255.
mask_color = np.array((128, 128, 255)) / 255.
low_snip_color = np.array((32, 192, 192)) / 255.
#low_snip_color = np.array((32, 128, 192)) / 255.
low_snip_lambda_color = np.array((32, 128, 192)) / 255.
#low_snip_lambda_color = np.array((32, 192, 192)) / 255.
high_snip_color = np.array((32, 192, 32)) / 255.
high_snip_lambda_color = np.array((128, 192, 128)) / 255.

class NormVisualizer:


    def __init__(self, plot_size=(1792, 1033), color_list=None, alpha_list=None, is_adjust_grayscale_to_view=True):
        self.plot_size = plot_size
        self.color_list = color_list if not color_list is None else [peak_color, hom_peak_color, hom_nonpeak_color,
                                                                     mask_color, low_snip_color, high_snip_color,
                                                                     #high_snip_lambda_color]
                                                                     low_snip_lambda_color]
        self.alpha_list = alpha_list if not alpha_list is None else [0.8, 0.5, 0.3,
                                                                     0.2, 1.0, 1.0,
                                                                     1.0]
        self.is_adjust_grayscale_to_view = is_adjust_grayscale_to_view

    def plot(self, image_arr, mask_arr, kde_grid, kde_bw,
             high_grad_cutoff_percentage,
             n_snippets_min, n_snippets_max, vol_snippets_min, lam,
             peak_prom_fraction, peak_selection_mode,
             plot_parameter_dict, voxel_spacing_arr, save_path, image_name=None, mask_name=None):
        n_dim = image_arr.ndim
        if voxel_spacing_arr is None:
            is_use_mm = False
            voxel_spacing_arr = np.ones(n_dim)
        else:
            is_use_mm = True

        fig, axs = plt.subplots(2, max(3, n_dim))

        title = "Normalization summary"
        if image_name is not None:
            title += f" for image {image_name}"
        if mask_name is not None:
            title += f" for mask {mask_name}"
        plt.suptitle(title)

        threshold_arr = plot_parameter_dict["threshold_arr"]
        jsd_low_arr = plot_parameter_dict["jsd_low_arr"]
        jsd_high_arr = plot_parameter_dict["jsd_high_arr"]
        n_snippets = plot_parameter_dict["n_snippets"]
        nonorm_low_snippet_pdf = plot_parameter_dict["nonorm_low_snippet_pdf"]
        nonorm_high_snippet_pdf = plot_parameter_dict["nonorm_high_snippet_pdf"]
        nonorm_chosen_pdf = plot_parameter_dict["nonorm_chosen_pdf"]
        nonorm_all_pdf = plot_parameter_dict["nonorm_all_pdf"]
        chosen_threshold = plot_parameter_dict["chosen_threshold"]
        homogeneous_mask = plot_parameter_dict["homogeneous_mask"]
        pdf_nonorm = plot_parameter_dict["pdf_nonorm"]
        chosen_peak_index = plot_parameter_dict["chosen_peak_index"]
        used_peak_int_arr = plot_parameter_dict["used_peak_int_arr"]
        used_peaks_pdf_nonorm_list = plot_parameter_dict["used_peaks_pdf_nonorm_list"]
        used_peaks_mask_list = plot_parameter_dict["used_peaks_mask_list"]

        # Plot snippet JSD curves in bottom left
        ax_jsd = axs[1,0]

        ax_jsd.set_title(f"JSD of snippet to highest and lowest grad snippets; n = {n_snippets}; lambda {lam} \nSettings: high_grad_cutoff = {high_grad_cutoff_percentage}%, n_min = {n_snippets_min},\nn_max = {n_snippets_max}, vol_min = {vol_snippets_min}")
        ax_jsd.plot(threshold_arr, jsd_low_arr, color=self.color_list[-3], marker='o', linestyle="-", label='distance to low grad snippet')
        ax_jsd.plot(threshold_arr, jsd_high_arr, color=self.color_list[-2], marker='o', linestyle="-", label='distance to high grad snippet')
        if lam != 1:
            ax_jsd.plot(threshold_arr, jsd_low_arr*lam, color=self.color_list[-1], marker='o', linestyle="-", label='distance to low grad snippet times lambda')
        ax_jsd.set_xlabel("Gradient threshold")
        ax_jsd.set_ylabel("JSD")
        ax_jsd.legend()

        # Plot snippet histograms in bottom middle
        ax_snp = axs[1, 1]

        ax_snp.set_title(f"Histograms of highest and lowest grad snippets,\nand homogeneous region selected at threshold {chosen_threshold:.2f}")
        ax_snp.plot(kde_grid, nonorm_low_snippet_pdf, color=self.color_list[-3], linestyle="-", label="low grad snippet")
        ax_snp.plot(kde_grid, nonorm_high_snippet_pdf, color=self.color_list[-2], linestyle="-", label="high grad snippet")
        ax_snp.plot(kde_grid, nonorm_chosen_pdf, color=self.color_list[2], label="chosen homogeneous region")
        ax_snp.plot(kde_grid, nonorm_all_pdf, color=self.color_list[3], label="entire selected mask")
        ax_snp.set_xlabel("Intensity")
        ax_snp.set_ylabel("Probability Density")
        ax_snp.legend()

        # Plot peak detection in bottom right
        ax_pks = axs[1, 2]

        n_bandwiths_threshold = 4

        ax_pks.set_title(f"Histograms of detected peaks: Chosen peak at intensity: {used_peak_int_arr[chosen_peak_index]:.2f} \n"
                         f"Settings: peak_prom_fraction = {peak_prom_fraction}, peak_selection_mode = {peak_selection_mode}")
        ax_pks.plot(kde_grid, nonorm_all_pdf, color=self.color_list[3], label="entire selected mask")
        ax_pks.plot(kde_grid, nonorm_chosen_pdf, color="lightgray", label="chosen homogeneous region")
        for i, used_peaks_pdf_nonorm in enumerate(used_peaks_pdf_nonorm_list):
            if i == chosen_peak_index:
                continue
            ax_pks.plot(kde_grid, used_peaks_pdf_nonorm, color=self.color_list[2], label=f"peak {i+1}")

        peak_left_bound = used_peak_int_arr[chosen_peak_index]-n_bandwiths_threshold*kde_bw
        peak_right_bound = used_peak_int_arr[chosen_peak_index]+n_bandwiths_threshold*kde_bw
        on_peak_grid_indices = (kde_grid >= peak_left_bound) & (kde_grid <= peak_right_bound)
        on_peak_chosen_pdf_nonorm = used_peaks_pdf_nonorm_list[chosen_peak_index].copy()
        on_peak_chosen_pdf_nonorm[~on_peak_grid_indices] = np.nan
        off_peak_chosen_pdf_nonorm = used_peaks_pdf_nonorm_list[chosen_peak_index].copy()
        off_peak_chosen_pdf_nonorm[on_peak_grid_indices] = np.nan
        ax_pks.plot(kde_grid, off_peak_chosen_pdf_nonorm, color=self.color_list[1], label=f"region outside peak")
        ax_pks.fill_between(kde_grid, off_peak_chosen_pdf_nonorm, color=self.color_list[1], alpha=0.5)
        ax_pks.plot(kde_grid, on_peak_chosen_pdf_nonorm, color=self.color_list[0], label=f"region inside peak {i+1}")
        ax_pks.fill_between(kde_grid, on_peak_chosen_pdf_nonorm, color=self.color_list[0], alpha=0.5)

        secondary_used_peak_int_arr = np.array([el for i, el in enumerate(used_peak_int_arr) if not i == chosen_peak_index])
        if secondary_used_peak_int_arr.size > 0:
            ax_pks.vlines((secondary_used_peak_int_arr), np.amin(nonorm_all_pdf),
                          np.amax(nonorm_all_pdf), colors='gray', linestyles=":", label="found secondary intensitiy peaks")
        ax_pks.vlines((used_peak_int_arr[chosen_peak_index]), np.amin(nonorm_all_pdf),
                      np.amax(nonorm_all_pdf), colors='k', linestyles="--", label="final selected peak")
        ax_pks.set_xlabel("Intensity")
        ax_pks.set_ylabel("Probability Density")
        ax_pks.legend()

        mask_arr_list = [
            used_peaks_mask_list[chosen_peak_index] & (image_arr > peak_left_bound) & (image_arr < peak_right_bound),
            used_peaks_mask_list[chosen_peak_index] & ((image_arr < peak_left_bound) | (image_arr > peak_right_bound)),
            homogeneous_mask,
            mask_arr
        ]

        highlight_arr_list = []
        for mask_arr in mask_arr_list:
            # Define the highlight arr from the mask
            highlight_arr = np.ones_like(image_arr)
            highlight_arr[~mask_arr] = np.nan

            # Exclude voxels already included in other highlights from new highlights
            for previous_highlight_arr in highlight_arr_list:
                highlight_arr[~np.isnan(previous_highlight_arr)] = np.nan

            highlight_arr_list.append(highlight_arr)

        mask_colormaps_list = [
            ListedColormap([np.array((255, 255, 255)) / 255., self.color_list[0]]),
            ListedColormap([np.array((255, 255, 255)) / 255., self.color_list[1]]),
            ListedColormap([np.array((255, 255, 255)) / 255., self.color_list[2]]),
            ListedColormap([np.array((255, 255, 255)) / 255., self.color_list[3]])
        ]

        if not self.is_adjust_grayscale_to_view:
            norm = Normalize(vmin=np.nanmin(image_arr), vmax=np.nanmax(image_arr)) # Old way of doing it, have absolute gray scale across whole image
        else:
            view_minima_list = []
            view_maxima_list = []
            for i in range(n_dim):
                ax = axs[0, i]
                other_axes = tuple(j for j in range(n_dim) if not j == i)
                most_voxels_index = np.argmax(np.count_nonzero(mask_arr_list[0], axis=other_axes)) # First given mask is the one to decide most voxel index
                view_minima_list.append(np.nanmin(image_arr.take(indices=most_voxels_index, axis=i)))
                view_maxima_list.append(np.nanmax(image_arr.take(indices=most_voxels_index, axis=i)))
            all_view_minimum = np.nanmin(view_minima_list)
            all_view_maximum = np.nanmax(view_maxima_list)
            norm = Normalize(vmin=all_view_minimum, vmax=all_view_maximum) # New way, making sure the displayed images are used for gray scale

        cmap = "gray"
        highlight_norm = Normalize(vmin=0.9, vmax=1.0)
        highlight_cmap_list = mask_colormaps_list

        highlight_alpha_list = [alpha for i, alpha in enumerate(self.alpha_list) if i < len(highlight_cmap_list)]

        for i in range(n_dim):
            ax = axs[0, i]
            other_axes = tuple(j for j in range(n_dim) if not j == i)
            most_voxels_index = np.argmax(np.count_nonzero(mask_arr_list[0], axis=other_axes)) # First given mask is the one to decide most voxel index

            extent = [0, image_arr.shape[other_axes[1]]*voxel_spacing_arr[other_axes[1]], 0, image_arr.shape[other_axes[0]]*voxel_spacing_arr[other_axes[0]]]

            mm_string = " mm" if is_use_mm else ""
            res_string = str([f"{voxel_spacing_arr[index]:.2f}{mm_string}" for index in other_axes])
            ax.set_title(f"Index {most_voxels_index},\nResolution = {res_string}")

            ax.imshow(image_arr.take(indices=most_voxels_index, axis=i), extent=extent, cmap=cmap, norm=norm,
                      origin="lower", interpolation=INTERPOLATION_MODE)
            for highlight_arr, highlight_cmap, highlight_alpha in zip(highlight_arr_list[::-1], highlight_cmap_list[::-1], highlight_alpha_list[::-1]):
                ax.imshow(highlight_arr.take(indices=most_voxels_index, axis=i), extent=extent, cmap=highlight_cmap,
                          norm=highlight_norm, alpha=highlight_alpha, origin="lower", interpolation=INTERPOLATION_MODE)
            if is_use_mm:
                ax.set_xlabel("mm")
                ax.set_ylabel("mm")

        if not save_path is None:
            px = 1/plt.rcParams['figure.dpi']
            size = (self.plot_size[0]*px, self.plot_size[1]*px)
            fig.set_size_inches(size)
            plt.tight_layout()
            if not save_path.endswith(".png"):
                save_path += ".png"
            plt.savefig(save_path)
            plt.close()


class BrainSplitVisualizer:


    def __init__(self, plot_size=(800, 600), color_list=None):
        self.plot_size = plot_size
        self.color_list = color_list if not color_list is None else ["black", "green", "lime"]

    def visualize_histogram_split(self, kde_grid, all_pdf, grad_weighted_pdf, median_int, split_int, erosion_width,
                                  cutoff_percentiles, comparison_arr, comparison_arr_raw, save_path=None):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        plt.title(f"Split intensity: {split_int}\nSettings: Brain erosion width: {erosion_width}, edges cutoff {cutoff_percentiles}%"
                  #f"; softmax exponent {exponent}"
                  )
        ax1.plot(kde_grid, all_pdf, color="k", label="total histogram")
        ax1.plot(kde_grid, grad_weighted_pdf, color="lime", label="gradient weighted histogram")
        ax1.plot([kde_grid[0]], [all_pdf[0]], color="g", label="average gradient histogram")
        ax1.vlines((median_int), np.amin(all_pdf), np.amax(all_pdf), colors='gray', linestyles=":", label="median")
        ax1.vlines((split_int), np.amin(all_pdf), np.amax(all_pdf), colors='g', linestyles="--", label="found split")
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Probability Density")
        ax1.legend()
        ax2.plot(kde_grid, comparison_arr_raw, color="g", linestyle=":", label="non-truncated evaluation function")
        ax2.plot(kde_grid, comparison_arr, color="g", label="evaluation function")
        #ax2.set_ylabel("Evaluation Function")
        ax2.set_ylabel("Average Gradient")

        if not save_path is None:
            px = 1/plt.rcParams['figure.dpi']
            size = (self.plot_size[0]*px, self.plot_size[1]*px)
            fig.set_size_inches(size)
            plt.tight_layout()
            if not save_path.endswith(".png"):
                save_path += ".png"
            #plt.show()
            plt.savefig(save_path)
            plt.close()