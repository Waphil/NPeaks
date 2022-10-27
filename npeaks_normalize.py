import numpy as np
import os
from scipy.spatial.distance import jensenshannon
from scipy.interpolate import interp1d

import sys
sys.path.append('../')

from NPeaks.util import calculate_KDE, calculate_KDE_with_discretized_points, obtain_flat_arrays, \
    obtain_histogram_peaks, calculate_first_intersection_point, automatic_snippet_number_determination, \
    combine_dict_with_default, create_local_intensity_change_function, erode_brain
from NPeaks.visualize import NormVisualizer, BrainSplitVisualizer

DEFAULT_FIG_SIZE = (8, 6)

DEFAULT_KDE_PARAMS = {
    "discretization_count" : 1024,
    "grid_number" : 10_000,
    "bandwith_selector" : 'ISJ',
}

DEFAULT_LOCAL_INTENSITY_CHANGE_RADIUS = 1.5 # In mm

INVERTED_BRAIN_CONTRAST_SEQUENCE_LIST = ["t2"]
DEFAULT_BRAIN_SPLIT_SETTINGS = {
    "mr_sequence" : "t1", # T2 is treated differently
    "outlier_cutoff_percentage" : 0.5, # Cutoff from both sides
    "subarachnoid_removal_distance" : 20, # In mm
}

DEFAULT_LOCAL_INTENSITY_CHANGE_THRESHOLD_SETTINGS = {
    "high_Local_intensity_change_outlier_cutoff_percentage" : 10., # Cut off this fraction of high local intensiyy
    # change voxels for high local intensity change snippet selection
    "n_snippets_min" : 10, # The minimum number of snippets considered. Recommended: 10
    "n_snippets_max" : 100, # The maximum number of snippets considered. Recommended: 100
    "vol_snippets_min" : 500, # The minimum volume (in mm^3) that each snippet should have. Needs to be adjusted by body
    # region. Recommended in brain: about 500, outside of brain about 1000.
    "lam" : 1.0, # Weighting factor lambda
}

DEFAULT_PEAK_DETECTION_SETTINGS = {
    "peak_prom_fraction" : 0.1, #
    "peak_selection_mode" : 'most', # Decide rationale for peak selection in case multiple have been found
}

DEFAULT_PDF_DISTANCE_FUNCTION = jensenshannon


class NPeakNormalizer:

    def __init__(self, kde_param_dict=None, local_intensity_change_radius=None, histogram_distance_function=None, plot_folder=None):
        """
        Initializes the normalizer with the given parameters.
        :param kde_param_dict: Parameters that define the calculation of the kernel density estimation.
        :param local_intensity_change_radius: Radius (in mm) in which the local intensity change is calculated
        :param histogram_distance_function:  The function used to compare histograms. By default, use Jensen Shannon
        Distance
        :param plot_folder: The folder where summary plots of the normalization will be saved for each mask in each
        image. If none is given, do not plot any images.
        """
        kde_param_dict = combine_dict_with_default(kde_param_dict, DEFAULT_KDE_PARAMS).copy()
        self.kde_discretization_count = kde_param_dict.get("discretization_count")
        self.kde_grid_number = kde_param_dict.get("grid_number")
        self.kde_bandwith_selector = kde_param_dict.get("bandwith_selector")

        self.local_intensity_change_radius = local_intensity_change_radius if not local_intensity_change_radius is None else DEFAULT_LOCAL_INTENSITY_CHANGE_RADIUS
        self.local_intensity_change_function = create_local_intensity_change_function(self.local_intensity_change_radius)

        self.histogram_distance_function = histogram_distance_function if not histogram_distance_function is None else DEFAULT_PDF_DISTANCE_FUNCTION

        self.plot_folder = plot_folder
        self.peak_finder = PeakFinder(histogram_distance_function=self.histogram_distance_function, plot_folder=self.plot_folder)

    def smart_brain_histogram_split(self, image_arr, local_intensity_change_arr, mask, kde_grid, kde_bw, erosion_width=20,
                                    cutoff_percentiles=0.5, is_search_below_median=True, image_name="image"):
        """
        Creates two masks for a brain image (ventricles and rest of the brain) based on intensity threshold.
        :param image_arr:
        :param local_intensity_change_arr:
        :param mask:
        :param kde_grid:
        :param kde_bw:
        :param erosion_width:
        :param cutoff_percentiles:
        :param is_search_below_median:
        :param image_name:
        :return:
        """
        flat_image_arr, flat_local_intensity_change_arr = obtain_flat_arrays(image_arr, local_intensity_change_arr, mask)
        median_int = np.median(flat_image_arr)
        grid_median_index = np.searchsorted(kde_grid, median_int, 'right')

        lower_bound, upper_bound = np.percentile(flat_image_arr, (cutoff_percentiles, 100.-cutoff_percentiles))

        # Need to determine whether to search below or above median
        grid_lower_search_index = 0 if is_search_below_median else grid_median_index
        grid_higher_search_index = grid_median_index if is_search_below_median else kde_grid.size

        all_pdf = calculate_KDE(flat_image_arr, kde_grid, kde_bw)[1]
        lic_weighted_pdf = calculate_KDE(flat_image_arr, kde_grid, kde_bw, weights=flat_local_intensity_change_arr)[1]

        comparison_arr_raw = lic_weighted_pdf / all_pdf
        comparison_arr = comparison_arr_raw.copy()
        comparison_arr[(kde_grid < lower_bound) | (kde_grid > upper_bound)] = np.nan

        split_int = kde_grid[grid_lower_search_index:grid_higher_search_index][np.nanargmax(comparison_arr[grid_lower_search_index:grid_higher_search_index])]


        if not self.plot_folder is None:
            split_vis = BrainSplitVisualizer()
            split_save_path = os.path.join(self.plot_folder, f"{image_name}_brainsplit.png")
            split_vis.visualize_histogram_split(kde_grid, all_pdf, lic_weighted_pdf, median_int, split_int,
                                                erosion_width,
                                                cutoff_percentiles, comparison_arr, comparison_arr_raw,
                                                save_path=split_save_path)

        return split_int

    def split_brain_mask(self, image_arr, local_intensity_change_arr, kde_grid, kde_bw, voxel_spacing_arr=None,
                         brain_mask=None, exclusion_mask=None, split_settings_dict=None, image_name="image"):
        #Read parameters
        split_settings_dict = combine_dict_with_default(split_settings_dict, DEFAULT_BRAIN_SPLIT_SETTINGS).copy()

        mr_sequence = split_settings_dict.get("mr_sequence")
        outlier_cutoff_percentage = split_settings_dict.get("outlier_cutoff_percentage")
        subarachnoid_removal_distance = split_settings_dict.get("subarachnoid_removal_distance")

        is_search_below_median = not (mr_sequence.lower() in INVERTED_BRAIN_CONTRAST_SEQUENCE_LIST) # For T2-w, have to search above median, else below

        if brain_mask is None:
            brain_mask = ~np.isnan(image_arr)

        eroded_brain_mask = erode_brain(brain_mask, subarachnoid_removal_distance, voxel_spacing_arr)

        # If have something to exclude (e.g. cancer, artifacts), can do it here
        if not exclusion_mask is None:
            brain_mask[exclusion_mask] = False
            eroded_brain_mask[exclusion_mask] = False

        split_intensity = self.smart_brain_histogram_split(image_arr, local_intensity_change_arr, eroded_brain_mask, kde_grid, kde_bw,
                                                           subarachnoid_removal_distance, outlier_cutoff_percentage,
                                                           is_search_below_median, image_name)

        mask_1 = image_arr <= split_intensity
        mask_2 = image_arr > split_intensity
        if not is_search_below_median:
            # Have to change the order if using sequence where csf is higher intensity than median (and thus than wm)
            tmp = mask_1
            mask_1 = mask_2
            mask_2 = tmp

        # Make sure masks are in brain, make sure csf uses eroded mask
        mask_1 = mask_1 & eroded_brain_mask
        mask_2 = mask_2 & brain_mask

        return mask_1, mask_2

    def calculate_peak_locs(self, image_arr, mask_list, base_mask=None, lic_thresh_settings_list=None,
                            peak_det_settings_list=False, voxel_spacing_arr=None, image_name="test",
                            mask_name_list=None):
        mask_name_list = mask_name_list if not mask_name_list is None else [f"mask{i}" for i in range(len(mask_list))]
        if base_mask is None:
            # If no base mask exists, combine all given masks to a synthetic base mask
            base_mask = np.zeros_like(image_arr).astype(bool)
            for mask in mask_list:
                base_mask = base_mask | mask

        local_intensity_change_arr = self.local_intensity_change_function(image_arr)

        kde_grid, all_pdf, kde_bw = calculate_KDE_with_discretized_points(image_arr[base_mask], self.kde_grid_number,
                                                                          self.kde_bandwith_selector,
                                                                          self.kde_discretization_count)

        return [self.peak_finder.get_peak(image_arr, local_intensity_change_arr, mask, kde_grid, kde_bw,
                                          local_intensity_change_threshold_settings_dict=lic_thresh_settings,
                                          peak_detection_settings_dict=peak_det_settings,
                                          voxel_spacing_arr=voxel_spacing_arr,
                                          image_name=image_name, mask_name=mask_name) for mask, mask_name, lic_thresh_settings, peak_det_settings in zip(mask_list, mask_name_list, lic_thresh_settings_list, peak_det_settings_list)]

    def calculate_peak_locs_autosplit(self, image_arr, brain_mask=None, exclusion_mask=None, split_settings_dict=None,
                                      lic_thresh_settings_list=None, peak_det_settings_list=False,
                                      voxel_spacing_arr=None, image_name="test"):
        local_intensity_change_arr = self.local_intensity_change_function(image_arr)

        kde_grid, all_pdf, kde_bw = calculate_KDE_with_discretized_points(image_arr[brain_mask], self.kde_grid_number,
                                                                          self.kde_bandwith_selector,
                                                                          self.kde_discretization_count)


        mask_csf, mask_wm = self.split_brain_mask(image_arr, local_intensity_change_arr, kde_grid, kde_bw,
                                                  voxel_spacing_arr=voxel_spacing_arr,
                                                  brain_mask=brain_mask, exclusion_mask=exclusion_mask,
                                                  split_settings_dict=split_settings_dict, image_name=image_name)

        return [self.peak_finder.get_peak(image_arr, local_intensity_change_arr, mask, kde_grid, kde_bw,
                                          local_intensity_change_threshold_settings_dict=lic_thresh_settings,
                                          peak_detection_settings_dict=peak_det_settings,
                                          voxel_spacing_arr=voxel_spacing_arr,
                                          image_name=image_name, mask_name=mask_name) for mask, mask_name, lic_thresh_settings, peak_det_settings in zip([mask_csf, mask_wm], ["CSF", "WM"], lic_thresh_settings_list, peak_det_settings_list)]

    def normalize(self, image_arr, mask_list=None, peak_intensity_list=None, goal_intensity_list=None, base_mask=None,
                  lic_thresh_settings_list=None, peak_det_settings_list=False, voxel_spacing_arr=None,
                  image_name="test", mask_name_list=None, is_return_peaks=False):
        """
        Normalizes the image based on the peak intensities found in the masks in mask_list according to the settings
        provided.
        :param image_arr:
        :param mask_list:
        :param peak_intensity_list:
        :param goal_intensity_list:
        :param base_mask:
        :param lic_thresh_settings_list:
        :param peak_det_settings_list:
        :param voxel_spacing_arr:
        :param image_name:
        :param mask_name_list:
        :param is_return_peaks:
        :return:
        """
        # Can also give a peak list if it is already known to only perform normalization without peak detection
        if peak_intensity_list is None and not mask_list is None:
            peak_intensity_list = self.calculate_peak_locs(image_arr, mask_list, base_mask=base_mask,
                                                           lic_thresh_settings_list=lic_thresh_settings_list,
                                                           peak_det_settings_list=peak_det_settings_list,
                                                           voxel_spacing_arr=voxel_spacing_arr, image_name=image_name,
                                                           mask_name_list=mask_name_list)

        goal_intensity_list = goal_intensity_list if not goal_intensity_list is None else [index for index, m in enumerate(mask_list)]
        f = interp1d(peak_intensity_list, goal_intensity_list, fill_value="extrapolate")
        norm_image_arr = f(image_arr)
        if is_return_peaks:
            return norm_image_arr, peak_intensity_list
        return norm_image_arr

class PeakFinder:

    def __init__(self, histogram_distance_function=None, plot_folder=None):
        self.histogram_distance_function = histogram_distance_function if not histogram_distance_function is None else DEFAULT_HISTOGRAM_DISTANCE_FUNCTION
        self.plot_folder = plot_folder
        self.plot_parameter_dict = {}
        self.norm_vis = NormVisualizer()

    def get_peak(self, image_arr, local_intensity_change_arr, mask, kde_grid, kde_bw, voxel_spacing_arr=None,
                 local_intensity_change_threshold_settings_dict=None, peak_detection_settings_dict=None,
                 image_name="test", mask_name="1"):
        """
        Obtain the intensity peak of the image in the mask. Automatically detects it in the homogeneous region based
        on the input settings
        :param image_arr:
        :param local_intensity_change_arr:
        :param mask:
        :param kde_grid:
        :param kde_bw:
        :param voxel_spacing_arr:
        :param local_intensity_change_threshold_settings_dict:
        :param peak_detection_settings_dict:
        :param image_name:
        :param mask_name:
        :return:
        """
        #Read parameters
        local_intensity_change_threshold_settings_dict = combine_dict_with_default(local_intensity_change_threshold_settings_dict,
                                                                                   DEFAULT_LOCAL_INTENSITY_CHANGE_THRESHOLD_SETTINGS).copy()

        high_lic_cutoff_percentage = local_intensity_change_threshold_settings_dict.get("high_Local_intensity_change_outlier_cutoff_percentage")
        n_snippets_min = local_intensity_change_threshold_settings_dict.get("n_snippets_min")
        n_snippets_max = local_intensity_change_threshold_settings_dict.get("n_snippets_max")
        vol_snippets_min = local_intensity_change_threshold_settings_dict.get("vol_snippets_min")
        lam = local_intensity_change_threshold_settings_dict.get("lam")

        peak_detection_settings_dict = combine_dict_with_default(peak_detection_settings_dict,
                                                                 DEFAULT_PEAK_DETECTION_SETTINGS).copy()

        smooth_radius = peak_detection_settings_dict.get("smooth_radius")
        is_use_smooth_peaks = peak_detection_settings_dict.get("is_use_smooth_peaks")
        peak_prom_fraction = peak_detection_settings_dict.get("peak_prom_fraction")
        peak_selection_mode = peak_detection_settings_dict.get("peak_selection_mode")

        local_intensity_change_threshold = self.local_intensity_change_threshold_estimation(image_arr, local_intensity_change_arr, mask, kde_grid, kde_bw,
                                                            high_lic_cutoff_percentage,
                                                            n_snippets_min, n_snippets_max, vol_snippets_min, lam,
                                                            voxel_spacing_arr)

        homogeneous_mask = mask & (local_intensity_change_arr <= local_intensity_change_threshold)

        peak_int, peak_mask = self.obtain_peak_in_mask(image_arr, homogeneous_mask, kde_grid, kde_bw,
                                                       peak_prom_fraction, peak_selection_mode)

        if not self.plot_folder is None:
            plot_path = os.path.join(self.plot_folder, f"{image_name}_mask{mask_name}.png")
            # If a plot folder is given, assemble all the information into a single plot
            self.norm_vis.plot(image_arr, mask, kde_grid, kde_bw,
                               high_lic_cutoff_percentage,
                               n_snippets_min, n_snippets_max, vol_snippets_min, lam,
                               peak_prom_fraction, peak_selection_mode,
                               self.plot_parameter_dict, voxel_spacing_arr, plot_path,
                               image_name=image_name, mask_name=mask_name)
            self.plot_parameter_dict = {} # Clear plot params

        return peak_int

    def local_intensity_change_threshold_estimation(self, image_arr, local_intensity_change_arr, mask, kde_grid, kde_bw,
                                                    high_lic_cutoff_percentage, n_snippets_min, n_snippets_max,
                                                    vol_snippets_min, lam, voxel_spacing_arr=None):
        """
        Estimates the threshold of local intensity change which separates the homogeneous tissue from the inhomogeneous
        tissue within the mask.
        :param image_arr:
        :param local_intensity_change_arr:
        :param mask:
        :param kde_grid:
        :param kde_bw:
        :param high_lic_cutoff_percentage:
        :param n_snippets_min:
        :param n_snippets_max:
        :param vol_snippets_min:
        :param lam:
        :param voxel_spacing_arr:
        :return:
        """
        flat_image_arr, flat_local_intensity_change_arr = obtain_flat_arrays(image_arr, local_intensity_change_arr, mask)

        # Automatic determination of n_segments based on settings
        n_snippets = automatic_snippet_number_determination(flat_image_arr.size, voxel_spacing_arr=voxel_spacing_arr,
                                                            n_min=n_snippets_min, n_max=n_snippets_max,
                                                            vol_min=vol_snippets_min)

        percentile_arr = (np.arange(n_snippets) + 1) / n_snippets * 100
        boundary_local_intensity_change_arr = np.nanpercentile(flat_local_intensity_change_arr, percentile_arr)
        boundary_index_arr = np.hstack((0, np.searchsorted(flat_local_intensity_change_arr, boundary_local_intensity_change_arr, side="right")))
        boundary_index_arr = np.array(sorted(list(set(boundary_index_arr)))) # Exclude empty snippets

        # Have to use the actual amount of segments that was split, since that is not guaranteed with the used method
        n_snippets = len(boundary_index_arr)-1

        flat_local_intensity_change_arr_list = [flat_local_intensity_change_arr[boundary_index_arr[i]:boundary_index_arr[i+1]] for i in range(n_snippets)]
        flat_image_arr_list = [flat_image_arr[boundary_index_arr[i]:boundary_index_arr[i+1]] for i in range(n_snippets)]

        jsd_low_arr = np.zeros(n_snippets)
        jsd_high_arr = np.zeros(n_snippets)
        threshold_arr = np.array([lic_arr[-1] for lic_arr in flat_local_intensity_change_arr_list])

        # Here we calculate the index of the high local intensity change snippet. In order to get rid of outliers, exclude
        # the top high_lic_cutoff_percentage percent of snippets (rounded up)
        high_snippet_index = int(n_snippets - 1 - np.ceil(high_lic_cutoff_percentage/100*n_snippets))

        base_low_snippet_int = flat_image_arr_list[0]
        base_high_snippet_int = flat_image_arr_list[high_snippet_index]

        base_low_snippet_pdf = calculate_KDE(base_low_snippet_int, kde_grid, kde_bw)[1]
        base_high_snippet_pdf = calculate_KDE(base_high_snippet_int, kde_grid, kde_bw)[1]

        for index, i_arr in enumerate(flat_image_arr_list):
            pdf = calculate_KDE(i_arr, kde_grid, kde_bw)[1]

            dist_to_low_lic = self.histogram_distance_function(base_low_snippet_pdf, pdf)
            dist_to_high_lic = self.histogram_distance_function(base_high_snippet_pdf, pdf)

            jsd_low_arr[index] = dist_to_low_lic
            jsd_high_arr[index] = dist_to_high_lic

        # Since I started using the background, I have run into the problem of getting terrible jsd values.
        # This can help ignore those cases
        if np.any(np.isnan(jsd_low_arr)):
            last_index_closer_to_low = 0
        else:
            # Find which snippet is the last that is closer to the low local intensity change snippet than the high (divided by lambda)
            last_index_closer_to_low = max(0, calculate_first_intersection_point(lam*jsd_low_arr, jsd_high_arr)-1)
        chosen_threshold = threshold_arr[last_index_closer_to_low]

        if not self.plot_folder is None:
            chosen_int_arr = flat_image_arr[flat_local_intensity_change_arr <= chosen_threshold]
            chosen_pdf = calculate_KDE(chosen_int_arr, kde_grid, kde_bw)[1]
            all_pdf = calculate_KDE(flat_image_arr, kde_grid, kde_bw)[1]
            homogeneous_mask = mask & (local_intensity_change_arr <= chosen_threshold)

            self.plot_parameter_dict["threshold_arr"] = threshold_arr
            self.plot_parameter_dict["jsd_low_arr"] = jsd_low_arr
            self.plot_parameter_dict["jsd_high_arr"] = jsd_high_arr
            self.plot_parameter_dict["n_snippets"] = n_snippets
            self.plot_parameter_dict["nonorm_low_snippet_pdf"] = base_low_snippet_pdf*base_low_snippet_int.size
            self.plot_parameter_dict["nonorm_high_snippet_pdf"] = base_high_snippet_pdf*base_high_snippet_int.size
            self.plot_parameter_dict["nonorm_chosen_pdf"] = chosen_pdf*chosen_int_arr.size
            self.plot_parameter_dict["nonorm_all_pdf"] = all_pdf*flat_image_arr.size
            self.plot_parameter_dict["chosen_threshold"] = chosen_threshold
            self.plot_parameter_dict["homogeneous_mask"] = homogeneous_mask

        return chosen_threshold

    def obtain_peak_in_mask(self, image_arr, mask, kde_grid, kde_bw,
                            peak_prom_fraction, peak_selection_mode):
        """
        Obtains all pdf intensity peaks in the mask based on the peak prominence fraction criteroin.
        Selects one of them as the definitive peak based on the peak_selection_mode
        :param image_arr:
        :param mask:
        :param kde_grid:
        :param kde_bw:
        :param peak_prom_fraction:
        :param peak_selection_mode:
        :return:
        """
        # Calculate smooth and nonsmooth histograms
        pdf = calculate_KDE(image_arr[mask], kde_grid, kde_bw)[1]

        # Obtain peaks in the smooth histogram
        peak_ints, peak_heights, peak_l_bounds, peak_r_bounds = obtain_histogram_peaks(kde_grid, pdf,
                                                                                       prom_fraction=peak_prom_fraction)

        peak_mask_list = [mask & (image_arr > peak_l_b) & (image_arr < peak_r_b) for peak_l_b, peak_r_b
                          in zip(peak_l_bounds, peak_r_bounds)] # Find masks of each peak by isolating the intensity ranges


        peak_int_arr_list = [image_arr[peak_mask] for peak_mask in peak_mask_list]
        peak_pdf_list = [calculate_KDE(peak_int_arr, kde_grid, kde_bw)[1] for peak_int_arr in peak_int_arr_list]
        peaks_pdf_nonnorm_list = [pdf*int_arr.size for pdf, int_arr in zip(peak_pdf_list, peak_int_arr_list)]

        used_peak_int_arr = np.array(peak_ints)
        used_peak_height_arr = np.array(peak_heights)

        if peak_selection_mode.lower() == "most": # Select peak with most voxels
            peak_mask_size_list = [np.count_nonzero(peak_mask) for peak_mask in peak_mask_list]
            chosen_peak_index = np.argmax(peak_mask_size_list)
        elif peak_selection_mode.lower() == "highest": # Select peak with highest peak
            chosen_peak_index = np.argmax(used_peak_height_arr)
        elif peak_selection_mode.lower() == "right": # Select peak with most intenstiy
            chosen_peak_index = np.argmax(used_peak_int_arr)
        else: # Select peak with least intensity
            chosen_peak_index = np.argmin(used_peak_int_arr)


        if not self.plot_folder is None:
            self.plot_parameter_dict["pdf_nonorm"] = pdf*np.count_nonzero(mask)
            self.plot_parameter_dict["chosen_peak_index"] = chosen_peak_index
            self.plot_parameter_dict["used_peak_int_arr"] = used_peak_int_arr
            self.plot_parameter_dict["used_peaks_pdf_nonorm_list"] = peaks_pdf_nonnorm_list
            self.plot_parameter_dict["used_peaks_mask_list"] = peak_mask_list

        return_peak_intensity = used_peak_int_arr[chosen_peak_index]
        return_peak_mask = peak_mask_list[chosen_peak_index]
        return return_peak_intensity, return_peak_mask
