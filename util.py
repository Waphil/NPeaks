import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import maximum_filter, minimum_filter, binary_opening, binary_closing, binary_dilation, \
    binary_erosion
from KDEpy import FFTKDE


def combine_dict_with_default(new_dict, default_dict):
    new_dict = new_dict.copy() if not new_dict is None else {}
    for key in default_dict:
        new_dict[key] = new_dict.get(key, default_dict[key])
    return new_dict

def obtain_flat_arrays(image_arr, gradient_arr, mask=None, is_ignore_nan=True, is_return_coords=False):
    if mask is None:
        mask = np.ones_like(image_arr).astype(bool) # If no mask specified, take whole image

    if is_ignore_nan:
        mask = mask.copy() # If ignore nans, exclude objects from mask where image or gradient are nan
        mask = mask & (~np.isnan(image_arr)) & (~np.isnan(gradient_arr))

    # Obtain all gradients in mask, sort both intensities and gradients according to ascending gradients
    flat_image_arr = image_arr[mask]
    flat_gradient_arr = gradient_arr[mask]

    gradient_order = np.argsort(flat_gradient_arr)
    flat_image_arr = flat_image_arr[gradient_order]
    flat_gradient_arr = flat_gradient_arr[gradient_order]

    if not is_return_coords:
        return flat_image_arr, flat_gradient_arr
    else:
        flat_coord_arr = np.argwhere(mask)
        flat_coord_arr = flat_coord_arr[gradient_order]
        return flat_image_arr, flat_gradient_arr, flat_coord_arr

def calculate_KDE(array, grid, bw='ISJ', weights=None, is_return_bandwith=False):
    # Workaround because of numerical problems
    threshold = 1e+4
    rescale_factor = 1.

    array_intensity_range = np.nanmax(array)-np.nanmin(array)
    is_rescale = array_intensity_range > threshold
    if is_rescale:
        array = array.copy()
        rescale_factor = array_intensity_range/threshold
        array /= rescale_factor
        if not isinstance(grid, int):
            grid = grid.copy()
            grid /= rescale_factor
        if not isinstance(bw, str):
            bw /= rescale_factor

    # Suggested for bw: automatic bw selection using Improved Sheather Jones (ISJ)
    # Always use gaussian kernel
    KDE = FFTKDE(kernel='gaussian', bw=bw) # Define KDE
    KDE.fit(array, weights=weights) # Fit data

    # If grid is an integer, a new grid will be created by the evaluate funciton. Otherwise, the existing one is used
    if isinstance(grid, int):
        intensities, pdf_arr = KDE.evaluate(grid) # Evaluate KDE on grid
    else:
        pdf_arr = KDE.evaluate(grid) # Evaluate KDE on grid
        intensities = grid
    bw = KDE.bw

    if is_rescale:
        intensities *= rescale_factor
        pdf_arr /= rescale_factor
        bw *= rescale_factor

    if not is_return_bandwith:
        return intensities, pdf_arr
    return intensities, pdf_arr, bw #Return also KDE bandwith

def calculate_KDE_with_discretized_points(arr, kde_grid, kde_bandwith, n_discrete=1024):
    val_min = np.nanmin(arr)
    val_range = np.nanmax(arr)-val_min
    discretized_arr = (arr.copy() - val_min) / (val_range/n_discrete) if not val_range <= 0 else arr.copy() - val_min
    discretized_arr = np.around(discretized_arr)
    discretized_arr = discretized_arr * (val_range/n_discrete) + val_min
    grid, discretized_pdf, bw = calculate_KDE(discretized_arr, kde_grid, bw=kde_bandwith, is_return_bandwith=True)
    pdf = calculate_KDE(discretized_arr, grid, bw=bw) [1]
    return grid, pdf, bw

def calculate_voxel_volume(voxel_spacing_arr=None):
    # Calculate voxel volume.
    # If no voxel spacing given, default to counting the number of voxels. Thus 1 voxel has volume 1
    if voxel_spacing_arr is None:
        return 1.
    else:
        return np.prod(voxel_spacing_arr)

def automatic_snippet_number_determination(n_voxels, voxel_spacing_arr=None, n_min=10, n_max=100, vol_min=1000):
    # Calculate how many voxel a snippet should contain to fulfill the vol_min requirement
    voxel_number_min = vol_min/calculate_voxel_volume(voxel_spacing_arr)
    # Calculate how many snippet should be created. Want at least n_min and at most n_max. If in between, should pick
    # an amount that leaves every snippet with at least vol_min volume.
    return max(n_min, min(n_max, int(n_voxels/voxel_number_min)))

def calculate_structuring_element(radius, dim=3, voxel_spacing_arr=None):
    if voxel_spacing_arr is None:
        voxel_spacing_arr = np.ones(dim)
    int_radius_arr = np.floor(radius/voxel_spacing_arr).astype(np.int32)

    coordinate_list = [np.linspace(-int_r, int_r, 2*int_r+1)*d for d, int_r in zip(voxel_spacing_arr, int_radius_arr)]
    grid_arr = np.array(np.meshgrid(*coordinate_list, indexing="ij"))
    dist_arr = np.linalg.norm(grid_arr, axis=0)
    structuring_element = dist_arr <= radius
    return structuring_element

def dilate_mask(mask, radius=1.8, iterations=1, voxel_spacing_arr=None):
    structure = calculate_structuring_element(radius, dim=mask.ndim, voxel_spacing_arr=voxel_spacing_arr)
    return binary_dilation(mask, structure, iterations=iterations)

def erode_mask(mask, radius=1.8, iterations=1, voxel_spacing_arr=None):
    structure = calculate_structuring_element(radius, dim=mask.ndim, voxel_spacing_arr=voxel_spacing_arr)
    return binary_erosion(mask, structure, iterations=iterations)

def open_mask(mask, radius=1.8, iterations=1, voxel_spacing_arr=None):
    structure = calculate_structuring_element(radius, dim=mask.ndim, voxel_spacing_arr=voxel_spacing_arr)
    return binary_opening(mask, structure, iterations=iterations)

def close_mask(mask, radius=1.8, iterations=1, voxel_spacing_arr=None):
    structure = calculate_structuring_element(radius, dim=mask.ndim, voxel_spacing_arr=voxel_spacing_arr)
    return binary_closing(mask, structure, iterations=iterations)

def erode_brain(brain_mask, subarachnoid_removal_iterations, voxel_spacing_arr=None):
    eroded_brain_mask = brain_mask.copy()

    if not (subarachnoid_removal_iterations is None or subarachnoid_removal_iterations < 0.):
        # Calculate maximum voxel spacing. Use this as radius so that we certainly erode all three dimensions
        max_voxel_spacing = np.amax(voxel_spacing_arr) if not voxel_spacing_arr is None else 1.
        # Calculate how many times we have to erode to get at least the given distance
        n_iterations = int(np.ceil(subarachnoid_removal_iterations / max_voxel_spacing))
        # Erode boundary (mostly important at top)
        eroded_brain_mask = erode_mask(eroded_brain_mask, max_voxel_spacing,
                                       iterations=n_iterations, voxel_spacing_arr=voxel_spacing_arr)
    return eroded_brain_mask

def calculate_local_intensity_change_from_min_and_max(data, min_data, max_data):
    return max_data-min_data

def create_local_intensity_change_function(radius=1.8, is_correct_near_miss=True):
    # The normalizer needs a gradient function as input. This method provides that based on min and max
    # If is_correct_near_miss, increase the radius slightly so it doesn't miss voxels due to floating point inaccuracies
    if is_correct_near_miss:
        radius *= 1.001
    def grad_func(array, voxel_spacing_arr=None):
        structure_ball_mask = calculate_structuring_element(radius, dim=array.ndim, voxel_spacing_arr=voxel_spacing_arr)
        max_arr = maximum_filter(array, footprint=structure_ball_mask)
        min_arr = minimum_filter(array, footprint=structure_ball_mask)
        grad = calculate_local_intensity_change_from_min_and_max(array, min_arr, max_arr)
        return grad
    return grad_func

def calculate_voxel_spacing_from_affine(affine_mat):
    return np.linalg.norm(affine_mat[:-1,:-1], axis=0)

def cutoff_percentiles(image_arr, mask=None, bottom_perc=1, top_perc=99):
    if mask is None:
        mask = ~np.isnan(image_arr)
    bottom_val = np.percentile(image_arr[mask], bottom_perc, method="lower")
    top_val = np.percentile(image_arr[mask], top_perc, method="higher")
    image_arr = np.maximum(image_arr, bottom_val)
    image_arr = np.minimum(image_arr, top_val)
    return image_arr

def hellinger_distance(p, q):
    # According to this: https://en.wikipedia.org/wiki/Hellinger_distance
    return np.sqrt(1 - np.vdot(np.sqrt(p), np.sqrt(q)))

def obtain_histogram_peaks_indices(hist, prom_fraction=0.1):
    # Determine all peak locations
    peaks = find_peaks(hist)[0]

    # Determine prominences and left and right bases for all peaks
    peak_proms, peak_l_bases, peak_r_bases = peak_prominences(hist, peaks)

    # Determine the indices of the peaks that are above the prominence threshold compared to the largest peak
    prominent_indices = peak_proms >= prom_fraction * np.amax(peak_proms)

    # Select only the peaks above the prominece thresholds
    peaks = peaks[prominent_indices]
    peak_proms = peak_proms[prominent_indices]
    peak_l_bases = peak_l_bases[prominent_indices]
    peak_r_bases = peak_r_bases[prominent_indices]
    peak_heights = hist[peaks]

    # Determine the split of the interval among the peaks. Start with the bases from the prominence algorithm
    peak_l_bound_arr = peak_l_bases.copy()
    peak_r_bound_arr = peak_r_bases.copy()

    for i, peak_index in enumerate(peaks):
        # Take as left bound the maximum of all lower index right bounds or the left bound itself.
        peak_l_bound_arr[i] = np.amax(np.append(peak_r_bases[peak_r_bases < peak_index], peak_l_bound_arr[i]))
        # Analogous for right bound
        peak_r_bound_arr[i] = np.amin(np.append(peak_l_bound_arr[peak_l_bound_arr > peak_index], peak_r_bound_arr[i]))

    return peaks, peak_heights, peak_l_bound_arr, peak_r_bound_arr

def obtain_histogram_peaks(grid, hist, prom_fraction=0.1):
    hist_peaks, hist_peak_heights, hist_peak_l_bound_arr, hist_peak_r_bound_arr = obtain_histogram_peaks_indices(hist, prom_fraction)
    return grid[hist_peaks], hist_peak_heights, grid[hist_peak_l_bound_arr], grid[hist_peak_r_bound_arr]

def obtain_gradient_thresholded_mask(gradient_arr, gradient_threshold, mask=None):
    if mask is None:
        mask = np.ones_like(gradient_arr).astype(bool)
    return (gradient_arr <= gradient_threshold) & mask

def calculate_first_intersection_point(arr_1, arr_2):
    diff = arr_2-arr_1
    if np.any(diff <= 0):
        return np.amin(np.argwhere(diff <= 0.))
    else:
        print("Warning! there is no intersection between the two JSD curves")
        return 0
