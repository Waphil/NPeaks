# N-Peaks MR intensity normalization

## Overview

Python implementation of N-Peaks MR intensity normalization method, which allows for normalization of MR intensity units across images to facilitate more biologically meaningful intensity units in a data set.

| ![method_explanation_image_V001_cropped](https://github.com/user-attachments/assets/d8d4abd7-6654-41ba-ae08-cfba919e9010) |
|:--:| 
| A visualization of each step in the N-Peaks method. (a) Two MR images that exist on different intensity scales and their intensity histograms. (b) Masks of homogeneous tissue in each image used for normalization, depicted in light green and light purple. (c) Isolation of homogeneous tissue in each mask, depicted in dark green and dark purple, and detection of histogram peaks, which are shown as vertical colored lines. (d) Piecewise linear transformation of intensity scales such that tissue intensity histogram peaks are aligned across images. The aligned peaks are depicted as vertical gray lines. |


For more background details, check out the linked publication in the Credit section of this readme.

_Dependencies:_

numpy,
scipy,
[KDEpy](https://kdepy.readthedocs.io/en/latest/),
Matplotlib

## How to use it

The file npeaks_normalize.py contains a class called NPeaksNormalizer. To use the normalization, instantiate a new object of that class. Then, you can call the _normalize_ method of the object on an image.

The _normalize_ method needs an image, a list of binary masks (representing the tissues to be normalized) and a list of target intensities for those masks. For information on further input parameters, consult the docstring and the table below. The method will return the normalized image.

If you do not want to receive the normalized image as a return object, but the determined landmark peaks (e.g. if you want to dynamically determine the target intensities), you can use the _find_peaks_ method of the normalizer class instead. The function to perform a piecewise linear intensity rescaling is given in _transform_intensity_scale_.

Multiple visualization tools are implemented in visualize.py. You can instantiate a NormVisualizer object and pass it to the NPeaksNormalizer when you define it to obtain visualization plots when applying the normalization. It is strongly recommended to plot the results of the normalization to verify that the result is as desired. The plots can also give important insights into which parameters need to be adjusted to get a better result.

## Normalization for the Brain

The code features a method that can heuristically segment brain images into ventricles and parenchyma if the brain mask is given as input. That way, these two masks can be used for normalization in a subsequent step. The method finds an intensity interval which has particularly high average local intensity change and determines that this must correspond not to a homogeneous tissue, but to the partial volume voxels between CSF and parenchyma. It then uses thresholding in that interval to determine the two masks. The method can be used by calling _split_brain_mask_ for an NPeaksNormalizer object. It also features its own visualization options using the class BrainSplitVisualizer in visualize.py

Once the two masks have been determined, N-Peaks normalization can be applied to the images with those masks and the isolation of homogeneous tissue can be used to automatically isolate white matter from the parenchymal tissue. The default settings of the provided N-Peaks normalization should give sensible results for brain masks created with the _split_brain_mask_ tool.

## Advanced usage

For most functions in the NPeaksNormalizer class, there exist an implementation with the "_custom" postfix. If these are called, the user has more control over what functions are used exactly, for example if the user wants to use a custom implementation for the local intensity change determination or the histogram peak detection.

## Parameter Choice Guide

| Parameter name |	Description |	Sensible Values |	Explanation |
| -------------- |	----------- |	--------------- |	----------- |
| _Kernel density estimation_ |
| bandwidth_selector |	KDE bandwith estimator |	Improved Sheather-Jones |	Used to determine the bandwidth the KDE uses for calculation of the PDF. Other algorithms (such as Silverman) were not tested and are expected to give worse results if the intensity distribution in the image / region of interest is not unimodal. |
| grid_number |	Amount of grid points on the KDE |	10’000 |	The amount of grid points on which the PDF is evaluated. Choosing diferent values between 1000 and 100’000 did not appear to affect the results in a major way. |
| discretization_count  |	Number of points that the intensity values will be discretized to before calculating the KDE bandwidth |	1024 |	In order to avoid the underestimation of the bandwidth for floating-point intensity values, they are discretized. Should not have a large effect as long as it is not substantially larger or smaller than the amount of discrete intensity values in the original image. |
| _Local intensity change_ |
| r, local_intensity_change_radius |	Radius of local neighborhood for local intensity change map |	Needs to be larger than the shortest side of the voxels in the dataset. Approximately 1.5 to 2 times the in-plane resolution gives good results. A single value can be chosen even if images in data set have different resolutions |	PVE and artifacts can affect a voxel and its neighbors, potentially even diagonal neighbors. As a result, a value for r between 1.5 and 2 times the in-plane resolution sensitizes the local intensity change to these effects. Note that the radius also applies for determining the neighborhood in through-plane direction. |
| _Isolating homogeneous region in user-defined mask_ |
| local_intensity_change_threshold_percentile |	Percentile value of the local intensity change in a mask where the threshold for the homogeneous region is chosen |	Depends on the mask used. If mask contains almost exclusively voxels of the desired tissue, can choose a high value (like 90%). If mask has many undesired voxels, choose a low value (like 25%). | An inspection oft he output plot of N-Peaks can help identify if this value is chosen too small or too large. If the value is chosen too high, then the detected peaks may be overly affected by voxels in inhomogeneous regions (e.g. PVE). If the value is chosen too low, there may be too few voxels oft he same tissue type remaining to form a clear peak on the histogram. |
| _Detecting peaks in intensity histogram_ |
| c<sub>prom</sub>, peak_prom_fraction |	Fraction used to define the minimum topographic prominence that a local maximum needs to show in order to be considered a separate peak |	0.1 |	Empirically, 0.1 seemed to be a good threshold to detect separate peaks without being too sensitive to small fluctuations in the PDF. If not enough peaks are detected, the value can be set smaller. If too many peaks are detected, the value can be set higher. |
| peak_selection_mode |	Criterion with which peak is automatically selected if multiple PDF peaks are detected |	“most”, “highest”, “left”, “right” |	This depends on the mask. “most” selects the peak encompassing the most voxels, which is usually a good choice. “highest” selects the highest peak. “left” selects the peak furthest to the left on the histogram, i.e. the one with the lowest intensity and vice versa for “right”. Those choices may be sensible for example if the intended tissue is expected to have the highest intensity in the mask (for example fat in bSSFP images). |

## Credit

Created by Philipp Wallimann, Department of Radiation Oncology, University Hospital Zürich.

__Manuscript in submission. A link will be provided here as soon as it is available.__

_Contact: philipp.wallimann@usz.ch_
