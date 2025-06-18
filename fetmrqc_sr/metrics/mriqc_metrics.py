# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# This code is a modified version of the code from the MRIQC project
# available at https://github.com/nipreps/mriqc/blob/master/mriqc/qc/anatomical.py
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
from scipy.stats import kurtosis  # pylint: disable=E0611
import numpy as np
from math import sqrt
from skimage import measure 
from statsmodels import robust
from scipy.stats import chi2, norm 

# MRIQC's Dietrich factor for SNR
DIETRICH_FACTOR = np.sqrt(2 / (4.0 - np.pi))



# --- New/Adapted tissue_to_max_intensity_ratio (formerly wm2max) ---
def tissue_to_max_intensity_ratio(img: np.ndarray, tissue_mask: np.ndarray, percentile_max: float = 99.95) -> float:
    """
    Calculates the ratio of the mean intensity within a specific tissue mask
    to a high percentile of the intensity in the entire image.
    """
    if not np.any(tissue_mask):
        print("WARNING (tissue_to_max_intensity_ratio): tissue_mask is empty.")
        return -1.0 # Or np.nan
        
    tissue_voxels = img[tissue_mask > 0]
    if tissue_voxels.size == 0:
        print("WARNING (tissue_to_max_intensity_ratio): No voxels in tissue_mask after indexing img.")
        return -1.0

    mu_tissue = np.mean(tissue_voxels)
    
    if img.size == 0:
        print("WARNING (tissue_to_max_intensity_ratio): Input image for percentile calculation is empty.")
        return -1.0

    # Calculate percentile on non-zero voxels to be more robust for masked/cropped images
    img_non_zero = img[img > 1e-6] # Avoid pure zero background influencing percentile too much
    if img_non_zero.size < 10: # Need enough voxels for percentile to be meaningful
        print(f"WARNING (tissue_to_max_intensity_ratio): Very few non-zero voxels ({img_non_zero.size}) in image for percentile calculation. Using simple max instead.")
        if img.size > 0 :
            overall_max_intensity = np.max(img) if img.size > 0 else 0.0
        else: # Should not happen due to earlier check
            overall_max_intensity = 0.0
    else:
        overall_max_intensity = np.percentile(img_non_zero.flatten(), percentile_max)

    if overall_max_intensity < 1e-6:
        print("WARNING (tissue_to_max_intensity_ratio): Overall max intensity in image is near zero.")
        return -1.0 # Or handle as per specific needs, e.g., if mu_tissue also ~0, ratio could be 1 or undefined
        
    return float(mu_tissue / overall_max_intensity)

# --- New RPVE (Residual Partial Volume Effect) ---
def rpve_custom(img: np.ndarray, pure_tissue_mask: np.ndarray, pve_interface_mask: np.ndarray) -> float:
    """
    Calculates a measure of Residual Partial Volume Effect.
    RPVE = 1.0 - (mean_intensity_in_pve_interface / mean_intensity_in_pure_tissue)
    Lower values (closer to 0) are better if interface should be similar to pure.
    If interface is darker, RPVE can be > 1. If brighter, can be < 0.
    This definition implies higher intensity in pure_tissue is 'good'.
    Alternative MRIQC: uses sum(abs(pv_boundary - pv_pure))/sum(pv_pure).
    Let's use the 1 - ratio formula for now.
    """
    if not np.any(pure_tissue_mask) or not np.any(pve_interface_mask):
        print("WARNING (rpve_custom): Pure tissue mask or PVE interface mask is empty.")
        return -1.0 # Or np.nan

    pure_voxels = img[pure_tissue_mask > 0]
    pve_voxels = img[pve_interface_mask > 0]

    if pure_voxels.size == 0 or pve_voxels.size == 0:
        print("WARNING (rpve_custom): No voxels in pure or PVE mask after indexing image.")
        return -1.0

    mean_pure = np.mean(pure_voxels)
    mean_pve = np.mean(pve_voxels)

    if np.abs(mean_pure) < 1e-6: # Avoid division by zero
        print("WARNING (rpve_custom): Mean intensity of pure tissue is near zero.")
        # If mean_pve is also near zero, ratio is ill-defined or could be 1 (RPVE=0)
        # If mean_pve is not zero, RPVE would be largely negative.
        return -1.0 if np.abs(mean_pve) > 1e-6 else 0.0

    ratio = mean_pve / mean_pure
    rpve = 1.0 - ratio 
    # In MRIQC for brain, GM/WM, WM is brighter. PV between them is darker.
    # mean_PVE < mean_WM -> ratio < 1 -> RPVE > 0.
    # If lens is darker than surrounding globe tissue (for rpve_globe_interface_with_lens):
    # pure_globe_tissue (brighter), pve_interface (mix of lens/globe, darker than pure globe).
    # mean_PVE < mean_pure_globe -> ratio < 1 -> RPVE > 0.
    # This seems reasonable. Values closer to 0 would mean PVE interface has similar intensity to pure tissue.
    return float(rpve)

def globe_sphericity(globe_mask_data, voxel_spacing):
    """
    Calculates the sphericity of a given binary mask (assumed to be the globe).
    Sphericity = (pi^(1/3) * (6 * Volume)^(2/3)) / SurfaceArea

    Args:
        globe_mask_data (np.ndarray): Binary numpy array of the globe segmentation.
        voxel_spacing (tuple or list): Voxel dimensions (e.g., [0.8, 0.8, 0.8]).

    Returns:
        float: Sphericity value (between 0 and 1), or np.nan if calculation fails.
    """
    # Ensure voxel spacing has 3 dimensions
    if len(voxel_spacing) < 3:
        print(f"\tWARNING: Invalid voxel spacing provided for sphericity: {voxel_spacing}. Using [1,1,1].")
        voxel_spacing = (1.0, 1.0, 1.0)
    elif len(voxel_spacing) > 3:
         voxel_spacing = voxel_spacing[:3] # Take only the first 3

    # Check if the globe mask is empty
    if np.sum(globe_mask_data) == 0:
        print(f"\tWARNING: Empty globe mask provided for sphericity calculation.")
        return np.nan

    voxel_volume = np.prod(voxel_spacing)

    # Calculate Volume
    volume = np.sum(globe_mask_data) * voxel_volume

    # Calculate Surface Area using marching cubes
    try:
        # Ensure data is integer type for marching cubes if needed, though bool often works
        verts, faces, _, _ = measure.marching_cubes(
            globe_mask_data.astype(np.uint8), level=0.5, spacing=voxel_spacing
        )
        surface_area = measure.mesh_surface_area(verts, faces)
    except (ValueError, RuntimeError) as e:
         print(f"\tWARNING: Marching cubes/Surface area calculation failed for globe mask: {e}")
         return np.nan

    # Calculate Sphericity
    if surface_area > 1e-6: # Avoid division by zero
        sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area
        # Sphericity should be <= 1. Clamp if minor numerical issues occur.
        sphericity = min(sphericity, 1.0)
    else:
        print(f"\tWARNING: Calculated surface area is near zero for globe mask.")
        sphericity = np.nan

    return sphericity

def lens_aspect_ratio(lens_mask_data, voxel_spacing):
    """
    Calculates the aspect ratio of the lens based on the principal axes
    of the best-fit ellipsoid derived from its inertia tensor.
    The aspect ratio is defined as sqrt(smallest_eigenvalue) / sqrt(largest_eigenvalue)
    of the inertia tensor, which is proportional to minor_axis / major_axis.
    A value closer to 1 indicates a more spherical/circular profile.

    Args:
        lens_mask_data (np.ndarray): Binary numpy array of the LENS segmentation (uint8).
        voxel_spacing (tuple or list): Voxel dimensions (e.g., [0.8, 0.8, 0.8]).

    Returns:
        float: Lens aspect ratio (between 0 and 1), or np.nan if calculation fails.
    """
    if not isinstance(lens_mask_data, np.ndarray) or lens_mask_data.ndim != 3:
        print("\tWARNING (lens_aspect_ratio): lens_mask_data must be a 3D numpy array.")
        return np.nan

    if lens_mask_data.dtype != np.uint8 and lens_mask_data.dtype != bool:
        print(f"\tWARNING (lens_aspect_ratio): lens_mask_data dtype is {lens_mask_data.dtype}. Casting to uint8.")
        lens_mask_data = lens_mask_data.astype(np.uint8)

    # Ensure voxel spacing has 3 dimensions
    if voxel_spacing is None or len(voxel_spacing) != 3:
        print(f"\tWARNING (lens_aspect_ratio): Invalid voxel spacing: {voxel_spacing}. Using [1,1,1].")
        voxel_spacing = (1.0, 1.0, 1.0)

    # Check if the lens mask is empty
    if np.sum(lens_mask_data) == 0:
        print(f"\tWARNING (lens_aspect_ratio): Empty lens mask provided.")
        return np.nan

    try:
        # regionprops_table expects a label image. If mask is binary, it's label 1.
        # Pass voxel_spacing to get geometrically correct properties.
        props = measure.regionprops_table(
            lens_mask_data, # Should be a label image, binary mask works (label 1)
            spacing=voxel_spacing,
            properties=('inertia_tensor_eigvals', 'label') # 'label' to ensure region is found
        )

        if not props['label']: # Check if any region was found
            print("\tWARNING (lens_aspect_ratio): No region found by regionprops_table.")
            return np.nan

        # Eigenvalues of the inertia tensor.
        # These are proportional to the square of the lengths of the principal axes.
        eigvals = np.array([
            props['inertia_tensor_eigvals-0'][0],
            props['inertia_tensor_eigvals-1'][0],
            props['inertia_tensor_eigvals-2'][0]
        ])

        # Filter out near-zero eigenvalues to avoid division by zero or instability
        eigvals = eigvals[eigvals > 1e-6]
        if len(eigvals) < 2: # Need at least two axes to form a ratio
            print("\tWARNING (lens_aspect_ratio): Not enough valid eigenvalues from inertia tensor.")
            return np.nan

        min_eig_sqrt = np.sqrt(np.min(eigvals))
        max_eig_sqrt = np.sqrt(np.max(eigvals))

        if max_eig_sqrt == 0: # Avoid division by zero
            print("\tWARNING (lens_aspect_ratio): Max eigenvalue sqrt is zero.")
            return np.nan

        aspect_ratio = min_eig_sqrt / max_eig_sqrt
        # Aspect ratio should be <= 1. Clamp if minor numerical issues occur.
        aspect_ratio = min(max(aspect_ratio, 0.0), 1.0)

        return aspect_ratio

    except Exception as e:
        print(f"\tERROR (lens_aspect_ratio): Failed calculating lens aspect ratio: {e}")
        import traceback
        traceback.print_exc()
        return np.nan


def summary_stats(data, pvms, airmask=None, erode=True):
    r"""
    Estimates the mean, the median, the standard deviation,
    the kurtosis,the median absolute deviation (mad), the 95\%
    and the 5\% percentiles and the number of voxels (summary\_\*\_n)
    of each tissue distribution.
    .. warning ::
        Sometimes (with datasets that have been partially processed), the air
        mask will be empty. In those cases, the background stats will be zero
        for the mean, median, percentiles and kurtosis, the sum of voxels in
        the other remaining labels for ``n``, and finally the MAD and the
        :math:`\sigma` will be calculated as:
        .. math ::
            \sigma_\text{BG} = \sqrt{\sum \sigma_\text{i}^2}
    """
    from statsmodels.stats.weightstats import DescrStatsW
    from statsmodels.robust.scale import mad

    output = {}
    for label, probmap in pvms.items():
        wstats = DescrStatsW(
            data=data.reshape(-1), weights=probmap.reshape(-1)
        )
        nvox = probmap.sum()
        p05, median, p95 = wstats.quantile(
            np.array([0.05, 0.50, 0.95]),
            return_pandas=False,
        )
        thresholded = data[probmap > (0.5 * probmap.max())]

        output[label] = {
            "mean": float(wstats.mean),
            "median": float(median),
            "p95": float(p95),
            "p05": float(p05),
            "k": float(kurtosis(thresholded)),
            "stdv": float(wstats.std),
            "mad": float(mad(thresholded, center=median)),
            "n": float(nvox),
        }
    return output


def volume_fraction(pvms):
    r"""
    Computes the :abbr:`ICV (intracranial volume)` fractions
    corresponding to the (partial volume maps).
    .. math ::
        \text{ICV}^k = \frac{\sum_i p^k_i}{\sum\limits_{x \in X_\text{brain}} 1}
    :param list pvms: list of :code:`numpy.ndarray` of partial volume maps.
    """
    tissue_vfs = {}
    total = 0
    for k, seg in list(pvms.items()):
        if k == "BG":
            continue
        tissue_vfs[k] = seg.sum()
        total += tissue_vfs[k]

    for k in list(tissue_vfs.keys()):
        tissue_vfs[k] /= total
    return {k: float(v) for k, v in list(tissue_vfs.items())}


def snr(mu_fg, sigma_fg, n):
    r"""
    Calculate the :abbr:`SNR (Signal-to-Noise Ratio)`.
    The estimation may be provided with only one foreground region in
    which the noise is computed as follows:
    .. math::
        \text{SNR} = \frac{\mu_F}{\sigma_F\sqrt{n/(n-1)}},
    where :math:`\mu_F` is the mean intensity of the foreground and
    :math:`\sigma_F` is the standard deviation of the same region.
    :param float mu_fg: mean of foreground.
    :param float sigma_fg: standard deviation of foreground.
    :param int n: number of voxels in foreground mask.
    :return: the computed SNR
    """
    if n < 1 or sigma_fg == 0:
        return np.nan
    return float(mu_fg / (sigma_fg * sqrt(n / (n - 1))))


def cnr(mu_lens, mu_globe, sigma_lens, sigma_globe):
    r"""
    Calculate the :abbr:`CNR (Contrast-to-Noise Ratio)`.
    Adapted for eye tissues (LENS vs GLOBE). Higher values are better.
    .. math::
        \text{CNR} = \frac{|\mu_\text{LENS} - \mu_\text{GLOBE} |}{\sqrt{\sigma_\text{LENS}^2 + \sigma_\text{GLOBE}^2}}

    :param float mu_lens: mean of signal within LENS mask.
    :param float mu_globe: mean of signal within GLOBE mask.
    :param float sigma_lens: standard deviation within LENS mask.
    :param float sigma_globe: standard within GLOBE mask.
    :return: the computed CNR
    """
    denominator = sqrt(sigma_lens**2 + sigma_globe**2)
    if denominator == 0:
        return np.nan
    return float(abs(mu_lens - mu_globe) / denominator)


def cjv(mu_lens, mu_globe, sigma_lens, sigma_globe):
    r"""
    Calculate the :abbr:`CJV (coefficient of joint variation)`, a measure
    related to :abbr:`SNR (Signal-to-Noise Ratio)` and
    :abbr:`CNR (Contrast-to-Noise Ratio)` that is presented as a proxy for
    the :abbr:`INU (intensity non-uniformity)` artifact [Ganzetti2016]_.
    Adapted for eye tissues (LENS vs GLOBE). Lower is better.

    .. math::
        \text{CJV} = \frac{\sigma_\text{LENS} + \sigma_\text{GLOBE}}{|\mu_\text{LENS} - \mu_\text{GLOBE}|}.

    :param float mu_lens: mean of signal within LENS mask.
    :param float mu_globe: mean of signal within GLOBE mask.
    :param float sigma_lens: standard deviation of signal within LENS mask.
    :param float sigma_globe: standard deviation of signal within GLOBE mask.
    :return: the computed CJV
    """
    denominator = abs(mu_lens - mu_globe)
    if denominator == 0:
        return np.nan
    return float((sigma_lens + sigma_globe) / denominator)


# def wm2max(img, mu_wm):
#     r"""
#     Calculate the :abbr:`WM2MAX (white-matter-to-max ratio)`,
#     defined as the maximum intensity found in the volume w.r.t. the
#     mean value of the white matter tissue.
#     Values close to 1.0 are better:
#     .. math ::
#         \text{WM2MAX} = \frac{\mu_\text{WM}}{P_{99.95}(X)}
#     """
#     return float(mu_wm / np.percentile(img.reshape(-1), 99.95))
