# FetMRQC: Quality control for fetal brain MRI
#
# Copyright 2023 Medical Image Analysis Laboratory (MIAL)
#
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
import pdb
import numpy as np
import nibabel as ni # Changed from ni to nib for consistency
import skimage
import traceback
from .utils import (
    normalized_cross_correlation,
    shannon_entropy,
    joint_entropy,
    mutual_information,
    normalized_mutual_information,
    psnr,
    rmse,
    mae,
    nmae,
    nrmse,
    ssim,
    centroid,
    rank_error,
    mask_volume,
    compute_topological_features,
)
from skimage.filters import sobel, laplace
from inspect import getmembers, isfunction
from fetal_brain_qc.utils import squeeze_dim
from scipy.stats import kurtosis, variation
import pandas as pd
from .mriqc_metrics import (
    summary_stats,
    volume_fraction,
    snr,
    cnr,
    cjv,
    globe_sphericity,
    lens_aspect_ratio,
    tissue_to_max_intensity_ratio,
    rpve_custom
)
import sys
from functools import partial
from fetal_brain_utils import get_cropped_stack_based_on_mask # Explicit import
from nilearn.image import resample_img # Explicit import
import warnings
import os

from skimage.morphology import binary_dilation, binary_erosion, binary_closing, disk, ball
from statsmodels import robust

SKIMAGE_FCT = [fct for _, fct in getmembers(skimage.filters, isfunction)]

SEGM = {
    "LENS":             0,
    "GLOBE":            1,
    "OPTIQUE_NERVE":    2,
    "FAT":              3,
    "MUSCLE":           4,
}
segm_names = list(SEGM.keys())

EYE_MAP_SEG = [
    None, # Index 0: Raw Label 0 (Background) -> Ignore
    0,    # Index 1: Raw Label 1 (LENS)       -> Target 0 (LENS)
    1,    # Index 2: Raw Label 2 (GLOBE)      -> Target 1 (GLOBE)
    2,    # Index 3: Raw Label 3 (NERVE)      -> Target 2 (OPTIQUE_NERVE)
    3,    # Index 4: Raw Label 4 (FAT)        -> Target 3 (FAT)
    3,    # Index 5: Raw Label 5 (FAT)        -> Target 3 (FAT)
    4,    # Index 6: Raw Label 6 (MUSCLE)     -> Target 4 (MUSCLE)
    4,    # Index 7: Raw Label 7 (MUSCLE)     -> Target 4 (MUSCLE)
    4,    # Index 8: Raw Label 8 (MUSCLE)     -> Target 4 (MUSCLE)
    4,    # Index 9: Raw Label 9 (MUSCLE)     -> Target 4 (MUSCLE)
]


class SRMetrics:
    """Contains a battery of metrics that can be evaluated on individual
    pairs of super-resolution stacks and segmentations.
    """

    def __init__(
        self,
        verbose=False,
        robust_preprocessing=False,
        correct_bias=False,
        map_seg=EYE_MAP_SEG,
        counter=0,
    ):
        default_params = dict(
            compute_on_mask=True,
            mask_intersection=True,
            use_window=True,
            reduction="mean",
        )

        self.verbose = verbose
        self.robust_prepro = robust_preprocessing
        self.correct_bias = correct_bias
        self.metrics_func = {
            # --- Centroid Metrics (Corrected - Use imported function) ---
            "centroid": partial(centroid, central_third=True),
            "centroid_full": partial(centroid, central_third=False),

            # --- Rank Error Metrics ---
            "rank_error": partial(
                rank_error,
                threshold=0.99,
                relative_rank=False,
            ),
            "rank_error_relative": partial(
                rank_error,
                threshold=0.99,
                relative_rank=True,
            ),

            # --- Mask Volume Metric ---
            "mask_volume": mask_volume, # Assign directly

            # --- Reference-Based Metrics (using process_metric) ---
            "ncc_window": self.process_metric(
                metric=normalized_cross_correlation, **default_params
            ),
            "ncc_median": self.process_metric(
                metric=normalized_cross_correlation,
                mask_intersection=True,
                reduction="median",
            ),
            "joint_entropy_window": self.process_metric(
                metric=joint_entropy, **default_params
            ),
            "joint_entropy_median": self.process_metric(
                metric=joint_entropy,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="median",
            ),
            "mi_window": self.process_metric(
                metric=mutual_information, **default_params
            ),
            "mi_median": self.process_metric(
                metric=mutual_information,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="median",
            ),
            "nmi_window": self.process_metric(
                metric=normalized_mutual_information, **default_params
            ),
            "nmi_median": self.process_metric(
                metric=normalized_mutual_information,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="median",
            ),
            "psnr_window": self.process_metric(
                psnr,
                use_datarange=True,
                **default_params,
            ),
            "nrmse_window": self.process_metric(nrmse, **default_params),
            "rmse_window": self.process_metric(rmse, **default_params),
            "nmae_window": self.process_metric(nmae, **default_params),
            "mae_window": self.process_metric(mae, **default_params),
            "ssim_window": partial(self._ssim, **default_params), # Uses its own wrapper _ssim

            # --- Non-Reference Metrics (using process_metric) ---
            "shannon_entropy": self.process_metric(
                shannon_entropy,
                type="noref",
                compute_on_mask=True,
            ),
            "mean": self.process_metric(
                np.mean,
                type="noref",
                compute_on_mask=True,
            ),
            "std": self.process_metric(
                np.std,
                type="noref",
                compute_on_mask=True,
            ),
            "median": self.process_metric(
                np.median,
                type="noref",
                compute_on_mask=True,
            ),
            "percentile_5": self.process_metric(
                partial(np.percentile, q=5),
                type="noref",
                compute_on_mask=True,
            ),
            "percentile_95": self.process_metric(
                partial(np.percentile, q=95),
                type="noref",
                compute_on_mask=True,
            ),
            "kurtosis": self.process_metric(
                kurtosis,
                type="noref",
                compute_on_mask=True,
            ),
            "variation": self.process_metric(
                variation,
                type="noref",
                compute_on_mask=True,
            ),

            # WM2MAX (Tissue2Max)
            "lens_to_max_intensity": self._lens_to_max_intensity,
            "globe_to_max_intensity": self._globe_to_max_intensity,

            # RPVE
            "rpve_lens_boundary": self._rpve_lens_boundary,
            "rpve_globe_boundary": self._rpve_globe_boundary, # General globe boundary
            "rpve_globe_lens_interface": self._rpve_globe_lens_interface, # Globe at lens interface

            # --- Filter-Based Metrics ---
            "filter_laplace": partial(self._metric_filter, filter=laplace),
            "filter_sobel": partial(self._metric_filter, filter=sobel),

            # --- Segmentation Metrics (Corrected - Direct Assignment) ---
            "seg_sstats": self._seg_sstats,
            "seg_volume": self._seg_volume,
            "seg_snr": self._seg_snr,
            "seg_cnr": self._seg_cnr, # Uses adapted cnr internally
            "seg_cjv": self._seg_cjv, # Uses adapted cjv internally
            # "seg_wm2max": REMOVED
            "seg_topology": self._seg_topology,
            "seg_globe_sphericity": self._seg_globe_sphericity,
            "seg_lens_aspect_ratio": self._seg_lens_aspect_ratio,
        }
        self._metrics = self.get_all_metrics()
        self._check_metrics()
        self.map_seg = map_seg
        self._sstats = None
        self.counter=counter

    # def _create_pve_masks(self, tissue_mask_np: np.ndarray, structure_element_radius: int = 1):
    #     if not np.any(tissue_mask_np):
    #         return None, None
    #     tissue_mask_bin = (tissue_mask_np > 0)
    #     selem = ball(structure_element_radius)
    #     eroded_mask = binary_erosion(tissue_mask_bin, footprint=selem)
    #     pure_mask = eroded_mask
    #     pve_interface_mask = tissue_mask_bin & (~eroded_mask)
    #     if not np.any(pure_mask) or not np.any(pve_interface_mask):
    #         return None, None
    #     return pure_mask, pve_interface_mask

    def _lens_to_max_intensity(self, image, seg_dict, **kwargs):
        lens_mask = seg_dict.get("LENS")
        if lens_mask is None: return np.nan, True
        val = tissue_to_max_intensity_ratio(image, (lens_mask > 0))
        return val, np.isnan(val) or val == -1.0

    def _globe_to_max_intensity(self, image, seg_dict, **kwargs):
        globe_mask = seg_dict.get("GLOBE")
        if globe_mask is None: return np.nan, True
        val = tissue_to_max_intensity_ratio(image, (globe_mask > 0))
        return val, np.isnan(val) or val == -1.0

    def _rpve_lens_boundary(self, image, seg_dict, **kwargs):
        lens_mask = seg_dict.get("LENS")
        if lens_mask is None: return np.nan, True
        pure_lens_mask, pve_lens_boundary_mask = self._create_pve_masks(lens_mask)
        if pure_lens_mask is None or pve_lens_boundary_mask is None:
            return np.nan, True
        val = rpve_custom(image, pure_lens_mask, pve_lens_boundary_mask)
        return val, np.isnan(val) or val == -1.0

    def _rpve_globe_boundary(self, image, seg_dict, **kwargs):
        globe_mask = seg_dict.get("GLOBE")
        if globe_mask is None: return np.nan, True
        pure_globe_mask, pve_globe_boundary_mask = self._create_pve_masks(globe_mask)
        if pure_globe_mask is None or pve_globe_boundary_mask is None:
            return np.nan, True
        val = rpve_custom(image, pure_globe_mask, pve_globe_boundary_mask)
        return val, np.isnan(val) or val == -1.0

    def _rpve_globe_lens_interface(self, image, seg_dict, **kwargs):
        lens_mask = seg_dict.get("LENS")
        globe_mask = seg_dict.get("GLOBE")
        if lens_mask is None or globe_mask is None: return np.nan, True
        lens_mask_bin = (lens_mask > 0)
        globe_mask_bin = (globe_mask > 0)
        pure_globe_region = binary_erosion(globe_mask_bin & (~lens_mask_bin), footprint=ball(1))
        dilated_lens = binary_dilation(lens_mask_bin, footprint=ball(1))
        pve_interface_mask = dilated_lens & globe_mask_bin & (~lens_mask_bin)
        if not np.any(pure_globe_region) or not np.any(pve_interface_mask):
            return np.nan, True
        val = rpve_custom(image, pure_globe_region, pve_interface_mask)
        return val, np.isnan(val) or val == -1.0

    def _seg_lens_aspect_ratio(self, seg_dict, vx_size, **kwargs):
        isnan = False
        if self.verbose: print(f"--- Debugging Lens Aspect Ratio (Counter: {self.counter}) ---")
        if "LENS" not in seg_dict or not isinstance(seg_dict.get("LENS"), np.ndarray):
            if self.verbose: print(f"\tWARNING: LENS segmentation not found or not a numpy array in seg_dict.")
            return 0.0, True
        lens_mask = seg_dict["LENS"]
        if self.verbose:
            print(f"\tInput LENS mask: Sum={np.sum(lens_mask)}, Shape={lens_mask.shape}, Dtype={lens_mask.dtype}")
            print(f"\tInput vx_size: {vx_size}")
        if lens_mask.dtype != np.uint8:
             if self.verbose: print(f"\tCasting LENS mask from {lens_mask.dtype} to uint8.")
             lens_mask = lens_mask.astype(np.uint8)
             if np.sum(lens_mask) == 0:
                  if self.verbose: print(f"\tWARNING: LENS mask became empty after casting to uint8.")
                  return 0.0, True
        if np.sum(lens_mask) == 0:
            if self.verbose: print(f"\tWARNING: LENS mask is empty.")
            return 0.0, True
        try:
            aspect_ratio = lens_aspect_ratio(lens_mask, vx_size)
            if self.verbose: print(f"\tCalculated Lens Aspect Ratio: {aspect_ratio}")
            if np.isnan(aspect_ratio): isnan = True
        except Exception as e:
            if self.verbose:
                print(f"\tERROR: Failed calculating lens aspect ratio via wrapper: {e}")
                traceback.print_exc()
            aspect_ratio = np.nan
            isnan = True
        if self.verbose: print(f"--- End Lens Aspect Ratio Debug ---")
        return aspect_ratio, isnan

    def _seg_globe_sphericity(self, seg_dict, vx_size, **kwargs):
        isnan = False
        sphericity_value_to_return = 0.0
        if self.verbose: print(f"--- Debugging Sphericity (Subject/Counter: {self.counter}) ---")
        if not isinstance(seg_dict, dict):
            if self.verbose: print(f"\tERROR: seg_dict is not a dictionary!")
            return 0.0, True
        globe_mask = seg_dict.get("GLOBE")
        if globe_mask is None:
            if self.verbose: print(f"\tERROR: 'GLOBE' key not found in seg_dict.")
            return 0.0, True
        if not isinstance(globe_mask, np.ndarray):
            if self.verbose: print(f"\tERROR: GLOBE mask is not a numpy array.")
            return 0.0, True
        if self.verbose:
            print(f"\tInput GLOBE mask (pre-cast): Sum={np.sum(globe_mask)}, Shape={globe_mask.shape}, Dtype={globe_mask.dtype}")
            print(f"\tInput vx_size: {vx_size}")
        
        # Verbose checks for float64 mask values (removed for brevity in final response but useful for dev)

        globe_mask_uint8 = globe_mask.astype(np.uint8) if globe_mask.dtype != np.uint8 else globe_mask
        if self.verbose and globe_mask.dtype != np.uint8:
            print(f"\tCasting GLOBE mask from {globe_mask.dtype} to uint8. New Sum: {np.sum(globe_mask_uint8)}")
            if np.sum(globe_mask) > 0 and np.sum(globe_mask_uint8) == 0:
                print(f"\t\tWARNING: GLOBE mask Sum became zero after casting to uint8.")

        if np.sum(globe_mask_uint8) == 0:
            if self.verbose: print(f"\tWARNING: GLOBE mask (uint8) is empty. Original float sum was {np.sum(globe_mask)}.")
            # Optional saving of problematic mask (removed for brevity)
            return 0.0, True
        
        # Optional saving of uint8 mask for sphericity calc (removed for brevity)

        try:
            calculated_sphericity = globe_sphericity(globe_mask_uint8, vx_size)
            if self.verbose: print(f"\tCalculated Sphericity by globe_sphericity(): {calculated_sphericity}")
            if np.isnan(calculated_sphericity):
                isnan = True
                sphericity_value_to_return = 0.0
            else:
                sphericity_value_to_return = calculated_sphericity
                isnan = False
        except Exception as e:
            if self.verbose:
                print(f"\tERROR: Failed calculating sphericity via wrapper: {e}")
                traceback.print_exc()
            sphericity_value_to_return = 0.0
            isnan = True
        if self.verbose: print(f"--- End Sphericity Debug (Subject/Counter: {self.counter}) ---")
        final_sphericity = 0.0 if isnan else sphericity_value_to_return
        return final_sphericity, isnan

    def get_all_metrics(self):
        return list(self.metrics_func.keys())

    def set_metrics(self, metrics):
        self._metrics = metrics

    def get_nan_output(self, metric):
        sstats_keys = ["mean", "median", "p95", "p05", "k", "stdv", "mad", "n"]
        topo_keys = ["b1", "b2", "b3", "ec"]
        if "seg_" in metric:
            if "seg_sstats" in metric:
                return {f"{n}_{k}": np.nan for n in segm_names for k in sstats_keys}
            elif "seg_topology" in metric:
                return {f"{n}_{k}": np.nan for n in segm_names + ["mask"] for k in topo_keys}
            # For seg_volume, seg_snr which return dicts for each segm_name
            elif metric in ["seg_volume", "seg_snr"]:
                 base_dict = {n: np.nan for n in segm_names}
                 if metric == "seg_snr": base_dict["total"] = np.nan
                 return base_dict
            return {m: np.nan for m in segm_names} # Fallback for other dict-based seg metrics
        else:
            return (np.nan, True) # For single value metrics that expect (val, isnan)

    def _flatten_dict(self, d):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update({k + "_" + kk: vv for kk, vv in self._flatten_dict(v).items()})
            else:
                out[k] = v
        return out

    def eval_metrics_and_update_results(self, results, metric, args_dict):
        try:
            out = self.metrics_func[metric](**args_dict)
        except Exception:
            if self.verbose:
                print(f"EXCEPTION with {metric}\n" + traceback.format_exc(), file=sys.stderr)
            out = self.get_nan_output(metric) # This will return (np.nan, True) for single val metrics

        if isinstance(out, dict):
            out_flat = self._flatten_dict(out)
            for k, v in out_flat.items():
                is_v_nan = pd.isna(v)
                results[metric + "_" + k] = 0.0 if is_v_nan else v
                results[metric + "_" + k + "_nan"] = is_v_nan
        else: # Assumes out is a tuple (value, is_nan_flag)
            val, is_nan_flag = out
            default_val = 0.0
            if metric == "seg_cjv" and is_nan_flag: default_val = 1000.0
            results[metric] = default_val if is_nan_flag else val
            results[metric + "_nan"] = is_nan_flag
        return results

    def _scale_intensity_percentiles(self, im, q_low, q_up, to_low, to_up, clip=True):
        from warnings import warn
        if im is None or im.size == 0:
            warn("Image for intensity scaling is empty.", Warning)
            return im
        
        # Filter out non-positive values if they are not meaningful for percentile, e.g. for MRI
        im_positive = im[im > 1e-6] if np.any(im > 1e-6) else im
        if im_positive.size < 2: # Not enough data for percentiles
             a_min = a_max = np.min(im) if im.size > 0 else 0.0 # Fallback
        else:
            a_min: float = np.percentile(im_positive, q_low)
            a_max: float = np.percentile(im_positive, q_up)

        b_min = to_low
        b_max = to_up

        if abs(a_max - a_min) < 1e-9:
            warn("Divide by zero (a_min approx equal to a_max) in intensity scaling.", Warning)
            return np.full_like(im, (b_min + b_max) / 2.0 if (b_min is not None and b_max is not None) else 0.0)

        im_scaled = (im - a_min) / (a_max - a_min)
        if (b_min is not None) and (b_max is not None):
            im_scaled = im_scaled * (b_max - b_min) + b_min
        if clip:
            im_scaled = np.clip(im_scaled, b_min, b_max)
        return im_scaled

    # This is a new helper that needs to be defined, conceptually replacing parts of _preprocess_nifti
    # Or _preprocess_nifti itself needs to be refactored to return these.
    # For now, this is a placeholder showing what _load_and_prep_nifti would expect.
    def _execute_preprocessing_pipeline(self, image_ni_input, mask_ni_for_processing_and_crop, seg_path_to_load,
                                        additional_niftis_to_transform, resample_to):
        """
        Handles preprocessing of the main image, segmentation, and additional NIfTIs
        (like the raw air mask) consistently through scaling, bias correction (optional),
        resampling, and cropping.
        """
        if self.verbose: print(f"\tStarting _execute_preprocessing_pipeline (counter: {self.counter})")

        # 1. Load and format segmentation (if seg_path_to_load provided)
        seg_form_ni = None
        if seg_path_to_load and str(seg_path_to_load).lower() != 'nan':
            try:
                seg_form_ni = self.load_and_format_seg(seg_path_to_load)
                if self.verbose: print(f"\t\tSegmentation loaded and formatted from {seg_path_to_load}")
            except Exception as e:
                if self.verbose: print(f"\t\tWARNING: Could not load/format seg {seg_path_to_load}: {e}")
                seg_form_ni = None # Proceed without segmentation if loading fails

        # 2. Optional Bias Correction on image_ni_input using mask_ni_for_processing_and_crop
        image_ni_current = image_ni_input
        if self.correct_bias:
            if self.verbose: print(f"\t\tApplying N4 Bias Field Correction (counter: {self.counter}).")
            try:
                import SimpleITK as sitk # Ensure SimpleITK is available
                # --- N4 Bias Correction Logic Start (Simplified & adapted from original _preprocess_nifti) ---
                def ni2sitk(im_nifti_obj):
                    im_data = np.asanyarray(im_nifti_obj.dataobj)
                    # Ensure 3D data for SITK
                    if im_data.ndim > 3: im_data = np.squeeze(im_data)[...,0] # Basic squeeze if >3D
                    if im_data.ndim < 3: raise ValueError("Image data has fewer than 3 dimensions for SITK.")
                    
                    im_sitk = sitk.GetImageFromArray(im_data.transpose(2,1,0)) # Assuming RAS input
                    zooms = im_nifti_obj.header.get_zooms()[:3]
                    im_sitk.SetSpacing(tuple(float(z) for z in zooms))
                    origin = im_nifti_obj.affine[:3,3]
                    im_sitk.SetOrigin(tuple(float(o) for o in origin))
                    # Note: Full affine (direction matrix) not set in SITK object here,
                    # relying on data already being in a somewhat standard orientation for N4.
                    return im_sitk

                im_sitk = ni2sitk(image_ni_current)
                # Use mask_ni_for_processing_and_crop as the processing mask for N4
                mask_sitk_internal = ni2sitk(mask_ni_for_processing_and_crop)
                mask_sitk_internal = sitk.Cast(mask_sitk_internal, sitk.sitkUInt8)
                
                im_array_for_stats = sitk.GetArrayFromImage(im_sitk)
                im_max_orig, im_min_orig = im_array_for_stats.max(), im_array_for_stats.min()

                rescaler_to_norm = sitk.RescaleIntensityImageFilter()
                rescaler_to_norm.SetOutputMaximum(1.0)
                rescaler_to_norm.SetOutputMinimum(0.0)
                im_sitk_norm = rescaler_to_norm.Execute(im_sitk)

                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
                corrector.SetSplineOrder(3)
                corrector.SetConvergenceThreshold(1e-7) # MRIQC defaults
                corrector.SetMaximumNumberOfIterations([50] * 4) # 4 levels

                corrected_im_norm = corrector.Execute(im_sitk_norm, mask_sitk_internal)
                
                # Rescale corrected_im_norm to ensure it's [0,1] before mapping back to original range
                corrected_im_temp_rescaled = rescaler_to_norm.Execute(corrected_im_norm)
                img_corrected_array_norm = sitk.GetArrayFromImage(corrected_im_temp_rescaled)
                
                # Map back to original intensity range
                img_final_array = (img_corrected_array_norm * (im_max_orig - im_min_orig)) + im_min_orig
                
                image_ni_current = ni.Nifti1Image(img_final_array.transpose(2,1,0), image_ni_current.affine, image_ni_current.header)
                if self.verbose: print(f"\t\tN4 Bias Field Correction applied (counter: {self.counter}).")
            except ImportError:
                if self.verbose: print("\t\tWARNING: SimpleITK not found. Skipping N4 bias correction.")
            except Exception as e_n4:
                if self.verbose: print(f"\t\tWARNING: N4 bias correction failed: {e_n4}")
                # Continue with uncorrected image_ni_current

        # 3. Intensity Scaling (if self.robust_prepro)
        img_data_current_arr = np.asanyarray(image_ni_current.dataobj)
        mask_data_for_scaling_arr = np.asanyarray(mask_ni_for_processing_and_crop.dataobj)
        
        img_data_to_scale = img_data_current_arr
        if self.robust_prepro:
            if self.verbose: print(f"\t\tApplying robust intensity scaling (counter: {self.counter}).")
            # Apply scaling only within the ROI defined by mask_ni_for_processing_and_crop
            img_data_masked_for_scaling = img_data_current_arr * (mask_data_for_scaling_arr > 0.5).astype(img_data_current_arr.dtype)
            img_data_scaled = self._scale_intensity_percentiles(
                img_data_masked_for_scaling, 0.5, 99.5, 0.0, 1.0, clip=True
            )
            # Put scaled data back, keeping unmasked regions as they were (or zero if masked)
            # This ensures that scaling is only applied to the ROI.
            img_data_to_scale = np.where((mask_data_for_scaling_arr > 0.5), img_data_scaled, img_data_current_arr * (mask_data_for_scaling_arr <= 0.5) )

        image_ni_to_resample = ni.Nifti1Image(img_data_to_scale, image_ni_current.affine, image_ni_current.header)

        # 4. Resampling
        if self.verbose: print(f"\t\tResampling to {resample_to}mm isotropic (counter: {self.counter}).")
        target_zooms_tuple = (resample_to, resample_to, resample_to)
        current_affine_for_resample = image_ni_to_resample.affine
        rotation_part = current_affine_for_resample[:3, :3]
        origin_part = current_affine_for_resample[:3, 3]
        current_voxel_sizes = np.sqrt(np.sum(rotation_part**2, axis=0))
        new_voxel_sizes_arr = np.array(target_zooms_tuple)
        scaled_rotation_matrix = np.zeros_like(rotation_part)
        for i in range(3):
            if current_voxel_sizes[i] > 1e-6: # Avoid division by zero
                scaled_rotation_matrix[:, i] = rotation_part[:, i] * (new_voxel_sizes_arr[i] / current_voxel_sizes[i])
            else: # Fallback if a current voxel size is zero (should be rare for valid NIfTI)
                scaled_rotation_matrix[:, i] = 0 # Or handle error appropriately
        
        target_affine_for_resampling = np.eye(4)
        target_affine_for_resampling[:3, :3] = scaled_rotation_matrix
        target_affine_for_resampling[:3, 3] = origin_part

        image_ni_resampled, mask_ni_resampled, seg_ni_resampled_out = None, None, None
        transformed_additional_niftis_resampled = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                image_ni_resampled = resample_img(image_ni_to_resample, target_affine=target_affine_for_resampling, interpolation='continuous')
                if self.verbose: print(f"\t\tImage resampled, new shape: {image_ni_resampled.shape}")

                mask_ni_resampled = resample_img(mask_ni_for_processing_and_crop, target_affine=target_affine_for_resampling, interpolation='nearest')
                if self.verbose: print(f"\t\tROI mask resampled, new shape: {mask_ni_resampled.shape}")

                if seg_form_ni:
                    seg_ni_resampled_out = resample_img(seg_form_ni, target_affine=target_affine_for_resampling, interpolation='nearest')
                    if self.verbose: print(f"\t\tSegmentation resampled, new shape: {seg_ni_resampled_out.shape if seg_ni_resampled_out else 'None'}")
                
                if additional_niftis_to_transform:
                    for i, add_ni in enumerate(additional_niftis_to_transform):
                        if add_ni is not None:
                            resampled_add_ni = resample_img(add_ni, target_affine=target_affine_for_resampling, interpolation='nearest')
                            transformed_additional_niftis_resampled.append(resampled_add_ni)
                            if self.verbose: print(f"\t\tAdditional NIfTI {i} resampled, new shape: {resampled_add_ni.shape}")
                        else:
                            transformed_additional_niftis_resampled.append(None)
            except Exception as e_resample:
                if self.verbose: print(f"\t\tERROR during resampling: {e_resample}")
                # If resampling fails, we can't proceed with these objects
                return {
                    "image_cropped": None, "mask_cropped": None, "seg_dict_cropped": {},
                    "additional_transformed_cropped": [None]*len(additional_niftis_to_transform or []),
                    "resampled_target_affine": target_affine_for_resampling, # Still return for potential debugging
                    "final_cropping_mask": None
                }


        # 5. Cropping (using the resampled ROI mask)
        if self.verbose: print(f"\t\tCropping all resampled NIfTIs (counter: {self.counter}).")
        # The cropping mask is derived from mask_ni_resampled (the ROI mask)
        final_cropping_mask_data = (np.asanyarray(mask_ni_resampled.dataobj) > 0.5).astype(np.uint8)
        if not np.any(final_cropping_mask_data): # If cropping mask is empty, cropping will fail or produce empty
            if self.verbose: print(f"\t\tWARNING: Final cropping mask is empty. Cropped outputs will be empty/None.")
            # Return None for cropped image/mask, empty for dicts/lists
            return {
                "image_cropped": None, "mask_cropped": None, "seg_dict_cropped": {},
                "additional_transformed_cropped": [None]*len(additional_niftis_to_transform or []),
                "resampled_target_affine": target_affine_for_resampling,
                "final_cropping_mask": ni.Nifti1Image(final_cropping_mask_data, mask_ni_resampled.affine, mask_ni_resampled.header)
            }

        final_cropping_mask_ni_obj = ni.Nifti1Image(final_cropping_mask_data, mask_ni_resampled.affine, mask_ni_resampled.header)
        crop_params = {'boundary_i': 5, 'boundary_j': 5, 'boundary_k': 5}

        imagec_ni_cropped = get_cropped_stack_based_on_mask(image_ni_resampled, final_cropping_mask_ni_obj, **crop_params)
        maskc_ni_cropped = get_cropped_stack_based_on_mask(mask_ni_resampled, final_cropping_mask_ni_obj, **crop_params)

        seg_dict_ni_cropped = {}
        if seg_ni_resampled_out:
            seg_data_resampled_arr = np.asanyarray(seg_ni_resampled_out.dataobj)
            for k_seg, l_target_seg in SEGM.items(): # Assuming SEGM is available
                tissue_mask_data = (seg_data_resampled_arr == l_target_seg).astype(np.uint8)
                tissue_nifti = ni.Nifti1Image(tissue_mask_data, seg_ni_resampled_out.affine, seg_ni_resampled_out.header)
                seg_dict_ni_cropped[k_seg] = get_cropped_stack_based_on_mask(tissue_nifti, final_cropping_mask_ni_obj, **crop_params)
        
        transformed_additional_niftis_cropped = []
        if transformed_additional_niftis_resampled:
            for add_ni_resampled in transformed_additional_niftis_resampled:
                if add_ni_resampled is not None:
                    transformed_additional_niftis_cropped.append(
                        get_cropped_stack_based_on_mask(add_ni_resampled, final_cropping_mask_ni_obj, **crop_params)
                    )
                else:
                    transformed_additional_niftis_cropped.append(None)
        
        if self.verbose: print(f"\tFinished _execute_preprocessing_pipeline (counter: {self.counter})")
        return {
            "image_cropped": imagec_ni_cropped,
            "mask_cropped": maskc_ni_cropped,
            "seg_dict_cropped": seg_dict_ni_cropped,
            "additional_transformed_cropped": transformed_additional_niftis_cropped,
            "resampled_target_affine": target_affine_for_resampling,
            "final_cropping_mask": final_cropping_mask_ni_obj
        }
    
    # This function is part of the SRMetrics class in fetmrqc_sr/metrics/metrics_sr.py
    def _load_and_prep_nifti(self, sr_path, mask_path, seg_path, resample_to):
        # Load initial images
        image_ni = ni.load(sr_path)
        # zero_fill the Nan values
        image_ni = ni.Nifti1Image(
            np.nan_to_num(image_ni.get_fdata()),
            image_ni.affine,
            image_ni.header,
        )
        mask_ni = ni.load(mask_path) # Will load the seg file if mask path points to it
        seg_ni_orig = ni.load(seg_path) # Load the original segmentation

        # Create combined mask: Use loaded mask + binarized segmentation
        # Ensure mask_ni is treated as binary if it's the same as seg_ni_orig
        if mask_path == seg_path:
             mask_data_for_combine = (mask_ni.get_fdata() > 0).astype(int)
        else:
             mask_data_for_combine = mask_ni.get_fdata()

        # Combine and clip
        mask_combined_data = np.clip(
            mask_data_for_combine + (seg_ni_orig.get_fdata() > 0).astype(int), 0, 1
        )
        # Use combined mask with original mask's affine/header for consistency before preprocessing
        mask_ni_combined = ni.Nifti1Image(mask_combined_data, mask_ni.affine, mask_ni.header)

        # Preprocess image, combined mask, and segmentation
        # Pass the original seg_path to _preprocess_nifti, it handles loading/mapping/resampling seg internally
        imagec_ni, maskc_ni, seg_dict_ni = self._preprocess_nifti(
            image_ni, mask_ni_combined, seg_path, resample_to, robust=self.robust_prepro, bias_corr=self.correct_bias
        )
        # Note: _preprocess_nifti now returns NIfTI objects after cropping

        # Convert final cropped outputs to numpy arrays with correct orientation
        def squeeze_flip_tr(nifti_obj):
            """Squeeze_dim returns a numpy array"""
            if nifti_obj is None: return None
            # Make sure data is loaded if it's a proxy
            data = np.asanyarray(nifti_obj.dataobj)
            return squeeze_dim(data, -1)[::-1, ::-1, ::-1].transpose(2, 1, 0)

        imagec_np = squeeze_flip_tr(imagec_ni)
        maskc_np = squeeze_flip_tr(maskc_ni)
        # Apply squeeze_flip_tr to the data within the NIfTI objects in seg_dict_ni
        seg_dict_np = {
            k: squeeze_flip_tr(v_ni) for k, v_ni in seg_dict_ni.items() if v_ni is not None
        }

        # <<< --- START DEBUG BLOCK --- >>>
        # This block was present in your uploaded file for debugging purposes.
        # You can keep or remove it as needed.
        print(f"\n--- Debugging _load_and_prep_nifti Results for SR: {sr_path} ---")
        print(f"Image Cropped Shape: {imagec_np.shape if imagec_np is not None else 'None'}")
        print(f"Mask Cropped Shape: {maskc_np.shape if maskc_np is not None else 'None'}")
        print(f"Seg Dict Keys: {list(seg_dict_np.keys())}")
        for k, v_np in seg_dict_np.items():
             if v_np is not None:
                  print(f"Tissue '{k}': Sum = {np.sum(v_np)}, Shape = {v_np.shape}, Dtype = {v_np.dtype}")
             else:
                  print(f"Tissue '{k}': Mask is None")
        print(f"--- End Debugging --- \n")
        # <<< --- END DEBUG BLOCK --- >>>

        # Return numpy arrays
        return imagec_np, maskc_np, seg_dict_np

    # This function should be within your SRMetrics class in fetmrqc_sr/metrics/metrics_sr.py

    def evaluate_metrics(self, sr_path, mask_path, seg_path):
        """Evaluate the metrics for a given image, mask, and optional segmentation.

        Args:
            sr_path (str): Path to the input image.
            mask_path (str): Path to the brain/ROI mask.
            seg_path (str, optional): Path to the segmentation file. Defaults to None.

        Returns:
            dict: Dictionary containing the results of the metrics.
        """
        self._sstats = None # Reset summary statistics for each evaluation

        resample_to = 0.8 # Define resampling target voxel size

        # Load and preprocess image, mask, and segmentation data
        # This returns imagec (cropped image data), maskc (cropped mask data),
        # and seg_dict (dictionary of segmented tissue masks) as numpy arrays.
        imagec, maskc, seg_dict = self._load_and_prep_nifti(
            sr_path, mask_path, seg_path, resample_to
        )

        # Prepare dictionary of arguments to pass to individual metric functions
        args_dict = {
            "image": imagec,
            "mask": maskc,  # This is the (cropped) input ROI mask
            # "generated_foreground_mask": was removed
            "seg_dict": seg_dict,
            "vx_size": [resample_to] * 3,
            "sr_path_for_debug": sr_path # Useful for debugging specific cases
        }

        results = {}

        # Handle cases where imagec itself might be None (e.g., _load_and_prep_nifti failed)
        if imagec is None:
            print(f"WARNING: Image data (imagec) is None for {sr_path}. Cannot calculate IQMs. Returning NaNs.")
            # Populate results with NaNs for all metrics
            for m in self._metrics: # self._metrics should be the filtered list from definitions.py
                nan_output = self.get_nan_output(m)
                if isinstance(nan_output, dict):
                    for k_nan, v_nan in nan_output.items():
                        # Ensure correct naming for flattened dicts from get_nan_output
                        results[f"{m}_{k_nan}"] = v_nan # Should be np.nan or 0.0
                        results[f"{m}_{k_nan}_nan"] = True
                else: # Assuming (val, isnan_flag) structure or similar simple list/tuple
                    default_val = np.nan
                    if isinstance(nan_output, (list, tuple)) and len(nan_output) > 0:
                        default_val = nan_output[0]
                    results[m] = default_val
                    results[m + "_nan"] = True
            return results

        # Check for segmentation metrics if seg_path or seg_dict is problematic
        # This check is important if segmentation-based IQMs are still in self._metrics
        if any(["seg_" in m for m in self._metrics]):
            if seg_path is None:
                 print(f"INFO: Segmentation path not provided for {sr_path}, but segmentation metrics requested. These metrics will be NaN.")
            elif seg_dict is None or not seg_dict : # Check if seg_dict is empty or None
                 print(f"INFO: Segmentation dictionary (seg_dict) is empty or None for {sr_path}. Segmentation metrics will be NaN.")


        # Loop through the metrics to be evaluated
        for m in self._metrics: # self._metrics should now reflect the reduced list of IQMs
            if self.verbose:
                print("\tRunning", m)
            
            # Skip segmentation metrics if seg_dict is not valid and current metric is a seg metric
            # This condition ensures we don't try to run seg metrics if there's no valid seg data
            is_seg_metric = "seg_" in m
            seg_data_is_problematic = (seg_dict is None or not seg_dict)

            if is_seg_metric and seg_data_is_problematic:
                if self.verbose:
                    print(f"\tSkipping segmentation metric {m} due to missing/empty segmentation data.")
                nan_output = self.get_nan_output(m)
                if isinstance(nan_output, dict):
                    for k_nan, v_nan in nan_output.items():
                        results[f"{m}_{k_nan}"] = v_nan
                        results[f"{m}_{k_nan}_nan"] = True
                else:
                    default_val = np.nan
                    if isinstance(nan_output, (list, tuple)) and len(nan_output) > 0:
                        default_val = nan_output[0]
                    results[m] = default_val
                    results[m + "_nan"] = True
                continue # Skip to next metric

            # Call the helper function to evaluate the metric and update results
            results = self.eval_metrics_and_update_results(
                results, m, args_dict
            )
            
        return results
    
    def _valid_mask(self, mask_path): # Make sure to use nib
        try:
            mask_obj = ni.load(mask_path)
            mask = np.asanyarray(mask_obj.dataobj)
            return mask.sum() != 0
        except Exception as e:
            if self.verbose: print(f"Error validating mask {mask_path}: {e}")
            return False
        
    def _check_metrics(self):
        """Check that the metrics are valid."""

        for m in self._metrics:
            if m not in self.metrics_func.keys():
                raise RuntimeError(
                    f"Metric {m} is not part of the available metrics."
                )

    def process_metric(
        self,
        metric,
        type="ref",
        **kwargs,
    ):
        """Wrapper to process the different categories of metrics.

        Args:
            metric (str): Name of the metric (in the list of available metrics).
            type (str, optional): Type of metric. Defaults to "ref". Available types are:
                - "ref": metric that is computed by comparing neighbouring slices.
                - "noref": metric that relies on individual slices
                - "seg": metric that make use of a segmentation.
            **kwargs: Additional processing to be done before evaluating the metric, detailed in the docstring of the corresponding function.
        """

        if type == "ref":
            return partial(
                self.preprocess_and_evaluate_metric, metric=metric, **kwargs
            )
        elif type == "noref":
            return partial(
                self.preprocess_and_evaluate_noref_metric,
                noref_metric=metric,
                **kwargs,
            )
        elif type == "seg":
            return partial(
                self.preprocess_and_evaluate_seg_metric,
                seg_metric=metric,
                **kwargs,
            )
        else:
            raise RuntimeError(
                f"Unknown metric type {type}. Please choose among ['ref', 'noref', 'seg']"
            )

    def load_and_format_seg(self, seg_path):
        """Load segmentation and format it to be used by the metrics"""

        seg_path = str(seg_path).strip()
        if seg_path.endswith(".nii.gz"):
            seg_ni = ni.load(seg_path)
            seg = squeeze_dim(seg_ni.get_fdata(), -1).astype(np.uint8)
            seg_remapped = np.zeros_like(seg)
            for label, target in enumerate(self.map_seg):
                if target is not None:
                    seg_remapped[seg == label] = target
            seg_ni = ni.Nifti1Image(
                seg_remapped,
                seg_ni.affine,
            )
        else:
            raise ValueError(
                f"Unknown file format for segmentation file {seg_path}"
            )
        # We cannot return a nifti object as seg_path might be .npz
        return seg_ni

    def _scale_intensity_percentiles(
        self, im, q_low, q_up, to_low, to_up, clip=True
    ):
        from warnings import warn

        a_min: float = np.percentile(im, q_low)  # type: ignore
        a_max: float = np.percentile(im, q_up)  # type: ignore
        b_min = to_low
        b_max = to_up

        if a_max - a_min == 0.0:
            warn("Divide by zero (a_min == a_max)", Warning)
            if b_min is None:
                return im - a_min
            return im - a_min + b_min

        im = (im - a_min) / (a_max - a_min)
        if (b_min is not None) and (b_max is not None):
            im = im * (b_max - b_min) + b_min
        if clip:
            im = np.clip(im, b_min, b_max)

        return im

    def _preprocess_nifti(
        self, im_ni, mask_ni, seg_path, resample_to=0.8, mask_im=True, robust=True, bias_corr=False
    ):
        from nilearn.image import resample_img
        # ni.save(im_ni, f"test_{self.counter}.nii.gz")
        seg = self.load_and_format_seg(seg_path)
        if bias_corr:
            import SimpleITK as sitk
            def ni2sitk(im):
                im_sitk = sitk.GetImageFromArray(im.get_fdata().transpose(2,1,0))
                x,y,z = im.header.get_zooms()
                im_sitk.SetSpacing((float(x), float(y), float(z)))
                # get qform
                x,y,z = im.affine[:3,3]
                im_sitk.SetOrigin((float(x), float(y), float(z)))
                return im_sitk

            im_sitk = ni2sitk(im_ni)
            mask_sitk = ni2sitk(mask_ni)
            mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)
            im_max, im_min = sitk.GetArrayFromImage(im_sitk).max(), sitk.GetArrayFromImage(im_sitk).min()
            # Normalize im with sitk
            rescaler = sitk.RescaleIntensityImageFilter()
            rescaler.SetOutputMaximum(1)
            rescaler.SetOutputMinimum(0)
            im_sitk = rescaler.Execute(im_sitk)

            # Correct bias field
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetBiasFieldFullWidthAtHalfMaximum(0.25)
            corrector.SetSplineOrder(3)
            corrected_im = corrector.Execute(im_sitk, mask_sitk)
            corrected_im = rescaler.Execute(corrected_im)
            img  = sitk.GetArrayFromImage(corrected_im)
            #Map back to original range
            img = (img  * (im_max - im_min)) + im_min
            im_ni = ni.Nifti1Image(img.transpose(2,1,0), im_ni.affine, im_ni.header)
        # ni.save(im_ni, f"test_{self.counter}_after_bias.nii.gz")

        img = (
            im_ni.get_fdata() * mask_ni.get_fdata()
            if mask_im
            else im_ni.get_fdata()
        )
        
        
        if robust:
            img = self._scale_intensity_percentiles(
                img, 0.5, 99.5, 0, 1, clip=True
            )

            new_affine = np.diag([resample_to] * 3)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image_ni = resample_img(
                    ni.Nifti1Image(
                        img,
                        im_ni.affine,
                        im_ni.header,
                    ),
                    target_affine=new_affine,
                )
                mask_ni = ni.Nifti1Image(
                    mask_ni.get_fdata(), im_ni.affine, im_ni.header
                )
                mask_ni = resample_img(
                    mask_ni, target_affine=new_affine, interpolation="nearest"
                )
                seg = resample_img(
                    seg, target_affine=new_affine, interpolation="nearest"
                )
        else:
            image_ni = ni.Nifti1Image(
                        img,
                        im_ni.affine,
                        im_ni.header,
                    )
            

        def crop_stack(x, y):
            return get_cropped_stack_based_on_mask(
                x,
                y,
                boundary_i=5,
                boundary_j=5,
                boundary_k=5,
            )

        # seg_dict = {k: (seg == l).astype(np.uint8) for k, l in SEGM.items()}

        seg_dict = {
            k: crop_stack(
                ni.Nifti1Image(
                    (seg.get_fdata() == l).astype(np.uint8),
                    image_ni.affine,
                    image_ni.header,
                ),
                mask_ni,
            )
            for k, l in SEGM.items()
        }
        imagec = crop_stack(image_ni, mask_ni)
        #ni.save(imagec, f"test_{self.counter}_postproc.nii.gz")
        maskc = crop_stack(mask_ni, mask_ni)
        return imagec, maskc, seg_dict

    '''def _load_and_prep_nifti(self, sr_path, mask_path, seg_path, resample_to):
        image_ni = ni.load(sr_path)
        # zero_fill the Nan values
        image_ni = ni.Nifti1Image(
            np.nan_to_num(image_ni.get_fdata()),
            image_ni.affine,
            image_ni.header,
        )
        mask_ni = ni.load(mask_path)
        seg_ni = ni.load(seg_path)
        mask = np.clip(
            mask_ni.get_fdata() + (seg_ni.get_fdata() > 0).astype(int), 0, 1
        )
        mask_ni = ni.Nifti1Image(mask, mask_ni.affine, mask_ni.header)
        imagec, maskc, seg_dict = self._preprocess_nifti(
            image_ni, mask_ni, seg_path, resample_to, robust=self.robust_prepro, bias_corr=self.correct_bias
        )

        def squeeze_flip_tr(x):
            """Squeeze_dim returns a numpy array"""
            return squeeze_dim(x, -1)[::-1, ::-1, ::-1].transpose(2, 1, 0)

        if imagec is not None:
            imagec = squeeze_flip_tr(imagec.get_fdata())
            maskc = squeeze_flip_tr(maskc.get_fdata())
            seg_dict = {
                k: squeeze_flip_tr(v.get_fdata()) for k, v in seg_dict.items()
            }
        return imagec, maskc, seg_dict'''


    def _remove_empty_slices(self, image, mask):
        s = np.flatnonzero(
            mask.sum(
                axis=(
                    1,
                    2,
                )
            )
            > 50  # Remove negligible amounts
        )
        imin, imax = s[0], s[-1]
        return image[imin:imax], mask[imin:imax]

    def _ssim(
        self,
        image,
        mask,
        seg_path=None,
        compute_on_mask=True,
        mask_intersection=True,
        reduction="mean",
        use_window=True,
        window_size=3,
        **kwargs,
    ):

        if (
            image is None
            or mask is None
            or any([s < 7 for s in image.shape[1:]])
        ):
            # image is None when the mask is empty: nothing is computed.
            # Similarly, return None when the image is of size smaller than 7
            return np.nan, True
        metric_out = []
        isnan = False

        image, mask = self._remove_empty_slices(image, mask)

        datarange = image[mask > 0].max() - image[mask > 0].min()
        for i, img_i in enumerate(image):
            if use_window:
                l, r = window_size // 2, window_size - window_size // 2
                range_j = range(max(0, i - l), min(image.shape[0], i + r))
            else:
                range_j = range(0, image.shape[0])
            for j in range_j:
                im_i = img_i
                im_j = image[j]
                mask_curr = (
                    mask[i] * mask[j]
                    if mask_intersection
                    else ((mask[i] + mask[j]) > 0).astype(int)
                )
                m = (
                    ssim(im_i, im_j, mask_curr, datarange)
                    if compute_on_mask
                    else ssim(im_i, im_j, datarange)
                )
                # Try to not consider self-correlation
                if not np.isnan(m) and i != j:
                    metric_out.append(m)
                if np.isnan(m):
                    isnan = True

        if reduction == "mean":
            return np.mean(metric_out), isnan
        elif reduction == "median":
            return np.median(metric_out), isnan

    '''    def _seg_sstats(self, image, segmentation):
        self._sstats = summary_stats(image, segmentation)
        return self._sstats'''
    # Inside SRMetrics class
    def _seg_sstats(self, image, seg_dict, **kwargs):
        print("--- Debugging SStats ---")
        segmentation_uint8 = {}
        valid_masks = True
        for k, mask_float in seg_dict.items():
            if not isinstance(mask_float, np.ndarray):
                print(f"\tWARNING: Mask for {k} is not a numpy array in seg_dict.")
                valid_masks = False
                break
            print(f"\tSStats Input {k}: Sum={np.sum(mask_float)}, Dtype={mask_float.dtype}")
            segmentation_uint8[k] = mask_float.astype(np.uint8)
            if np.sum(segmentation_uint8[k]) == 0:
                print(f"\tWARNING: Mask for {k} is empty after casting.")
                # Decide how to handle - maybe skip stats for this tissue?
                # Or allow summary_stats to potentially handle empty weights?

        if not valid_masks:
            # Return NaN for all expected outputs if any mask was invalid
            # Need to construct the expected dictionary structure with NaNs
            # This part depends on how get_nan_output is structured for seg_sstats
            print("\tERROR: Invalid masks found, returning NaN for sstats.")
            return self.get_nan_output("seg_sstats") # Assuming this returns the dict structure with NaNs

        try:
            # Pass the dictionary with uint8 masks
            self._sstats = summary_stats(image, segmentation_uint8)
            print(f"\tSummary Stats Calculated: {list(self._sstats.keys())}")
        except Exception as e:
            print(f"\tERROR calculating summary_stats: {e}")
            import traceback
            traceback.print_exc()
            self._sstats = self.get_nan_output("seg_sstats") # Return structure with NaNs

        print("--- End SStats Debug ---")
        # The evaluate_metrics loop handles flattening and NaN flags
        return self._sstats
    '''    def _seg_volume(self, image, segmentation):
        return volume_fraction(segmentation)'''
    
    def _seg_volume(self, seg_dict, **kwargs):
        print("--- Debugging Volume ---")
        segmentation_uint8 = {}
        valid_masks = True
        for k, mask_float in seg_dict.items():
            if not isinstance(mask_float, np.ndarray):
                print(f"\tWARNING: Mask for {k} is not a numpy array.")
                valid_masks = False
                break
            print(f"\tVolume Input {k}: Sum={np.sum(mask_float)}, Dtype={mask_float.dtype}")
            segmentation_uint8[k] = mask_float.astype(np.uint8)

        if not valid_masks:
            print("\tERROR: Invalid masks found, returning NaN for volume.")
            return self.get_nan_output("seg_volume")

        try:
            vol_frac = volume_fraction(segmentation_uint8)
            print(f"\tVolume Fractions Calculated: {vol_frac}")
        except Exception as e:
            print(f"\tERROR calculating volume_fraction: {e}")
            import traceback
            traceback.print_exc()
            vol_frac = self.get_nan_output("seg_volume")

        print("--- End Volume Debug ---")
        return vol_frac
    '''    def _seg_snr(self, image, segmentation):
        if self._sstats is None:
            self._sstats = summary_stats(image, segmentation)
        snr_dict = {}
        for tlabel in segmentation.keys():
            snr_dict[tlabel] = snr(
                self._sstats[tlabel]["median"],
                self._sstats[tlabel]["stdv"],
                self._sstats[tlabel]["n"],
            )
        snr_dict["total"] = float(np.mean(list(snr_dict.values())))
        return snr_dict'''

    # Inside SRMetrics class
    def _seg_snr(self, image, seg_dict, **kwargs): # Added image, seg_dict args
        print("--- Debugging SNR ---")
        # Ensure sstats are calculated (and handle potential previous errors)
        if self._sstats is None or not isinstance(self._sstats, dict) or not self._sstats:
            print("\tWARNING: _sstats not available or invalid, attempting recalculation.")
            # Use the already prepared uint8 casting logic from _seg_sstats if possible
            # Or recalculate here, ensuring uint8 casting
            self._sstats = self._seg_sstats(image, seg_dict, **kwargs) # Recalculate
            if not isinstance(self._sstats, dict) or not self._sstats:
                print("\tERROR: Failed to get valid _sstats for SNR calculation.")
                return self.get_nan_output("seg_snr") # Return NaN structure

        snr_dict = {}
        all_snr_nan = True
        for tlabel in seg_dict.keys(): # Iterate through expected labels
            # Check if stats exist for this label
            if tlabel not in self._sstats or not isinstance(self._sstats[tlabel], dict):
                print(f"\tWARNING: Stats for {tlabel} not found in _sstats for SNR.")
                snr_dict[tlabel] = np.nan
                continue # Skip calculation for this label

            stats = self._sstats[tlabel]
            # Check for necessary keys within the stats
            if not all(k in stats for k in ["median", "stdv", "n"]):
                print(f"\tWARNING: Missing required keys in stats for {tlabel}.")
                snr_dict[tlabel] = np.nan
                continue

            try:
                # Calculate SNR using the function from mriqc_metrics
                snr_val = snr(
                    stats["median"],
                    stats["stdv"],
                    stats["n"],
                )
                snr_dict[tlabel] = snr_val
                if not np.isnan(snr_val):
                    all_snr_nan = False
                print(f"\tSNR {tlabel}: {snr_val}")
            except Exception as e:
                print(f"\tERROR calculating SNR for {tlabel}: {e}")
                snr_dict[tlabel] = np.nan

        print("--- End SNR Debug ---")

        # Return the dictionary, NaNs will be handled by the main loop
        return snr_dict

    '''    def _seg_cnr(self, image, segmentation):
        if self._sstats is None:
            self._sstats = summary_stats(image, segmentation)
        out = cnr(
            self._sstats["LENS"]["median"],
            self._sstats["GLOBE"]["median"],
            self._sstats["LENS"]["stdv"],
            self._sstats["GLOBE"]["stdv"],
        )
        is_nan = np.isnan(out)
        return 0.0 if is_nan else out, is_nan'''

    # Inside SRMetrics class
    def _seg_cnr(self, image, seg_dict, **kwargs): # Added args
        print("--- Debugging CNR ---")
        # Ensure sstats are calculated correctly first
        if self._sstats is None or not isinstance(self._sstats, dict) or not self._sstats:
            print("\tWARNING: _sstats not available or invalid for CNR, attempting recalculation.")
            self._sstats = self._seg_sstats(image, seg_dict, **kwargs) # Recalculate
            if not isinstance(self._sstats, dict) or not self._sstats:
                print("\tERROR: Failed to get valid _sstats for CNR calculation.")
                return np.nan, True # Return NaN and True flag

        # Check if LENS and GLOBE stats exist and are valid dictionaries
        if not all(k in self._sstats and isinstance(self._sstats[k], dict) for k in ["LENS", "GLOBE"]):
            print("\tWARNING: LENS or GLOBE statistics not found/valid in _sstats for CNR calculation.")
            return np.nan, True

        # Check for necessary keys within LENS and GLOBE stats
        required_keys = ["median", "stdv"]
        if not all(key in self._sstats["LENS"] for key in required_keys) or \
        not all(key in self._sstats["GLOBE"] for key in required_keys):
            print("\tWARNING: Missing 'median' or 'stdv' in LENS/GLOBE stats for CNR.")
            return np.nan, True

        try:
            out = cnr(
                self._sstats["LENS"]["median"],   # Correct key
                self._sstats["GLOBE"]["median"],  # Correct key
                self._sstats["LENS"]["stdv"],     # Correct key
                self._sstats["GLOBE"]["stdv"],    # Correct key
            )
            is_nan = np.isnan(out)
            print(f"\tCalculated CNR: {out}")
        except Exception as e:
            print(f"\tERROR calculating CNR: {e}")
            import traceback
            traceback.print_exc()
            out = np.nan
            is_nan = True

        print("--- End CNR Debug ---")
        # Return 0 if NaN, otherwise the value, plus NaN flag
        return 0.0 if is_nan else out, is_nan

    '''    def _seg_cjv(self, image, segmentation):
        if self._sstats is None:
            self._sstats = summary_stats(image, segmentation)
        out = cjv(
            # mu_wm, mu_gm, sigma_wm, sigma_gm
            self._sstats["LENS"]["median"],
            self._sstats["GLOBE"]["median"],
            self._sstats["LENS"]["mad"],
            self._sstats["GLOBE"]["mad"],
        )
        is_nan = np.isnan(out)
        return 1000 if is_nan else out, is_nan'''
    
    # Inside SRMetrics class
    def _seg_cjv(self, image, seg_dict, **kwargs): # Added args
        print("--- Debugging CJV ---")
        # Ensure sstats are calculated correctly first
        if self._sstats is None or not isinstance(self._sstats, dict) or not self._sstats:
            print("\tWARNING: _sstats not available or invalid for CJV, attempting recalculation.")
            self._sstats = self._seg_sstats(image, seg_dict, **kwargs) # Recalculate
            if not isinstance(self._sstats, dict) or not self._sstats:
                print("\tERROR: Failed to get valid _sstats for CJV calculation.")
                return np.nan, True # Return NaN and True flag

        # Check if LENS and GLOBE stats exist and are valid dictionaries
        if not all(k in self._sstats and isinstance(self._sstats[k], dict) for k in ["LENS", "GLOBE"]):
            print("\tWARNING: LENS or GLOBE statistics not found/valid in _sstats for CJV calculation.")
            return np.nan, True

        # Check for necessary keys within LENS and GLOBE stats (using MAD here based on original code)
        required_keys = ["median", "mad"] # Original CJV used MAD, ensure it's calculated in summary_stats
        if not all(key in self._sstats["LENS"] for key in required_keys) or \
        not all(key in self._sstats["GLOBE"] for key in required_keys):
            print("\tWARNING: Missing 'median' or 'mad' in LENS/GLOBE stats for CJV.")
            # Fallback to stdv if mad is missing? Or return NaN? Let's return NaN for now.
            # Alternatively, modify summary_stats to ensure 'mad' is always present or use 'stdv' here.
            # Check your summary_stats implementation if 'mad' calculation might fail.
            return np.nan, True

        try:
            # Using MAD based on original implementation of CJV
            out = cjv(
                self._sstats["LENS"]["median"],   # Correct key
                self._sstats["GLOBE"]["median"],  # Correct key
                self._sstats["LENS"]["mad"],      # Using MAD
                self._sstats["GLOBE"]["mad"],     # Using MAD
            )
            is_nan = np.isnan(out)
            print(f"\tCalculated CJV: {out}")
        except Exception as e:
            print(f"\tERROR calculating CJV: {e}")
            import traceback
            traceback.print_exc()
            out = np.nan
            is_nan = True

        print("--- End CJV Debug ---")
        # Return default value if NaN (original code returned 1000 for NaN), plus NaN flag
        default_cjv = 1000.0 # Match original behavior on NaN if desired
        return default_cjv if is_nan else out, is_nan

    # def _seg_wm2max(self, image, segmentation):
    #     if self._sstats is None:
    #         self._sstats = summary_stats(image, segmentation)
    #     out = wm2max(image, self._sstats["WM"]["median"])
    #     is_nan = np.isnan(out)
    #     return 0.0 if is_nan else out, is_nan

    '''    def _seg_topology(self, image, segmentation):
        topo_dict = {}
        
        for tlabel in segmentation.keys():
            betti_numbers, ec = compute_topological_features(
                segmentation[tlabel]
            )
            b1, b2, b3 = betti_numbers
            topo_dict[f"{tlabel}_b1"] = b1
            topo_dict[f"{tlabel}_b2"] = b2
            topo_dict[f"{tlabel}_b3"] = b3
            topo_dict[f"{tlabel}_ec"] = ec
        mask = np.zeros_like(image)
        for tlabel in segmentation.keys():
            mask += segmentation[tlabel]

        betti_numbers, ec = compute_topological_features(mask)
        topo_dict["mask_b1"] = betti_numbers[0]
        topo_dict["mask_b2"] = betti_numbers[1]
        topo_dict["mask_b3"] = betti_numbers[2]
        topo_dict["mask_ec"] = ec
        return topo_dict'''
    
    def _seg_topology(self, seg_dict, **kwargs):
        print(f"--- Debugging Topology ---")
        # Ensure compute_topological_features is imported if not already at top
        # from .utils import compute_topological_features

        segmentation = seg_dict # Use the passed dictionary

        topo_keys = ["b1", "b2", "b3", "ec"]
        all_metric_keys = {} # Initialize empty dictionary

        # Process individual tissue masks
        processed_labels = list(segmentation.keys())
        for tlabel in processed_labels:
            # Initialize metrics for this label with 0
            for tk in topo_keys:
                all_metric_keys[f"{tlabel}_{tk}"] = 0

            current_mask_float = segmentation.get(tlabel)

            if not isinstance(current_mask_float, np.ndarray):
                print(f"\tWARNING: Mask for {tlabel} is not a numpy array. Skipping topology.")
                continue

            # --- Explicitly cast to uint8 ---
            if current_mask_float.dtype != np.uint8:
                current_mask = current_mask_float.astype(np.uint8)
            else:
                current_mask = current_mask_float

            if np.sum(current_mask) == 0:
                print(f"\tSkipping topology for empty mask {tlabel}.")
                continue # Skip, leaving the initialized 0s

            # --- Call compute_topological_features ---
            try:
                # compute_topological_features now returns integers or 0s on error
                betti_numbers, ec = compute_topological_features(current_mask)
                print(f"\tTissue {tlabel}: Betti={betti_numbers}, EC={ec}")
                # Assign results (already integers or 0)
                all_metric_keys[f"{tlabel}_b1"] = betti_numbers[0]
                all_metric_keys[f"{tlabel}_b2"] = betti_numbers[1]
                all_metric_keys[f"{tlabel}_b3"] = betti_numbers[2]
                all_metric_keys[f"{tlabel}_ec"] = ec
            except Exception as e:
                # This exception block might be redundant if utils handles it, but keep for safety
                print(f"\tERROR assigning topology results for {tlabel}: {e}")
                # Values remain 0 as initialized

        # Process combined mask topology
        # Initialize combined mask metrics with 0
        for tk in topo_keys:
            all_metric_keys[f"mask_{tk}"] = 0

        try:
            first_key = list(segmentation.keys())[0]
            combined_mask = np.zeros_like(segmentation[first_key], dtype=np.uint8)
            for tlabel in segmentation.keys():
                mask_to_add = segmentation.get(tlabel)
                if isinstance(mask_to_add, np.ndarray):
                    if mask_to_add.dtype != np.uint8:
                        mask_to_add = mask_to_add.astype(np.uint8)
                    combined_mask += mask_to_add
            combined_mask = np.clip(combined_mask, 0, 1)

            if np.sum(combined_mask) > 0:
                # --- Call compute_topological_features for combined mask ---
                betti_numbers, ec = compute_topological_features(combined_mask)
                print(f"\tCombined Mask: Betti={betti_numbers}, EC={ec}")
                # Assign results (already integers or 0)
                all_metric_keys["mask_b1"] = betti_numbers[0]
                all_metric_keys["mask_b2"] = betti_numbers[1]
                all_metric_keys["mask_b3"] = betti_numbers[2]
                all_metric_keys["mask_ec"] = ec
            else:
                print(f"\tSkipping topology for empty combined mask")
                # Values remain 0 as initialized

        except Exception as e:
            print(f"\tERROR creating or assigning topology for combined mask: {e}")
            # Values remain 0 as initialized

        print(f"--- End Topology Debug ---")

        # Return dictionary containing only integers (or 0 for failures)
        return all_metric_keys

    def preprocess_and_evaluate_metric(
        self,
        metric,
        image,
        mask,
        *,
        compute_on_mask=True,
        mask_intersection=True,
        reduction="mean",
        use_window=True,
        window_size=3,
        use_datarange=False,
        **kwargs,
    ):
        VALID_REDUCTIONS = ["mean", "median"]
        assert reduction in VALID_REDUCTIONS, (
            f"Unknown reduction function {reduction}."
            f"Choose from {VALID_REDUCTIONS}"
        )

        if image is None or mask is None:
            # image is None when the mask is empty: nothing is computed.
            return np.nan, True

        image, mask = self._remove_empty_slices(image, mask)
        metric_out = []
        isnan = False
        if use_datarange:
            if compute_on_mask:
                datarange = image[mask > 0].max() - image[mask > 0].min()
            else:
                datarange = image.max() - image.min()
        for i, img_i in enumerate(image):
            if use_window:
                l, r = window_size // 2, window_size - window_size // 2
                range_j = range(max(0, i - l), min(image.shape[0], i + r))
            else:
                range_j = range(0, image.shape[0])
            for j in range_j:
                im_i = img_i
                im_j = image[j]
                if compute_on_mask:
                    idx = (
                        np.where(mask[i] * mask[j])
                        if mask_intersection
                        else np.where(mask[i] + mask[j])
                    )
                    im_i, im_j = im_i[idx], im_j[idx]
                if use_datarange:
                    m = metric(im_i, im_j, datarange)
                else:
                    m = metric(im_i, im_j)

                if not np.isnan(m) and i != j:
                    metric_out.append(m)
                if np.isnan(m):
                    isnan = True
        if reduction == "mean":
            return np.mean(metric_out), isnan
        elif reduction == "median":
            return np.median(metric_out), isnan

    def preprocess_and_evaluate_noref_metric(
        self,
        noref_metric,
        image,
        mask,
        seg_path=None,
        *,
        compute_on_mask=True,
        flatten=True,
        **kwargs,
    ):
        if compute_on_mask:
            image = image[np.where(mask)]
        if flatten:
            metric = noref_metric(image.flatten())
        return metric, np.isnan(metric)

    def preprocess_and_evaluate_seg_metric(
        self, seg_metric, image, seg_dict, **kwargs
    ):
        return seg_metric(image, seg_dict)

    ### Filter-based metrics

    def _metric_filter(
        self,
        image,
        filter=None,
        **kwargs,
    ) -> np.ndarray:
        """Given a path to a LR image and its corresponding image,
        loads and processes the LR image, filters it with a `filter` from
        skimage.filters and returns the mean of the absolute value.

        Inputs
        ------
        filter:
            A filter from skimage.filters to be applied to the mask

        Output
        ------
        """

        assert (
            filter in SKIMAGE_FCT
        ), f"ERROR: {filter} is not a function from `skimage.filters`"

        filtered = filter(image)
        res = np.mean(abs(filtered - image))
        return res, np.isnan(res)
