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
import nibabel as ni
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
    compute_topological_features
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
    # wm2max,
    globe_sphericity,
    lens_aspect_ratio
)
import sys
from functools import partial
from fetal_brain_utils import get_cropped_stack_based_on_mask
import warnings

SKIMAGE_FCT = [fct for _, fct in getmembers(skimage.filters, isfunction)]

SEGM = {
    "LENS":             0,
    "GLOBE":            1,
    "OPTIQUE_NERVE":    2,
    "FAT":              3,
    "MUSCLE":           4,
}

# SEGM = {
#     "LENS":             2,
#     "GLOBE":            0,
#     "OPTIQUE_NERVE":    4,
#     "FAT":              3,
#     "MUSCLE":           1,
# }

#SEGM = {"CSF": 1, "GM": 2, "WM": 3, "BS": 4, "CBM": 5}
# Re-mapping to do for FeTA labels: ventricles as CSF, dGM as GM.

FETA_LABELS = [None, 1, 2, 3, 1, None, 2, None, None]

segm_names = list(SEGM.keys())

EYE_MAP_SEG = [
    None, # Index 0: Raw Label 0 (Background) -> Ignore
    0,    # Index 1: Raw Label 1 (LENS)       -> Target 0 (LENS)
    1,    # Index 2: Raw Label 2 (GLOBE)      -> Target 1 (GLOBE)
    2,    # Index 3: Raw Label 3 (NERVE)      -> Target 2 (OPTIQUE_NERVE)
    3,    # Index 4: Raw Label 4 (FAT)        -> Target 3 (FAT)
    3,    # Index 5: Raw Label 5 (FAT)        -> Target 3 (FAT) <-- Group FAT
    4,    # Index 6: Raw Label 6 (MUSCLE)     -> Target 4 (MUSCLE)
    4,    # Index 7: Raw Label 7 (MUSCLE)     -> Target 4 (MUSCLE) <-- Group MUSCLE
    4,    # Index 8: Raw Label 8 (MUSCLE)     -> Target 4 (MUSCLE) <-- Group MUSCLE
    4,    # Index 9: Raw Label 9 (MUSCLE)     -> Target 4 (MUSCLE) <-- Group MUSCLE
]

BOUNTI_LABELS = [
    None,
    1,
    1,
    2,
    2,
    3,
    3,
    1,
    1,
    1,
    4,
    5,
    5,
    5,
    2,
    2,
    2,
    2,
    1,
    1,
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
        '''        self.metrics_func = {
            "centroid": partial(centroid),
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
            "mask_volume": mask_volume,
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
            "shannon_entropy": self.process_metric(
                shannon_entropy,
                type="noref",
                compute_on_mask=True,
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
            "ssim_window": partial(self._ssim, **default_params),
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
            ## Filter-based metrics
            "filter_laplace": partial(self._metric_filter, filter=laplace),
            "filter_sobel": partial(self._metric_filter, filter=sobel),
            "seg_sstats": self.process_metric(self._seg_sstats, type="seg"),
            "seg_volume": self.process_metric(self._seg_volume, type="seg"),
            "seg_snr": self.process_metric(self._seg_snr, type="seg"),
            "seg_cnr": self.process_metric(self._seg_cnr, type="seg"),
            "seg_cjv": self.process_metric(self._seg_cjv, type="seg"),
            #"seg_wm2max": self.process_metric(self._seg_wm2max, type="seg"),
            "seg_topology": self.process_metric(self._seg_topology, type="seg"),



            # For eye
            "centroid": partial(centroid, central_third=True),
            "centroid_full": partial(centroid, central_third=False),
            "seg_globe_sphericity": self.process_metric(self._seg_globe_sphericity, type="seg"),

            "im_size_vx_size": self._get_voxel_size,
        }'''
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

            # --- Image Size Metrics ---
            # "im_size_vx_size": self._get_voxel_size, 
        }
        self._metrics = self.get_all_metrics()
        self._check_metrics()
        self.map_seg = map_seg
        # Summary statistics from the segmentation, used for computing a bunch of metrics
        # besides being a metric itself
        self._sstats = None
        self.counter=counter

    '''   def _seg_globe_sphericity(self, seg_dict, vx_size, **kwargs):
        """
        Wrapper to calculate globe sphericity using pre-loaded segmentation data.
        """
        isnan = False

        globe_mask = seg_dict["GLOBE"]

        try:
            sphericity = globe_sphericity(globe_mask, vx_size)
            if np.isnan(sphericity):
                isnan = True
                sphericity = 0.0
        except Exception as e:
            sphericity = np.nan
            isnan = True

        value_to_return = 0.0 if isnan else sphericity
        return value_to_return, isnan'''
    
    def _seg_lens_aspect_ratio(self, seg_dict, vx_size, **kwargs):
        """
        Wrapper to calculate lens aspect ratio using pre-loaded segmentation data.
        """
        isnan = False
        print(f"--- Debugging Lens Aspect Ratio ---")

        # Check if LENS mask exists in the dictionary
        if "LENS" not in seg_dict or not isinstance(seg_dict.get("LENS"), np.ndarray):
            print(f"\tWARNING: LENS segmentation not found or not a numpy array in seg_dict.")
            return 0.0, True # Return 0 and mark as NaN

        lens_mask = seg_dict["LENS"]
        print(f"\tInput LENS mask: Sum={np.sum(lens_mask)}, Shape={lens_mask.shape}, Dtype={lens_mask.dtype}")
        print(f"\tInput vx_size: {vx_size}")

        # Ensure mask is uint8 as expected by the core function's internal checks/casting
        if lens_mask.dtype != np.uint8:
             print(f"\tCasting LENS mask from {lens_mask.dtype} to uint8.")
             lens_mask = lens_mask.astype(np.uint8)
             # Re-check sum after casting
             if np.sum(lens_mask) == 0:
                  print(f"\tWARNING: LENS mask became empty after casting to uint8.")
                  return 0.0, True

        if np.sum(lens_mask) == 0: # Check if mask is empty
            print(f"\tWARNING: LENS mask is empty.")
            return 0.0, True

        try:
            # Call the actual calculation function from mriqc_metrics.py
            aspect_ratio = lens_aspect_ratio(lens_mask, vx_size)
            print(f"\tCalculated Lens Aspect Ratio: {aspect_ratio}")

            if np.isnan(aspect_ratio):
                isnan = True
                # The core function already returns np.nan, so this might be redundant
                # but good for explicit handling.
                # If it's NaN, the main loop will convert it to 0.0 and set _nan=True.
            # No need to set aspect_ratio = 0.0 here if isnan, main loop handles it.

        except Exception as e:
            print(f"\tERROR: Failed calculating lens aspect ratio via wrapper: {e}")
            import traceback
            traceback.print_exc()
            aspect_ratio = np.nan # Ensure it's NaN on error
            isnan = True

        print(f"--- End Lens Aspect Ratio Debug ---")
        # Return the calculated value (or NaN) and the isnan flag
        return aspect_ratio, isnan

    def _seg_globe_sphericity(self, seg_dict, vx_size, **kwargs):
        """
        Wrapper to calculate globe sphericity using pre-loaded segmentation data.
        """
        isnan = False
        print(f"--- Debugging Sphericity ---")

        # --- More detailed check ---
        print(f"\tChecking seg_dict type: {type(seg_dict)}")
        if isinstance(seg_dict, dict):
             print(f"\tseg_dict keys: {list(seg_dict.keys())}")
             globe_mask = seg_dict.get("GLOBE") # Use .get() for safer access
             print(f"\tType of seg_dict['GLOBE']: {type(globe_mask)}")
        else:
             print(f"\tERROR: seg_dict is not a dictionary!")
             globe_mask = None

        # Check if the mask exists AND is a numpy array AND has non-zero sum
        if not isinstance(globe_mask, np.ndarray) or np.sum(globe_mask) == 0:
            print(f"\tWARNING: GLOBE mask is not a valid ndarray or is empty. Sum={np.sum(globe_mask) if isinstance(globe_mask, np.ndarray) else 'N/A'}")
            return 0.0, True # Return 0 and mark as NaN
        # --- End Detailed Check ---

        # If we get here, globe_mask is a non-empty numpy array
        print(f"\tInput GLOBE mask: Sum={np.sum(globe_mask)}, Shape={globe_mask.shape}, Dtype={globe_mask.dtype}")
        print(f"\tInput vx_size: {vx_size}")

        # Explicitly cast to uint8
        if globe_mask.dtype != np.uint8:
             print(f"\tCasting GLOBE mask from {globe_mask.dtype} to uint8.")
             globe_mask = globe_mask.astype(np.uint8)
             # Re-check sum after casting in case of issues
             if np.sum(globe_mask) == 0:
                  print(f"\tWARNING: GLOBE mask became empty after casting to uint8.")
                  return 0.0, True

        try:
            # Call the actual calculation function
            sphericity = globe_sphericity(globe_mask, vx_size)
            print(f"\tCalculated Sphericity: {sphericity}")

            if np.isnan(sphericity):
                isnan = True
                sphericity = 0.0
        except Exception as e:
            print(f"\tERROR: Failed calculating sphericity via wrapper: {e}")
            import traceback
            traceback.print_exc()
            sphericity = 0.0
            isnan = True

        print(f"--- End Sphericity Debug ---")
        value_to_return = 0.0 if isnan else sphericity
        return value_to_return, isnan
    def _get_voxel_size(self, vx_size, **kwargs):
        """
        Returns the voxel size passed during metric evaluation.
        Note: Returns the list [vx, vy, vz]. The original 'im_size' IQM
                might have expected separate x, y, z, etc. This needs
                adjustment if individual dimensions are required as separate IQMs.
        """
        # vx_size is expected to be a list like [0.8, 0.8, 0.8]
        # For now, return the list directly. Needs flattening if individual IQMs are needed.
        # Returning a dummy value and NaN status for compatibility for now.
        # Proper handling would involve splitting vx_size into x, y, z components
        # and potentially adding image dimensions if needed, then updating definitions.py
        # For simplicity, let's just return a known value (e.g., product) and False for NaN
        if vx_size is None or not isinstance(vx_size, (list, tuple, np.ndarray)) or len(vx_size) != 3:
                print("\tWARNING: Invalid vx_size encountered in _get_voxel_size.")
                return 0.0, True # Return 0 and indicate NaN

        # Example: return voxel volume, needs update if individual dims are needed
        value = float(np.prod(vx_size))
        return value, np.isnan(value)

    def get_all_metrics(self):
        return list(self.metrics_func.keys())

    def set_metrics(self, metrics):
        self._metrics = metrics

    def _valid_mask(self, mask_path):
        mask = ni.load(mask_path).get_fdata()
        if mask.sum() == 0:
            return False
        else:
            return True

    def get_nan_output(self, metric):
        sstats_keys = [
            "mean",
            "median",
            "median",
            "p95",
            "p05",
            "k",
            "stdv",
            "mad",
            "n",
        ]

        topo_keys = [
            "b1",
            "b2",
            "b3",
            "ec",
        ]
        if "seg_" in metric:
            metrics = segm_names
            if "seg_sstats" in metric:
                metrics = [f"{n}_{k}" for n in segm_names for k in sstats_keys]
                return {m: np.nan for m in metrics}
            elif "seg_topology" in metric:
                metrics = [f"{n}_{k}" for n in segm_names + ["mask"] for k in topo_keys]
                return {m: np.nan for m in metrics}
            return {m: np.nan for m in metrics}
        else:
            return [np.nan]

    def get_default_output(self, metric):
        """Return the default output for a given metric when the mask is invalid and metrics cannot be computed."""
        METRIC_DEFAULT = {"cjv": 0}
        if metric not in METRIC_DEFAULT.keys():
            return [0.0, False]
        else:
            return METRIC_DEFAULT[metric]

    def _flatten_dict(self, d):
        """Flatten a nested dictionary by concatenating the keys with '_'."""
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(
                    {
                        k + "_" + kk: vv
                        for kk, vv in self._flatten_dict(v).items()
                    }
                )
            else:
                out[k] = v
        return out

    def eval_metrics_and_update_results(
        self,
        results,
        metric,
        args_dict,
    ):
        """Evaluate a metric and update the results dictionary."""
        try:
            out = self.metrics_func[metric](**args_dict)
        except Exception:
            if self.verbose:

                print(
                    f"EXCEPTION with {metric}\n" + traceback.format_exc(),
                    file=sys.stderr,
                )
            out = self.get_nan_output(metric)
        # Checking once more that if the metric is nan, we replace it with 0
        if isinstance(out, dict):
            out = self._flatten_dict(out)
            for k, v in out.items():
                results[metric + "_" + k] = v if not np.isnan(v) else 0.0
                results[metric + "_" + k + "_nan"] = np.isnan(v)
        else:
            if np.isnan(out[0]):
                out = (0, True)
            results[metric], results[metric + "_nan"] = out
        return results

    def evaluate_metrics(self, sr_path, mask_path, seg_path):
        """Evaluate the metrics for a given LR image and mask.

        Args:
            sr_path (str): Path to the LR image.
            seg_path (str, optional): Path to the segmentation. Defaults to None.

        Returns:
            dict: Dictionary containing the results of the metrics.
        """

        # Remark: Could do something better with a class here: giving flexible
        # features as input.
        # Reset the summary statistics
        self._sstats = None

        resample_to = 0.8
        imagec, maskc, seg_dict = self._load_and_prep_nifti(
            sr_path, mask_path, seg_path, resample_to
        )

        args_dict = {
            "image": imagec,
            "mask": maskc,
            "seg_dict": seg_dict,
            "vx_size": [resample_to] * 3,
        }
        if any(["seg_" in m for m in self._metrics]):
            assert seg_path is not None, (
                "Segmentation path should be provided "
                "when evaluating segmentation metrics."
            )
        results = {}
        for m in self._metrics:
            if self.verbose:
                print("\tRunning", m)
            results = self.eval_metrics_and_update_results(
                results, m, args_dict
            )
        return results

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

        # Calculate total SNR
        valid_snrs = [v for v in snr_dict.values() if not np.isnan(v)]
        if valid_snrs:
            snr_dict["total"] = float(np.mean(valid_snrs))
            all_snr_nan = False
        else:
            snr_dict["total"] = np.nan

        print(f"\tSNR Total: {snr_dict.get('total', 'N/A')}")
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
