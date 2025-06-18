# FetMRQC SR: Quality control for fetal brain MRI
#
# Copyright 2025 Medical Image Analysis Laboratory (MIAL)
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
# fetmrqc_sr/definitions.py

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Adapted list of IQMs for Eye Tissue Analysis
IQMS = [

    # --- Non-Segmentation Metrics ---
    'centroid', 'rank_error', 'rank_error_relative', 'mask_volume',
    'ncc_window', 'ncc_median',
    'joint_entropy_window', 'joint_entropy_median',
    'mi_window', 'mi_median',
    'nmi_window', 'nmi_median',
    'shannon_entropy',
    'psnr_window',
    'nrmse_window', 'rmse_window',
    'nmae_window', 'mae_window',
    'ssim_window',
    'mean', 'std', 'median', 'percentile_5', 'percentile_95',
    'kurtosis', 'variation',
    'filter_laplace', 'filter_sobel',

    # --- Segmentation-Based Metrics  ---

    # Summary Stats (sstats) for each eye tissue
    'seg_sstats_LENS_mean', 'seg_sstats_LENS_median', 'seg_sstats_LENS_p95', 'seg_sstats_LENS_p05', 'seg_sstats_LENS_k', 'seg_sstats_LENS_stdv', 'seg_sstats_LENS_mad', 'seg_sstats_LENS_n',
    'seg_sstats_GLOBE_mean', 'seg_sstats_GLOBE_median', 'seg_sstats_GLOBE_p95', 'seg_sstats_GLOBE_p05', 'seg_sstats_GLOBE_k', 'seg_sstats_GLOBE_stdv', 'seg_sstats_GLOBE_mad', 'seg_sstats_GLOBE_n',
    'seg_sstats_OPTIQUE_NERVE_mean', 'seg_sstats_OPTIQUE_NERVE_median', 'seg_sstats_OPTIQUE_NERVE_p95', 'seg_sstats_OPTIQUE_NERVE_p05', 'seg_sstats_OPTIQUE_NERVE_k', 'seg_sstats_OPTIQUE_NERVE_stdv', 'seg_sstats_OPTIQUE_NERVE_mad', 'seg_sstats_OPTIQUE_NERVE_n',
    'seg_sstats_FAT_mean', 'seg_sstats_FAT_median', 'seg_sstats_FAT_p95', 'seg_sstats_FAT_p05', 'seg_sstats_FAT_k', 'seg_sstats_FAT_stdv', 'seg_sstats_FAT_mad', 'seg_sstats_FAT_n',
    'seg_sstats_MUSCLE_mean', 'seg_sstats_MUSCLE_median', 'seg_sstats_MUSCLE_p95', 'seg_sstats_MUSCLE_p05', 'seg_sstats_MUSCLE_k', 'seg_sstats_MUSCLE_stdv', 'seg_sstats_MUSCLE_mad', 'seg_sstats_MUSCLE_n',

    # Volume Fraction for each eye tissue
    'seg_volume_LENS', 'seg_volume_GLOBE', 'seg_volume_OPTIQUE_NERVE', 'seg_volume_FAT', 'seg_volume_MUSCLE',

    # Signal-to-Noise Ratio (SNR) for each eye tissue
    'seg_snr_LENS', 'seg_snr_GLOBE', 'seg_snr_OPTIQUE_NERVE', 'seg_snr_FAT', 'seg_snr_MUSCLE',

    # Contrast-to-Noise Ratio (CNR - Adapted: LENS vs GLOBE)
    'seg_cnr',

    # Coefficient of Joint Variation (CJV - Adapted: LENS vs GLOBE)
    'seg_cjv',

    # Topology Features for each eye tissue and the combined mask
    'seg_topology_LENS_b1', 'seg_topology_LENS_b2', 'seg_topology_LENS_b3', 'seg_topology_LENS_ec',
    'seg_topology_GLOBE_b1', 'seg_topology_GLOBE_b2', 'seg_topology_GLOBE_b3', 'seg_topology_GLOBE_ec',
    'seg_topology_OPTIQUE_NERVE_b1', 'seg_topology_OPTIQUE_NERVE_b2', 'seg_topology_OPTIQUE_NERVE_b3', 'seg_topology_OPTIQUE_NERVE_ec',
    'seg_topology_FAT_b1', 'seg_topology_FAT_b2', 'seg_topology_FAT_b3', 'seg_topology_FAT_ec',
    'seg_topology_MUSCLE_b1', 'seg_topology_MUSCLE_b2', 'seg_topology_MUSCLE_b3', 'seg_topology_MUSCLE_ec',
    'seg_topology_mask_b1', 'seg_topology_mask_b2', 'seg_topology_mask_b3', 'seg_topology_mask_ec',

    # Custom Eye Metrics
    'seg_globe_sphericity', 'seg_lens_aspect_ratio',
    
    # 'seg_snr_GLOBE',
    # 'seg_cjv',
    # 'seg_sstats_GLOBE_mad',
    # 'seg_sstats_GLOBE_stdv',
    # 'seg_sstats_GLOBE_k',
    # 'seg_snr_total',
    # 'seg_cnr',
    # 'seg_sstats_GLOBE_p95',
    # 'centroid_full',
]

# --- Add _nan versions for all metrics ---

_IQMS_NO_NAN = IQMS[:] # Take a copy
for iqm in _IQMS_NO_NAN:
    IQMS.append(iqm + '_nan')