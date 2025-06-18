
# MReyeQC

MReyeQC is a tool for creating a regression model to predict the quality of eye MRI images. This project is inspired by and based on the [MRIQC](https://mriqc.readthedocs.io/) framework.

The main workflow consists of:
1.  Preprocessing a dataset of ocular MRI images and their segmentations.
2.  Computing a set of Image Quality Metrics (IQMs) using MRIQC tools.
3.  Training models (either regression or binary classification) to predict an image's quality from its IQMs.

## Installation

To install MReyeQC, start by creating a new `conda` environment. Python 3.9 is recommended.

```bash
conda create --name MReyeQC python=3.9
conda activate MReyeQC
```

Then, install the project and its dependencies by cloning this repository and running:
```bash
pip install -e .
```

## Usage

### 1. Adding Data and Computing IQMs

This section describes the steps required to add new images to the dataset, preprocess them, and compute the Image Quality Metrics (IQMs).

1.  **Add Images**
    Place your MRI images in NIfTI format into the `data/img/` directory.

2.  **Index New Images**
    Run the `add_new_img.py` script. This script prepares the new data for the subsequent steps.
    **Warning**: Do not run this script twice in a row, as it deletes tracking columns that are only generated on the first run.

3.  **Prepare Segmentation Masks**
    Run the following scripts sequentially to generate and format the binary masks required for IQM calculation:
    * `add_new_seg.py`
    * `convert_to_binary_mask.py`

4.  **Generate BIDS List**
    Use the `qc_list_bids_csv` command to create a CSV file listing your images and their corresponding masks. Replace `<path_to_project>` with the absolute path to the project's root directory.

    ```bash
    qc_list_bids_csv \
        --bids_dir "<path_to_project>/data/img" \
        --mask_patterns_base "<path_to_project>/data/mask" \
        --mask_patterns "sub-{subject}_mask.nii.gz" \
        --out_csv "<path_to_project>/data/bids_csv/bids_csv.csv" \
        --suffix T1w \
        --no-anonymize_name
    ```

5.  **Add Ratings**
    Run the following scripts to integrate the quality ratings (manual or existing) into the CSV file:
    * `add_average_rating_to_bids.py`
    * `add_new_rating.py`

6.  **Compute IQMs**
    Finally, run the `srqc_compute_iqms` command to extract the quality metrics from the images and masks. The output will be saved to `IQA.csv`.

    ```bash
    srqc_compute_iqms \
        --bids_csv "<path_to_project>/data/bids_csv/bids_csv_rating.csv" \
        --out_csv "<path_to_project>/data/IQA.csv"
    ```

### 2. Model Training

Once the IQMs are computed and the ratings have been added, you can train the quality prediction models.

To do this, run one of the following Jupyter notebooks, depending on the desired model type:
* `train_model_MREye_3_models_regression.ipynb`: To train a regression model that predicts a continuous quality score.
* `train_model_MREye_3_models_binary.ipynb`: To train a classification model that predicts a binary quality label (e.g., "good" vs. "bad").

## Citation
> Esteban O, Birman D, Schaer M, Koyejo OO, Poldrack RA, Gorgolewski KJ; MRIQC: Advancing the Automatic Prediction of Image Quality in MRI from Unseen Sites; PLOS ONE 12(9):e0184661; doi:10.1371/journal.pone.0184661.
