# FetMRQC: Quality control for fetal brain MRI
#
# Copyright 2023 Medical Image Analysis Laboratory (MIAL)
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
""" Compute segmentation on low resolution clinical acquisitions.
By default the segmentation is computed using a nnUNet-v2 model pretrained on FeTA data,
but other methods can be used.
"""

# Import libraries

import pandas as pd
import os
from pathlib import Path
import numpy as np
import time
import shutil


def format_sub_ses(sub, ses):
    """
    Format the subject and session to be used in the path.
    """

    ses = (
        ses
        if isinstance(ses, str)
        else None if ses is None else ses if not np.isnan(ses) else None
    )
    sub = f"{int(sub):03d}" if str(sub).isdigit() else str(sub)
    ses = (
        ses
        if isinstance(ses, str)
        else f"{int(ses):02d}" if ses is not None else None
    )
    return sub, ses


def get_sub_ses_path(base, sub, ses, f=None):
    """
    Get the path to the subject/session folder.
    """
    sub, ses = format_sub_ses(sub, ses)
    if ses is None:
        if f is None:
            return base / f"sub-{sub}" / "anat"
        else:
            return base / f"sub-{sub}" / "anat" / f
    else:
        if f is None:
            return base / f"sub-{sub}" / f"ses-{ses}" / "anat"
        else:
            return base / f"sub-{sub}" / f"ses-{ses}" / "anat" / f


def create_tmp_dir(suffix=""):
    """
    Create a temporary directory to store the cropped images.
    """
    suffix = suffix if suffix == "" else f"_{suffix}"
    tmp_dir = os.path.abspath(f"./tmp_{time.time()}{suffix}")
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def bidsify_and_save_seg(df, seg_path, out_path):
    seg_path = Path(seg_path).absolute()
    out_path = Path(out_path).absolute()
    files = sorted([f for f in os.listdir(seg_path) if f.endswith(".nii.gz")])
    for f in files:
        if f.endswith("-mask-bet-1.nii.gz"):
            f_im = f.replace("-mask-bet-1.nii.gz", ".nii.gz")
            # Find in df["im"] the image that corresponds to f_im
            f_out = f_im.replace("_T2w.nii.gz", "_mask.nii.gz")
            df_row = df[df["im"].str.contains(f_im)]
            idx = df_row.index[0]
            col = "mask"
        elif f.endswith("-mask-brain_bounti-19.nii.gz"):
            f_im = f.replace("-mask-brain_bounti-19.nii.gz", ".nii.gz")
            df_row = df[df["im"].str.contains(f_im)]
            idx = df_row.index[0]
            f_out = f_im.replace("_T2w.nii.gz", "_dseg.nii.gz")
            col = "seg"
        sub, ses = df_row["sub"].values[0], df_row["ses"].values[0]
        sub, ses = format_sub_ses(sub, ses)
        os.makedirs(get_sub_ses_path(out_path, sub, ses), exist_ok=True)
        shutil.move(
            seg_path / f,
            get_sub_ses_path(out_path, sub, ses, f_out),
        )
        df.loc[idx, col] = get_sub_ses_path(out_path, sub, ses, f_out)
    return df


def add_to_df_if_done(df, out_path):
    out_path = Path(out_path).absolute()
    # Add columns to df if they don't exist
    if "mask" not in df.columns:
        df["mask"] = None
    if "seg" not in df.columns:
        df["seg"] = None

    for idx, row in df.iterrows():
        sub, ses, im = row["sub"], row["ses"], row["im"]
        mask, seg = row["mask"], row["seg"]
        mask = None if pd.isnull(mask) else mask
        seg = None if pd.isnull(seg) else seg
        if mask is not None or seg is not None:
            if os.path.exists(mask) and os.path.exists(seg):
                print(
                    f"Segmentation for {sub} {ses} {os.path.basename(im)} already exists."
                )
                continue
            else:
                raise RuntimeError(
                    f"Segmentation and/or mask that was supposed to exist for {sub} {ses} was not found. Check the provided paths. Aborting."
                )
        im = os.path.basename(im)
        row_mask = get_sub_ses_path(
            out_path, sub, ses, im.replace("_T2w.nii.gz", "_mask.nii.gz")
        )
        row_seg = get_sub_ses_path(
            out_path, sub, ses, im.replace("_T2w.nii.gz", "_dseg.nii.gz")
        )
        if os.path.exists(row_mask) and os.path.exists(row_seg):
            print(f"Segmentation for {sub} {ses} {im} found.")
            df.loc[idx, "mask"] = row_mask
            df.loc[idx, "seg"] = row_seg
    return df


def save_df(df, df_path):
    if df_path.endswith(".csv"):
        df.to_csv(df_path, index=False)
    elif df_path.endswith(".tsv"):
        df.to_csv(df_path, sep="\t", index=False)
    else:
        raise ValueError("bids_csv must be either csv or tsv.")


def run_bounti(bids_csv, out_path, chunk_size, device):
    """
    Loads the data from bids_csv, checks whether the segmentation has already been computed
    and if not, computes the segmentation, saves it to the <out_path> folder and updates the
    DataFrame with the segmentation paths.

    Args:
        bids_csv (str): Path to the bids_csv file.
        out_path (str): Path to the output folder.
        nnunet_res_path (str): Path to the nnunet folder containing the model checkpoint.
        nnunet_env_path (str): Path to the environment in which nnunet was installed (from `conda env list`)
    """
    if bids_csv.endswith(".csv"):
        df = pd.read_csv(bids_csv)
    elif bids_csv.endswith(".tsv"):
        df = pd.read_csv(bids_csv, sep="\t")
    else:
        raise ValueError("bids_csv must be either csv or tsv.")

    os.makedirs(out_path, exist_ok=True)
    df = add_to_df_if_done(df, out_path)
    df_todo = df[df["mask"].isnull() | df["seg"].isnull()]

    files = df_todo["im"].values
    # Check if files exist.
    # Work in chunks of 200 files.
    files_chunks = [
        files[i : i + chunk_size] for i in range(0, len(files), chunk_size)
    ]

    for chunk in files_chunks:
        tmp_dir = create_tmp_dir()
        tmp_dir2 = create_tmp_dir("out")
        for f in chunk:
            os.system(f"cp {f} {tmp_dir}")

        if device == "cpu":
            cmd = (
                "docker run --rm "
                f"-v {tmp_dir}:/home/data "
                f"-v {tmp_dir2}:/home/out "
                "fetalsvrtk/segmentation:general_auto_amd bash /home/auto-proc-svrtk/scripts/auto-brain-bounti-segmentation-fetal.sh "
                f"/home/data/ /home/out"
            )
        else:
            cmd = (
                "docker run --rm --gpus all "
                f"-v {tmp_dir}:/home/data "
                f"-v {tmp_dir2}:/home/out "
                "fetalsvrtk/segmentation:general_auto_amd bash /home/auto-proc-svrtk/scripts/auto-brain-bounti-segmentation-fetal-gpu.sh "
                f"/home/data/ /home/out"
            )
        print(cmd)
        os.system(cmd)

        df = bidsify_and_save_seg(df, tmp_dir2, out_path)
        # Check if size of new df is smaller than the original df
        shutil.rmtree(tmp_dir)
        shutil.rmtree(tmp_dir2)
        if len(df) < len(pd.read_csv(bids_csv)):
            save_df(df, f"{out_path}/bids_csv_error.csv")
            raise RuntimeError(
                "Something went wrong: the updated dataframe is smaller than the original one. Saving the updated dataframe to <out_path>/bids_csv_error.csv and aborting."
            )
        save_df(df, bids_csv)

    return 0


def main():
    import argparse
    from pathlib import Path
    from fetal_brain_utils import print_title

    p = argparse.ArgumentParser(
        description=("Compute a brain mask and segmentation using BOUNTI."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--bids_csv",
        help="Path where the bids config csv file is located.",
        required=True,
    )

    p.add_argument(
        "--out_path",
        help="Path where the segmentations will be stored. (if not specified in bids_csv)",
    )

    p.add_argument(
        "--chunk_size",
        help="Number of files to process at once in BOUNTI.",
        default=25,
        type=int,
    )
    p.add_argument(
        "--device",
        help="Device to use for the segmentation. Options: 'cpu' or 'cuda'.",
        default="cuda",
        choices=["cpu", "cuda"],
    )

    args = p.parse_args()
    print_title("Running BOUNTI segmentation on SR volumes.")
    out_path = Path(args.out_path).absolute()
    run_bounti(args.bids_csv, out_path, args.chunk_size, args.device)
    return 0


if __name__ == "__main__":
    main()
