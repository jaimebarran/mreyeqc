# FetMRQC_SR: Quality control for fetal brain MRI
#
# Copyright 2025 Medical Image Analysis Laboratory (MIAL)
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

import pandas as pd
import os
#from fetal_brain_qc.qc_evaluation import METRICS, METRICS_SEG
from joblib import dump
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from datetime import datetime
#from fetal_brain_qc.definitions import FETMRQC20
import json
from fetmrqc_sr import ROOT_DIR, IQMS

DATASET = os.path.join(
    ROOT_DIR, "data", "QC_IQMs.csv"
)  
OUT_DIR = os.path.join(
    ROOT_DIR, "data", "models"
)  
def load_dataset(dataset, first_iqm):
    df = pd.read_csv(dataset)
    xy_index = df.columns.tolist().index(first_iqm)

    train_x = df[df.columns[xy_index:]].copy()
    train_y = df[df.columns[:xy_index]].copy()

    return train_x, train_y


def get_rating(rating, class_threshold=1.0):
    """Format the rating: if it is a classification task,
    binarize the rating at the class_threshold
    """
    if isinstance(rating, list):
        return [int(r > class_threshold) for r in rating]
    elif isinstance(rating, pd.DataFrame):
        return (rating > class_threshold).astype(int)
    else:
        return rating > class_threshold


def model_name(iqms, target):
    if iqms == IQMS:
        iqms = "_full"
    else:
        iqms = "_custom"
    return f"fetmrqc_SR{iqms}_{target}.joblib"

# Root of package fetmrqc_sr
#data_folder = os.path.dirname(os.path.abspath(__file__))


def main():
    # Parser version of the code below
    import argparse

    parser = argparse.ArgumentParser("Train a FetMRQC_SR classification model.")
    parser.add_argument(
        "--dataset",
        help="Path to the csv file dataset.",
        default=DATASET,
    )
    parser.add_argument(
        "--first_iqm",
        help="First IQM in the csv of the dataset.",
        default="centroid",
    )
    parser.add_argument(
        "--target",
        help="Target rating to use as ground truth.",
        default="qcglobal",
        choices=[
            "qcglobal",
            "is_reconstructed",
            "geom_artefact",
            "recon_artefact",
            "noise",
            "intensity_gm",
            "intensity_dgm",
        ]
    )

    parser.add_argument(
        "--iqms_list",
        help="Custom list of IQMs to use. By default, all IQMs are used.",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--threshold",
        help="Threshold for classification.",
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--model_path", help="Where to save the model.", default=None
    )
    args = parser.parse_args()

    iqms = IQMS
    if args.iqms_list is not None:
        iqms = args.iqms_list
        print(f"Using custom IQMs: {iqms}")

    save_path = args.model_path
    if args.model_path is None:
        save_path = os.path.join(
            OUT_DIR, model_name(iqms, args.target)
        )
    else:
        assert save_path.endswith(".joblib"), "Model path must end with .joblib"
            
    
    save_dir = os.path.dirname(save_path)
    

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    train_x, train_y = load_dataset(args.dataset, args.first_iqm)
    model = RandomForestClassifier()
    rating = get_rating(train_y[args.target], args.threshold)
    model.fit(train_x[iqms], rating)

    curr_time = datetime.now().strftime("%d%m%y_%H%M%S")
    dump(model, save_path)
    config = {
        "dataset": args.dataset,
        "timestamp": curr_time,
        "iqms": iqms,
    }

    with open(save_path.replace(".joblib", ".json"), "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main()
