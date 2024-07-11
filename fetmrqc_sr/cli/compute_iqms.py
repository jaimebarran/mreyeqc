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
""" Command line interface for the extraction of IQMs from fetal brain images
using FetMRQC
"""
import os
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from fetal_brain_utils import csv_to_list, print_title
from fetmrqc_sr.metrics import SRMetrics
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def process_subject(idx, run, verbose):
    name = Path(run["im"]).name

    print(f"Processing subject {name} (PID: {os.getpid()})")
    lr_metrics = SRMetrics(
        verbose=verbose,
    )
    res = lr_metrics.evaluate_metrics(run["im"], run["mask"], run["seg"])
    return idx, res


def main(argv=None):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    p = argparse.ArgumentParser(
        description=("Computes quality metrics from given images."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--bids_csv",
        help="Path where the bids config csv file is located.",
        required=True,
    )

    p.add_argument(
        "--out_csv",
        help="Path where the IQA results will be stored.",
        required=True,
    )

    p.add_argument(
        "--continue_run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether QC run should re-use existing results if a metrics.csv file at "
            "`out_path`/metrics.csv."
        ),
    )

    p.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Enable verbose."),
    )
    p.add_argument(
        "--nworkers", type=int, default=5, help="Number of workers."
    )

    args = p.parse_args(argv)
    df_base = pd.read_csv(args.bids_csv)
    df_base = df_base.set_index("name")
    print_title("Computing IQMs")

    metrics_dict = {}

    os.makedirs(Path(args.out_csv).parent, exist_ok=True)

    # If a file is found, continue.
    if os.path.isfile(args.out_csv) and args.continue_run:
        print("\tCONTINUING FROM A PREVIOUSLY FOUND RUN.")
        df = pd.read_csv(args.out_csv).set_index("name")
        metrics_dict = df.to_dict(orient="index")
        # Remove duplicate keys
        metrics_dict = {
            k: {k2: v2 for k2, v2 in v.items() if k2 not in df_base.columns}
            for k, v in metrics_dict.items()
        }

    df_run = df_base[~df_base.index.isin(metrics_dict.keys())]

    start = time.time()
    with ProcessPoolExecutor(args.nworkers) as executor:
        # submit tasks and collect futures
        futures = [
            executor.submit(process_subject, idx, run, args.verbose)
            for idx, run in df_run.iterrows()
        ]
        # process task results as they are available
        for i, future in enumerate(as_completed(futures)):
            idx, res = future.result()
            metrics_dict[idx] = res
            df = pd.DataFrame.from_dict(metrics_dict, orient="index")
            df = pd.concat([df_base, df], axis=1, join="inner")
            df.index.name = "name"

            df.to_csv(args.out_csv)
    print(f"Finished in {time.time() - start:.2f} seconds")
    return 0


if __name__ == "__main__":
    main()
