# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MReyeQC predicts the quality of ocular (eye) MRI from Image Quality Metrics (IQMs). It is a fork of **FetMRQC-SR** (fetal brain QC), adapted to eye tissue. The Python package is still named `fetmrqc_sr`, and a lot of the fetal-brain code is still here. When something mentions fetal or brain anatomy (BOUNTI, `qcglobal`, `fetmrqc_SR_*` models, `QC_IQMs.csv`, `Dockerfile`, `run_docker.py`), it is probably an upstream leftover and not part of the eye pipeline.

## Setup and tooling

```bash
conda create --name MReyeQC python=3.9 && conda activate MReyeQC
pip install -e .
```

- `pyproject.toml` takes precedence over `setup.py`. It pulls in the upstream `fetal_brain_qc` package from git, and this repo needs that package at runtime: `qc_list_bids_csv`, `fetal_brain_utils`, `squeeze_dim`, and `qc_evaluation.preprocess` all come from it. `requirements.txt` is a conda export pinned with `=` and is not valid as pip input.
- The console scripts are defined in `[project.scripts]`: `srqc_compute_iqms`, `srqc_segmentation`, `srqc_train_model`, `srqc_inference`. `setup.py` lists a different name, `srqc_training`.
- Linting: run `pre-commit run --all-files`. It runs black 22.12 and flake8, and flake8 reads `.flake8` (max line length 100 and a long ignore list).
- There is no test suite.

## Eye pipeline (data flow)

Working data lives in `/data/` at the repo root, which is gitignored. The flow, in order:

1. **Data prep scripts** in `fetmrqc_sr/cli/custom_file/` and `fetmrqc_sr/cli/path_generator.py`. These are plain scripts, not entry points. Their input and output paths are hardcoded as module-level constants at the bottom of each file, and they point to the original author's machine (`/Users/cyriltelley/...`). Edit those constants before running `python <script>.py`. Order: `add_new_img.py` → `add_new_seg.py` (copies nnU-Net eye segmentations, which are produced outside this repo) → `convert_to_binary_mask.py` → `qc_list_bids_csv` (upstream CLI) → `average_rating.py` / `add_average_rating_to_bids.py` / `add_new_rating.py`.
   - Don't run `add_new_img.py` twice in a row. The README says it deletes tracking columns that only the first run generates.
2. **IQM extraction**: `srqc_compute_iqms --bids_csv ... --out_csv data/IQA.csv`. The input CSV needs `name`, `sub`, `im`, `mask` and `seg` columns. `--continue_run` is on by default and skips any `name` already in `out_csv`, so delete the output or pass `--no-continue_run` to recompute. It runs subjects in a `ProcessPoolExecutor` (`--nworkers`) and rewrites the whole CSV after each subject.
3. **Model training** happens in `analysis/train_model_MREye_3_models_{regression,binary}.ipynb`, not through the `srqc_train_model` CLI. The notebooks import `analysis/utils.py` with `from utils import ...`, so start them from `analysis/`. `DATA_CSV_PATH` is hardcoded too. Each run writes `model_*.joblib`, `scaler_*.joblib`, `features_*.json` and `config_*.json` to `MODEL_OUTPUT_DIR`; `analysis/trained_models_MREye_REGRESSION/` is an example. `load_and_format_data` splits the CSV at `FIRST_IQM_COLUMN_NAME` (`"centroid"`): columns to its left are metadata and ratings, and columns from it onward are features.

`srqc_train_model` and `srqc_inference` default to the bundled fetal-brain `fetmrqc_sr/data/QC_IQMs.csv`, which has no eye IQM columns. Pass `--dataset` (and probably `--iqms_list`) to use them on eye data.

## IQM engine: `fetmrqc_sr/metrics/metrics_sr.py`

`SRMetrics` holds all the metric logic. Points that take reading several files to piece together:

- **Segmentation labels.** `EYE_MAP_SEG` maps raw nnU-Net labels 0–9 onto the 5 tissue classes in `SEGM` (`LENS`, `GLOBE`, `OPTIQUE_NERVE`, `FAT`, `MUSCLE`). Several raw labels merge into FAT and MUSCLE. The IQM column names come from these class names.
- **Preprocessing.** Images are resampled to 0.8 mm isotropic, cropped to the mask, and given optional robust intensity clipping and bias correction (`_load_and_prep_nifti` / `_preprocess_nifti`).
- **Metric registry.** Metrics are registered in the `self.metrics_func` dict in `__init__`. By default every key is computed; `--metrics` selects a subset. `process_metric(type="ref"|"noref"|"seg")` wraps generic functions from `metrics/utils.py` and `metrics/mriqc_metrics.py`. Slice-wise "ref" metrics compare neighbouring slices.
- **Output columns.** Each scalar metric produces `<name>` plus a boolean `<name>_nan`. A dict-returning metric (any `seg_*`) is flattened to `<name>_<key>[_<subkey>]` and gets its own `_nan` columns. NaN values are replaced with `0.0`, except `seg_cjv`, which uses `1000.0`. Exceptions inside a metric are caught and turned into NaN output, so use `--verbose` to see them. When a metric's return shape changes, update `get_nan_output` to match.
- **Feature list.** `fetmrqc_sr/definitions.py:IQMS` is a separate, hand-maintained feature list that the train and inference CLIs use. It auto-appends a `_nan` twin for every entry. Adding a metric to `metrics_func` does not add it to `IQMS`. For example, `lens_to_max_intensity` and the `rpve_*` metrics are computed but not listed there.

## Upstream code that isn't used

- `fetmrqc_sr/report.py` and `fetmrqc_sr/index.py` import modules that don't exist here (`.plotting`, `.utils`).
- `fetmrqc_sr/data/config.py` templates resolve against the `fetal_brain_qc` package.
- `fetmrqc_sr/qc_evaluation/` is upstream's model-selection framework, built around fetal brain IQM names.
- `srqc_segmentation` runs fetal-brain BOUNTI through Docker.
- `Dockerfile` copies a `fetal_brain_qc/` directory that isn't in this repo.
- `analysis/models.py` and `analysis/example_train_model.ipynb` are upstream experiments.
