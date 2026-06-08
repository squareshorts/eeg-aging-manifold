# EEG Aging Manifold

This repository contains the manuscript, reproducible scripts, frozen model
artifacts, and generated tables for a longitudinal resting-state EEG aging
trajectory analysis.

## Datasets

Development dataset:

- OpenNeuro `ds005385`, version 1.0.3
- DOI: <https://doi.org/10.18112/openneuro.ds005385.v1.0.3>
- Baseline model fitting uses session-1 eyes-closed pre-task EEG only.

External validation dataset:

- OpenNeuro `ds003775`, SRM Resting-state EEG, version 1.2.1
- DOI: <https://doi.org/10.18112/openneuro.ds003775.v1.2.1>
- Descriptor DOI: <https://doi.org/10.1016/j.dib.2022.108647>
- External validation uses first-session eyes-closed resting EEG in the adult
  subset age 20 years or older.

Raw EEG data are not committed. See `data/README.md` and
`scripts/00_download_or_link_data.md` for download and placement instructions.

## Repository Structure

- `data/` - data-access notes plus ignored local raw/derived data locations.
- `notebooks/` - exploratory notebook retained for provenance.
- `scripts/` - numbered reproducibility pipeline.
- `src/eeg_age/` - reusable feature, modeling, trajectory, stats, and plotting code.
- `results/models/` - versioned frozen development model parameters.
- `results/features/` - derived external feature table used for validation.
- `results/projections/` - fixed-projection coordinates.
- `results/tables/` - CSV tables regenerated from scripts.
- `figures/` - regenerated external-validation figures.
- `tables/` - LaTeX table fragments for the manuscript.
- `manuscript/` - LaTeX manuscript and publication figures.
- `outputs/` - existing ds005385-derived feature tables and legacy outputs.
- `robustness_outputs/` - previously generated robustness analyses.

## Environment

Using pip:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Using conda:

```powershell
conda env create -f environment.yml
conda activate eeg-aging-manifold
```

## Exact Run Order

Run from the repository root.

1. Download or link raw data as described in:

```powershell
Get-Content scripts\00_download_or_link_data.md
```

2. If regenerating ds005385 features from raw EDF files:

```powershell
python scripts\01_extract_features.py --dataset ds005385 --session ses-1
python scripts\01_extract_features.py --dataset ds005385 --session ses-2
```

The repository already includes the ds005385-derived feature tables under
`outputs/`, so this step is optional unless starting from raw data.

3. Fit frozen development parameters from ds005385 session-1:

```powershell
python scripts\02_train_development_models.py
```

4. Project ds005385 session-2 follow-up recordings without refitting:

```powershell
python scripts\03_project_followup.py
```

5. Extract SRM external-validation features:

```powershell
python scripts\01_extract_features.py --dataset ds003775 --force
```

6. Run external validation with frozen ds005385 parameters:

```powershell
python scripts\04_external_validation.py
```

7. Regenerate figures and tables:

```powershell
python scripts\05_make_figures.py
python scripts\06_make_tables.py
```

8. Run reproducibility sanity checks:

```powershell
python scripts\07_sanity_checks.py
```

## Frozen Model Artifacts

`scripts/02_train_development_models.py` saves the following under
`results/models/ds005385_ses1_v1/`:

- `feature_list.json`
- `feature_transform_params.csv`
- `pca_loadings.csv`
- `pca_params.json`
- `trajectory_params.json`
- `trajectory_age_calibration.json`
- `ridge_coefficients.csv`
- `ridge_params.json`
- `manifest.json`

External validation loads these files directly. PCA, scaling, trajectory
centroids, and ridge coefficients are never refit on session-2 or external data.

## Current External Validation Summary

The independent SRM ds003775 adult subset contained 100 participants aged 20-71
years. The fixed-projection coordinate showed weak external linear association
with chronological age (Pearson `r=0.228`, 95% bootstrap CI `0.010-0.414`),
but the rank association was not significant (Spearman `rho=0.112`, 95% CI
`-0.099 to 0.314`, `p=0.269`). The transferred ridge brain-age model did not
generalize as a calibrated predictor (MAE `31.36` years, negative `R2`).

## Key Generated Tables

- `tables/cohort_summary.tex`
- `results/tables/cohort_summary.csv`
- `tables/development_external_performance.tex`
- `results/tables/development_external_performance.csv`
- `results/tables/reproducibility_sanity_checks.csv`
