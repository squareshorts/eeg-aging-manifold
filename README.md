# EEG Aging Manifold

This repository contains code, manuscript materials, figures, and derived summary outputs for a longitudinal resting-state EEG aging manifold analysis.

## Dataset

Raw EEG data are publicly available from OpenNeuro:

- Dataset: ds005385
- DOI: 10.18112/openneuro.ds005385.v1.0.3
- Descriptor: Getzmann et al., Scientific Data, 2024

Raw EDF files are not included in this repository.

## Main analyses

The workflow estimates:

1. EEG spectral features from eyes-closed resting-state recordings.
2. EEG brain-age prediction using ridge regression.
3. PCA and PLS latent aging representations.
4. A low-dimensional EEG aging trajectory.
5. Test-retest feature stability.
6. Longitudinal trajectory displacement.
7. Baseline trajectory position as a predictor of future trajectory change.

## Repository structure

notebooks/              Colab notebook for the full analysis
manuscript/             LaTeX manuscript and references
manuscript/figures/     Final manuscript figures
outputs/                Small derived summary CSV files
src/                    Optional reusable Python modules

## Reproducibility

Run the notebook in `notebooks/` using Google Colab. The notebook is configured to use OpenNeuro ds005385 as the raw data source and to avoid storing raw EEG files in this repository.

## Data policy

This repository does not store raw EEG data. Users should download the dataset directly from OpenNeuro.

## Citation

If using this repository, cite the manuscript and the OpenNeuro dataset ds005385.
