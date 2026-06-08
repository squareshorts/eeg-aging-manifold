"""Project-wide paths and analysis constants."""

from __future__ import annotations

from pathlib import Path


SEED = 20260605
BOOTSTRAP_N = 5000
MODEL_VERSION = "ds005385_ses1_v1"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models" / MODEL_VERSION
FEATURES_DIR = RESULTS_DIR / "features"
PROJECTIONS_DIR = RESULTS_DIR / "projections"
EXTERNAL_RESULTS_DIR = RESULTS_DIR / "external"
TABLES_DIR = ROOT / "tables"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = ROOT / "figures"
MANUSCRIPT_DIR = ROOT / "manuscript"

DEVELOPMENT_FEATURES = OUTPUTS_DIR / "baseline_s3_corrected_features.csv"
FOLLOWUP_FEATURES = OUTPUTS_DIR / "ses2_eyesclosed_pre_features.csv"
PARTICIPANTS = OUTPUTS_DIR / "participants_clean.csv"

METADATA_COLS = {
    "participant_id",
    "subject",
    "session",
    "task",
    "acquisition",
    "run",
    "eeg_file",
    "filename",
    "n_eeg_channels",
    "sex",
    "gender",
    "handedness",
    "session1",
    "late_ses1",
    "session2",
    "late_ses2",
    "age",
}

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "low_alpha": (8.0, 10.0),
    "high_alpha": (10.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

REGIONS = {
    "frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "central": ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4"],
    "temporal": ["T7", "T8", "TP9", "TP10"],
    "parietal": ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8"],
    "occipital": ["O1", "Oz", "O2"],
}

SCALP_10_10_CHANNELS = {
    "Fp1",
    "Fpz",
    "Fp2",
    "AF9",
    "AF7",
    "AF5",
    "AF3",
    "AF1",
    "AFz",
    "AF2",
    "AF4",
    "AF6",
    "AF8",
    "AF10",
    "F9",
    "F7",
    "F5",
    "F3",
    "F1",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "F10",
    "FT9",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "FT10",
    "T9",
    "T7",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "T10",
    "TP9",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "TP10",
    "P9",
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO9",
    "PO7",
    "PO5",
    "PO3",
    "PO1",
    "POz",
    "PO2",
    "PO4",
    "PO6",
    "PO8",
    "PO10",
    "O1",
    "Oz",
    "O2",
    "Iz",
}

MODEL_FILES = {
    "manifest": "manifest.json",
    "feature_list": "feature_list.json",
    "transform": "feature_transform_params.csv",
    "pca_loadings": "pca_loadings.csv",
    "pca_params": "pca_params.json",
    "trajectory": "trajectory_params.json",
    "ridge": "ridge_coefficients.csv",
    "ridge_params": "ridge_params.json",
    "t_age_calibration": "trajectory_age_calibration.json",
}
