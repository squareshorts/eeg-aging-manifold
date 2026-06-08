"""External validation of the frozen ds005385 projection on ds003690 and ds004148."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_age.config import DATA_DIR, EXTERNAL_RESULTS_DIR, MODELS_DIR, RESULTS_TABLES_DIR, SEED  # noqa: E402
from eeg_age.io import write_csv, write_json  # noqa: E402
from eeg_age.models import FrozenModel  # noqa: E402
from eeg_age.stats import metric_table_for_association, metric_table_for_predictions  # noqa: E402
from eeg_age.features import extract_feature_table, read_participants_tsv  # noqa: E402


def validate_dataset(
    model: FrozenModel,
    dataset_id: str,
    eeg_files: list[Path],
    participants: pd.DataFrame,
    dataset_label: str,
    dataset_summary: dict,
    min_age: float | None = None,
) -> pd.DataFrame:
    print(f"\nProcessing {dataset_id}...")
    print(f"Found {len(eeg_files)} EEG files.")
    
    features_csv = DATA_DIR / "derived" / f"{dataset_id}_features.csv"
    if features_csv.exists():
        print(f"Loading existing features: {features_csv}")
        external = pd.read_csv(features_csv)
    else:
        print(f"Extracting features...")
        external, failures = extract_feature_table(eeg_files, participants)
        features_csv.parent.mkdir(parents=True, exist_ok=True)
        external.to_csv(features_csv, index=False)
        if len(failures):
            failures.to_csv(features_csv.with_name(features_csv.stem + "_failures.csv"), index=False)
            print(f"Failed to extract features for {len(failures)} files.")

    if "age" not in external.columns:
        raise ValueError("External feature table must contain age.")

    external["age"] = pd.to_numeric(external["age"], errors="coerce")
    original_n = len(external)
    missing_age_n = int(external["age"].isna().sum())
    
    if min_age is not None:
        external = external[external["age"] >= min_age].copy()
    
    external = external.dropna(subset=["age"]).reset_index(drop=True)
    if len(external) < 10:
        raise ValueError(f"Too few external participants after age filtering: {len(external)}")

    print(f"Applying frozen model transfer to {len(external)} participants...")
    projection = model.project(external, dataset_label)
    
    meta_cols = [col for col in ["participant_id", "age", "sex", "session", "filename", "n_eeg_channels"] if col in external.columns]
    out = external[meta_cols].reset_index(drop=True).join(projection.reset_index(drop=True))
    
    EXTERNAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    projection_path = EXTERNAL_RESULTS_DIR / f"{dataset_id}_external_projection.csv"
    write_csv(out, projection_path)

    metrics = pd.concat(
        [
            metric_table_for_association(
                "External T vs chronological age",
                out["T"],
                out["age"],
            ),
            metric_table_for_predictions(
                "External T-calibrated age prediction",
                out["age"],
                out["T_predicted_age"],
            ),
            metric_table_for_predictions(
                "External ridge age prediction",
                out["age"],
                out["ridge_predicted_age"],
            ),
        ],
        ignore_index=True,
    )
    metrics["dataset"] = dataset_id
    metrics["validation_protocol"] = (
        "Full 93-feature fixed transfer: imputation, winsorization, scaler, PCA, "
        "trajectory axis, T-age calibration, and ridge coefficients loaded from "
        "ds005385 session-1 model artifacts without refitting."
    )
    metrics_path = EXTERNAL_RESULTS_DIR / f"{dataset_id}_validation_metrics.csv"
    write_csv(metrics, metrics_path)
    write_csv(metrics, RESULTS_TABLES_DIR / f"{dataset_id}_validation_metrics.csv")

    summary = dataset_summary.copy()
    summary["raw_participants_n"] = int(original_n)
    summary["analysis_n"] = int(len(out))
    summary["excluded_below_min_age"] = int(original_n - missing_age_n - len(out))
    summary["missing_age_n"] = missing_age_n
    summary["analysis_age_min"] = float(out["age"].min())
    summary["analysis_age_max"] = float(out["age"].max())
    summary["analysis_age_mean"] = float(out["age"].mean())
    summary["analysis_age_sd"] = float(out["age"].std(ddof=1))
    
    if "sex" in out.columns:
        summary["female_n"] = int((out["sex"].astype(str).str.upper().str[0] == "F").sum())
        summary["male_n"] = int((out["sex"].astype(str).str.upper().str[0] == "M").sum())
    else:
        summary["female_n"] = None
        summary["male_n"] = None
        
    write_json(summary, EXTERNAL_RESULTS_DIR / f"{dataset_id}_dataset_summary.json")

    print(f"Saved external projections: {projection_path}")
    print(f"Saved external metrics: {metrics_path}")
    print(metrics.to_string(index=False))
    return metrics


def main() -> None:
    np.random.seed(SEED)
    model = FrozenModel.load(MODELS_DIR)

    all_metrics = []

    # ds003690 (Healthy Aging)
    ds003690_dir = DATA_DIR / "external" / "ds003690"
    if ds003690_dir.exists():
        eeg_files = sorted(ds003690_dir.glob("*/eeg/*task-passive*eeg.set"))
        participants_file = ds003690_dir / "participants.tsv"
        if participants_file.exists() and eeg_files:
            participants = read_participants_tsv(participants_file)
            metrics = validate_dataset(
                model=model,
                dataset_id="ds003690",
                eeg_files=eeg_files,
                participants=participants,
                dataset_label="OpenNeuro ds003690 Healthy Aging Resting-state EEG",
                dataset_summary={
                    "dataset": "OpenNeuro ds003690, Healthy Aging",
                    "doi": "10.18112/openneuro.ds003690.v1.0.0",
                    "resting_condition": "Resting-state passive",
                    "suitability": "Independent public resting-state EEG dataset with young and older healthy adults.",
                },
                min_age=None,
            )
            all_metrics.append(metrics)
    else:
        print("ds003690 data not found, skipping...")

    # ds004148 (Healthy Young Adults)
    ds004148_dir = DATA_DIR / "external" / "ds004148"
    if ds004148_dir.exists():
        eeg_files = sorted(ds004148_dir.glob("*/ses-session1/eeg/*task-eyesclosed*eeg.vhdr"))
        participants_file = ds004148_dir / "participants.tsv"
        if participants_file.exists() and eeg_files:
            participants = read_participants_tsv(participants_file)
            metrics = validate_dataset(
                model=model,
                dataset_id="ds004148",
                eeg_files=eeg_files,
                participants=participants,
                dataset_label="OpenNeuro ds004148 Healthy Young Adults Resting-state EEG",
                dataset_summary={
                    "dataset": "OpenNeuro ds004148, Healthy Young Adults",
                    "doi": "10.18112/openneuro.ds004148.v1.0.0",
                    "resting_condition": "Resting-state eyes closed",
                    "suitability": "Independent public resting-state EEG dataset with young healthy adults.",
                },
                min_age=None,
            )
            all_metrics.append(metrics)
    else:
        print("ds004148 data not found, skipping...")

if __name__ == "__main__":
    main()
