"""External validation of the frozen ds005385 projection on SRM ds003775."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-features",
        type=Path,
        default=DATA_DIR / "derived" / "ds003775_t1_features.csv",
    )
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--min-age", type=float, default=20.0)
    parser.add_argument("--dataset-label", default="OpenNeuro ds003775 SRM Resting-state EEG")
    return parser.parse_args()


def main() -> None:
    np.random.seed(SEED)
    args = parse_args()
    model = FrozenModel.load(args.model_dir)
    external = pd.read_csv(args.external_features)
    if "age" not in external.columns:
        raise ValueError("External feature table must contain age.")

    external["age"] = pd.to_numeric(external["age"], errors="coerce")
    original_n = len(external)
    missing_age_n = int(external["age"].isna().sum())
    if args.min_age is not None:
        external = external[external["age"] >= args.min_age].copy()
    external = external.dropna(subset=["age"]).reset_index(drop=True)
    if len(external) < 10:
        raise ValueError(f"Too few external participants after age filtering: {len(external)}")

    projection = model.project(external, args.dataset_label)
    meta_cols = [col for col in ["participant_id", "age", "sex", "session", "filename", "n_eeg_channels"] if col in external.columns]
    out = external[meta_cols].reset_index(drop=True).join(projection.reset_index(drop=True))
    EXTERNAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    projection_path = EXTERNAL_RESULTS_DIR / "ds003775_external_projection.csv"
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
    metrics["dataset"] = "ds003775"
    metrics["validation_protocol"] = (
        "Full 93-feature fixed transfer: imputation, winsorization, scaler, PCA, "
        "trajectory axis, T-age calibration, and ridge coefficients loaded from "
        "ds005385 session-1 model artifacts without refitting."
    )
    metrics_path = EXTERNAL_RESULTS_DIR / "ds003775_validation_metrics.csv"
    write_csv(metrics, metrics_path)
    write_csv(metrics, RESULTS_TABLES_DIR / "ds003775_validation_metrics.csv")

    dataset_summary = {
        "dataset": "OpenNeuro ds003775, SRM Resting-state EEG",
        "doi": "10.18112/openneuro.ds003775.v1.2.1",
        "descriptor_doi": "10.1016/j.dib.2022.108647",
        "url": "https://openneuro.org/datasets/ds003775/versions/1.2.1",
        "license": "CC0",
        "raw_participants_n": int(original_n),
        "analysis_n": int(len(out)),
        "excluded_below_min_age": int(original_n - missing_age_n - len(out)),
        "missing_age_n": missing_age_n,
        "analysis_age_min": float(out["age"].min()),
        "analysis_age_max": float(out["age"].max()),
        "analysis_age_mean": float(out["age"].mean()),
        "analysis_age_sd": float(out["age"].std(ddof=1)),
        "female_n": int((out.get("sex", pd.Series(dtype=str)).astype(str).str.upper().str[0] == "F").sum()) if "sex" in out else None,
        "male_n": int((out.get("sex", pd.Series(dtype=str)).astype(str).str.upper().str[0] == "M").sum()) if "sex" in out else None,
        "resting_condition": "Four minutes uninterrupted eyes-closed resting-state EEG.",
        "montage": "BioSemi ActiveTwo, 64 electrodes, extended 10-20/10-10 positions.",
        "suitability": (
            "Independent public resting-state EEG dataset with age/sex metadata, "
            "64-channel scalp montage, average-referenced raw EDF files, and "
            "sufficient adult age variation for fixed-projection transfer."
        ),
    }
    write_json(dataset_summary, EXTERNAL_RESULTS_DIR / "ds003775_dataset_summary.json")

    print(f"Saved external projections: {projection_path}")
    print(f"Saved external metrics: {metrics_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
