"""Fit and save frozen development transformations from ds005385 session-1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_age.config import DEVELOPMENT_FEATURES, MODELS_DIR, PROJECTIONS_DIR, RESULTS_TABLES_DIR, SEED  # noqa: E402
from eeg_age.io import write_csv  # noqa: E402
from eeg_age.models import assert_no_metadata_leakage, feature_columns, fit_frozen_model  # noqa: E402
from eeg_age.stats import metric_table_for_association, metric_table_for_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-features", type=Path, default=DEVELOPMENT_FEATURES)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    return parser.parse_args()


def main() -> None:
    np.random.seed(SEED)
    args = parse_args()
    baseline = pd.read_csv(args.baseline_features)
    features = feature_columns(baseline)
    assert_no_metadata_leakage(features)

    model = fit_frozen_model(baseline, model_dir=args.model_dir, seed=SEED)
    projection = model.project(baseline, "ds005385 session-1")
    meta_cols = [col for col in ["participant_id", "age", "sex"] if col in baseline.columns]
    projection = baseline[meta_cols].reset_index(drop=True).join(projection.reset_index(drop=True))

    write_csv(projection, PROJECTIONS_DIR / "development_baseline_projection.csv")

    t_assoc = metric_table_for_association(
        "Development fixed T vs chronological age",
        projection["T"],
        projection["age"],
    )
    t_pred = metric_table_for_predictions(
        "Development T-calibrated age prediction (apparent)",
        projection["age"],
        projection["T_predicted_age"],
    )
    ridge_pred = metric_table_for_predictions(
        "Development ridge age prediction (apparent)",
        projection["age"],
        projection["ridge_predicted_age"],
    )
    metrics = pd.concat([t_assoc, t_pred, ridge_pred], ignore_index=True)
    write_csv(metrics, RESULTS_TABLES_DIR / "development_fixed_projection_metrics.csv")

    print(f"Saved frozen model parameters to {args.model_dir}")
    print(f"Saved development projections to {PROJECTIONS_DIR / 'development_baseline_projection.csv'}")
    print(f"Feature count: {len(model.feature_list)}")


if __name__ == "__main__":
    main()
