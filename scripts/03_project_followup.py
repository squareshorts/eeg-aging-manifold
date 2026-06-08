"""Project ds005385 session-2 into the frozen session-1 coordinate system."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_age.config import DEVELOPMENT_FEATURES, FOLLOWUP_FEATURES, MODELS_DIR, PROJECTIONS_DIR, RESULTS_TABLES_DIR, SEED  # noqa: E402
from eeg_age.io import write_csv  # noqa: E402
from eeg_age.models import FrozenModel  # noqa: E402
from eeg_age.stats import safe_spearman  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-features", type=Path, default=DEVELOPMENT_FEATURES)
    parser.add_argument("--followup-features", type=Path, default=FOLLOWUP_FEATURES)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    return parser.parse_args()


def main() -> None:
    np.random.seed(SEED)
    args = parse_args()
    model = FrozenModel.load(args.model_dir)
    baseline = pd.read_csv(args.baseline_features)
    followup = pd.read_csv(args.followup_features)

    id_col = "participant_id"
    followup_ids = followup[id_col].astype(str)
    baseline_subset = baseline[baseline[id_col].astype(str).isin(followup_ids)].copy()
    baseline_subset = baseline_subset.set_index(id_col).loc[followup_ids].reset_index()

    base_projection = model.project(baseline_subset, "ds005385 session-1 longitudinal subset")
    fu_projection = model.project(followup, "ds005385 session-2 follow-up")

    meta_cols = [col for col in [id_col, "age", "sex"] if col in followup.columns]
    base_out = baseline_subset[meta_cols].reset_index(drop=True).join(base_projection.add_prefix("base_").reset_index(drop=True))
    fu_out = followup[meta_cols].reset_index(drop=True).join(fu_projection.add_prefix("fu_").reset_index(drop=True))

    long_df = base_out[[id_col, "age", "sex"]].copy()
    for col in ["T", "D_signed", "D_abs", "ridge_predicted_age", "T_predicted_age"]:
        long_df[f"base_{col}"] = base_projection[col].to_numpy(float)
        long_df[f"fu_{col}"] = fu_projection[col].to_numpy(float)
        long_df[f"delta_{col}"] = long_df[f"fu_{col}"] - long_df[f"base_{col}"]

    write_csv(base_out, PROJECTIONS_DIR / "followup_subset_baseline_projection.csv")
    write_csv(fu_out, PROJECTIONS_DIR / "followup_session2_projection.csv")
    write_csv(long_df, PROJECTIONS_DIR / "followup_longitudinal_projection.csv")

    rows = []
    for metric in ["T", "D_signed", "D_abs"]:
        delta = long_df[f"delta_{metric}"].replace([np.inf, -np.inf], np.nan).dropna()
        w = wilcoxon(delta[delta != 0]) if np.any(delta != 0) else None
        rho, p, n = safe_spearman(long_df["age"], long_df[f"delta_{metric}"])
        rows.append(
            {
                "metric": metric,
                "n": len(delta),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "std_delta": float(delta.std(ddof=1)),
                "wilcoxon_p": float(w.pvalue) if w is not None else np.nan,
                "spearman_age_delta_rho": rho,
                "spearman_age_delta_p": p,
                "spearman_n": n,
            }
        )
    write_csv(pd.DataFrame(rows), RESULTS_TABLES_DIR / "followup_projection_summary.csv")
    print(f"Saved follow-up projections to {PROJECTIONS_DIR}")


if __name__ == "__main__":
    main()
