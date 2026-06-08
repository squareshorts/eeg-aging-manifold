"""Generate reproducible figures from saved projection outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_age.config import EXTERNAL_RESULTS_DIR, FIGURES_DIR, SEED  # noqa: E402
from eeg_age.plotting import scatter_age_prediction  # noqa: E402


def main() -> None:
    np.random.seed(SEED)
    external_path = EXTERNAL_RESULTS_DIR / "ds003775_external_projection.csv"
    if external_path.exists():
        df = pd.read_csv(external_path)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        ax.scatter(df["age"], df["T"], alpha=0.78, edgecolor="none")
        ax.set_xlabel("Chronological age")
        ax.set_ylabel("External fixed-projection T")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "external_ds003775_T_vs_age.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "external_ds003775_T_vs_age.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        scatter_age_prediction(
            df,
            "age",
            "ridge_predicted_age",
            FIGURES_DIR / "external_ds003775_ridge_age_prediction.pdf",
            xlabel="Chronological age",
            ylabel="Transferred ridge predicted age",
        )
        print(f"Saved external validation figures to {FIGURES_DIR}")
    else:
        print(f"External projection file not found, skipping external figures: {external_path}")


if __name__ == "__main__":
    main()
