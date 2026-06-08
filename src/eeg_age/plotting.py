"""Small plotting helpers for reproducibility scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def scatter_age_prediction(df: pd.DataFrame, x: str, y: str, path: Path, *, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df[x], df[y], alpha=0.75, edgecolor="none")
    lo = min(float(df[x].min()), float(df[y].min()))
    hi = max(float(df[x].max()), float(df[y].max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if path.suffix.lower() == ".pdf":
        fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
