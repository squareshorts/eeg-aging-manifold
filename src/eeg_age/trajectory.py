"""Fixed aging-axis geometry for PCA projections."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_age_axis(
    pc_scores: pd.DataFrame,
    age,
    *,
    components: tuple[str, str] = ("PC1", "PC4"),
    lower_quantile: float = 0.20,
    upper_quantile: float = 0.80,
) -> dict[str, object]:
    age_arr = np.asarray(age, dtype=float)
    z = pc_scores.loc[:, list(components)].to_numpy(float)
    q_low = float(np.nanquantile(age_arr, lower_quantile))
    q_high = float(np.nanquantile(age_arr, upper_quantile))
    young = np.nanmean(z[age_arr <= q_low], axis=0)
    old = np.nanmean(z[age_arr >= q_high], axis=0)
    direction = old - young
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Cannot define aging axis because old-young centroid difference is zero.")
    direction = direction / norm
    signed_axis = np.array([-direction[1], direction[0]])
    return {
        "components": list(components),
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "young_age_threshold": q_low,
        "old_age_threshold": q_high,
        "young_centroid": young.tolist(),
        "old_centroid": old.tolist(),
        "direction": direction.tolist(),
        "signed_axis": signed_axis.tolist(),
    }


def project_trajectory(pc_scores: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
    components = list(params["components"])
    z = pc_scores.loc[:, components].to_numpy(float)
    young = np.asarray(params["young_centroid"], dtype=float)
    direction = np.asarray(params["direction"], dtype=float)
    signed_axis = np.asarray(params["signed_axis"], dtype=float)
    centered = z - young
    t = centered @ direction
    d_signed = centered @ signed_axis
    d_abs = np.abs(d_signed)
    return pd.DataFrame(
        {
            "T": t,
            "D_signed": d_signed,
            "D_abs": d_abs,
        },
        index=pc_scores.index,
    )
