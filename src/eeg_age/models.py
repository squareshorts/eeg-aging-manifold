"""Frozen feature transformations, PCA projection, and ridge prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

from .config import METADATA_COLS, MODEL_FILES
from .io import read_json, write_csv, write_json
from .trajectory import fit_age_axis, project_trajectory


def feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include="number").columns.tolist()
    return [
        col
        for col in numeric
        if col not in METADATA_COLS
        and not any(term in col.lower() for term in ["session", "ses", "late"])
    ]


def assert_no_metadata_leakage(features: list[str]) -> None:
    bad = [
        col
        for col in features
        if col in METADATA_COLS or any(term in col.lower() for term in ["age", "sex", "session", "participant"])
    ]
    if bad:
        raise ValueError(f"Metadata-like columns would enter the model: {bad}")


def assert_feature_order(df: pd.DataFrame, feature_list: list[str], label: str) -> None:
    missing = [col for col in feature_list if col not in df.columns]
    extra_numeric = [col for col in feature_columns(df) if col not in feature_list]
    if missing:
        raise ValueError(f"{label} is missing frozen model features: {missing}")
    observed = [col for col in df.columns if col in feature_list]
    if observed != feature_list:
        raise ValueError(
            f"{label} feature order differs from frozen model order. "
            "Reindex with the saved feature list before projection."
        )
    if extra_numeric:
        raise ValueError(f"{label} contains unexpected numeric EEG features: {extra_numeric}")


def reindex_features(df: pd.DataFrame, feature_list: list[str], label: str) -> pd.DataFrame:
    missing = [col for col in feature_list if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing frozen model features: {missing}")
    return df.loc[:, feature_list].copy()


def basic_transform(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    x = df.loc[:, feature_list].copy()
    for col in feature_list:
        if "_abs_" in col:
            x[col] = np.log10(x[col].astype(float) + 1e-30)
    return x.replace([np.inf, -np.inf], np.nan)


@dataclass
class FrozenModel:
    feature_list: list[str]
    transform_params: pd.DataFrame
    pca_loadings: pd.DataFrame
    pca_params: dict[str, object]
    trajectory_params: dict[str, object]
    ridge_coefficients: pd.DataFrame
    ridge_params: dict[str, object]
    t_age_calibration: dict[str, float]

    @classmethod
    def load(cls, model_dir: Path) -> "FrozenModel":
        feature_list = read_json(model_dir / MODEL_FILES["feature_list"])["features"]
        return cls(
            feature_list=feature_list,
            transform_params=pd.read_csv(model_dir / MODEL_FILES["transform"]),
            pca_loadings=pd.read_csv(model_dir / MODEL_FILES["pca_loadings"]),
            pca_params=read_json(model_dir / MODEL_FILES["pca_params"]),
            trajectory_params=read_json(model_dir / MODEL_FILES["trajectory"]),
            ridge_coefficients=pd.read_csv(model_dir / MODEL_FILES["ridge"]),
            ridge_params=read_json(model_dir / MODEL_FILES["ridge_params"]),
            t_age_calibration=read_json(model_dir / MODEL_FILES["t_age_calibration"]),
        )

    def transform_features(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        x = reindex_features(df, self.feature_list, label)
        x = basic_transform(x, self.feature_list)
        params = self.transform_params.set_index("feature").loc[self.feature_list]
        x = x.fillna(params["imputation_median"])
        x = x.clip(params["winsor_lower"], params["winsor_upper"], axis=1)
        scaled = (x - params["scaler_mean"]) / params["scaler_sd"].replace(0, np.nan)
        return scaled.fillna(0.0)

    def pca_scores(self, scaled: pd.DataFrame) -> pd.DataFrame:
        loadings = self.pca_loadings.set_index("feature")
        pc_cols = [col for col in loadings.columns if col.startswith("PC")]
        components = loadings.loc[self.feature_list, pc_cols].to_numpy(float)
        pca_mean = np.asarray(self.pca_params["mean"], dtype=float)
        scores = (scaled.to_numpy(float) - pca_mean) @ components
        return pd.DataFrame(scores, columns=pc_cols, index=scaled.index)

    def ridge_predict(self, scaled: pd.DataFrame) -> np.ndarray:
        coefs = self.ridge_coefficients.set_index("feature").loc[self.feature_list, "coefficient"].to_numpy(float)
        intercept = float(self.ridge_params["intercept"])
        return scaled.to_numpy(float) @ coefs + intercept

    def project(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        scaled = self.transform_features(df, label)
        pc = self.pca_scores(scaled)
        traj = project_trajectory(pc, self.trajectory_params)
        out = pc.join(traj)
        out["ridge_predicted_age"] = self.ridge_predict(scaled)
        out["T_predicted_age"] = (
            float(self.t_age_calibration["intercept"])
            + float(self.t_age_calibration["slope"]) * out["T"].to_numpy(float)
        )
        return out


def fit_frozen_model(
    baseline_df: pd.DataFrame,
    *,
    model_dir: Path,
    seed: int,
    n_components: int = 10,
) -> FrozenModel:
    model_dir.mkdir(parents=True, exist_ok=True)
    features = feature_columns(baseline_df)
    assert_no_metadata_leakage(features)
    if "age" not in baseline_df.columns:
        raise ValueError("Development feature table must contain age.")

    x_basic = basic_transform(baseline_df, features)
    medians = x_basic.median(axis=0)
    x_imputed = x_basic.fillna(medians)
    lower = x_imputed.quantile(0.01, axis=0)
    upper = x_imputed.quantile(0.99, axis=0)
    x_winsor = x_imputed.clip(lower, upper, axis=1)
    scaler = StandardScaler()
    x_scaled_arr = scaler.fit_transform(x_winsor)
    x_scaled = pd.DataFrame(x_scaled_arr, columns=features, index=baseline_df.index)

    pca = PCA(n_components=n_components, random_state=seed)
    scores_arr = pca.fit_transform(x_scaled)
    pc_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    scores = pd.DataFrame(scores_arr, columns=pc_cols, index=baseline_df.index)

    age = baseline_df["age"].to_numpy(float)
    trajectory_params = fit_age_axis(scores, age)
    traj = project_trajectory(scores, trajectory_params)

    ridge = RidgeCV(alphas=np.logspace(-2, 6, 80))
    ridge.fit(x_scaled, age)

    t_age_model = LinearRegression()
    t_age_model.fit(traj[["T"]], age)

    transform_params = pd.DataFrame(
        {
            "feature": features,
            "log10_abs_power": ["_abs_" in col for col in features],
            "imputation_median": medians.loc[features].to_numpy(float),
            "winsor_lower": lower.loc[features].to_numpy(float),
            "winsor_upper": upper.loc[features].to_numpy(float),
            "scaler_mean": scaler.mean_,
            "scaler_sd": scaler.scale_,
        }
    )
    pca_loadings = pd.DataFrame(pca.components_.T, columns=pc_cols)
    pca_loadings.insert(0, "feature", features)
    ridge_coefficients = pd.DataFrame({"feature": features, "coefficient": ridge.coef_})

    pca_params = {
        "n_components": n_components,
        "mean": pca.mean_.tolist(),
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pc_cols,
    }
    ridge_params = {
        "alpha": float(ridge.alpha_),
        "intercept": float(ridge.intercept_),
        "target": "chronological age",
        "input_space": "development-fitted transformed and standardized EEG features",
    }
    t_age_calibration = {
        "intercept": float(t_age_model.intercept_),
        "slope": float(t_age_model.coef_[0]),
        "target": "chronological age",
        "input": "fixed-projection trajectory coordinate T",
    }
    manifest = {
        "model_version": model_dir.name,
        "seed": seed,
        "development_dataset": "OpenNeuro ds005385 session-1 eyes-closed pre-task",
        "n_development": int(len(baseline_df)),
        "n_features": int(len(features)),
        "age_min": float(np.nanmin(age)),
        "age_max": float(np.nanmax(age)),
        "fit_scope": "All transformations, PCA, age-axis centroids, T-age calibration, and ridge coefficients are fitted only on ds005385 session-1.",
    }

    write_json({"features": features}, model_dir / MODEL_FILES["feature_list"])
    write_csv(transform_params, model_dir / MODEL_FILES["transform"])
    write_csv(pca_loadings, model_dir / MODEL_FILES["pca_loadings"])
    write_json(pca_params, model_dir / MODEL_FILES["pca_params"])
    write_json(trajectory_params, model_dir / MODEL_FILES["trajectory"])
    write_csv(ridge_coefficients, model_dir / MODEL_FILES["ridge"])
    write_json(ridge_params, model_dir / MODEL_FILES["ridge_params"])
    write_json(t_age_calibration, model_dir / MODEL_FILES["t_age_calibration"])
    write_json(manifest, model_dir / MODEL_FILES["manifest"])

    return FrozenModel.load(model_dir)
