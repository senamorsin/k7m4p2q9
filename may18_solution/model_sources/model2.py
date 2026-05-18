#!/usr/bin/env python3
"""
Compact, deterministic wind-power prediction pipeline.
Optimized for MAE/90.09*100 metric, CPU-only, fallback-ready.
"""
from __future__ import annotations

import os
import argparse
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

CAPACITY = 90.09
SEEDS = [42, 123, 777]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Wind directions stored as degrees/1000 in competition CSV
    for c in df.columns:
        if c.startswith("wind_direction"):
            df[c] = df[c].astype(float) * 1000.0
            rad = np.deg2rad(df[c])
            df[f"{c}_sin"] = np.sin(rad)
            df[f"{c}_cos"] = np.cos(rad)
            df[f"{c}_sin2"] = np.sin(2.0 * rad)
            df[f"{c}_cos2"] = np.cos(2.0 * rad)

    for c in df.columns:
        if c.startswith("wind_speed"):
            df[c] = df[c].astype(float)
            df[f"{c}_sq"] = df[c] ** 2
            df[f"{c}_log"] = np.log1p(df[c])

    if "wind_speed_180m" in df.columns and "wind_speed_10m" in df.columns:
        df["shear_180_10"] = df["wind_speed_180m"] / (df["wind_speed_10m"] + 1e-3)
    if "wind_speed_120m" in df.columns and "wind_speed_80m" in df.columns:
        df["shear_120_80"] = df["wind_speed_120m"] / (df["wind_speed_80m"] + 1e-3)
    if "wind_gusts_10m" in df.columns and "wind_speed_10m" in df.columns:
        df["gust_ratio"] = df["wind_gusts_10m"] / (df["wind_speed_10m"] + 1e-3)

    if "pressure_msl" in df.columns and "temperature_80m" in df.columns:
        df["density_proxy"] = df["pressure_msl"] / (df["temperature_80m"] + 273.15)

    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour_of_day"] / 24.0)
    df["month_sin"] = np.sin(2.0 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * df["month"] / 12.0)

    if "Кол-во_ВЭУ_в_ремонте" in df.columns:
        df["repair_factor"] = 1.0 - df["Кол-во_ВЭУ_в_ремонте"] / 26.0

    if "wind_speed_80m" in df.columns:
        ws80 = df["wind_speed_80m"]
        df["regime_ramp"] = ((ws80 >= 3.0) & (ws80 <= 12.0)).astype(float)
        df["regime_rated"] = (ws80 > 12.0).astype(float)

    return df


def train_model(train_path: str, artifact_dir: str) -> None:
    os.makedirs(artifact_dir, exist_ok=True)
    df = pd.read_csv(train_path, parse_dates=["METEOFORECASTHOUR_OPENM_Datetime"])
    target = "Выработка. Результирующий расчет"
    y = df[target].values
    X_raw = df.drop(columns=[target, "METEOFORECASTHOUR_OPENM_Datetime"])
    X = engineer_features(X_raw)

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    feat_cols = [c for c in X_imp.columns if c not in (target, "METEOFORECASTHOUR_OPENM_Datetime")]

    models = []
    backend = "lgb"
    try:
        import lightgbm as lgb
        lib_cls = lgb.LGBMRegressor
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        lib_cls = HistGradientBoostingRegressor
        backend = "skl"

    for seed in SEEDS:
        if backend == "lgb":
            mdl = lib_cls(
                objective="regression_l1", metric="mae",
                num_leaves=31, learning_rate=0.05,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                min_child_samples=20, max_depth=-1,
                device="cpu", n_jobs=1, deterministic=True, force_col_wise=True,
                random_state=seed, verbose=-1
            )
        else:
            mdl = lib_cls(
                learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
                random_state=seed, verbose=0
            )
        mdl.fit(X_imp[feat_cols], y)
        models.append(mdl)

    joblib.dump({
        "imputer": imputer,
        "models": models,
        "feat_cols": feat_cols,
        "backend": backend
    }, os.path.join(artifact_dir, "model2_pipeline.pkl"))
    print(f"[Train] Saved {len(models)} {backend} models to {artifact_dir}")


def predict_day(
    features_path: str,
    artifact_dir: str,
    output_path: str,
    external_weather_dir: Optional[str] = None
) -> None:
    if external_weather_dir and os.path.isdir(external_weather_dir):
        # Placeholder for external weather merging if schema is known.
        pass

    art = joblib.load(os.path.join(artifact_dir, "model2_pipeline.pkl"))
    imputer = art["imputer"]
    models = art["models"]
    feat_cols = art["feat_cols"]

    df = pd.read_csv(features_path, parse_dates=["METEOFORECASTHOUR_OPENM_Datetime"])
    X_raw = df.drop(columns=["METEOFORECASTHOUR_OPENM_Datetime"], errors="ignore")
    X = engineer_features(X_raw)
    X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns)
    X_pred = X_imp[feat_cols]

    preds = np.array([m.predict(X_pred) for m in models])
    final = np.median(preds, axis=0)
    final = np.clip(final, 0.0, CAPACITY)

    out = pd.DataFrame({"Выработка": final})
    assert len(out) == len(df), f"Row count mismatch: {len(out)} vs {len(df)}"
    out.to_csv(output_path, index=False)
    print(f"[Predict] Saved {len(out)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wind Power Judge-Day Pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--train_path", required=True)
    t.add_argument("--artifact_dir", required=True)

    p = sub.add_parser("predict")
    p.add_argument("--features_path", required=True)
    p.add_argument("--artifact_dir", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--external_weather_dir", default=None)

    args = parser.parse_args()
    if args.cmd == "train":
        train_model(args.train_path, args.artifact_dir)
    else:
        predict_day(args.features_path, args.artifact_dir, args.output_path, args.external_weather_dir)
