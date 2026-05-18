#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model1.py

Deterministic wind-power judge-day predictor.

Public interface:
    train_model(train_path: str, artifact_dir: str) -> None
    predict_day(features_path: str, artifact_dir: str, output_path: str,
                external_weather_dir: str | None = None) -> None

CLI:
    python model1.py train --train data/raw/train_dataset_plus_may18_recent.csv --artifact-dir model_weights/model1
    python model1.py predict --features data/raw/may18_features_actualized_bestmatch.csv \
        --artifact-dir model_weights/model1 --output submissions/model1.csv

Design:
    - deterministic preprocessing and model seeds;
    - physical/empirical power-curve baseline;
    - residual tree ensemble under MAE/median objectives;
    - hierarchical median residual correction tables;
    - analog median component for robust future-day inference;
    - optional external-weather defensive ingestion and test-time alternate forecast path;
    - strict output validation: one column named "Выработка", finite values clipped to [0, 90.09].

No internet calls are made by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Determinism before numeric imports where possible
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error

try:  # optional but recommended
    import lightgbm as lgb
except Exception:  # pragma: no cover - fallback is intentional
    lgb = None

try:  # optional second model family
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - fallback is intentional
    CatBoostRegressor = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.modules.setdefault("model1", sys.modules[__name__])
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings("ignore", category=PerformanceWarning)
except Exception:
    pass

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

DATE_COL = "METEOFORECASTHOUR_OPENM_Datetime"
TARGET_COL = "Выработка. Результирующий расчет"
SUBMISSION_COL = "Выработка"

N_INST = 90.09
N_TURBINES = 26.0
P_RATED_PER_TURBINE = 3.465
H_HUB = 80.0
LAT_DEG = 46.8268455973
LON_DEG = 38.7179393185
UTC_OFFSET_H = 3.0

RHO_0 = 1.225
R_DRY = 287.05
G = 9.81
V_CUT_IN = 3.0
V_RATED = 12.0
V_CUT_OUT = 25.0
WD_SCALE = 1000.0

ARTIFACT_NAME = "model1_bundle.joblib"
META_NAME = "model1_meta.json"

# Target-day prior: the final private day is May 18.  This is problem statement
# information, not a fitted target-value constant.  It only affects sample weights,
# not any post-hoc target-day offset.
JUDGE_MONTH = 5

# All optional external-weather features are always present so the model schema is
# stable even when no external files are supplied.
EXT_COLS = [
    "ext_available", "ext_source_count",
    "ext_ws80_mean", "ext_ws80_std", "ext_ws80_min", "ext_ws80_max", "ext_ws80_spread",
    "ext_ws120_mean", "ext_ws120_std", "ext_ws180_mean", "ext_ws180_std",
    "ext_wd80_mean", "ext_wd80_sin_mean", "ext_wd80_cos_mean",
    "ext_temp80_mean", "ext_pressure_mean", "ext_gust_mean",
    "ext_rain_mean", "ext_cloud_low_mean",
    "ext_ws80_delta", "ext_ws80_abs_delta", "ext_gust_delta", "ext_temp80_delta",
    "ext_pressure_delta", "ext_cloud_delta",
]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def _sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_date_col(cols: Iterable[str]) -> str:
    cols = list(cols)
    if DATE_COL in cols:
        return DATE_COL
    for c in cols:
        lc = c.lower()
        if lc in {"datetime", "date", "time", "timestamp"} or "datetime" in lc:
            return c
    raise ValueError(f"Could not find datetime column. Expected {DATE_COL!r} or a datetime-like column.")


def _safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c.startswith("__") or c == "datetime":
            continue
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _as_deg_if_scaled(s: pd.Series) -> pd.Series:
    """Competition wind directions are stored as degrees/1000. Convert only if needed."""
    x = pd.to_numeric(s, errors="coerce").astype(float)
    q = float(np.nanquantile(np.abs(x.values), 0.99)) if x.notna().any() else 0.0
    if q <= 3.6:  # 0.275 -> 275 deg
        x = x * WD_SCALE
    return np.mod(x, 360.0)


def _circular_mean_deg(values: np.ndarray, axis: int = 0) -> np.ndarray:
    rad = np.deg2rad(values)
    s = np.nanmean(np.sin(rad), axis=axis)
    c = np.nanmean(np.cos(rad), axis=axis)
    return np.mod(np.rad2deg(np.arctan2(s, c)), 360.0)


def _angle_diff_deg(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    """Smallest signed difference a-b in degrees."""
    return ((np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0) - 180.0


def _clip_power(x: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 0.0, N_INST)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def air_density(pressure_hpa: np.ndarray, temp_celsius: np.ndarray) -> np.ndarray:
    """Air density at hub height from MSL pressure [hPa] and temperature [C]."""
    pressure_hpa = np.asarray(pressure_hpa, dtype=float)
    temp_celsius = np.asarray(temp_celsius, dtype=float)
    p_hub_hpa = pressure_hpa * (1.0 - 0.0065 * H_HUB / 288.15) ** 5.2561
    p_pa = p_hub_hpa * 100.0
    temp_k = temp_celsius + 273.15
    return p_pa / (R_DRY * np.maximum(temp_k, 150.0))


def effective_wind(v_hub: np.ndarray, density: np.ndarray) -> np.ndarray:
    return np.asarray(v_hub, dtype=float) * np.power(np.maximum(density, 0.7) / RHO_0, 1.0 / 3.0)


def power_curve(v: np.ndarray,
                v_cut_in: float = V_CUT_IN,
                v_rated: float = V_RATED,
                v_cut_out: float = V_CUT_OUT,
                p_rated: float = 1.0) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    p = np.zeros_like(v, dtype=float)
    ramp = (v >= v_cut_in) & (v < v_rated)
    denom = v_rated ** 3 - v_cut_in ** 3
    p[ramp] = p_rated * (v[ramp] ** 3 - v_cut_in ** 3) / denom
    plateau = (v >= v_rated) & (v <= v_cut_out)
    p[plateau] = p_rated
    return p


def physical_power(v_eff: np.ndarray, n_avail: np.ndarray) -> np.ndarray:
    return power_curve(v_eff) * np.asarray(n_avail, dtype=float) * P_RATED_PER_TURBINE


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def _read_competition_csv(path: str | Path, has_target: bool) -> Tuple[pd.DataFrame, int]:
    raw = pd.read_csv(path)
    n_original = len(raw)
    date_col = _find_date_col(raw.columns)
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    if raw[date_col].isna().any():
        bad = int(raw[date_col].isna().sum())
        raise ValueError(f"{path}: {bad} rows have invalid datetimes")

    raw = raw.rename(columns={date_col: "datetime"})
    raw["__row_order"] = np.arange(len(raw), dtype=np.int64)

    if has_target and TARGET_COL not in raw.columns:
        raise ValueError(f"Training file must contain target column {TARGET_COL!r}")
    if not has_target and TARGET_COL in raw.columns:
        raw = raw.drop(columns=[TARGET_COL])

    raw = _safe_numeric_df(raw)
    raw = raw.sort_values("datetime", kind="mergesort")
    # For training duplicates, keep the first timestamp.  For prediction, duplicate
    # timestamps are allowed but uncommon; later row-order restoration keeps output length.
    if has_target:
        raw = raw.drop_duplicates("datetime", keep="first")
    raw = raw.set_index("datetime", drop=True).sort_index(kind="mergesort")

    # Fix wind directions once, immediately after reading.
    for c in [c for c in raw.columns if c.startswith("wind_direction_")]:
        raw[c] = _as_deg_if_scaled(raw[c])

    # Canonical calendar columns from timestamp, never trusting stale CSV values.
    raw["month"] = raw.index.month.astype(np.int16)
    raw["hour_of_day"] = raw.index.hour.astype(np.int16)

    # Required repair column is rare but critical; if absent, assume all turbines available.
    if "Кол-во_ВЭУ_в_ремонте" not in raw.columns:
        raw["Кол-во_ВЭУ_в_ремонте"] = 0.0

    # Fill weather columns.  Keep target untouched.
    fill_cols = [c for c in raw.columns if c not in {TARGET_COL, "__row_order"}]
    for c in fill_cols:
        if not pd.api.types.is_numeric_dtype(raw[c]):
            continue
        if raw[c].isna().any():
            raw[c] = raw[c].interpolate(method="time", limit_direction="both")
            raw[c] = raw[c].ffill().bfill()

    # Defensive fallbacks for 180m data, missing in older rows.
    if "wind_speed_180m" in raw.columns:
        raw["wind_speed_180m"] = raw["wind_speed_180m"].fillna(raw.get("wind_speed_120m", raw.get("wind_speed_80m")))
    if "wind_direction_180m" in raw.columns:
        raw["wind_direction_180m"] = raw["wind_direction_180m"].fillna(raw.get("wind_direction_120m", raw.get("wind_direction_80m")))

    return raw, n_original


# ---------------------------------------------------------------------------
# External-weather ingestion
# ---------------------------------------------------------------------------

def _read_json_weather(path: Path) -> Optional[pd.DataFrame]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None

    # Open-Meteo-like {"hourly": {"time": [...], "wind_speed_80m": [...]}}
    if isinstance(obj, dict) and isinstance(obj.get("hourly"), dict):
        hourly = obj["hourly"]
        if any(k in hourly for k in ["time", "datetime", "date"]):
            return pd.DataFrame(hourly)
    if isinstance(obj, dict):
        try:
            return pd.json_normalize(obj)
        except Exception:
            return None
    if isinstance(obj, list):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return None
    return None


def _read_external_file(path: Path) -> Optional[pd.DataFrame]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".json", ".jsonl"}:
            if suffix == ".jsonl":
                return pd.read_json(path, lines=True)
            return _read_json_weather(path)
    except Exception:
        return None
    return None


def _pick_col(cols: List[str], patterns: List[str]) -> Optional[str]:
    lower = {c.lower().replace(" ", "_"): c for c in cols}
    for p in patterns:
        p = p.lower()
        # exact-ish first
        for lc, orig in lower.items():
            if lc == p:
                return orig
        for lc, orig in lower.items():
            if p in lc:
                return orig
    return None


def _normalise_external_source(df: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    date_col = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"time", "datetime", "date", "timestamp"} or "datetime" in lc:
            date_col = c
            break
    if date_col is None:
        return None

    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(df[date_col], errors="coerce")
    if out["datetime"].isna().all():
        return None

    cols = [str(c) for c in df.columns]
    mapping = {
        "ws80": ["wind_speed_80m", "windspeed_80m", "wind_speed_100m", "windspeed_100m", "ws_80", "ws80", "wind_speed"],
        "ws120": ["wind_speed_120m", "windspeed_120m", "ws120"],
        "ws180": ["wind_speed_180m", "windspeed_180m", "ws180"],
        "wd80": ["wind_direction_80m", "winddirection_80m", "wind_direction_100m", "winddirection_100m", "wd_80", "wd80", "wind_direction"],
        "temp80": ["temperature_80m", "temperature_100m", "temp_80", "t_80", "temperature_2m", "temperature"],
        "pressure": ["pressure_msl", "surface_pressure", "pressure"],
        "gust": ["wind_gusts_10m", "windgusts_10m", "gust", "wind_gusts"],
        "rain": ["rain", "precipitation", "precip"],
        "cloud_low": ["cloud_cover_low", "low_cloud_cover", "cloudcover_low", "cloud_cover"],
    }
    got_any = False
    for name, pats in mapping.items():
        col = _pick_col(cols, pats)
        if col is not None and col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            if name.startswith("wd"):
                x = _as_deg_if_scaled(x)
            out[name] = x
            got_any = True
    if not got_any:
        return None

    out = out.dropna(subset=["datetime"]).sort_values("datetime")
    out = out.drop_duplicates("datetime", keep="first").set_index("datetime")
    # Prefix values with source identity to avoid collisions before aggregation.
    out = out.add_prefix(f"{source_name}__")
    return out


def load_external_weather(external_weather_dir: str | None, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load and aggregate any CSV/JSON weather files from a directory.

    The model remains deterministic and defensive: unknown files/columns are skipped.
    Returned frame always has EXT_COLS and is aligned to index.
    """
    ext = pd.DataFrame(index=index)
    for c in EXT_COLS:
        ext[c] = 0.0
    if external_weather_dir is None:
        return ext

    root = Path(external_weather_dir)
    if not root.exists() or not root.is_dir():
        return ext

    frames: List[pd.DataFrame] = []
    for i, path in enumerate(sorted(root.rglob("*"))):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
            continue
        raw = _read_external_file(path)
        norm = _normalise_external_source(raw, f"src{i:02d}")
        if norm is not None and not norm.empty:
            frames.append(norm)

    if not frames:
        return ext

    joined = pd.concat(frames, axis=1).sort_index()
    # Align exactly to forecast timestamps using nearest same-time reindexing; no extrapolating
    # beyond provided source coverage except one-hour tolerance.
    joined = joined.reindex(index, method=None)

    def cols_for(kind: str) -> List[str]:
        return [c for c in joined.columns if c.endswith(f"__{kind}")]

    def set_stats(kind: str, prefix: str) -> None:
        cs = cols_for(kind)
        if not cs:
            return
        vals = joined[cs].astype(float)
        count = vals.notna().sum(axis=1).astype(float)
        ext["ext_source_count"] = np.maximum(ext["ext_source_count"].values, count.values)
        ext[f"{prefix}_mean"] = vals.mean(axis=1).fillna(0.0).values
        if f"{prefix}_std" in ext.columns:
            ext[f"{prefix}_std"] = vals.std(axis=1).fillna(0.0).values
        if f"{prefix}_min" in ext.columns:
            ext[f"{prefix}_min"] = vals.min(axis=1).fillna(0.0).values
        if f"{prefix}_max" in ext.columns:
            ext[f"{prefix}_max"] = vals.max(axis=1).fillna(0.0).values
            ext[f"{prefix}_spread"] = (vals.max(axis=1) - vals.min(axis=1)).fillna(0.0).values

    set_stats("ws80", "ext_ws80")
    set_stats("ws120", "ext_ws120")
    set_stats("ws180", "ext_ws180")
    set_stats("temp80", "ext_temp80")
    set_stats("pressure", "ext_pressure")
    set_stats("gust", "ext_gust")
    set_stats("rain", "ext_rain")
    set_stats("cloud_low", "ext_cloud_low")

    wd_cols = cols_for("wd80")
    if wd_cols:
        vals = joined[wd_cols].astype(float).values
        ext["ext_wd80_mean"] = _circular_mean_deg(vals, axis=1)
        rad = np.deg2rad(vals)
        ext["ext_wd80_sin_mean"] = np.nanmean(np.sin(rad), axis=1)
        ext["ext_wd80_cos_mean"] = np.nanmean(np.cos(rad), axis=1)
        ext[["ext_wd80_mean", "ext_wd80_sin_mean", "ext_wd80_cos_mean"]] = ext[
            ["ext_wd80_mean", "ext_wd80_sin_mean", "ext_wd80_cos_mean"]
        ].fillna(0.0)

    ext["ext_available"] = (ext["ext_source_count"] > 0).astype(float)
    return ext.reindex(index).fillna(0.0)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _solar_elevation(index: pd.DatetimeIndex) -> np.ndarray:
    doy = index.dayofyear.values.astype(float)
    hour = index.hour.values.astype(float) + index.minute.values.astype(float) / 60.0
    local_solar_time = hour + LON_DEG / 15.0 - UTC_OFFSET_H
    hour_angle = np.deg2rad(15.0 * (local_solar_time - 12.0))
    decl = np.deg2rad(23.45 * np.sin(np.deg2rad(360.0 / 365.0 * (doy - 81.0))))
    lat = np.deg2rad(LAT_DEG)
    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.arcsin(np.clip(sin_elev, -1.0, 1.0))


def _add_required_weather_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "wind_speed_10m": np.nan,
        "wind_speed_80m": np.nan,
        "wind_speed_120m": np.nan,
        "wind_speed_180m": np.nan,
        "wind_direction_10m": np.nan,
        "wind_direction_80m": np.nan,
        "wind_direction_120m": np.nan,
        "wind_direction_180m": np.nan,
        "wind_gusts_10m": np.nan,
        "temperature_80m": 15.0,
        "temperature_120m": 14.7,
        "pressure_msl": 1013.25,
        "rain": 0.0,
        "showers": 0.0,
        "snowfall": 0.0,
        "cloud_cover_low": 0.0,
        "Кол-во_ВЭУ_в_ремонте": 0.0,
    }
    for c, val in defaults.items():
        if c not in out.columns:
            out[c] = val

    # Cross-fill wind speeds and directions if a level is missing.
    for c in ["wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if out["wind_speed_80m"].isna().all():
        out["wind_speed_80m"] = out["wind_speed_120m"].fillna(out["wind_speed_10m"])
    out["wind_speed_10m"] = out["wind_speed_10m"].fillna(out["wind_speed_80m"] * 0.75)
    out["wind_speed_120m"] = out["wind_speed_120m"].fillna(out["wind_speed_80m"] * 1.05)
    out["wind_speed_180m"] = out["wind_speed_180m"].fillna(out["wind_speed_120m"] * 1.05)

    for c in ["wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m"]:
        out[c] = _as_deg_if_scaled(out[c])
    if out["wind_direction_80m"].isna().all():
        out["wind_direction_80m"] = out["wind_direction_120m"].fillna(out["wind_direction_10m"])
    out["wind_direction_10m"] = out["wind_direction_10m"].fillna(out["wind_direction_80m"])
    out["wind_direction_120m"] = out["wind_direction_120m"].fillna(out["wind_direction_80m"])
    out["wind_direction_180m"] = out["wind_direction_180m"].fillna(out["wind_direction_120m"])

    for c in defaults:
        if out[c].isna().any():
            out[c] = out[c].interpolate(method="time", limit_direction="both")
            out[c] = out[c].ffill().bfill().fillna(defaults[c])
    return out


def make_features(raw_df: pd.DataFrame, external_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build deterministic weather/physics/calendar features.

    raw_df must be indexed by datetime and may contain target / __row_order.
    """
    df = _add_required_weather_defaults(raw_df)
    out = df.copy()
    idx = out.index

    # Availability
    repair = pd.to_numeric(out["Кол-во_ВЭУ_в_ремонте"], errors="coerce").fillna(0.0).clip(0.0, N_TURBINES)
    out["n_avail"] = (N_TURBINES - repair).clip(0.0, N_TURBINES)
    out["avail_ratio"] = out["n_avail"] / N_TURBINES

    # Core wind and density physics
    ws80 = out["wind_speed_80m"].astype(float).clip(lower=0.0, upper=60.0)
    out["ws_hub"] = ws80
    out["ws_hub_sq"] = ws80 ** 2
    out["ws_hub_cu"] = ws80 ** 3
    out["density"] = air_density(out["pressure_msl"].values, out["temperature_80m"].values)
    out["ws_eff"] = effective_wind(ws80.values, out["density"].values)
    out["ws_eff_sq"] = out["ws_eff"] ** 2
    out["ws_eff_cu"] = out["ws_eff"] ** 3
    out["wind_power_density"] = 0.5 * out["density"] * out["ws_eff"] ** 3
    out["power_phys"] = physical_power(out["ws_eff"].values, out["n_avail"].values)

    # Wind direction and vector components
    for h in [10, 80, 120, 180]:
        wd_col = f"wind_direction_{h}m"
        ws_col = f"wind_speed_{h}m"
        wd = _as_deg_if_scaled(out[wd_col])
        out[wd_col] = wd
        rad = np.deg2rad(wd.values)
        out[f"wd_sin_{h}"] = np.sin(rad)
        out[f"wd_cos_{h}"] = np.cos(rad)
        out[f"wd_sector_{h}"] = np.floor(wd / 30.0).fillna(-1).astype(np.int16)
        if ws_col in out.columns:
            out[f"u_wind_{h}"] = -out[ws_col].astype(float).values * np.sin(rad)
            out[f"v_wind_{h}"] = -out[ws_col].astype(float).values * np.cos(rad)

    out["wd_diff_10_80"] = _angle_diff_deg(out["wind_direction_80m"], out["wind_direction_10m"])
    out["wd_diff_80_120"] = _angle_diff_deg(out["wind_direction_120m"], out["wind_direction_80m"])
    out["wd_diff_80_180"] = _angle_diff_deg(out["wind_direction_180m"], out["wind_direction_80m"])
    out["wd_coherence_10_80"] = np.cos(np.deg2rad(out["wd_diff_10_80"]))
    out["wd_coherence_80_180"] = np.cos(np.deg2rad(out["wd_diff_80_180"]))

    # Shear / vertical profile
    eps = 1e-4
    out["shear_10_80"] = out["wind_speed_80m"] - out["wind_speed_10m"]
    out["shear_80_120"] = out["wind_speed_120m"] - out["wind_speed_80m"]
    out["shear_120_180"] = out["wind_speed_180m"] - out["wind_speed_120m"]
    out["shear_10_180"] = out["wind_speed_180m"] - out["wind_speed_10m"]
    out["shear_ratio_10_80"] = out["shear_10_80"] / out["wind_speed_10m"].clip(lower=0.5)
    out["shear_ratio_80_180"] = (out["wind_speed_180m"] - out["wind_speed_80m"]) / out["wind_speed_80m"].clip(lower=0.5)
    out["alpha_10_80"] = np.log((out["wind_speed_80m"] + eps) / (out["wind_speed_10m"] + eps)) / np.log(80.0 / 10.0)
    out["alpha_80_180"] = np.log((out["wind_speed_180m"] + eps) / (out["wind_speed_80m"] + eps)) / np.log(180.0 / 80.0)
    out["alpha_diff"] = out["alpha_80_180"] - out["alpha_10_80"]
    ws_profile = np.vstack([
        out["wind_speed_10m"].values,
        out["wind_speed_80m"].values,
        out["wind_speed_120m"].values,
        out["wind_speed_180m"].values,
    ]).T
    out["ws_profile_mean"] = np.nanmean(ws_profile, axis=1)
    out["ws_profile_std"] = np.nanstd(ws_profile, axis=1)
    out["ws_profile_range"] = np.nanmax(ws_profile, axis=1) - np.nanmin(ws_profile, axis=1)
    out["ws_profile_cv"] = out["ws_profile_std"] / out["ws_hub"].clip(lower=0.5)
    out["ws_upper_lower_ratio"] = out["wind_speed_180m"] / out["wind_speed_10m"].clip(lower=0.5)

    # Gusts and precipitation/clouds
    out["gust_ratio"] = out["wind_gusts_10m"] / out["ws_hub"].clip(lower=0.5)
    out["gust_excess"] = (out["wind_gusts_10m"] - out["ws_hub"]).clip(lower=0.0)
    out["precip_total"] = out["rain"].fillna(0.0) + out["showers"].fillna(0.0) + out["snowfall"].fillna(0.0)
    # Competition low cloud is often encoded 0..0.1 for 0..100%; normalize softly.
    cloud = out["cloud_cover_low"].astype(float)
    out["cloud_low_frac"] = np.where(cloud.max() <= 1.0, np.clip(cloud * 10.0, 0.0, 1.0), np.clip(cloud / 100.0, 0.0, 1.0))

    # Thermal/stability features
    out["temp_grad_80_120"] = out["temperature_80m"] - out["temperature_120m"]
    dz = 40.0
    t_avg_k = ((out["temperature_80m"] + out["temperature_120m"]) / 2.0 + 273.15).clip(lower=150.0)
    dtdz = out["temp_grad_80_120"] / dz
    dudz = (out["wind_speed_120m"] - out["wind_speed_80m"]) / dz + 1e-6
    out["richardson_number"] = ((G / t_avg_k) * dtdz / (dudz ** 2)).clip(-5.0, 5.0)
    out["atm_stable"] = (out["richardson_number"] > 0.25).astype(np.int8)
    out["atm_unstable"] = (out["richardson_number"] < -0.50).astype(np.int8)
    out["atm_neutral"] = ((out["richardson_number"] >= -0.50) & (out["richardson_number"] <= 0.25)).astype(np.int8)

    # Calendar / solar geometry
    out["hour"] = idx.hour.astype(np.int16)
    out["month"] = idx.month.astype(np.int16)
    out["dayofyear"] = idx.dayofyear.astype(np.int16)
    out["dayofmonth"] = idx.day.astype(np.int16)
    out["weekofyear"] = idx.isocalendar().week.astype(np.int16).to_numpy()
    out["is_weekend"] = (idx.dayofweek >= 5).astype(np.int8)
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["month_sin"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["month"] / 12.0)
    out["doy_sin"] = np.sin(2.0 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * out["dayofyear"] / 365.25)
    elev = _solar_elevation(idx)
    out["solar_elevation"] = elev
    out["solar_elevation_sin"] = np.sin(elev)
    out["is_daytime"] = (elev > 0.0).astype(np.int8)
    out["insolation_proxy"] = np.clip(np.sin(elev), 0.0, None) * (1.0 - 0.75 * out["cloud_low_frac"])
    out["night_stable"] = ((elev < -0.1) & (out["richardson_number"] > 0.1)).astype(np.int8)

    # Power curve regime and sensitivity
    v = out["ws_eff"].values
    out["v_norm"] = np.clip((v - V_CUT_IN) / (V_RATED - V_CUT_IN), 0.0, 1.0)
    denom = V_RATED ** 3 - V_CUT_IN ** 3
    slope = np.where((v >= V_CUT_IN) & (v < V_RATED), 3.0 * v ** 2 / denom, 0.0)
    out["pc_slope"] = slope
    regime = np.zeros(len(out), dtype=np.int8)
    regime[(v >= V_CUT_IN) & (v < V_RATED)] = 1
    regime[(v >= V_RATED) & (v <= V_CUT_OUT)] = 2
    regime[v > V_CUT_OUT] = 3
    out["pc_regime"] = regime
    out["dist_to_cutin"] = np.clip(v - V_CUT_IN, 0.0, None)
    out["dist_to_rated"] = np.abs(v - V_RATED)
    out["margin_to_cutout"] = np.clip(V_CUT_OUT - v, 0.0, None)
    out["in_ramp_zone"] = ((v >= V_CUT_IN) & (v < V_RATED)).astype(np.int8)
    out["ramp_uncertainty"] = out["pc_slope"] * out["ws_profile_std"]
    out["turb_corr_2nd"] = 0.5 * (out["ws_profile_std"] ** 2) * np.where(
        out["in_ramp_zone"].astype(bool), 6.0 * N_INST * v / (V_RATED ** 3), 0.0
    )

    # Temporal rolling/forecast-window features. For judge-day inference, the full day of
    # weather forecasts is known, so centered weather windows and leads are legitimate.
    for w in [3, 6, 12, 24]:
        out[f"ws_mean_{w}h"] = out["ws_hub"].rolling(w, min_periods=1).mean()
        out[f"ws_std_{w}h"] = out["ws_hub"].rolling(w, min_periods=2).std().fillna(0.0)
        out[f"ws_min_{w}h"] = out["ws_hub"].rolling(w, min_periods=1).min()
        out[f"ws_max_{w}h"] = out["ws_hub"].rolling(w, min_periods=1).max()
        out[f"ws_diff_{w}h"] = out["ws_hub"].diff(w).fillna(0.0)
    for hl in [3, 12, 48]:
        out[f"ws_ewm_{hl}h"] = out["ws_hub"].ewm(halflife=hl, min_periods=1, adjust=False).mean()
        out[f"ws_dev_from_ewm{hl}"] = out["ws_hub"] - out[f"ws_ewm_{hl}h"]
    out["ws_ramp_1h"] = out["ws_hub"].diff(1).fillna(0.0)
    out["ws_ramp_3h"] = out["ws_hub"].diff(3).fillna(0.0)
    out["ws_ramp_abs_3h"] = out["ws_ramp_3h"].abs()
    out["ti_proxy_6h"] = out["ws_std_6h"] / out["ws_mean_6h"].clip(lower=0.5)
    out["ws_ratio_24h"] = out["ws_hub"] / out["ws_mean_24h"].clip(lower=0.5)
    out["ws_min_center_3h"] = out["ws_eff"].rolling(3, center=True, min_periods=1).min()
    out["ws_max_center_3h"] = out["ws_eff"].rolling(3, center=True, min_periods=1).max()
    out["ws_local_range_3h"] = out["ws_max_center_3h"] - out["ws_min_center_3h"]
    for lead in [1, 2, 3]:
        lead_v = out["ws_eff"].shift(-lead).fillna(out["ws_eff"])
        out[f"ws_eff_lead_{lead}h"] = lead_v
        out[f"ramp_lead_{lead}h"] = np.where((lead_v >= V_CUT_IN) & (lead_v < V_RATED), lead_v, 0.0)
    for lag in [1, 2, 3, 6, 12, 24]:
        out[f"ws_eff_lag_{lag}h"] = out["ws_eff"].shift(lag).fillna(out["ws_eff"])

    # Sector features: southern directions are explicitly represented because prior
    # technical notes marked them as hard sectors.
    wd80 = out["wind_direction_80m"].astype(float)
    out["southerly_component"] = np.clip(-out["v_wind_80"], 0.0, None)
    out["easterly_component"] = np.clip(out["u_wind_80"], 0.0, None)
    out["sector_NE"] = ((wd80 >= 45.0) & (wd80 < 67.5)).astype(np.int8)
    out["sector_SE"] = ((wd80 >= 135.0) & (wd80 < 202.5)).astype(np.int8)
    out["sector_SW"] = ((wd80 >= 202.5) & (wd80 < 270.0)).astype(np.int8)
    out["sector_S_all"] = ((wd80 >= 135.0) & (wd80 < 270.0)).astype(np.int8)
    out["night_x_south"] = ((out["sector_S_all"] == 1) & ((out["hour"] >= 20) | (out["hour"] < 6))).astype(np.int8)
    du = out["u_wind_180"] - out["u_wind_80"]
    dv = out["v_wind_180"] - out["v_wind_80"]
    out["hodograph_span"] = np.sqrt(du ** 2 + dv ** 2)
    out["llj_ratio"] = out["wind_speed_180m"] / out["wind_speed_10m"].clip(lower=0.5)
    out["llj_flag"] = ((out["llj_ratio"] > 2.5) & (elev < 0.0)).astype(np.int8)
    out["llj_excess"] = np.where(elev < 0.0, (out["wind_speed_180m"] - out["wind_speed_80m"]).clip(lower=0.0), 0.0)

    # External weather features and deltas
    if external_features is None:
        external_features = pd.DataFrame(index=idx)
    external_features = external_features.reindex(idx).copy()
    for c in EXT_COLS:
        if c not in external_features.columns:
            external_features[c] = 0.0
    out = out.join(external_features[EXT_COLS], how="left")
    out[EXT_COLS] = out[EXT_COLS].fillna(0.0)
    mask_ext = out["ext_available"] > 0
    out["ext_ws80_delta"] = np.where(mask_ext, out["ext_ws80_mean"] - out["wind_speed_80m"], 0.0)
    out["ext_ws80_abs_delta"] = np.abs(out["ext_ws80_delta"])
    out["ext_gust_delta"] = np.where(mask_ext, out["ext_gust_mean"] - out["wind_gusts_10m"], 0.0)
    out["ext_temp80_delta"] = np.where(mask_ext, out["ext_temp80_mean"] - out["temperature_80m"], 0.0)
    out["ext_pressure_delta"] = np.where(mask_ext, out["ext_pressure_mean"] - out["pressure_msl"], 0.0)
    out["ext_cloud_delta"] = np.where(mask_ext, out["ext_cloud_low_mean"] - out["cloud_cover_low"], 0.0)

    # Replace remaining problematic infinities/NaNs later using train medians.
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# ---------------------------------------------------------------------------
# Baseline calibration and correction tables
# ---------------------------------------------------------------------------

@dataclass
class BaselineArtifacts:
    iso_curve: IsotonicRegression
    global_median: float
    bias_tables: Dict[str, Dict[Tuple[int, ...], float]]
    table_counts: Dict[str, Dict[Tuple[int, ...], int]]


def _speed_bin(ws_eff: pd.Series | np.ndarray) -> np.ndarray:
    bins = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 60], dtype=float)
    return np.digitize(np.asarray(ws_eff, dtype=float), bins, right=False).astype(np.int16)


def _month_distance(month: np.ndarray, target_month: int = JUDGE_MONTH) -> np.ndarray:
    m = np.asarray(month, dtype=float)
    d = np.abs(m - target_month)
    return np.minimum(d, 12.0 - d)


def _fit_baseline_artifacts(feat: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> BaselineArtifacts:
    avail = feat["avail_ratio"].astype(float).clip(lower=0.2).values
    y_full_equiv = np.clip(y / avail, 0.0, N_INST)
    x = feat["ws_eff"].astype(float).values
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=float)

    order = np.argsort(x, kind="mergesort")
    iso = IsotonicRegression(y_min=0.0, y_max=N_INST, increasing=True, out_of_bounds="clip")
    iso.fit(x[order], y_full_equiv[order], sample_weight=np.asarray(sample_weight)[order])

    base = _predict_baseline_from_iso(feat, iso)
    resid = y - base
    global_median = float(np.nanmedian(resid))

    tmp = pd.DataFrame(index=feat.index)
    tmp["resid"] = resid
    tmp["ws_bin"] = _speed_bin(feat["ws_eff"])
    tmp["wd_sector"] = np.floor(feat["wind_direction_80m"].astype(float) / 30.0).astype(np.int16).values
    tmp["hour"] = feat["hour"].astype(np.int16).values
    tmp["month"] = feat["month"].astype(np.int16).values
    tmp["regime"] = feat["pc_regime"].astype(np.int16).values
    tmp["is_daytime"] = feat["is_daytime"].astype(np.int16).values
    tmp["sector_s"] = feat["sector_S_all"].astype(np.int16).values

    specs = {
        "speed": ["ws_bin"],
        "speed_dir": ["ws_bin", "wd_sector"],
        "speed_dir_hour": ["ws_bin", "wd_sector", "hour"],
        "speed_month": ["ws_bin", "month"],
        "speed_regime_day": ["ws_bin", "regime", "is_daytime"],
        "south_hour": ["sector_s", "hour", "ws_bin"],
    }
    tables: Dict[str, Dict[Tuple[int, ...], float]] = {}
    counts: Dict[str, Dict[Tuple[int, ...], int]] = {}
    for name, keys in specs.items():
        grp = tmp.groupby(keys, dropna=False)["resid"].agg(["median", "count"])
        # Shrink medians toward global to avoid brittle sparse cells.
        table: Dict[Tuple[int, ...], float] = {}
        count_table: Dict[Tuple[int, ...], int] = {}
        for key, row in grp.iterrows():
            if not isinstance(key, tuple):
                key = (key,)
            key_int = tuple(int(k) for k in key)
            n = int(row["count"])
            shrink = n / (n + 30.0)
            val = float(shrink * row["median"] + (1.0 - shrink) * global_median)
            table[key_int] = val
            count_table[key_int] = n
        tables[name] = table
        counts[name] = count_table

    return BaselineArtifacts(iso_curve=iso, global_median=global_median, bias_tables=tables, table_counts=counts)


def _predict_baseline_from_iso(feat: pd.DataFrame, iso: IsotonicRegression) -> np.ndarray:
    full = iso.predict(feat["ws_eff"].astype(float).values)
    iso_power = full * feat["avail_ratio"].astype(float).clip(lower=0.0, upper=1.0).values
    phys = feat["power_phys"].astype(float).values
    # A convex physical+empirical baseline. Physical is useful at boundaries;
    # isotonic captures farm-specific curve and forecast bias.
    base = 0.68 * iso_power + 0.32 * phys
    return _clip_power(base)


def _table_lookup(table: Dict[Tuple[int, ...], float], keys: List[Tuple[int, ...]], default: float) -> np.ndarray:
    return np.array([float(table.get(k, default)) for k in keys], dtype=float)


def _predict_bias_correction(feat: pd.DataFrame, artifacts: BaselineArtifacts) -> np.ndarray:
    ws_bin = _speed_bin(feat["ws_eff"])
    wd_sector = np.floor(feat["wind_direction_80m"].astype(float).values / 30.0).astype(np.int16)
    hour = feat["hour"].astype(np.int16).values
    month = feat["month"].astype(np.int16).values
    regime = feat["pc_regime"].astype(np.int16).values
    day = feat["is_daytime"].astype(np.int16).values
    sector_s = feat["sector_S_all"].astype(np.int16).values
    default = artifacts.global_median

    k_speed = [(int(a),) for a in ws_bin]
    k_speed_dir = [(int(a), int(b)) for a, b in zip(ws_bin, wd_sector)]
    k_speed_dir_hour = [(int(a), int(b), int(c)) for a, b, c in zip(ws_bin, wd_sector, hour)]
    k_speed_month = [(int(a), int(b)) for a, b in zip(ws_bin, month)]
    k_speed_regime_day = [(int(a), int(b), int(c)) for a, b, c in zip(ws_bin, regime, day)]
    k_south_hour = [(int(a), int(b), int(c)) for a, b, c in zip(sector_s, hour, ws_bin)]

    c_speed = _table_lookup(artifacts.bias_tables.get("speed", {}), k_speed, default)
    c_dir = _table_lookup(artifacts.bias_tables.get("speed_dir", {}), k_speed_dir, default)
    c_dir_hour = _table_lookup(artifacts.bias_tables.get("speed_dir_hour", {}), k_speed_dir_hour, default)
    c_month = _table_lookup(artifacts.bias_tables.get("speed_month", {}), k_speed_month, default)
    c_regime = _table_lookup(artifacts.bias_tables.get("speed_regime_day", {}), k_speed_regime_day, default)
    c_south = _table_lookup(artifacts.bias_tables.get("south_hour", {}), k_south_hour, default)
    corr = 0.18 * c_speed + 0.24 * c_dir + 0.22 * c_dir_hour + 0.16 * c_month + 0.12 * c_regime + 0.08 * c_south
    return np.clip(corr, -18.0, 18.0)


# ---------------------------------------------------------------------------
# Model fitting / prediction helpers
# ---------------------------------------------------------------------------

def _sample_weights(feat: pd.DataFrame) -> np.ndarray:
    # Recency: two-year half life. Season: gentle May similarity boost.
    max_date = feat.index.max()
    age_days = (max_date - feat.index).days.astype(float)
    recency = np.power(0.5, age_days / 730.0)
    season = 0.90 + 0.30 * np.cos(2.0 * np.pi * _month_distance(feat["month"].values, JUDGE_MONTH) / 12.0)
    # Ramp-zone hours matter most for MAE sensitivity.
    ramp = 1.0 + 0.15 * feat["in_ramp_zone"].astype(float).values + 0.10 * feat["sector_S_all"].astype(float).values
    w = recency * season * ramp
    return np.asarray(w / np.nanmean(w), dtype=float)


def _fit_feature_schema(feat: pd.DataFrame) -> Tuple[List[str], Dict[str, float]]:
    blocked = {TARGET_COL, SUBMISSION_COL, "__row_order"}
    num = feat.select_dtypes(include=[np.number]).copy()
    cols = [c for c in num.columns if c not in blocked]
    # Explicitly remove degenerate columns after feature engineering.
    good_cols: List[str] = []
    medians: Dict[str, float] = {}
    for c in cols:
        s = pd.to_numeric(num[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = float(s.median()) if s.notna().any() else 0.0
        medians[c] = med
        nunique = int(s.fillna(med).nunique(dropna=False))
        if nunique > 1 or c in {"power_phys", "ws_eff", "n_avail", "avail_ratio"}:
            good_cols.append(c)
    return good_cols, medians


def _make_X(feat: pd.DataFrame, feature_cols: List[str], medians: Dict[str, float]) -> pd.DataFrame:
    X = pd.DataFrame(index=feat.index)
    for c in feature_cols:
        if c in feat.columns:
            X[c] = pd.to_numeric(feat[c], errors="coerce")
        else:
            X[c] = medians.get(c, 0.0)
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in feature_cols:
        X[c] = X[c].fillna(medians.get(c, 0.0))
    return X.astype(np.float32)


def _train_lgb_models(X: pd.DataFrame,
                      resid: np.ndarray,
                      sample_weight: np.ndarray,
                      valid_mask: np.ndarray) -> Tuple[List[Any], Dict[str, Any]]:
    models: List[Any] = []
    meta: Dict[str, Any] = {"backend": "lightgbm" if lgb is not None else "sklearn_fallback", "models": []}

    if lgb is None:
        # Fallback: deterministic HistGradientBoosting + ExtraTrees residual ensemble.
        hgb = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.035,
            max_iter=700,
            max_leaf_nodes=31,
            min_samples_leaf=35,
            l2_regularization=0.05,
            random_state=SEED,
        )
        hgb.fit(X, resid, sample_weight=sample_weight)
        models.append(("hgb_abs", hgb))
        et = ExtraTreesRegressor(
            n_estimators=320,
            max_depth=18,
            min_samples_leaf=8,
            max_features=0.72,
            bootstrap=False,
            random_state=SEED,
            n_jobs=1,
        )
        et.fit(X, resid, sample_weight=sample_weight)
        models.append(("extratrees", et))
        meta["models"] = ["hgb_abs", "extratrees"]
        return models, meta

    base_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.035,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 45,
        "feature_fraction": 0.90,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 0.05,
        "lambda_l2": 1.25,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 1,
    }

    # Fixed all-data training.  Avoiding early-stopping retrains keeps the module fast
    # and deterministic in CPU-only judge environments.  The validation mask is used
    # only for reporting diagnostics in train_model().
    seeds = [42]
    for seed in seeds:
        params = dict(base_params)
        params["seed"] = seed
        dfull = lgb.Dataset(X, resid, weight=sample_weight, free_raw_data=False)
        rounds = 520
        final = lgb.train(params, dfull, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
        print(f"  trained lgb_l1_seed{seed} ({rounds} rounds)", flush=True)
        models.append((f"lgb_l1_seed{seed}", final))
        meta["models"].append({"name": f"lgb_l1_seed{seed}", "rounds": int(rounds)})


    return models, meta


def _train_cat_model(X: pd.DataFrame,
                     resid: np.ndarray,
                     sample_weight: np.ndarray,
                     valid_mask: np.ndarray) -> Tuple[Optional[Any], Dict[str, Any]]:
    if CatBoostRegressor is None or os.environ.get("CHATGPT55_USE_CATBOOST", "0") != "1":
        return None, {"backend": "disabled_or_unavailable"}
    tr_idx = ~valid_mask
    va_idx = valid_mask
    have_val = int(va_idx.sum()) >= 100 and int(tr_idx.sum()) >= 1000
    params = dict(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=500,
        learning_rate=0.035,
        depth=7,
        l2_leaf_reg=4.0,
        random_seed=SEED,
        random_strength=0.0,
        bootstrap_type="No",
        allow_writing_files=False,
        thread_count=1,
        verbose=False,
    )
    model = CatBoostRegressor(**params)
    if have_val:
        model.fit(
            X.loc[tr_idx], resid[tr_idx],
            sample_weight=sample_weight[tr_idx],
            eval_set=(X.loc[va_idx], resid[va_idx]),
            early_stopping_rounds=80,
            use_best_model=True,
            verbose=False,
        )
    else:
        model.fit(X, resid, sample_weight=sample_weight, verbose=False)
    return model, {"backend": "catboost", "tree_count": int(model.tree_count_)}


def _predict_residual_models(models: List[Any], cat_model: Optional[Any], X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}
    for name, model in models:
        if lgb is not None and hasattr(model, "predict") and name.startswith("lgb"):
            preds[name] = np.asarray(model.predict(X), dtype=float)
        else:
            preds[name] = np.asarray(model.predict(X), dtype=float)
    if cat_model is not None:
        preds["catboost"] = np.asarray(cat_model.predict(X), dtype=float)

    centers: List[np.ndarray] = []
    for name, pred in preds.items():
        if name.startswith("lgb_q"):
            continue
        centers.append(pred)
    if not centers:
        centers = list(preds.values())
    center = np.mean(np.vstack(centers), axis=0)

    # If quantile shoulders exist, report spread.  Also gently shrink residual in highly
    # uncertain hours instead of extrapolating aggressively.
    q40 = preds.get("lgb_q40")
    q60 = preds.get("lgb_q60")
    if q40 is not None and q60 is not None:
        width = np.maximum(q60 - q40, 0.0)
        qmid = 0.5 * (q40 + q60)
        center = 0.88 * center + 0.12 * qmid
    else:
        width = np.zeros(len(X), dtype=float)
    return center, width


def _fit_analog_store(feat: pd.DataFrame, y: np.ndarray, max_rows: int = 30000) -> Dict[str, Any]:
    cols = [
        "ws_eff", "ws_hub", "power_phys", "wd_sin_80", "wd_cos_80",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "avail_ratio",
        "pc_slope", "gust_ratio", "shear_ratio_10_80", "sector_S_all", "llj_flag",
    ]
    A = feat[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    med = A.median().fillna(0.0)
    A = A.fillna(med)
    # Deterministic thinning if needed: keep all recent rows and an even sample of older rows.
    if len(A) > max_rows:
        keep_idx = np.linspace(0, len(A) - 1, max_rows).round().astype(int)
        A = A.iloc[keep_idx]
        y_store = y[keep_idx]
    else:
        y_store = y
    mean = A.mean(axis=0).values.astype(np.float32)
    std = A.std(axis=0).replace(0.0, 1.0).values.astype(np.float32)
    mat = ((A.values.astype(np.float32) - mean) / std).astype(np.float32)
    return {
        "cols": cols,
        "median": med.to_dict(),
        "mean": mean,
        "std": std,
        "X": mat,
        "y": y_store.astype(np.float32),
    }


def _predict_analog(feat: pd.DataFrame, store: Dict[str, Any], k: int = 64, chunk_size: int = 512) -> np.ndarray:
    cols = store["cols"]
    A = pd.DataFrame(index=feat.index)
    for c in cols:
        A[c] = pd.to_numeric(feat[c], errors="coerce") if c in feat.columns else store["median"].get(c, 0.0)
        A[c] = A[c].replace([np.inf, -np.inf], np.nan).fillna(store["median"].get(c, 0.0))
    Xq = ((A.values.astype(np.float32) - store["mean"]) / store["std"]).astype(np.float32)
    Xt = store["X"]
    yt = store["y"]
    out = np.empty(Xq.shape[0], dtype=np.float32)
    k = min(k, Xt.shape[0])
    for start in range(0, Xq.shape[0], chunk_size):
        q = Xq[start:start + chunk_size]
        # squared Euclidean distances via dot product; deterministic top-k.
        q_norm = np.sum(q * q, axis=1, keepdims=True)
        t_norm = np.sum(Xt * Xt, axis=1)[None, :]
        dist = q_norm + t_norm - 2.0 * q @ Xt.T
        idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        vals = yt[idx]
        # Median is robust under MAE; mean would be more sensitive to curtailment/outliers.
        out[start:start + len(q)] = np.median(vals, axis=1)
    return _clip_power(out)


# ---------------------------------------------------------------------------
# Prediction assembly
# ---------------------------------------------------------------------------

def _base_components(feat: pd.DataFrame, baseline: BaselineArtifacts) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = _predict_baseline_from_iso(feat, baseline.iso_curve)
    corr = _predict_bias_correction(feat, baseline)
    table_pred = _clip_power(base + corr)
    return base, corr, table_pred


def _predict_once(feat: pd.DataFrame, bundle: Dict[str, Any]) -> Dict[str, np.ndarray]:
    baseline: BaselineArtifacts = bundle["baseline"]
    feature_cols: List[str] = bundle["feature_cols"]
    medians: Dict[str, float] = bundle["feature_medians"]
    X = _make_X(feat, feature_cols, medians)
    base, corr, table_pred = _base_components(feat, baseline)
    resid_pred, q_width = _predict_residual_models(bundle["residual_models"], bundle.get("cat_model"), X)
    ml_pred = _clip_power(base + np.clip(resid_pred, -35.0, 35.0))
    analog_pred = _predict_analog(feat, bundle["analog_store"])

    # Dynamic robust blend.  In ramp regime, ML is strongest. Near cut-in/rated/cut-out,
    # physics and analog components prevent pathological extrapolation.
    ramp = feat["in_ramp_zone"].astype(float).values
    uncertainty = np.clip(q_width / 20.0, 0.0, 1.0)
    w_ml = 0.52 + 0.08 * ramp - 0.08 * uncertainty
    w_table = 0.16
    w_analog = 0.10 + 0.04 * uncertainty
    w_base = 1.0 - w_ml - w_table - w_analog
    pred = w_ml * ml_pred + w_table * table_pred + w_analog * analog_pred + w_base * base

    # Conservative boundary rules grounded in turbine physics.
    ws_eff = feat["ws_eff"].astype(float).values
    pred = np.where(ws_eff > V_CUT_OUT, 0.0, pred)
    # Do not force cut-in to zero: forecast wind is noisy and historic data has non-zero
    # production under nominally low forecast speed. Only soften extreme calm cases.
    pred = np.where(ws_eff < 1.2, np.minimum(pred, 3.0), pred)
    pred = _clip_power(pred)
    return {
        "pred": pred,
        "base": base,
        "table": table_pred,
        "ml": ml_pred,
        "analog": analog_pred,
        "q_width": q_width,
    }


def _make_external_replaced_raw(raw_df: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """Create an alternate forecast state by blending raw NWP with external consensus."""
    out = raw_df.copy()
    ext = ext.reindex(out.index).fillna(0.0)
    avail = ext["ext_available"].values > 0
    if not np.any(avail):
        return out

    def blend_col(raw_col: str, ext_col: str, weight: float = 0.50) -> None:
        if raw_col in out.columns and ext_col in ext.columns:
            raw = pd.to_numeric(out[raw_col], errors="coerce").astype(float).values
            ex = ext[ext_col].astype(float).values
            good = avail & np.isfinite(ex) & (ex != 0.0)
            raw[good] = (1.0 - weight) * raw[good] + weight * ex[good]
            out[raw_col] = raw

    blend_col("wind_speed_80m", "ext_ws80_mean", 0.50)
    blend_col("wind_speed_120m", "ext_ws120_mean", 0.45)
    blend_col("wind_speed_180m", "ext_ws180_mean", 0.40)
    blend_col("wind_gusts_10m", "ext_gust_mean", 0.35)
    blend_col("temperature_80m", "ext_temp80_mean", 0.35)
    blend_col("temperature_120m", "ext_temp80_mean", 0.30)
    blend_col("pressure_msl", "ext_pressure_mean", 0.25)
    blend_col("cloud_cover_low", "ext_cloud_low_mean", 0.25)
    blend_col("rain", "ext_rain_mean", 0.25)

    if "wind_direction_80m" in out.columns and "ext_wd80_mean" in ext.columns:
        raw = _as_deg_if_scaled(out["wind_direction_80m"]).values
        ex = ext["ext_wd80_mean"].astype(float).values
        good = avail & np.isfinite(ex) & (ex != 0.0)
        # Circular blend using sin/cos.
        rr = np.deg2rad(raw[good])
        ee = np.deg2rad(ex[good])
        s = 0.5 * np.sin(rr) + 0.5 * np.sin(ee)
        c = 0.5 * np.cos(rr) + 0.5 * np.cos(ee)
        raw[good] = np.mod(np.rad2deg(np.arctan2(s, c)), 360.0)
        for h in [10, 80, 120, 180]:
            col = f"wind_direction_{h}m"
            if col in out.columns:
                vals = _as_deg_if_scaled(out[col]).values
                vals[good] = raw[good]
                out[col] = vals
    return out


def _validate_output_file(output_path: str | Path, expected_rows: int) -> None:
    df = pd.read_csv(output_path)
    if list(df.columns) != [SUBMISSION_COL]:
        raise ValueError(f"Output must have exactly one column named {SUBMISSION_COL!r}; got {list(df.columns)!r}")
    if len(df) != expected_rows:
        raise ValueError(f"Output row count {len(df)} != expected {expected_rows}")
    vals = pd.to_numeric(df[SUBMISSION_COL], errors="coerce")
    if vals.isna().any():
        raise ValueError(f"Output contains {int(vals.isna().sum())} NaN/non-numeric predictions")
    if not np.isfinite(vals.values).all():
        raise ValueError("Output contains non-finite predictions")
    if (vals < -1e-9).any() or (vals > N_INST + 1e-9).any():
        raise ValueError(f"Output predictions are outside [0, {N_INST}]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_model(train_path: str, artifact_dir: str) -> None:
    """Train deterministic model bundle and save it into artifact_dir."""
    _seed_everything(SEED)
    artifact_path = _ensure_dir(artifact_dir)

    print("Loading training data...", flush=True)
    raw, n_original = _read_competition_csv(train_path, has_target=True)
    ext_empty = load_external_weather(None, raw.index)
    print("Building features...", flush=True)
    feat = make_features(raw, ext_empty)

    y = pd.to_numeric(feat[TARGET_COL], errors="coerce").values.astype(float)
    train_mask = np.isfinite(y) & (y >= 0.0) & (y <= N_INST + 5.0)
    # Remove only severe non-physical anomalies; do not over-clean curtailment/low-wind
    # points because judge-day data can contain forecast/model mismatch.
    calm_bad = (feat["ws_hub"].values < 1.0) & (y > 0.35 * N_INST)
    train_mask &= ~calm_bad
    feat_train = feat.loc[train_mask].copy()
    y_train = _clip_power(y[train_mask])

    if len(feat_train) < 1000:
        raise ValueError(f"Too few usable training rows: {len(feat_train)}")

    weights = _sample_weights(feat_train)
    print("Fitting physical/empirical baseline...", flush=True)
    baseline = _fit_baseline_artifacts(feat_train, y_train, weights)
    base, corr, table_pred = _base_components(feat_train, baseline)
    feat_train["base_iso_phys"] = base
    feat_train["bias_table_corr"] = corr
    feat_train["table_pred"] = table_pred

    feature_cols, feature_medians = _fit_feature_schema(feat_train)
    X = _make_X(feat_train, feature_cols, feature_medians)
    resid = y_train - base

    # Fixed time validation on an early-2025 window if present, otherwise final 20% by time.
    idx = feat_train.index
    valid_mask = np.asarray((idx >= pd.Timestamp("2025-01-01")) & (idx <= pd.Timestamp("2025-03-31 23:00:00")))
    if valid_mask.sum() < 100 or (~valid_mask).sum() < 1000:
        cutoff = idx[int(len(idx) * 0.80)]
        valid_mask = np.asarray(idx >= cutoff)

    print("Training residual models...", flush=True)
    residual_models, lgb_meta = _train_lgb_models(X, resid, weights, valid_mask)
    cat_model, cat_meta = _train_cat_model(X, resid, weights, valid_mask)
    print("Residual models trained.", flush=True)

    # Diagnostics on validation only; saved as metadata, not used for target leakage.
    diagnostics: Dict[str, Any] = {}
    if valid_mask.sum() > 0:
        pred_train_parts = _predict_residual_models(residual_models, cat_model, X.loc[valid_mask])
        pred_valid = _clip_power(base[valid_mask] + pred_train_parts[0])
        pred_valid_blend = 0.70 * pred_valid + 0.18 * table_pred[valid_mask] + 0.12 * base[valid_mask]
        diagnostics["valid_rows"] = int(valid_mask.sum())
        diagnostics["valid_start"] = str(idx[valid_mask].min())
        diagnostics["valid_end"] = str(idx[valid_mask].max())
        diagnostics["mae_mw"] = float(mean_absolute_error(y_train[valid_mask], pred_valid_blend))
        diagnostics["mae_norm_pct"] = float(diagnostics["mae_mw"] / N_INST * 100.0)

    print("Building analog store...", flush=True)
    analog_store = _fit_analog_store(feat_train, y_train)

    bundle: Dict[str, Any] = {
        "version": "model1_v1",
        "created_by": "model1.py",
        "seed": SEED,
        "constants": {
            "capacity_mw": N_INST,
            "n_turbines": N_TURBINES,
            "p_rated_per_turbine_mw": P_RATED_PER_TURBINE,
            "hub_height_m": H_HUB,
            "latitude": LAT_DEG,
            "longitude": LON_DEG,
        },
        "baseline": baseline,
        "feature_cols": feature_cols,
        "feature_medians": feature_medians,
        "residual_models": residual_models,
        "cat_model": cat_model,
        "analog_store": analog_store,
        "diagnostics": diagnostics,
        "model_meta": {"lgb": lgb_meta, "catboost": cat_meta},
        "train_summary": {
            "train_path": str(train_path),
            "train_sha256": _sha256_file(train_path),
            "raw_rows": int(n_original),
            "usable_rows": int(len(feat_train)),
            "removed_calm_bad": int(calm_bad.sum()),
            "date_min": str(feat_train.index.min()),
            "date_max": str(feat_train.index.max()),
            "target_mean": float(np.mean(y_train)),
            "target_median": float(np.median(y_train)),
        },
    }

    out_file = artifact_path / ARTIFACT_NAME
    print("Saving artifact...", flush=True)
    joblib.dump(bundle, out_file, compress=3)
    meta = {
        "artifact": str(out_file),
        "artifact_sha256": _sha256_file(out_file),
        "diagnostics": diagnostics,
        "train_summary": bundle["train_summary"],
        "n_feature_cols": len(feature_cols),
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": getattr(lgb, "__version__", None) if lgb is not None else None,
            "catboost": "available" if CatBoostRegressor is not None else None,
        },
    }
    with (artifact_path / META_NAME).open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"Saved artifact: {out_file}")
    if diagnostics:
        print(f"Validation MAE_norm: {diagnostics['mae_norm_pct']:.4f}% ({diagnostics['mae_mw']:.4f} MW)")
    print(f"Feature columns: {len(feature_cols)}")


def predict_day(features_path: str,
                artifact_dir: str,
                output_path: str,
                external_weather_dir: str | None = None) -> None:
    """Predict one valid-like feature file and write one-column submission CSV."""
    _seed_everything(SEED)
    artifact_path = Path(artifact_dir) / ARTIFACT_NAME
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing artifact {artifact_path}. Run train first.")
    bundle: Dict[str, Any] = joblib.load(artifact_path)

    raw, expected_rows = _read_competition_csv(features_path, has_target=False)
    # Keep original row order exactly for output.
    row_order = raw["__row_order"].copy()

    ext = load_external_weather(external_weather_dir, raw.index)
    feat = make_features(raw, ext)
    feat["base_iso_phys"], feat["bias_table_corr"], feat["table_pred"] = _base_components(feat, bundle["baseline"])
    pred_parts = _predict_once(feat, bundle)
    pred = pred_parts["pred"]

    # If external weather is available, run deterministic alternate-weather TTA and blend
    # cautiously. The model was trained on competition features, so external weather is a
    # secondary source, not a full replacement.
    if external_weather_dir is not None and float(ext["ext_available"].sum()) > 0:
        raw_alt = _make_external_replaced_raw(raw, ext)
        feat_alt = make_features(raw_alt, ext)
        feat_alt["base_iso_phys"], feat_alt["bias_table_corr"], feat_alt["table_pred"] = _base_components(feat_alt, bundle["baseline"])
        pred_alt = _predict_once(feat_alt, bundle)["pred"]
        spread = np.clip(feat["ext_ws80_abs_delta"].astype(float).values / 3.0, 0.0, 1.0)
        # More external influence when raw-vs-external disagreement is material but not huge.
        w_ext = np.where(ext["ext_available"].values > 0, 0.10 + 0.08 * spread, 0.0)
        pred = (1.0 - w_ext) * pred + w_ext * pred_alt
        # If sources disagree strongly in the steep ramp zone, soften toward median/table.
        high_unc = (feat["in_ramp_zone"].values == 1) & (feat["ext_ws80_abs_delta"].values > 2.5)
        pred[high_unc] = 0.92 * pred[high_unc] + 0.08 * pred_parts["table"][high_unc]

    pred = _clip_power(pred)

    # Restore original feature-file row order, even if input was descending or shuffled.
    out = pd.DataFrame({"__row_order": row_order.values, SUBMISSION_COL: pred})
    out = out.sort_values("__row_order", kind="mergesort")[[SUBMISSION_COL]]

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path_obj, index=False, float_format="%.6f")
    _validate_output_file(output_path_obj, expected_rows)

    meta = {
        "features_path": str(features_path),
        "features_sha256": _sha256_file(features_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256_file(artifact_path),
        "external_weather_dir": str(external_weather_dir) if external_weather_dir else None,
        "rows": int(expected_rows),
        "output_path": str(output_path_obj),
        "output_sha256": _sha256_file(output_path_obj),
        "prediction_min": float(out[SUBMISSION_COL].min()),
        "prediction_max": float(out[SUBMISSION_COL].max()),
        "prediction_mean": float(out[SUBMISSION_COL].mean()),
        "date_min": str(raw.index.min()),
        "date_max": str(raw.index.max()),
        "validation": "passed",
    }
    with output_path_obj.with_suffix(output_path_obj.suffix + ".meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"Wrote: {output_path_obj}")
    print(f"Rows: {expected_rows}; min={meta['prediction_min']:.4f}; max={meta['prediction_max']:.4f}; mean={meta['prediction_mean']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic wind-power May-18 predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train model bundle")
    p_train.add_argument("--train", required=True, help="path to train_dataset.csv")
    p_train.add_argument("--artifact-dir", required=True, help="directory to write model artifact")

    p_pred = sub.add_parser("predict", help="predict a valid-like feature CSV")
    p_pred.add_argument("--features", required=True, help="path to valid-like/may18 feature CSV")
    p_pred.add_argument("--artifact-dir", required=True, help="directory containing model artifact")
    p_pred.add_argument("--output", required=True, help="submission CSV path")
    p_pred.add_argument("--external-weather", default=None, help="optional directory with external weather CSV/JSON files")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        train_model(args.train, args.artifact_dir)
    elif args.command == "predict":
        predict_day(args.features, args.artifact_dir, args.output, args.external_weather)
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
