#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features.py — общие функции инженерии признаков, изотонический бейзлайн и климатология.
Используется и train.py, и inference.py, и evaluate_holdout.py.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

DATETIME = 'METEOFORECASTHOUR_OPENM_Datetime'
TOTAL_TURBINES = 26
HUB_BIN = 0.5
WS_BIN = 0.5
DOY_BIN = 5

# ============================================================
# Поиск служебных колонок (robust к кодировке кириллицы)
# ============================================================
def find_col(df, substr_lower):
    for c in df.columns:
        if substr_lower in c.lower():
            return c
    return None

def find_target_col(df):
    c = find_col(df, 'выработка')
    if c is None:
        raise KeyError("Target column with 'выработка' not found")
    return c

def find_repair_col(df):
    return find_col(df, 'ремонт')

# ============================================================
# Импьютация wind_speed_180m (21% NaN)
# ============================================================
def impute_ws180m(df):
    if 'wind_speed_180m' not in df.columns:
        return df
    mask = df['wind_speed_180m'].isna()
    df['ws180m_was_imputed'] = mask.astype(int)
    if mask.sum() == 0:
        return df
    if 'wind_speed_120m' in df.columns and 'wind_speed_80m' in df.columns:
        extrap = df['wind_speed_120m'] + (df['wind_speed_120m'] - df['wind_speed_80m']) * (60.0 / 40.0)
        extrap = extrap.clip(lower=0)
        df.loc[mask, 'wind_speed_180m'] = extrap.loc[mask]
    if 'wind_direction_180m' in df.columns and 'wind_direction_120m' in df.columns:
        mask2 = df['wind_direction_180m'].isna()
        df.loc[mask2, 'wind_direction_180m'] = df.loc[mask2, 'wind_direction_120m']
    return df

# ============================================================
# Временные признаки
# ============================================================
def add_time_features(df):
    dt = df[DATETIME]
    df['hour'] = dt.dt.hour
    df['dayofweek'] = dt.dt.dayofweek
    df['month_feat'] = dt.dt.month
    df['dayofyear'] = dt.dt.dayofyear
    df['is_pm'] = (df['hour'] >= 12).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month_feat'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_feat'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    return df

# ============================================================
# Направления ветра (sin/cos + сдвиги по высоте)
# ============================================================
def add_wind_direction_features(df):
    for h in [10, 80, 120, 180]:
        wd_col = f'wind_direction_{h}m'
        if wd_col in df.columns:
            rad = np.radians(df[wd_col])
            df[f'{wd_col}_sin'] = np.sin(rad)
            df[f'{wd_col}_cos'] = np.cos(rad)
    for low, high in [(80, 120), (120, 180), (10, 80)]:
        sl = f'wind_direction_{low}m_sin'
        sh = f'wind_direction_{high}m_sin'
        cl = f'wind_direction_{low}m_cos'
        ch = f'wind_direction_{high}m_cos'
        if sl in df.columns and sh in df.columns:
            df[f'wind_dir_sin_diff_{low}_{high}'] = df[sh] - df[sl]
            df[f'wind_dir_cos_diff_{low}_{high}'] = df[ch] - df[cl]
    return df

# ============================================================
# Физика и производные признаки скорости
# ============================================================
def add_physics_features(df, repair_col=None):
    for h in [10, 80, 120, 180]:
        col = f'wind_speed_{h}m'
        if col in df.columns:
            df[f'{col}_cubed'] = df[col] ** 3
    for low, high in [(10, 80), (80, 120), (120, 180), (10, 120)]:
        cl, ch = f'wind_speed_{low}m', f'wind_speed_{high}m'
        if cl in df.columns and ch in df.columns:
            df[f'wind_shear_{low}_{high}'] = df[ch] - df[cl]

    if 'pressure_msl' in df.columns and 'temperature_80m' in df.columns:
        df['air_density'] = df['pressure_msl'] * 100 / (287.058 * (df['temperature_80m'] + 273.15))
    else:
        df['air_density'] = 1.225

    if repair_col is not None and repair_col in df.columns:
        df['n_eff'] = (TOTAL_TURBINES - df[repair_col]).astype(float).clip(lower=1.0)
    else:
        df['n_eff'] = float(TOTAL_TURBINES)
    df['available_ratio'] = df['n_eff'] / TOTAL_TURBINES

    speed_cols = [f'wind_speed_{h}m' for h in [10, 80, 120, 180] if f'wind_speed_{h}m' in df.columns]
    if speed_cols:
        df['avg_wind_speed'] = df[speed_cols].mean(axis=1)
        df['avg_wind_speed_cubed'] = df['avg_wind_speed'] ** 3

    if 'wind_speed_80m' in df.columns and 'wind_speed_120m' in df.columns:
        df['hub_wind_speed'] = 0.5 * (df['wind_speed_80m'] + df['wind_speed_120m'])
        df['hub_wind_speed_cubed'] = df['hub_wind_speed'] ** 3
    elif 'wind_speed_120m' in df.columns:
        df['hub_wind_speed'] = df['wind_speed_120m'].astype(float)
        df['hub_wind_speed_cubed'] = df['hub_wind_speed'] ** 3
    elif 'wind_speed_80m' in df.columns:
        df['hub_wind_speed'] = df['wind_speed_80m'].astype(float)
        df['hub_wind_speed_cubed'] = df['hub_wind_speed'] ** 3

    for h in [10, 80, 120, 180]:
        ws_col, wd_col = f'wind_speed_{h}m', f'wind_direction_{h}m'
        if ws_col in df.columns and wd_col in df.columns:
            rad = np.radians(df[wd_col])
            df[f'{ws_col}_u'] = df[ws_col] * np.sin(rad)
            df[f'{ws_col}_v'] = df[ws_col] * np.cos(rad)

    if 'wind_speed_80m' in df.columns:
        df['ws80_below_cutin'] = (df['wind_speed_80m'] < 3.5).astype(int)
        df['ws80_above_cutout'] = (df['wind_speed_80m'] > 23.0).astype(int)
        df['ws80_in_rated'] = ((df['wind_speed_80m'] >= 12.0) & (df['wind_speed_80m'] <= 23.0)).astype(int)

    if 'air_density' in df.columns and 'avg_wind_speed_cubed' in df.columns:
        df['power_potential_avg'] = df['air_density'] * df['avg_wind_speed_cubed']
        df['power_potential_avg_n_eff'] = df['power_potential_avg'] * df['n_eff']
    if 'air_density' in df.columns and 'hub_wind_speed_cubed' in df.columns:
        df['power_potential_hub'] = df['air_density'] * df['hub_wind_speed_cubed']
        df['power_potential_hub_n_eff'] = df['power_potential_hub'] * df['n_eff']

    if 'temperature_80m' in df.columns and 'wind_speed_80m_cubed' in df.columns:
        df['wind_cubed_inv_temp'] = df['wind_speed_80m_cubed'] / (df['temperature_80m'] + 273.15)

    # ============ ФИЗИКА АТМОСФЕРЫ (вдохновлено v41-v44) ============
    # Richardson number: stability via lapse / shear
    if 'temperature_80m' in df.columns and 'temperature_120m' in df.columns and 'wind_speed_80m' in df.columns and 'wind_speed_120m' in df.columns:
        # dT/dz over 40m (К/м)
        dT_dz = (df['temperature_120m'] - df['temperature_80m']) / 40.0
        # dU/dz over 40m (м/с/м)
        dU_dz = (df['wind_speed_120m'] - df['wind_speed_80m']) / 40.0
        # T в K
        T_mean = (df['temperature_80m'] + df['temperature_120m'])/2 + 273.15
        # Ri = (g/T) * dT/dz / (dU/dz)^2
        g = 9.81
        df['richardson'] = (g / T_mean) * dT_dz / (dU_dz**2 + 1e-3)
        df['richardson'] = df['richardson'].clip(-10, 10)
        df['atm_stable']   = (df['richardson'] > 0.25).astype(int)
        df['atm_unstable'] = (df['richardson'] < 0).astype(int)
        df['atm_neutral']  = ((df['richardson'] >= 0) & (df['richardson'] <= 0.25)).astype(int)
        df['lapse_rate']   = dT_dz  # сам lapse как фича

    # LLJ (Low-Level Jet): ws на высоких слоях МЕНЬШЕ чем на низких → reverse profile
    if all(f'wind_speed_{h}m' in df.columns for h in [10, 80, 120, 180]):
        # Стандартный профиль: ws увеличивается с высотой. LLJ — нарушение.
        # llj если есть локальный максимум между 80м и 120м
        df['llj_excess_80_120'] = (df['wind_speed_80m'] - df['wind_speed_120m']).clip(lower=0)
        df['llj_excess_80_180'] = (df['wind_speed_80m'] - df['wind_speed_180m']).clip(lower=0)
        df['llj_flag'] = (df['llj_excess_80_120'] > 0.5).astype(int)
        df['llj_ratio_80_120'] = df['wind_speed_80m'] / (df['wind_speed_120m'] + 1e-3)
        df['llj_ratio_80_180'] = df['wind_speed_80m'] / (df['wind_speed_180m'] + 1e-3)
        # Aiken-Bonin: альфа в power-law профиле
        # ws_h = ws_ref * (h/h_ref)^alpha
        # alpha = log(ws_h/ws_ref) / log(h/h_ref)
        df['alpha_80_120'] = np.log(df['wind_speed_120m'].clip(lower=0.1) / df['wind_speed_80m'].clip(lower=0.1)) / np.log(120.0/80.0)
        df['alpha_10_80']  = np.log(df['wind_speed_80m'].clip(lower=0.1)  / df['wind_speed_10m'].clip(lower=0.1))  / np.log(80.0/10.0)

    # Hodograph: rotation of wind with height (frontal passage indicator)
    if all(f'wind_direction_{h}m' in df.columns for h in [10, 80, 120, 180]):
        def angular_diff(a, b):
            d = (a - b + 180) % 360 - 180
            return d
        df['wd_veer_10_80']   = angular_diff(df['wind_direction_80m'].values, df['wind_direction_10m'].values)
        df['wd_veer_80_120']  = angular_diff(df['wind_direction_120m'].values, df['wind_direction_80m'].values)
        df['wd_veer_120_180'] = angular_diff(df['wind_direction_180m'].values, df['wind_direction_120m'].values)
        df['wd_veer_total']   = angular_diff(df['wind_direction_180m'].values, df['wind_direction_10m'].values)
        df['hodograph_curve'] = np.abs(df['wd_veer_10_80']) + np.abs(df['wd_veer_80_120']) + np.abs(df['wd_veer_120_180'])

    # Effective wind (с поправкой на плотность)
    if 'avg_wind_speed' in df.columns and 'air_density' in df.columns:
        # ws_eff = ws * (rho/rho0)^(1/3), rho0=1.225
        df['ws_eff'] = df['avg_wind_speed'] * (df['air_density'] / 1.225) ** (1/3)
        df['ws_eff_cubed'] = df['ws_eff'] ** 3

    # Pressure derivatives (already in rolling, but explicit short-term)
    if 'pressure_msl' in df.columns:
        df['pres_diff_3h'] = df['pressure_msl'].diff(3)
        df['pres_diff_6h'] = df['pressure_msl'].diff(6)
        df['pres_diff_12h'] = df['pressure_msl'].diff(12)

    # Gust ratio: max gust / mean wind
    if 'wind_gusts_10m' in df.columns and 'wind_speed_10m' in df.columns:
        df['gust_excess_ratio'] = (df['wind_gusts_10m'] - df['wind_speed_10m']) / (df['wind_speed_10m'] + 1e-3)

    # Turbulence intensity proxy (rolling std / mean of ws80 over 6h)
    if 'wind_speed_80m' in df.columns:
        std6 = df['wind_speed_80m'].rolling(window=6, min_periods=1, center=True).std()
        mean6 = df['wind_speed_80m'].rolling(window=6, min_periods=1, center=True).mean()
        df['ti_proxy_6h'] = std6 / (mean6 + 1e-3)

    # Ramp rate (wind speed change over 3h — indicates frontal passages)
    if 'wind_speed_80m' in df.columns:
        df['ws_ramp_3h'] = df['wind_speed_80m'].diff(3)
        df['ws_ramp_abs_3h'] = df['ws_ramp_3h'].abs()

    # Solar elevation (proxy via doy + hour)
    if 'dayofyear' in df.columns and 'hour' in df.columns:
        # Approximation: sin solar declination + hour-angle
        decl = 23.45 * np.sin(np.deg2rad(360 * (df['dayofyear'] - 81) / 365))
        ha = 15 * (df['hour'] - 12)  # hour angle
        lat = 46.83
        sin_elev = (np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(decl)) +
                    np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(ha)))
        df['solar_elevation_sin'] = sin_elev
        df['is_daytime'] = (sin_elev > 0).astype(int)
        df['is_night'] = (sin_elev < -0.1).astype(int)
        df['night_stable'] = df['is_night'] * df.get('atm_stable', 0)

    # Ice risk: cold + humid + precipitation
    if 'temperature_80m' in df.columns and 'snowfall' in df.columns and 'rain' in df.columns:
        df['ice_risk'] = ((df['temperature_80m'] < 1) & ((df['snowfall'] + df['rain']) > 0.1)).astype(int)

    # === НОВЫЕ ФИЧИ: турбулентность и порывы ===
    if 'wind_gusts_10m' in df.columns and 'wind_speed_10m' in df.columns:
        df['gust_factor_10m'] = df['wind_gusts_10m'] / (df['wind_speed_10m'] + 1e-3)
        df['gust_excess_10m'] = (df['wind_gusts_10m'] - df['wind_speed_10m']).clip(lower=0)
    # Температурный градиент 80m-120m (атм. устойчивость)
    if 'temperature_80m' in df.columns and 'temperature_120m' in df.columns:
        df['temp_gradient_80_120'] = df['temperature_80m'] - df['temperature_120m']
        # Положительный → неустойчивая стратификация → турбулентность
        df['stable_atmosphere'] = (df['temp_gradient_80_120'] < -0.5).astype(int)
    # Замерзание + осадки (риск обледенения лопастей)
    if 'temperature_80m' in df.columns:
        df['is_freezing'] = (df['temperature_80m'] < 0).astype(int)
        df['is_very_cold'] = (df['temperature_80m'] < -10).astype(int)
    if 'temperature_80m' in df.columns and ('snowfall' in df.columns or 'rain' in df.columns):
        precip = (df.get('snowfall', 0) + df.get('rain', 0) + df.get('showers', 0))
        df['freezing_precip'] = ((df['temperature_80m'] < 1) & (precip > 0.1)).astype(int)
        df['total_precip'] = precip
    # Разница направлений ветра на двух высотах (модуль вектора)
    if 'wind_speed_80m_u' in df.columns and 'wind_speed_120m_u' in df.columns:
        du = df['wind_speed_120m_u'] - df['wind_speed_80m_u']
        dv = df['wind_speed_120m_v'] - df['wind_speed_80m_v']
        df['wind_vec_diff_80_120'] = np.sqrt(du**2 + dv**2)
    if 'wind_speed_80m_u' in df.columns and 'wind_speed_180m_u' in df.columns:
        du = df['wind_speed_180m_u'] - df['wind_speed_80m_u']
        dv = df['wind_speed_180m_v'] - df['wind_speed_80m_v']
        df['wind_vec_diff_80_180'] = np.sqrt(du**2 + dv**2)
    # Квадрат сдвига (kinetic energy proxy turbulence)
    if 'wind_shear_80_120' in df.columns:
        df['wind_shear_80_120_sq'] = df['wind_shear_80_120'] ** 2
    if 'wind_shear_10_80' in df.columns:
        df['wind_shear_10_80_sq'] = df['wind_shear_10_80'] ** 2
    return df

# ============================================================
# Прошлые / будущие скользящие окна по NWP (без таргета)
# ============================================================
def add_rolling_nwp(df):
    cols = [c for c in ['wind_speed_80m', 'wind_speed_120m', 'hub_wind_speed'] if c in df.columns]
    for col in cols:
        df[f'{col}_past_mean_6h']  = df[col].rolling(window=6,  min_periods=1).mean().shift(1)
        df[f'{col}_past_mean_24h'] = df[col].rolling(window=24, min_periods=1).mean().shift(1)
        df[f'{col}_past_max_24h']  = df[col].rolling(window=24, min_periods=1).max().shift(1)
        df[f'{col}_past_std_24h']  = df[col].rolling(window=24, min_periods=1).std().shift(1)
        df[f'{col}_diff_1h']       = df[col].diff(1)
        df[f'{col}_diff_3h']       = df[col].diff(3)
        df[f'{col}_diff_6h']       = df[col].diff(6)
        df[f'{col}_future_mean_6h']  = df[col].rolling(window=6,  min_periods=1).mean().shift(-5)
        df[f'{col}_future_mean_12h'] = df[col].rolling(window=12, min_periods=1).mean().shift(-11)
        df[f'{col}_future_mean_24h'] = df[col].rolling(window=24, min_periods=1).mean().shift(-23)
        df[f'{col}_future_std_24h']  = df[col].rolling(window=24, min_periods=1).std().shift(-23)
        df[f'{col}_future_max_24h']  = df[col].rolling(window=24, min_periods=1).max().shift(-23)
        df[f'{col}_future_min_24h']  = df[col].rolling(window=24, min_periods=1).min().shift(-23)
    # Скользящие фичи направления (стабильность ветра)
    for h in [80, 120]:
        u_col = f'wind_speed_{h}m_u'; v_col = f'wind_speed_{h}m_v'
        if u_col in df.columns and v_col in df.columns:
            u_mean = df[u_col].rolling(window=12, min_periods=1).mean()
            v_mean = df[v_col].rolling(window=12, min_periods=1).mean()
            speed_mean = df[f'wind_speed_{h}m'].rolling(window=12, min_periods=1).mean()
            df[f'wind_dir_stability_{h}m_12h'] = np.sqrt(u_mean**2 + v_mean**2) / (speed_mean + 1e-3)
    # Скользящая std порывов (если есть)
    if 'wind_gusts_10m' in df.columns:
        df['gusts_past_std_24h'] = df['wind_gusts_10m'].rolling(window=24, min_periods=1).std().shift(1)
    return df

# ============================================================
# OpenMeteo actuals — мульти-источник погоды (ERA5 + 4 других)
# ============================================================
ACTUALS_PATH = 'data/actuals_multisource.parquet'

def attach_actuals(df, path=ACTUALS_PATH):
    """Прикрепляет actual погоду из 5 источников к df по DATETIME."""
    import os
    if not os.path.exists(path):
        return df
    actuals = pd.read_parquet(path)
    # OpenMeteo возвращает wind в km/h, конвертируем в m/s (как в train data)
    KMH_TO_MS = 1.0 / 3.6
    for c in actuals.columns:
        if 'wind_speed' in c or 'wind_gusts' in c:
            actuals[c] = actuals[c] * KMH_TO_MS
    # actuals: index = datetime; columns = e.g. 'wind_speed_100m__era5', ...

    # Compute consensus (median across 5 models) for each variable
    models = ['era5', 'era5_land', 'ecmwf_ifs', 'dwd_icon_eu', 'gfs_seamless']
    var_groups = {
        'wind_speed_10m':       [f'wind_speed_10m__{m}' for m in models],
        'wind_speed_100m':      [f'wind_speed_100m__{m}' for m in models],
        'wind_direction_10m':   [f'wind_direction_10m__{m}' for m in models],
        'wind_direction_100m':  [f'wind_direction_100m__{m}' for m in models],
        'wind_gusts_10m':       [f'wind_gusts_10m__{m}' for m in models],
        'temperature_2m':       [f'temperature_2m__{m}' for m in models],
        'surface_pressure':     [f'surface_pressure__{m}' for m in models],
        'precipitation':        [f'precipitation__{m}' for m in models],
        'cloud_cover_low':      [f'cloud_cover_low__{m}' for m in models],
    }
    actuals_features = pd.DataFrame(index=actuals.index)
    for var, cols in var_groups.items():
        cols_present = [c for c in cols if c in actuals.columns]
        if len(cols_present) == 0:
            continue
        arr = actuals[cols_present].astype(float).values
        if 'direction' in var:
            # Angular median: convert to radians, mean sin/cos, then atan2
            rad = np.deg2rad(arr)
            mean_sin = np.nanmean(np.sin(rad), axis=1)
            mean_cos = np.nanmean(np.cos(rad), axis=1)
            actuals_features[f'act_{var}_cons'] = (np.rad2deg(np.arctan2(mean_sin, mean_cos))) % 360
            actuals_features[f'act_{var}_std']  = np.nanstd(rad, axis=1)
        else:
            actuals_features[f'act_{var}_cons'] = np.nanmedian(arr, axis=1)
            actuals_features[f'act_{var}_std']  = np.nanstd(arr, axis=1)

    # Hub wind (80m) from actuals via power law: ws_at_h = ws_10m * (h/10)^alpha, alpha=0.143
    # Or interpolate from 10 and 100m. Use power-law between two known heights.
    ws10 = actuals_features['act_wind_speed_10m_cons']
    ws100 = actuals_features['act_wind_speed_100m_cons']
    # Hub at 80m: linear interp on log10 height
    actuals_features['act_wind_speed_80m_hub']  = ws10 * (8.0 ** 0.143) * (ws100 / (ws10 + 1e-6)) ** (np.log(80/10) / np.log(100/10))
    # Simpler: direct linear interpolation in height-space (often good enough)
    actuals_features['act_wind_speed_80m_lin'] = ws10 + (ws100 - ws10) * (80 - 10) / (100 - 10)

    # Join to df by datetime
    df = df.copy()
    df = df.merge(actuals_features, how='left', left_on=DATETIME, right_index=True)

    # Build residual / disagreement features (forecast - actual)
    if 'wind_speed_80m' in df.columns:
        df['ws80_resid_vs_actcons'] = df['wind_speed_80m'] - df['act_wind_speed_80m_lin']
        df['ws80_resid_abs']        = df['ws80_resid_vs_actcons'].abs()
        # Forecast quality proxy
        df['fcst_quality'] = df['ws80_resid_abs'] / df['wind_speed_80m'].clip(lower=1.0)
    if 'wind_speed_10m' in df.columns:
        df['ws10_resid_vs_actcons'] = df['wind_speed_10m'] - df['act_wind_speed_10m_cons']
    if 'wind_speed_120m' in df.columns:
        df['ws120_resid_vs_act100'] = df['wind_speed_120m'] - df['act_wind_speed_100m_cons']
    # Pressure residual (Pa to hPa: act is in hPa likely)
    if 'pressure_msl' in df.columns:
        # Note: surface_pressure ≠ pressure_msl but close. Use diff for signal.
        df['pres_resid'] = df['pressure_msl'] - df['act_surface_pressure_cons']
    # Temperature residual
    if 'temperature_80m' in df.columns:
        df['temp_resid'] = df['temperature_80m'] - df['act_temperature_2m_cons']

    # ============ Temporal context: lags/leads на actuals (Stage E из плана) ============
    # Сортируем по времени для корректных скользящих
    df = df.sort_values(DATETIME).reset_index(drop=True)
    for col in ['act_wind_speed_80m_lin', 'act_wind_speed_100m_cons',
                'act_wind_speed_10m_cons', 'act_temperature_2m_cons', 'act_surface_pressure_cons']:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag3'] = df[col].shift(3)
            df[f'{col}_lag6'] = df[col].shift(6)
            df[f'{col}_lag12'] = df[col].shift(12)
            df[f'{col}_lead1'] = df[col].shift(-1)
            df[f'{col}_lead3'] = df[col].shift(-3)
            df[f'{col}_lead6'] = df[col].shift(-6)
            df[f'{col}_lead12'] = df[col].shift(-12)
            df[f'{col}_diff_3h'] = df[col] - df[col].shift(3)
            df[f'{col}_diff_6h'] = df[col] - df[col].shift(6)
            df[f'{col}_diff_12h'] = df[col] - df[col].shift(12)
            df[f'{col}_mean_6h'] = df[col].rolling(window=6, min_periods=1, center=True).mean()
            df[f'{col}_std_6h']  = df[col].rolling(window=6, min_periods=1, center=True).std()
            df[f'{col}_mean_12h'] = df[col].rolling(window=12, min_periods=1, center=True).mean()
            df[f'{col}_std_12h']  = df[col].rolling(window=12, min_periods=1, center=True).std()
            df[f'{col}_mean_24h'] = df[col].rolling(window=24, min_periods=1, center=True).mean()

    return df


# ============================================================
# Multi-NWP forecast evolution (Stage F)
# Несколько независимых NWP-прогнозов для тех же часов.
# Разница между NWP-моделями = неуверенность прогноза.
# ============================================================
NWP_EVO_PATH = 'data/forecast_evolution.parquet'

def attach_nwp_evolution(df, path=NWP_EVO_PATH):
    """Attach multi-NWP forecast features (ecmwf, icon, gfs) + consensus + spread."""
    import os
    if not os.path.exists(path):
        return df
    nwp = pd.read_parquet(path)
    KMH_TO_MS = 1.0 / 3.6
    for c in nwp.columns:
        if 'wind_speed' in c or 'wind_gusts' in c:
            nwp[c] = nwp[c] * KMH_TO_MS

    models = ['ecmwf_ifs', 'dwd_icon_eu', 'gfs_seamless']
    nwp_feat = pd.DataFrame(index=nwp.index)
    # WS100 by source
    for m in models:
        c = f'wind_speed_100m__hfc_{m}'
        if c in nwp.columns:
            nwp_feat[f'nwp_ws100_{m}'] = nwp[c]
        c = f'wind_direction_100m__hfc_{m}'
        if c in nwp.columns:
            nwp_feat[f'nwp_wd100_{m}'] = nwp[c]

    ws_cols = [c for c in nwp_feat.columns if 'nwp_ws100_' in c]
    if ws_cols:
        nwp_feat['nwp_ws100_cons'] = nwp_feat[ws_cols].median(axis=1)
        nwp_feat['nwp_ws100_spread'] = nwp_feat[ws_cols].std(axis=1)
        nwp_feat['nwp_ws100_max_min_diff'] = nwp_feat[ws_cols].max(axis=1) - nwp_feat[ws_cols].min(axis=1)
    # Temperature
    t_cols = []
    for m in models:
        c = f'temperature_2m__hfc_{m}'
        if c in nwp.columns:
            nwp_feat[f'nwp_t_{m}'] = nwp[c]
            t_cols.append(f'nwp_t_{m}')
    if t_cols:
        nwp_feat['nwp_t_cons'] = nwp_feat[t_cols].median(axis=1)
        nwp_feat['nwp_t_spread'] = nwp_feat[t_cols].std(axis=1)
    # Pressure
    p_cols = []
    for m in models:
        c = f'surface_pressure__hfc_{m}'
        if c in nwp.columns:
            nwp_feat[f'nwp_p_{m}'] = nwp[c]
            p_cols.append(f'nwp_p_{m}')
    if p_cols:
        nwp_feat['nwp_p_cons'] = nwp_feat[p_cols].median(axis=1)
        nwp_feat['nwp_p_spread'] = nwp_feat[p_cols].std(axis=1)
    # Wind gusts
    g_cols = []
    for m in models:
        c = f'wind_gusts_10m__hfc_{m}'
        if c in nwp.columns:
            nwp_feat[f'nwp_gust_{m}'] = nwp[c]
            g_cols.append(f'nwp_gust_{m}')
    if g_cols:
        nwp_feat['nwp_gust_cons'] = nwp_feat[g_cols].median(axis=1)
        nwp_feat['nwp_gust_spread'] = nwp_feat[g_cols].std(axis=1)

    df = df.copy()
    df = df.merge(nwp_feat, how='left', left_on=DATETIME, right_index=True)

    # Residuals: train forecast (NWP) vs each external NWP
    if 'wind_speed_120m' in df.columns:
        for m in models:
            c = f'nwp_ws100_{m}'
            if c in df.columns:
                df[f'train_resid_vs_{m}'] = df['wind_speed_120m'] - df[c]  # 120m ~ 100m
    if 'wind_speed_80m' in df.columns and 'nwp_ws100_cons' in df.columns:
        df['train_resid_vs_nwp_cons_80'] = df['wind_speed_80m'] - df['nwp_ws100_cons'] * 0.94  # interpolate to 80m

    # CRITICAL: comparison of our forecast (train) vs actuals (ERA5) vs alternative NWPs
    # If actuals are between our forecast and another NWP — that NWP is better aligned
    if 'act_wind_speed_100m_cons' in df.columns and 'nwp_ws100_cons' in df.columns:
        # Distance from actuals to OUR forecast vs distance to alt NWP
        df['our_minus_act_100'] = df['wind_speed_120m'] - df['act_wind_speed_100m_cons'] if 'wind_speed_120m' in df.columns else 0
        df['nwp_minus_act_100'] = df['nwp_ws100_cons'] - df['act_wind_speed_100m_cons']
        df['who_is_closer_to_act'] = np.abs(df['our_minus_act_100']) - np.abs(df['nwp_minus_act_100'])

    return df


# ============================================================
# Spatial actuals: 9 точек вокруг ВЭС (Stage D из плана)
# ============================================================
SPATIAL_PATH = 'data/actuals_spatial.parquet'

def attach_spatial(df, path=SPATIAL_PATH):
    """Прикрепляет spatial погоду из 9 точек (R=15км)."""
    import os
    if not os.path.exists(path):
        return df
    spatial = pd.read_parquet(path)
    KMH_TO_MS = 1.0 / 3.6
    for c in spatial.columns:
        if 'wind_speed' in c:
            spatial[c] = spatial[c] * KMH_TO_MS

    dirs = ['center','N','NE','E','SE','S','SW','W','NW']
    spatial_features = pd.DataFrame(index=spatial.index)
    # WS at 100m for each direction
    for d in dirs:
        col = f'wind_speed_100m__{d}__era5'
        if col in spatial.columns:
            spatial_features[f'spat_ws100_{d}'] = spatial[col]
    # WD at 100m for each direction
    for d in dirs:
        col = f'wind_direction_100m__{d}__era5'
        if col in spatial.columns:
            spatial_features[f'spat_wd100_{d}'] = spatial[col]

    # Spatial gradients
    if all(f'spat_ws100_{d}' in spatial_features.columns for d in ['N','S','E','W']):
        spatial_features['spat_grad_NS'] = (spatial_features['spat_ws100_N'] - spatial_features['spat_ws100_S']) / 30.0
        spatial_features['spat_grad_EW'] = (spatial_features['spat_ws100_E'] - spatial_features['spat_ws100_W']) / 30.0
    # Marine (south, southwest, southeast — Azov sea) vs continental (N, NE, NW)
    if all(f'spat_ws100_{d}' in spatial_features.columns for d in ['S','SE','SW','N','NE','NW']):
        spatial_features['spat_ws_marine'] = (spatial_features['spat_ws100_S'] + spatial_features['spat_ws100_SE'] + spatial_features['spat_ws100_SW']) / 3
        spatial_features['spat_ws_continental'] = (spatial_features['spat_ws100_N'] + spatial_features['spat_ws100_NE'] + spatial_features['spat_ws100_NW']) / 3
        spatial_features['spat_marine_continental_contrast'] = spatial_features['spat_ws_marine'] - spatial_features['spat_ws_continental']
    # Std across all 9 points (spatial heterogeneity)
    ws_cols = [c for c in spatial_features.columns if c.startswith('spat_ws100_')]
    if ws_cols:
        spatial_features['spat_ws_std9'] = spatial_features[ws_cols].std(axis=1)
        spatial_features['spat_ws_mean9'] = spatial_features[ws_cols].mean(axis=1)
        spatial_features['spat_ws_max9'] = spatial_features[ws_cols].max(axis=1)
        spatial_features['spat_ws_min9'] = spatial_features[ws_cols].min(axis=1)
    # Temperature contrast (Azov sea vs land)
    for d in dirs:
        col = f'temperature_2m__{d}__era5'
        if col in spatial.columns:
            spatial_features[f'spat_t_{d}'] = spatial[col]
    if all(f'spat_t_{d}' in spatial_features.columns for d in ['S','N']):
        spatial_features['spat_t_NS_contrast'] = spatial_features['spat_t_S'] - spatial_features['spat_t_N']

    df = df.copy()
    df = df.merge(spatial_features, how='left', left_on=DATETIME, right_index=True)
    return df


# ============================================================
# Мастер-конвейер инженерии (без баселайна и климатологии)
# ============================================================
def engineer_features(df, repair_col=None, with_actuals=False, with_spatial=False, with_nwp_evo=False):
    df = df.copy()
    df = impute_ws180m(df)
    df = add_time_features(df)
    df = add_wind_direction_features(df)
    df = add_physics_features(df, repair_col=repair_col)
    df = add_rolling_nwp(df)
    if with_actuals:
        df = attach_actuals(df)
    if with_spatial:
        df = attach_spatial(df)
    if with_nwp_evo:
        df = attach_nwp_evolution(df)
    return df

# ============================================================
# Изотонический бейзлайн: hub_wind_speed -> power_per_turbine
# ============================================================
def fit_baseline_curve(train_df, target_col, repair_col=None):
    df = train_df
    if repair_col is not None and repair_col in df.columns:
        n_eff = (TOTAL_TURBINES - df[repair_col]).astype(float).clip(lower=1.0)
    else:
        n_eff = np.full(len(df), float(TOTAL_TURBINES))
    power_per = df[target_col].values / n_eff.values
    hub = df['hub_wind_speed'].values
    mask = (~np.isnan(hub)) & (~np.isnan(power_per))
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0)
    iso.fit(hub[mask], power_per[mask])
    return iso

def apply_baseline_curve(df, iso, repair_col=None):
    if repair_col is not None and repair_col in df.columns:
        n_eff = (TOTAL_TURBINES - df[repair_col]).astype(float).clip(lower=1.0).values
    else:
        n_eff = np.full(len(df), float(TOTAL_TURBINES))
    hub_arr = df['hub_wind_speed'].fillna(method='ffill').fillna(0).values
    per_turbine = iso.predict(hub_arr)
    df['baseline_iso_per_turbine'] = per_turbine
    df['baseline_iso'] = per_turbine * n_eff
    return df

# ============================================================
# Per-month изотонический бейзлайн: 12 кривых, по одной на месяц
# ============================================================
def fit_baseline_curve_per_month(train_df, target_col, repair_col=None, min_rows=200):
    """Фитит 12 изотоник, по одной на каждый месяц.
    Для месяцев с малым кол-вом данных fallback к глобальной кривой."""
    if repair_col is not None and repair_col in train_df.columns:
        n_eff = (TOTAL_TURBINES - train_df[repair_col]).astype(float).clip(lower=1.0)
    else:
        n_eff = pd.Series(float(TOTAL_TURBINES), index=train_df.index)
    power_per = train_df[target_col].values / n_eff.values
    hub = train_df['hub_wind_speed'].values
    months_arr = train_df[DATETIME].dt.month.values

    global_iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0)
    g_mask = (~np.isnan(hub)) & (~np.isnan(power_per))
    global_iso.fit(hub[g_mask], power_per[g_mask])

    per_month = {}
    for m in range(1, 13):
        mm = (months_arr == m) & (~np.isnan(hub)) & (~np.isnan(power_per))
        if mm.sum() < min_rows:
            per_month[m] = global_iso
        else:
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0)
            iso.fit(hub[mm], power_per[mm])
            per_month[m] = iso
    return {'global': global_iso, 'per_month': per_month}

def apply_baseline_curve_per_month(df, iso_dict, repair_col=None):
    if repair_col is not None and repair_col in df.columns:
        n_eff = (TOTAL_TURBINES - df[repair_col]).astype(float).clip(lower=1.0).values
    else:
        n_eff = np.full(len(df), float(TOTAL_TURBINES))
    hub_arr = df['hub_wind_speed'].fillna(method='ffill').fillna(0).values
    months_arr = df[DATETIME].dt.month.values

    per_turbine = np.zeros(len(df), dtype=np.float64)
    for m in range(1, 13):
        mask = months_arr == m
        if mask.sum() == 0:
            continue
        iso = iso_dict['per_month'].get(m, iso_dict['global'])
        per_turbine[mask] = iso.predict(hub_arr[mask])

    df['baseline_iso_per_turbine'] = per_turbine
    df['baseline_iso'] = per_turbine * n_eff
    # Также добавим глобальную кривую как доп.фичу — модель сможет учиться сезонной поправке
    df['baseline_iso_global'] = iso_dict['global'].predict(hub_arr) * n_eff
    return df

# ============================================================
# Климатологические таблицы (медиана таргета по бинам)
# ============================================================
def fit_climatology_tables(train_df, target_col):
    df = train_df.copy()
    dt = df[DATETIME]
    df['_month'] = dt.dt.month.values
    df['_hour'] = dt.dt.hour.values
    df['_doy_bin'] = (dt.dt.dayofyear // DOY_BIN).astype(int).values
    if 'hub_wind_speed' in df.columns:
        df['_hub_bin'] = (df['hub_wind_speed'] / HUB_BIN).round() * HUB_BIN
    else:
        df['_hub_bin'] = (df['wind_speed_80m'] / HUB_BIN).round() * HUB_BIN
    df['_ws80_bin'] = (df['wind_speed_80m'] / WS_BIN).round() * WS_BIN

    g = float(df[target_col].median())
    by_mh = df.groupby(['_month', '_hour'])[target_col].median().to_dict()
    by_dh = df.groupby(['_doy_bin', '_hour'])[target_col].median().to_dict()
    by_hm = df.groupby(['_hub_bin', '_month'])[target_col].median().to_dict()
    by_hmh = df.groupby(['_hub_bin', '_month', '_hour'])[target_col].median().to_dict()
    by_wm = df.groupby(['_ws80_bin', '_month'])[target_col].median().to_dict()
    return {
        'global_median': g,
        'by_month_hour': by_mh,
        'by_doy_hour': by_dh,
        'by_hub_month': by_hm,
        'by_hub_month_hour': by_hmh,
        'by_ws80_month': by_wm,
    }

def apply_climatology(df, tables):
    dt = df[DATETIME]
    months = dt.dt.month.values
    hours = dt.dt.hour.values
    doy_bins = (dt.dt.dayofyear // DOY_BIN).astype(int).values
    if 'hub_wind_speed' in df.columns:
        hub_bins = (df['hub_wind_speed'].fillna(0).values / HUB_BIN).round() * HUB_BIN
    else:
        hub_bins = (df['wind_speed_80m'].fillna(0).values / HUB_BIN).round() * HUB_BIN
    ws_bins = (df['wind_speed_80m'].fillna(0).values / WS_BIN).round() * WS_BIN

    g = tables['global_median']
    bmh = tables['by_month_hour']; bdh = tables['by_doy_hour']
    bhm = tables['by_hub_month']; bhmh = tables['by_hub_month_hour']
    bwm = tables['by_ws80_month']

    df['clim_month_hour']     = [bmh.get((m, h), g) for m, h in zip(months, hours)]
    df['clim_doy_hour']       = [bdh.get((d, h), g) for d, h in zip(doy_bins, hours)]
    df['clim_hub_month']      = [bhm.get((b, m), g) for b, m in zip(hub_bins, months)]
    df['clim_hub_month_hour'] = [bhmh.get((b, m, h), g) for b, m, h in zip(hub_bins, months, hours)]
    df['clim_ws80_month']     = [bwm.get((b, m), g) for b, m in zip(ws_bins, months)]
    return df

# ============================================================
# Аномалии (курилмент / сенсорные ошибки) — для очистки train
# ============================================================
def anomaly_mask(df, target_col):
    if target_col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    target = df[target_col].values
    ws = df['wind_speed_80m'].values
    a = (target < 1.0) & (ws > 5.0)
    b = (target > 70.0) & (ws < 4.0)
    return a | b

# ============================================================
# Список «всех потенциально полезных» признаков
# ============================================================
def build_feature_list(df, exclude_extra=()):
    """Возвращает имена колонок, пригодных для модели (числовые, без target/datetime/служебных)."""
    drop = {DATETIME, 'Результат расчёта'}
    drop.update(exclude_extra)
    # отбрасываем target колонку, если она в df
    tcol = find_col(df, 'выработка')
    if tcol is not None:
        drop.add(tcol)
    rcol = find_col(df, 'ремонт')
    if rcol is not None:
        drop.add(rcol)
    feats = []
    for c in df.columns:
        if c in drop: continue
        if df[c].dtype == 'O':
            continue
        feats.append(c)
    return feats
