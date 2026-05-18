#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — обучение трёхгоризонтного ансамбля (near / far / vfar), residual-постановка.

Использование:
    python train.py                  # обучает на полном train, сохраняет в model/
    python train.py --holdout        # обучает на train[:-90d], сохраняет в model_holdout/
"""
import os, json, warnings, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from tqdm import tqdm

import features as F

warnings.filterwarnings('ignore')

HISTORY_TARGET_LEN = 24
NEAR_HORIZON = 12
FAR_HORIZON  = 48                       # near=[1..12], far=[13..48], vfar=>48
HOLDOUT_DAYS = 90
VAL_FRACTION = 0.10
SAMPLE_WEIGHT_HALFLIFE_YEARS = 2.0
USE_CATBOOST = True

try:
    from catboost import CatBoostRegressor
    CATBOOST_OK = True
except Exception:
    CATBOOST_OK = False

LGB_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'n_estimators': 3000,
    'learning_rate': 0.015,
    'num_leaves': 63,
    'max_depth': -1,
    'subsample': 0.8,
    'subsample_freq': 5,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'min_child_samples': 60,
    'verbosity': -1,
    'n_jobs': -1,
}
DEFAULT_SEEDS = [42, 123, 777, 2024, 9999]
EXTENDED_SEEDS = [42, 123, 777, 2024, 9999, 31337, 65535, 1024, 88, 1791]
LGB_NEAR_SEEDS = DEFAULT_SEEDS
LGB_FAR_SEEDS  = DEFAULT_SEEDS
LGB_VFAR_SEEDS = DEFAULT_SEEDS
CB_SEEDS = [42, 123]


def load_train(path):
    df = pd.read_csv(path, parse_dates=[F.DATETIME])
    df.sort_values(F.DATETIME, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_direct_dataset(df_feat, target_col, baseline_col, feature_cols,
                         max_horizon, history_len, step=1, anomaly_mask=None, use_lags=True):
    n = len(df_feat)
    target_arr = df_feat[target_col].values.astype(np.float32)
    baseline_arr = df_feat[baseline_col].values.astype(np.float32)
    feature_mat = df_feat[feature_cols].values.astype(np.float32)
    n_feat = len(feature_cols)
    lag_dim = history_len if use_lags else 0
    full_dim = n_feat + lag_dim + 1   # +lags? +hours_ahead

    origins = list(range(history_len, n - max_horizon, step))
    if anomaly_mask is None:
        anomaly_mask = np.zeros(n, dtype=bool)

    total = len(origins) * max_horizon
    print(f"  build direct: {total} examples ({len(origins)} origins x {max_horizon}h)  use_lags={use_lags}")
    X = np.zeros((total, full_dim), dtype=np.float32)
    y = np.zeros(total, dtype=np.float32)
    ha = np.zeros(total, dtype=np.int16)
    oidx = np.zeros(total, dtype=np.int32)
    keep = np.zeros(total, dtype=bool)

    idx = 0
    for i in tqdm(origins, desc='  origins'):
        if use_lags:
            lags = target_arr[i - history_len:i][::-1]
        for h in range(1, max_horizon + 1):
            t = i + h
            X[idx, :n_feat] = feature_mat[t]
            if use_lags:
                X[idx, n_feat:n_feat + lag_dim] = lags
            X[idx, -1] = float(h)
            y[idx] = target_arr[t] - baseline_arr[t]
            ha[idx] = h
            oidx[idx] = i
            keep[idx] = not anomaly_mask[t]
            idx += 1
    valid = keep & ~np.isnan(y)
    return X[valid], y[valid], ha[valid], oidx[valid]


def compute_sample_weights(origin_idx, datetime_series, half_life_years=SAMPLE_WEIGHT_HALFLIFE_YEARS):
    dt0 = pd.Timestamp(datetime_series.iloc[0])
    sel = datetime_series.iloc[origin_idx].values
    days = (sel - np.datetime64(dt0)) / np.timedelta64(1, 'D')
    return np.exp(days.astype(np.float64) / (half_life_years * 365.0) * np.log(2.0)).astype(np.float32)


def time_split_mask(origin_idx, val_fraction=VAL_FRACTION):
    cutoff = np.quantile(origin_idx, 1.0 - val_fraction)
    val_mask = origin_idx >= cutoff
    return ~val_mask, val_mask


def train_lgb(X_tr, y_tr, X_val, y_val, w_tr=None, w_val=None, seed=42, params=None):
    p = dict(params or LGB_PARAMS)
    p['random_state'] = seed
    p['seed'] = seed
    model = lgb.LGBMRegressor(**p)
    eval_kwargs = dict(
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
    )
    if w_val is not None:
        eval_kwargs['eval_sample_weight'] = [w_val]
    model.fit(X_tr, y_tr, sample_weight=w_tr, **eval_kwargs)
    return model


def train_catboost(X_tr, y_tr, X_val, y_val, w_tr=None, w_val=None, seed=42):
    if not CATBOOST_OK:
        return None
    model = CatBoostRegressor(
        loss_function='MAE',
        iterations=3000,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=0,
        thread_count=-1,
        early_stopping_rounds=150,
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=(X_val, y_val), use_best_model=True, verbose=0)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout', action='store_true', help='Не использовать последние 90 дней')
    parser.add_argument('--no-catboost', action='store_true', help='Не обучать CatBoost')
    parser.add_argument('--no-sample-weights', action='store_true', help='Отключить sample weights')
    parser.add_argument('--exclude-from', type=str, default=None, help='YYYY-MM-DD: исключить из train с этой даты')
    parser.add_argument('--exclude-to', type=str, default=None, help='YYYY-MM-DD: исключить из train по эту дату (исключительно)')
    parser.add_argument('--num-leaves', type=int, default=None)
    parser.add_argument('--min-child', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--reg-alpha', type=float, default=None)
    parser.add_argument('--reg-lambda', type=float, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--step', type=int, default=1, help='Шаг по origin (по умолчанию 1)')
    parser.add_argument('--no-climatology', action='store_true', help='Не использовать клим-фичи (clim_*)')
    parser.add_argument('--loss', type=str, choices=['l1','l2','huber','quantile'], default='l1', help='LGB objective')
    parser.add_argument('--drop-power-potential', action='store_true', help='Убрать power_potential_* (дубли baseline_iso)')
    parser.add_argument('--drop-target-lag', action='store_true', help='Не использовать target_lag_* в near/far')
    parser.add_argument('--drop-feature-prefixes', type=str, default='', help='Запятая-разделённый список префиксов для удаления')
    parser.add_argument('--per-month-iso', action='store_true', help='Использовать per-month isotonic baseline (12 кривых)')
    parser.add_argument('--season-weight', type=float, default=1.0, help='Множитель веса для Q1 (Jan-Mar) строк (1.0=без эффекта)')
    parser.add_argument('--seeds', type=str, default='default', choices=['default', 'extended'], help='Набор сидов: default=5, extended=10')
    parser.add_argument('--no-turbulence', action='store_true', help='Убрать новые турбулентные фичи (для fair-теста)')
    parser.add_argument('--boosting', type=str, default='gbdt', choices=['gbdt','dart','goss'], help='LGB boosting type')
    parser.add_argument('--no-residual', action='store_true', help='Predict target directly (zero baseline)')
    parser.add_argument('--log-target', action='store_true', help='Train on log(target+1)')
    parser.add_argument('--smooth-target', type=int, default=0, help='Smooth target with rolling mean of W hours (0 = no smooth)')
    parser.add_argument('--mult-residual', action='store_true', help='Train on multiplicative residual (target / max(baseline, eps))')
    parser.add_argument('--with-actuals', action='store_true', help='Attach OpenMeteo multi-source actual weather features')
    parser.add_argument('--with-spatial', action='store_true', help='Attach spatial weather (9 points around station)')
    parser.add_argument('--with-nwp-evo', action='store_true', help='Attach multi-NWP forecast features (Stage F)')
    args = parser.parse_args()
    global LGB_NEAR_SEEDS, LGB_FAR_SEEDS, LGB_VFAR_SEEDS
    if args.seeds == 'extended':
        LGB_NEAR_SEEDS = EXTENDED_SEEDS
        LGB_FAR_SEEDS = EXTENDED_SEEDS
        LGB_VFAR_SEEDS = EXTENDED_SEEDS

    if args.loss == 'l2':
        LGB_PARAMS['objective'] = 'regression'
    elif args.loss == 'huber':
        LGB_PARAMS['objective'] = 'huber'
        LGB_PARAMS['alpha'] = 0.9
    elif args.loss == 'quantile':
        LGB_PARAMS['objective'] = 'quantile'
        LGB_PARAMS['alpha'] = 0.5  # медиана
    # 'l1' is default (regression_l1)
    if args.boosting != 'gbdt':
        LGB_PARAMS['boosting_type'] = args.boosting
        if args.boosting == 'dart':
            LGB_PARAMS['drop_rate'] = 0.1
            LGB_PARAMS['skip_drop'] = 0.5
            LGB_PARAMS['n_estimators'] = 1500  # dart медленнее, делаем меньше
        if args.boosting == 'goss':
            # GOSS не поддерживает bagging — убираем subsample
            LGB_PARAMS.pop('subsample', None)
            LGB_PARAMS.pop('subsample_freq', None)
            LGB_PARAMS['top_rate'] = 0.2
            LGB_PARAMS['other_rate'] = 0.1

    # Override LGB params if CLI specified
    if args.num_leaves is not None: LGB_PARAMS['num_leaves'] = args.num_leaves
    if args.min_child is not None: LGB_PARAMS['min_child_samples'] = args.min_child
    if args.lr is not None: LGB_PARAMS['learning_rate'] = args.lr
    if args.reg_alpha is not None: LGB_PARAMS['reg_alpha'] = args.reg_alpha
    if args.reg_lambda is not None: LGB_PARAMS['reg_lambda'] = args.reg_lambda

    use_catboost = USE_CATBOOST and CATBOOST_OK and not args.no_catboost
    train_path = 'data/train_dataset.csv'
    out_dir = args.out_dir if args.out_dir else ('model_holdout' if args.holdout else 'model')
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Загрузка {train_path} ===")
    df = load_train(train_path)
    target_col = F.find_target_col(df)
    repair_col = F.find_repair_col(df)
    print(f"  target_col={target_col!r}  repair_col={repair_col!r}")
    print(f"  rows={len(df)}  span: {df[F.DATETIME].iloc[0]} -> {df[F.DATETIME].iloc[-1]}")

    if args.holdout:
        last_dt = df[F.DATETIME].iloc[-1]
        cutoff_dt = last_dt - pd.Timedelta(days=HOLDOUT_DAYS)
        df_use = df[df[F.DATETIME] < cutoff_dt].reset_index(drop=True)
        print(f"  HOLDOUT: cutoff={cutoff_dt}, train rows={len(df_use)}")
    else:
        df_use = df.reset_index(drop=True)

    # Исключение периода (cross-year holdout): держим строки ВНЕ интервала [exclude-from, exclude-to)
    if args.exclude_from and args.exclude_to:
        ex_from = pd.Timestamp(args.exclude_from)
        ex_to = pd.Timestamp(args.exclude_to)
        m_exclude = (df_use[F.DATETIME] >= ex_from) & (df_use[F.DATETIME] < ex_to)
        n_before = len(df_use)
        df_use = df_use[~m_exclude].reset_index(drop=True)
        print(f"  EXCLUDE [{ex_from} .. {ex_to}): {n_before - len(df_use)} строк убрано, осталось {len(df_use)}")

    # ============ Инженерия признаков ТОЛЬКО на обучающей части ============
    print(f"=== Инженерия признаков на {len(df_use)} строках ===")
    df_use = F.engineer_features(df_use, repair_col=repair_col, with_actuals=args.with_actuals, with_spatial=args.with_spatial, with_nwp_evo=args.with_nwp_evo)

    # ============ Изотонический бейзлайн ============
    print(f"=== Фит изотонической кривой мощности ===")
    if args.per_month_iso:
        iso = F.fit_baseline_curve_per_month(df_use, target_col=target_col, repair_col=repair_col)
        joblib.dump(iso, os.path.join(out_dir, 'empirical_power_curve.joblib'))
        df_use = F.apply_baseline_curve_per_month(df_use, iso, repair_col=repair_col)
        print(f"  per-month isotonic baselines: {len(iso['per_month'])} curves")
    else:
        iso = F.fit_baseline_curve(df_use, target_col=target_col, repair_col=repair_col)
        joblib.dump(iso, os.path.join(out_dir, 'empirical_power_curve.joblib'))
        df_use = F.apply_baseline_curve(df_use, iso, repair_col=repair_col)
    b = df_use['baseline_iso'].values
    y_check = df_use[target_col].values
    m_ok = ~np.isnan(b) & ~np.isnan(y_check)
    mae_baseline = float(np.mean(np.abs(y_check[m_ok] - b[m_ok])))
    print(f"  MAE изотонического бейзлайна (in-sample): {mae_baseline:.3f}")

    # ============ Direct target prediction: zero out baseline (no residual) ============
    if args.no_residual or args.log_target:
        df_use['baseline_iso_orig'] = df_use['baseline_iso'].values
        df_use['baseline_iso'] = 0.0
        if args.no_residual:
            print(f"  NO RESIDUAL mode: baseline_iso=0, models predict target directly")

    # ============ Log target transform ============
    if args.log_target:
        df_use[target_col + '_orig'] = df_use[target_col].values
        df_use[target_col] = np.log1p(df_use[target_col].clip(lower=0))
        print(f"  LOG TARGET mode: y = log(1 + target), baseline forced to 0")

    # ============ Smooth target (rolling mean) ============
    if args.smooth_target > 1:
        df_use[target_col + '_orig'] = df_use[target_col].values
        df_use[target_col] = df_use[target_col].rolling(window=args.smooth_target, min_periods=1, center=True).mean().values
        print(f"  SMOOTH TARGET mode: y = rolling_mean(target, window={args.smooth_target})")

    # ============ Multiplicative residual ============
    if args.mult_residual:
        df_use[target_col + '_orig'] = df_use[target_col].values
        df_use['baseline_orig'] = df_use['baseline_iso'].values
        # target = ratio = target / max(baseline, eps), clip to reasonable range
        eps = 0.5
        ratio = df_use[target_col].values / np.maximum(df_use['baseline_iso'].values, eps)
        ratio = np.clip(ratio, 0, 5.0)  # cap extreme ratios
        df_use[target_col] = ratio
        df_use['baseline_iso'] = 0.0
        print(f"  MULT RESIDUAL mode: y = target / max(baseline, {eps}), models predict ratio")

    # ============ Климатологические таблицы ============
    if args.no_climatology:
        print(f"=== Климатология ОТКЛЮЧЕНА ===")
        clim_tables = {'global_median': float(df_use[target_col].median()),
                       'by_month_hour': {}, 'by_doy_hour': {},
                       'by_hub_month': {}, 'by_hub_month_hour': {}, 'by_ws80_month': {}}
        joblib.dump(clim_tables, os.path.join(out_dir, 'climatology_tables.joblib'))
        # Pad columns with global_median so apply_climatology in inference still works,
        # but they will all be the same value (no signal).
        for c in ['clim_month_hour','clim_doy_hour','clim_hub_month','clim_hub_month_hour','clim_ws80_month']:
            df_use[c] = clim_tables['global_median']
    else:
        print(f"=== Климатологические таблицы ===")
        clim_tables = F.fit_climatology_tables(df_use, target_col=target_col)
        joblib.dump(clim_tables, os.path.join(out_dir, 'climatology_tables.joblib'))
        df_use = F.apply_climatology(df_use, clim_tables)
        print(f"  global_median={clim_tables['global_median']:.2f}")

    # ============ Аномалии ============
    anomaly = F.anomaly_mask(df_use, target_col)
    print(f"  Аномальных строк: {anomaly.sum()} ({100*anomaly.mean():.2f}%)")

    # ============ Выбор признаков ============
    exclude = {F.DATETIME, target_col, 'Результат расчёта'}
    if repair_col is not None:
        exclude.add(repair_col)
    all_feats = [c for c in df_use.columns
                 if c not in exclude
                 and df_use[c].dtype != 'O'
                 and not c.startswith('target_lag')
                 and c != 'hours_ahead']
    if args.no_climatology:
        all_feats = [c for c in all_feats if not c.startswith('clim_')]
        print(f"  Клим-фичи исключены из all_feats")
    if args.no_turbulence:
        # точечный список новых фич, добавленных позже (turbulence/gust/freezing/extra rolling)
        turb_exact = {
            'gust_factor_10m', 'gust_excess_10m', 'gusts_past_std_24h',
            'temp_gradient_80_120', 'stable_atmosphere',
            'is_freezing', 'is_very_cold', 'freezing_precip', 'total_precip',
            'wind_vec_diff_80_120', 'wind_vec_diff_80_180',
            'wind_shear_80_120_sq', 'wind_shear_10_80_sq',
            'wind_dir_stability_80m_12h', 'wind_dir_stability_120m_12h',
        }
        turb_prefixes_endings = []  # rolling новые (suffixes)
        rolling_new_suffixes = ['_past_std_24h', '_diff_3h', '_future_max_24h', '_future_min_24h']
        rolling_base = ['wind_speed_80m', 'wind_speed_120m', 'hub_wind_speed']
        for b in rolling_base:
            for s in rolling_new_suffixes:
                turb_exact.add(f'{b}{s}')
        before = len(all_feats)
        all_feats = [c for c in all_feats if c not in turb_exact]
        print(f"  Турбулентные фичи исключены: {before - len(all_feats)} штук")
    if args.drop_power_potential:
        all_feats = [c for c in all_feats if not c.startswith('power_potential')]
        print(f"  power_potential_* исключены")
    if args.drop_feature_prefixes:
        for pref in args.drop_feature_prefixes.split(','):
            pref = pref.strip()
            if pref:
                before = len(all_feats)
                all_feats = [c for c in all_feats if not c.startswith(pref)]
                print(f"  Префикс '{pref}': убрано {before - len(all_feats)} фич")
    print(f"  Признаков: {len(all_feats)}")
    df_use[all_feats] = df_use[all_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df_use['baseline_iso'] = df_use['baseline_iso'].fillna(0.0)

    # ============ Direct dataset для near + far ============
    print(f"=== Direct-датасет h=1..{FAR_HORIZON}, step={args.step} ===")
    use_lags = not args.drop_target_lag
    X_all, y_all, ha_all, oidx_all = build_direct_dataset(
        df_use, target_col=target_col, baseline_col='baseline_iso',
        feature_cols=all_feats, max_horizon=FAR_HORIZON,
        history_len=HISTORY_TARGET_LEN, step=args.step, anomaly_mask=anomaly,
        use_lags=use_lags,
    )
    print(f"  итого: {len(y_all)} примеров")
    if use_lags:
        feature_names_direct = all_feats + [f'target_lag_{k}' for k in range(1, HISTORY_TARGET_LEN + 1)] + ['hours_ahead']
    else:
        feature_names_direct = all_feats + ['hours_ahead']

    if args.no_sample_weights:
        weights_direct = np.ones(len(oidx_all), dtype=np.float32)
        print(f"  sample weights ОТКЛЮЧЕНЫ (uniform)")
    else:
        weights_direct = compute_sample_weights(oidx_all, df_use[F.DATETIME])
        print(f"  sample weights: min={weights_direct.min():.3f}, max={weights_direct.max():.3f}")
    if args.season_weight != 1.0:
        # умножаем на season_weight строки где origin приходится на Q1 (Jan-Mar)
        dt_origins = df_use[F.DATETIME].iloc[oidx_all].dt.month.values
        q1_mask = (dt_origins >= 1) & (dt_origins <= 3)
        weights_direct = weights_direct.copy()
        weights_direct[q1_mask] *= args.season_weight
        print(f"  season weight={args.season_weight} применён к {q1_mask.sum()} Q1-строкам (из {len(weights_direct)})")

    tr_mask, val_mask = time_split_mask(oidx_all)
    print(f"  time-split: train={tr_mask.sum()} val={val_mask.sum()}")

    # ============ NEAR ============
    mask_near = (ha_all >= 1) & (ha_all <= NEAR_HORIZON)
    Xn = X_all[mask_near]; yn = y_all[mask_near]
    wn = weights_direct[mask_near]
    tn = tr_mask[mask_near]; vn = val_mask[mask_near]
    print(f"=== NEAR (1..{NEAR_HORIZON}ч): train={tn.sum()} val={vn.sum()} ===")
    near_files = []
    for seed in LGB_NEAR_SEEDS:
        print(f"  LGB near seed={seed}")
        m = train_lgb(Xn[tn], yn[tn], Xn[vn], yn[vn], w_tr=wn[tn], w_val=wn[vn], seed=seed)
        path = os.path.join(out_dir, f'lgb_near_seed_{seed}.lgb')
        m.booster_.save_model(path)
        near_files.append(f'lgb_near_seed_{seed}.lgb')
        print(f"    best_iter={m.booster_.best_iteration}")
    if use_catboost:
        for seed in CB_SEEDS:
            print(f"  CatBoost near seed={seed}")
            cb = train_catboost(Xn[tn], yn[tn], Xn[vn], yn[vn], w_tr=wn[tn], w_val=wn[vn], seed=seed)
            if cb is not None:
                path = os.path.join(out_dir, f'cb_near_seed_{seed}.cbm')
                cb.save_model(path)
                near_files.append(f'cb_near_seed_{seed}.cbm')
    del Xn, yn, wn, tn, vn

    # ============ FAR ============
    mask_far = (ha_all > NEAR_HORIZON) & (ha_all <= FAR_HORIZON)
    Xf = X_all[mask_far]; yf = y_all[mask_far]
    wf = weights_direct[mask_far]
    tf = tr_mask[mask_far]; vf = val_mask[mask_far]
    print(f"=== FAR ({NEAR_HORIZON+1}..{FAR_HORIZON}ч): train={tf.sum()} val={vf.sum()} ===")
    far_files = []
    for seed in LGB_FAR_SEEDS:
        print(f"  LGB far seed={seed}")
        m = train_lgb(Xf[tf], yf[tf], Xf[vf], yf[vf], w_tr=wf[tf], w_val=wf[vf], seed=seed)
        path = os.path.join(out_dir, f'lgb_far_seed_{seed}.lgb')
        m.booster_.save_model(path)
        far_files.append(f'lgb_far_seed_{seed}.lgb')
        print(f"    best_iter={m.booster_.best_iteration}")
    if use_catboost:
        for seed in CB_SEEDS:
            print(f"  CatBoost far seed={seed}")
            cb = train_catboost(Xf[tf], yf[tf], Xf[vf], yf[vf], w_tr=wf[tf], w_val=wf[vf], seed=seed)
            if cb is not None:
                path = os.path.join(out_dir, f'cb_far_seed_{seed}.cbm')
                cb.save_model(path)
                far_files.append(f'cb_far_seed_{seed}.cbm')
    del Xf, yf, wf, tf, vf, X_all, y_all, ha_all, oidx_all

    # ============ VFAR: каждая строка train — один пример ============
    print(f"=== VFAR (>{FAR_HORIZON}ч): per-row датасет ===")
    Xv_all = df_use[all_feats].values.astype(np.float32)
    yv_all = (df_use[target_col].values - df_use['baseline_iso'].values).astype(np.float32)
    valid_v = (~anomaly) & ~np.isnan(yv_all)
    Xv = Xv_all[valid_v]
    yv = yv_all[valid_v]
    dt_v = df_use.loc[valid_v, F.DATETIME].reset_index(drop=True)
    n_v = len(yv)
    cutoff_v = int(n_v * (1.0 - VAL_FRACTION))
    tr_v = np.arange(n_v) < cutoff_v
    vl_v = ~tr_v
    days_v = (dt_v.values - dt_v.values[0]) / np.timedelta64(1, 'D')
    days_v = days_v.astype(np.float64)
    if args.no_sample_weights:
        w_v = np.ones(n_v, dtype=np.float32)
    else:
        w_v = np.exp(days_v / (SAMPLE_WEIGHT_HALFLIFE_YEARS * 365.0) * np.log(2.0)).astype(np.float32)
    if args.season_weight != 1.0:
        v_months = dt_v.dt.month.values
        q1m = (v_months >= 1) & (v_months <= 3)
        w_v = w_v.copy()
        w_v[q1m] *= args.season_weight
    print(f"  VFAR train={tr_v.sum()} val={vl_v.sum()}")
    vfar_files = []
    for seed in LGB_VFAR_SEEDS:
        print(f"  LGB vfar seed={seed}")
        m = train_lgb(Xv[tr_v], yv[tr_v], Xv[vl_v], yv[vl_v], w_tr=w_v[tr_v], w_val=w_v[vl_v], seed=seed)
        path = os.path.join(out_dir, f'lgb_vfar_seed_{seed}.lgb')
        m.booster_.save_model(path)
        vfar_files.append(f'lgb_vfar_seed_{seed}.lgb')
        print(f"    best_iter={m.booster_.best_iteration}")
    if use_catboost:
        for seed in CB_SEEDS:
            print(f"  CatBoost vfar seed={seed}")
            cb = train_catboost(Xv[tr_v], yv[tr_v], Xv[vl_v], yv[vl_v], w_tr=w_v[tr_v], w_val=w_v[vl_v], seed=seed)
            if cb is not None:
                path = os.path.join(out_dir, f'cb_vfar_seed_{seed}.cbm')
                cb.save_model(path)
                vfar_files.append(f'cb_vfar_seed_{seed}.cbm')

    # ============ last_target_history для inference ============
    last_target_history = df_use[target_col].iloc[-HISTORY_TARGET_LEN:].tolist()
    last_train_datetime = str(df_use[F.DATETIME].iloc[-1])

    # ============ Артефакты ============
    joblib.dump(all_feats, os.path.join(out_dir, 'features_base.joblib'))
    joblib.dump(feature_names_direct, os.path.join(out_dir, 'features_direct.joblib'))

    meta = {
        'target_col': target_col,
        'repair_col': repair_col,
        'history_target_len': HISTORY_TARGET_LEN,
        'near_horizon': NEAR_HORIZON,
        'far_horizon': FAR_HORIZON,
        'total_turbines': F.TOTAL_TURBINES,
        'last_target_history': [float(x) for x in last_target_history],
        'last_train_datetime': last_train_datetime,
        'features_base': all_feats,
        'features_direct': feature_names_direct,
        'use_target_lag': use_lags,
        'log_target': bool(args.log_target),
        'no_residual': bool(args.no_residual),
        'smooth_target': int(args.smooth_target),
        'mult_residual': bool(args.mult_residual),
        'with_actuals': bool(args.with_actuals),
        'with_spatial': bool(args.with_spatial),
        'with_nwp_evo': bool(args.with_nwp_evo),
        'near_models': near_files,
        'far_models': far_files,
        'vfar_models': vfar_files,
        'holdout_days': HOLDOUT_DAYS if args.holdout else 0,
        'baseline_col': 'baseline_iso',
        'baseline_in_sample_mae': mae_baseline,
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n=== Готово. Артефакты в {out_dir}/ ===")
    print(f"  near={len(near_files)}  far={len(far_files)}  vfar={len(vfar_files)}")


if __name__ == '__main__':
    main()
