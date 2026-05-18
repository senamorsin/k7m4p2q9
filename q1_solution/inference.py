#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py — инференс трёхгоризонтного ансамбля.

Использование:
    python inference.py                            # модель из model/, valid_features.csv -> submission.csv
    python inference.py --model-dir model_holdout  # для холдоут-оценки
    python inference.py --out submission.csv
"""
import os, json, argparse, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

import features as F

warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostRegressor
    CATBOOST_OK = True
except Exception:
    CATBOOST_OK = False


def load_test(path):
    df = pd.read_csv(path, parse_dates=[F.DATETIME])
    # valid в исходном файле в обратном порядке — приводим к хронологии
    df = df.sort_values(F.DATETIME).reset_index(drop=True)
    return df


def load_models(model_dir, files):
    out = []
    for fn in files:
        path = os.path.join(model_dir, fn)
        if fn.endswith('.lgb'):
            out.append(('lgb', lgb.Booster(model_file=path)))
        elif fn.endswith('.cbm'):
            if not CATBOOST_OK:
                print(f"  WARNING: CatBoost не установлен, пропускаю {fn}")
                continue
            cb = CatBoostRegressor()
            cb.load_model(path)
            out.append(('cb', cb))
    return out


def ensemble_predict(models, X):
    """Среднее по всем моделям. X — numpy float32."""
    preds = []
    for kind, m in models:
        if kind == 'lgb':
            preds.append(m.predict(X))
        else:
            preds.append(m.predict(X))
    return np.mean(preds, axis=0)


def physical_caps(pred, ws80, max_target=89.5):
    pred = max(0.0, float(pred))
    pred = min(pred, max_target)
    if ws80 < 2.0:
        pred = min(pred, 1.0)
    if ws80 > 26.0:
        pred = 0.0
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='model')
    parser.add_argument('--test', default='data/valid_features.csv')
    parser.add_argument('--train', default='data/train_dataset.csv')
    parser.add_argument('--out', default='submission.csv')
    args = parser.parse_args()

    print(f"=== Inference: model_dir={args.model_dir} ===")
    with open(os.path.join(args.model_dir, 'metadata.json'), encoding='utf-8') as f:
        meta = json.load(f)
    target_col = meta['target_col']
    repair_col = meta['repair_col']
    history_len = meta['history_target_len']
    near_horizon = meta['near_horizon']
    far_horizon = meta['far_horizon']
    features_base = meta['features_base']
    use_target_lag = meta.get('use_target_lag', True)
    is_log_target = meta.get('log_target', False)
    is_no_residual = meta.get('no_residual', False)
    is_mult_residual = meta.get('mult_residual', False)
    smooth_target_w = meta.get('smooth_target', 0)
    with_actuals = meta.get('with_actuals', False)
    with_spatial = meta.get('with_spatial', False)
    with_nwp_evo = meta.get('with_nwp_evo', False)
    last_target_history = list(meta['last_target_history'])
    last_train_dt = pd.Timestamp(meta['last_train_datetime'])

    iso = joblib.load(os.path.join(args.model_dir, 'empirical_power_curve.joblib'))
    is_per_month_iso = isinstance(iso, dict) and 'per_month' in iso
    clim_tables = joblib.load(os.path.join(args.model_dir, 'climatology_tables.joblib'))

    print(f"  загрузка моделей: near={len(meta['near_models'])}, far={len(meta['far_models'])}, vfar={len(meta['vfar_models'])}")
    near_models = load_models(args.model_dir, meta['near_models'])
    far_models = load_models(args.model_dir, meta['far_models'])
    vfar_models = load_models(args.model_dir, meta['vfar_models'])

    # ---------- Загрузка данных ----------
    print(f"=== Подготовка test ({args.test}) ===")
    train_df = pd.read_csv(args.train, parse_dates=[F.DATETIME])
    train_df = train_df.sort_values(F.DATETIME).reset_index(drop=True)
    test_df = load_test(args.test)
    print(f"  train rows={len(train_df)}, test rows={len(test_df)}")
    print(f"  train end = {train_df[F.DATETIME].iloc[-1]}, test start = {test_df[F.DATETIME].iloc[0]}")

    # Удаляем target из train (его нет в test), затем concat для общей инженерии
    cols_for_concat = [c for c in train_df.columns if c != target_col]
    train_for_concat = train_df[cols_for_concat]
    test_for_concat = test_df.copy()
    # Добавим target_col=NaN в test, чтобы можно было сконкатенировать
    if target_col not in test_for_concat.columns:
        test_for_concat[target_col] = np.nan
    if target_col not in train_for_concat.columns:
        # train без target тоже добавим NaN — для симметрии
        train_for_concat = train_for_concat.copy()
        train_for_concat[target_col] = np.nan

    combined = pd.concat([train_for_concat, test_for_concat], ignore_index=True)
    combined = combined.sort_values(F.DATETIME).reset_index(drop=True)
    # NB: дубликаты по времени мы не ожидаем, но на всякий случай
    combined = combined.drop_duplicates(subset=[F.DATETIME], keep='last').reset_index(drop=True)

    print(f"  combined rows = {len(combined)} (после concat и дедупа)")

    # ---------- Feature engineering на конкатенированном df ----------
    combined = F.engineer_features(combined, repair_col=repair_col, with_actuals=with_actuals, with_spatial=with_spatial, with_nwp_evo=with_nwp_evo)
    if is_per_month_iso:
        combined = F.apply_baseline_curve_per_month(combined, iso, repair_col=repair_col)
    else:
        combined = F.apply_baseline_curve(combined, iso, repair_col=repair_col)
    combined = F.apply_climatology(combined, clim_tables)

    # Сохраним baseline_iso ДО заполнения NaN — он понадобится
    # Заполним NaN/inf в фичах
    for c in features_base:
        if c not in combined.columns:
            combined[c] = 0.0
    combined[features_base] = combined[features_base].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    combined['baseline_iso'] = combined['baseline_iso'].fillna(0.0)

    # ---------- Берём только test-часть ----------
    test_feat = combined[combined[F.DATETIME] > last_train_dt].reset_index(drop=True)
    print(f"  test_feat rows = {len(test_feat)}")
    assert len(test_feat) == len(test_df), f"Несовпадение длин: {len(test_feat)} vs {len(test_df)}"

    # ---------- Постройка прогнозов ----------
    base_n = len(features_base)
    lag_dim = history_len if use_target_lag else 0
    X_direct = np.zeros((len(test_feat), base_n + lag_dim + 1), dtype=np.float32)
    feat_mat = test_feat[features_base].values.astype(np.float32)
    X_direct[:, :base_n] = feat_mat
    if use_target_lag:
        lags = np.array(last_target_history[::-1][:history_len], dtype=np.float32)
        X_direct[:, base_n:base_n + history_len] = lags

    hours_ahead = ((test_feat[F.DATETIME] - last_train_dt).dt.total_seconds() / 3600.0).values
    hours_ahead = np.round(hours_ahead).astype(np.int32)
    X_direct[:, -1] = hours_ahead.astype(np.float32)

    # Для vfar — без лагов и без hours_ahead
    X_vfar = feat_mat  # уже выровнено

    # baseline_iso по строкам
    baseline_arr = test_feat['baseline_iso'].values
    ws80 = test_feat['wind_speed_80m'].values if 'wind_speed_80m' in test_feat.columns else np.zeros(len(test_feat))

    print(f"=== Predict: near=[1..{near_horizon}], far=[{near_horizon+1}..{far_horizon}], vfar=>{far_horizon} ===")
    preds = np.zeros(len(test_feat), dtype=np.float32)
    mask_near = (hours_ahead >= 1) & (hours_ahead <= near_horizon)
    mask_far = (hours_ahead > near_horizon) & (hours_ahead <= far_horizon)
    mask_vfar = hours_ahead > far_horizon
    print(f"  near={mask_near.sum()}  far={mask_far.sum()}  vfar={mask_vfar.sum()}")

    # При no_residual / log_target / mult_residual модель предсказывает target напрямую
    eff_baseline = np.zeros_like(baseline_arr) if (is_no_residual or is_log_target or is_mult_residual) else baseline_arr
    if mask_near.any() and near_models:
        resid = ensemble_predict(near_models, X_direct[mask_near])
        preds[mask_near] = eff_baseline[mask_near] + resid
    if mask_far.any() and far_models:
        resid = ensemble_predict(far_models, X_direct[mask_far])
        preds[mask_far] = eff_baseline[mask_far] + resid
    if mask_vfar.any() and vfar_models:
        resid = ensemble_predict(vfar_models, X_vfar[mask_vfar])
        preds[mask_vfar] = eff_baseline[mask_vfar] + resid

    if is_log_target:
        preds = np.expm1(preds.clip(min=0))
        print(f"  Применено expm1 (log_target=True)")

    if is_mult_residual:
        # preds сейчас = ratio (предсказание модели). Умножаем на baseline.
        print(f"  DEBUG: preds before mult: mean={preds.mean():.3f}, baseline_arr mean={baseline_arr.mean():.3f}")
        preds = preds * baseline_arr
        print(f"  DEBUG: preds after mult: mean={preds.mean():.3f}")
        print(f"  Применено ratio*baseline (mult_residual=True)")

    # ---------- Физические капы ----------
    final = np.array([physical_caps(p, ws80[i]) for i, p in enumerate(preds)], dtype=np.float32)

    # ---------- Возвращаем порядок submission ----------
    # Исходный test_df был в обратном порядке. test_feat — в хронологии.
    # submission должен соответствовать порядку строк valid_features.csv (как было).
    test_orig = pd.read_csv(args.test, parse_dates=[F.DATETIME])
    orig_order = test_orig[F.DATETIME].values
    # Сопоставим test_feat -> original order
    dt_to_pred = dict(zip(test_feat[F.DATETIME].values.astype('datetime64[ns]'), final))
    final_orig_order = np.array([dt_to_pred[np.datetime64(t)] for t in orig_order], dtype=np.float32)

    sub_df = pd.DataFrame(final_orig_order, columns=[target_col])
    sub_df.to_csv(args.out, index=False, encoding='utf-8')
    print(f"=== Сохранено: {args.out}, строк {len(sub_df)} ===")
    print(f"  предсказания: min={final.min():.2f}, mean={final.mean():.2f}, max={final.max():.2f}")


if __name__ == '__main__':
    main()
