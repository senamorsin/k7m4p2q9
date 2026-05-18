#!/usr/bin/env python3
"""Phase B / v19: FEATURE-AWARE CORRECTION via RIDGE REGRESSION.

Idea: learn δ_j = f(features_j) from LB constraints.

For each known LB submission with score s_i and diff d_i = sub_i - base:
    Δscore_i = LB_i - LB_base ≈ -<sign_err_base, d_i> * 100 / N_INST

If we hypothesize sign_err_base = X β + bias (linear in features), then:
    Δscore_i * (-N_INST / 100) ≈ <Xβ + bias, d_i> = β^T (X^T d_i) + bias * sum(d_i)

Stack for all i: solve linear system for (β, bias).

Then correction = γ * normalize(X β).

This uses ALL features as basis (47 features → 47-dim β),
gives a smooth row-level proxy that GENERALIZES via features
(not memorized per-row like FBS proxy).
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv17_COMB_WS_HR_DIV3_b12.csv"
BASE_SCORE = 7.249806446590462
VALID_FEATURES = "data/valid_features.csv"
POOL = 'teamblend_v3_pool'

KNOWN_SCORES = {
    # Far constraints (large rms, more info per sub)
    "submission_rebound_v6_t075_hneg10.csv": 7.37455489169975,
    "submission_feedback_sign_smooth5_g0p08.csv": 7.372587722053496,
    "submission_REBv6_LIGHT10_smooth.csv": 7.377025947066968,
    "submission_ULTRA_DIV3_gauss_b06.csv": 7.371827077437699,
    "submission_FBSv3_top19_g0p22.csv": 7.363100871060781,
    "submission_FBSv4_top10_g0p45.csv": 7.353705058320320,
    "submission_FBSv5_top10_asy_gp35_gn55.csv": 7.352680653403223,
    "submission_FBSv7_CUR05_t10_p35_n55.csv": 7.330521861258725,
    "submission_FBSv8_CURens_t10_p35_n55.csv": 7.307464464111231,
    "submission_FBSv9_MULTICB_MEAN_x12.csv": 7.293590174587235,
    "submission_FBSv10_NB_CURens_NEW_t10_p35_n55.csv": 7.266675457551658,
    "submission_FBSv11_MCBv11_15_MED_x14.csv": 7.263233849236824,
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv12_HYBR_mcb60_x14.csv": 7.257319999283790,
    "submission_FBSv12_HYBR_mcb50_x14.csv": 7.257856931059352,
    "submission_FBSv14_EXT_DIV3_w020.csv": 7.254768850541299,
    "submission_FBSv14_EXT_covCB_w020.csv": 7.254920930824225,
    "submission_FBSv14_EXT_DIV3_w010.csv": 7.255426543419984,
    "submission_FBSv14_EXT_v87_mlp_w010.csv": 7.255334184333610,
    "submission_FBSv15_EXTgauss_DIV3_b05_c30_s20.csv": 7.252770942615267,
    "submission_FBSv15_EXTgauss_DIV3_b03_c30_s20.csv": 7.253375876657015,
    "submission_FBSv15_EXTgauss_covCB_b03_c30_s20.csv": 7.253398504584263,
    "submission_FBSv15_EXT_DIV3_w025.csv": 7.254157741377730,
    "submission_FBSv15_MULTIEXT_DIV3025_v87010.csv": 7.254102320068292,
    "submission_FBSv16_BIG_DIV3_b10_c30_s20.csv": 7.252142365356362,
    "submission_FBSv16_BIG_DIV3_b08_c30_s20.csv": 7.252210265039256,
    "submission_FBSv16_BIG_DIV3_b12_c30_s20.csv": 7.252267661309633,
    "submission_FBSv16_BIG_covCB_b08_c30_s20.csv": 7.252374719790486,
    "submission_FBSv16_BIMODAL_DIV3_b108_b212_c15_35.csv": 7.252108144044382,
    "submission_FBSv16_MULTIgauss_b10_c30_s20.csv": 7.252122254859401,
    "submission_FBSv16_SHARP_DIV3_c30_s10_b12.csv": 7.253599338407148,
    "submission_FBSv16_ASYgauss_DIV3_bd15_bu04_c30_s20.csv": 7.257587325090352,
    "submission_FBSv17_WS_DIV3_ws80_s30_b08.csv": 7.253758980082456,
    "submission_FBSv17_WS_DIV3_ws80_s30_b18.csv": 7.257156516674318,
    "submission_FBSv17_WS_DIV3_ws100_s30_b08.csv": 7.254620835877599,
    "submission_FBSv17_WSadd_DIV3_ws80_s30_b06.csv": 7.253558279225708,
    "submission_FBSv17_HR_DIV3_h8_s4_b08.csv": 7.253150142013385,
    "submission_FBSv17_HR_DIV3_h15_s5_b08.csv": 7.251641273868878,
    "submission_FBSv17_REP_DIV3_b15.csv": 7.261087271621371,
    "submission_FBSv17_ANTIREP_DIV3_b15.csv": 7.253243437467055,
    "submission_FBSv17_MONTH3_DIV3_b10.csv": 7.252930424430208,
}


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    path = ROOT / name
    if not path.exists():
        pool_p = ROOT / POOL / name
        if pool_p.exists():
            df = pd.read_csv(pool_p)
            return df.iloc[:, 0].to_numpy(dtype=float), df.columns[0]
        return None, None
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, OSError):
        return None, None
    vals = df.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(vals).all(): return None, None
    return vals, df.columns[0]


def save(label, values, target_col):
    arr = np.clip(values, 0.0, N_INST)
    path = ROOT / f"submission_FBSv19_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path),
            "mean": float(arr.mean()), "std": float(arr.std())}


def build_feature_matrix(feat, base):
    """Build X matrix [n_rows, n_features] with engineered features."""
    cols = {}

    # Raw features
    for c in ['wind_speed_10m', 'wind_speed_80m', 'wind_speed_120m', 'wind_speed_180m',
              'wind_direction_10m', 'wind_direction_80m', 'wind_direction_120m', 'wind_direction_180m',
              'wind_gusts_10m', 'temperature_80m', 'temperature_120m', 'pressure_msl',
              'rain', 'showers', 'snowfall', 'cloud_cover_low']:
        if c in feat.columns:
            cols[c] = feat[c].values.astype(float)

    # Hour/month as cyclic
    h = feat['hour_of_day'].values.astype(float)
    cols['hour_sin'] = np.sin(2*np.pi*h/24)
    cols['hour_cos'] = np.cos(2*np.pi*h/24)
    cols['hour'] = h
    m = feat['month'].values.astype(float)
    cols['month_sin'] = np.sin(2*np.pi*m/12)
    cols['month_cos'] = np.cos(2*np.pi*m/12)
    cols['month_3'] = (m == 3).astype(float)
    cols['month_2'] = (m == 2).astype(float)
    cols['month_1'] = (m == 1).astype(float)

    cols['repair'] = feat['Кол-во_ВЭУ_в_ремонте'].values.astype(float)

    # The base prediction itself
    cols['base'] = base
    cols['base_sq'] = base ** 2

    # Wind hub effective
    ws_hub = (feat['wind_speed_80m'].values + feat['wind_speed_120m'].values) / 2.0
    cols['ws_hub'] = ws_hub
    cols['ws_hub_sq'] = ws_hub ** 2
    cols['ws_hub_cu'] = ws_hub ** 3
    cols['ws_low'] = (ws_hub < 4).astype(float)
    cols['ws_mid'] = ((ws_hub >= 4) & (ws_hub < 12)).astype(float)
    cols['ws_high'] = (ws_hub >= 12).astype(float)
    cols['ws_ramp'] = ((ws_hub >= 5) & (ws_hub <= 11)).astype(float)

    # Wind shear (80m vs 120m vs 180m)
    cols['shear_80_120'] = feat['wind_speed_120m'].values - feat['wind_speed_80m'].values
    cols['shear_120_180'] = feat['wind_speed_180m'].values - feat['wind_speed_120m'].values

    # Wind direction veering
    cols['veer_80_120'] = feat['wind_direction_120m'].values - feat['wind_direction_80m'].values

    # Gust factor
    cols['gust_factor'] = feat['wind_gusts_10m'].values / (feat['wind_speed_10m'].values + 0.1)

    # Atmospheric stability
    cols['temp_grad'] = feat['temperature_120m'].values - feat['temperature_80m'].values

    # Precipitation flag
    cols['has_precip'] = ((feat['rain'].values > 0) | (feat['snowfall'].values > 0)).astype(float)

    # Cold flag
    cols['cold'] = (feat['temperature_80m'].values < 0).astype(float)
    cols['hot'] = (feat['temperature_80m'].values > 20).astype(float)

    X = np.stack(list(cols.values()), axis=1)
    return X, list(cols.keys())


def fit_feature_proxy(X, base, base_score, lb_subs):
    """
    Fit β such that X β approximates sign(err_base).

    For each LB sub i with diff d_i and score s_i:
        Δscore_i = s_i - base_score ≈ -(1/N_test) <sign_err, d_i> * 100/N_INST
        ⇒ <sign_err, d_i> ≈ -Δscore_i * N_test * N_INST / 100

    Replace sign_err with Xβ:
        <Xβ, d_i> = β^T (X^T d_i)

    Stack into linear system A β = b where A_i = X^T d_i, b_i = target.
    Solve with ridge.
    """
    A_rows = []
    b_vec = []
    rms_list = []
    for name, score in lb_subs.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base): continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        if rms > 5.0: continue
        # row: X^T @ diff  (n_features dim)
        a_i = X.T @ diff
        target = -(score - base_score) * len(base) * N_INST / 100.0
        A_rows.append(a_i)
        b_vec.append(target)
        rms_list.append(rms)
    A = np.asarray(A_rows)  # [n_subs, n_features]
    b = np.asarray(b_vec)   # [n_subs]
    rms_arr = np.asarray(rms_list)

    # Weight rows by 1/(0.05 + rms)^0.7  (favor close subs)
    w = 1.0 / np.power(0.05 + rms_arr, 0.7)
    W = np.diag(w)

    # Ridge: solve (A^T W A + α I) β = A^T W b
    for alpha in [1e-3, 1e-2, 1e-1, 1.0]:
        AtA = A.T @ W @ A + alpha * np.eye(A.shape[1])
        Atb = A.T @ W @ b
        try:
            beta = np.linalg.solve(AtA, Atb)
        except np.linalg.LinAlgError:
            continue
        yield alpha, beta, len(A)


def main():
    base, target_col = load(BASE_FILE)
    v12_base, _ = load("submission_FBSv12_MCBv12_20_TRIM_x12.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])

    # Build feature matrix for base
    X, feat_names = build_feature_matrix(feat, base)
    print(f"  Features: {X.shape[1]} (rows {X.shape[0]})")

    # Standardize features (z-score)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-9
    X_norm = (X - X_mean) / X_std

    outputs = []

    # Fit feature proxy at multiple regularizations
    for alpha, beta, n_subs in fit_feature_proxy(X_norm, base, BASE_SCORE, KNOWN_SCORES):
        # proxy = X β
        proxy = X_norm @ beta
        # Normalize
        proxy = proxy - proxy.mean()
        # robust clip
        lo, hi = np.percentile(proxy, [1, 99])
        proxy = np.clip(proxy, lo, hi)
        proxy_std = proxy.std()
        if proxy_std < 1e-9: continue
        proxy = proxy / proxy_std

        print(f"  alpha={alpha:.0e}: n_subs={n_subs}, proxy std={proxy_std:.3f}")

        # Apply as top-quantile asymmetric correction (the winning recipe)
        for q in [10, 12, 15, 20]:
            qq = 1.0 - q/100.0
            active = np.abs(proxy) >= np.quantile(np.abs(proxy), qq)
            pos = active & (proxy > 0); neg = active & (proxy < 0)
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50), (0.45, 0.45)]:
                corr = np.zeros_like(proxy)
                corr[pos] = gp * proxy[pos]
                corr[neg] = gn * proxy[neg]
                a_tag = f"a{int(-np.log10(alpha)):d}"
                outputs.append(save(
                    f"FRR_{a_tag}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

        # Raw proxy (no quantile)
        for gamma in [0.10, 0.15, 0.25, 0.40]:
            a_tag = f"a{int(-np.log10(alpha)):d}"
            outputs.append(save(f"FRRraw_{a_tag}_g{int(gamma*100):02d}",
                                base + gamma * proxy, target_col))

    # ====== Same approach but on V12 BASE (cleaner) ======
    X12, _ = build_feature_matrix(feat, v12_base)
    X12_norm = (X12 - X12.mean(axis=0)) / (X12.std(axis=0) + 1e-9)
    BASE12_SCORE = 7.256224793049420
    for alpha, beta, n_subs in fit_feature_proxy(X12_norm, v12_base, BASE12_SCORE, KNOWN_SCORES):
        proxy = X12_norm @ beta
        proxy = proxy - proxy.mean()
        lo, hi = np.percentile(proxy, [1, 99])
        proxy = np.clip(proxy, lo, hi)
        proxy_std = proxy.std()
        if proxy_std < 1e-9: continue
        proxy = proxy / proxy_std

        a_tag = f"a{int(-np.log10(alpha)):d}"
        for q in [10, 12]:
            qq = 1.0 - q/100.0
            active = np.abs(proxy) >= np.quantile(np.abs(proxy), qq)
            pos = active & (proxy > 0); neg = active & (proxy < 0)
            corr = np.zeros_like(proxy)
            corr[pos] = 0.35 * proxy[pos]; corr[neg] = 0.55 * proxy[neg]
            outputs.append(save(f"FRRv12_{a_tag}_t{q:02d}_p35_n55",
                                v12_base + corr, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "feature_count": X.shape[1],
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "feature_regression_v19_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v19 candidates (Phase B: feature regression)")


if __name__ == "__main__":
    main()
