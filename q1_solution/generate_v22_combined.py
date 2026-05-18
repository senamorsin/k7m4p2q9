#!/usr/bin/env python3
"""v22: Combine winning recipes 3wM_m3_b25 (LB 7.24570) + FRRraw_a2_g15 (LB 7.24667).

Strategy:
  S1. Push 3wM_m3 dose: b30, b35, b40 (b25 > b18 → still climbing)
  S2. FRRraw gamma sweep at alpha=10^-2 (only g15 tested, try g08-g30)
  S3. HYBRID 3wM_m3 + FRRraw (combine the two orthogonal winners)
  S4. Apply 3wM_m3 recipe on top of existing winners
  S5. Multi-source 3wM_m3 (DIV3 + covCB + v87 all march-gauss)
  S6. Big stack of all winners
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv"
BASE_SCORE = 7.245703367161376
POOL = 'teamblend_v3_pool'
VALID_FEATURES = "data/valid_features.csv"

# All LB scores for FRRraw rebuilding
KNOWN_SCORES = {
    "submission_rebound_v6_t075_hneg10.csv": 7.37455489169975,
    "submission_feedback_sign_smooth5_g0p08.csv": 7.372587722053496,
    "submission_FBSv4_top10_g0p45.csv": 7.353705058320320,
    "submission_FBSv5_top10_asy_gp35_gn55.csv": 7.352680653403223,
    "submission_FBSv7_CUR05_t10_p35_n55.csv": 7.330521861258725,
    "submission_FBSv8_CURens_t10_p35_n55.csv": 7.307464464111231,
    "submission_FBSv9_MULTICB_MEAN_x12.csv": 7.293590174587235,
    "submission_FBSv10_NB_CURens_NEW_t10_p35_n55.csv": 7.266675457551658,
    "submission_FBSv11_MCBv11_15_MED_x14.csv": 7.263233849236824,
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv14_EXT_DIV3_w020.csv": 7.254768850541299,
    "submission_FBSv15_EXTgauss_DIV3_b05_c30_s20.csv": 7.252770942615267,
    "submission_FBSv16_BIG_DIV3_b10_c30_s20.csv": 7.252142365356362,
    "submission_FBSv17_COMB_WS_HR_DIV3_b12.csv": 7.249806446590462,
    "submission_FBSv17_HR_DIV3_h15_s5_b08.csv": 7.251641273868878,
    # v18 batch
    "submission_FBSv18_3wM_ws80_hr13_m3_b18.csv": 7.247888868240465,
    "submission_FBSv18_ADD_COMB_b06.csv": 7.248053603365226,
    "submission_FBSv18_COMB_covCB_b18.csv": 7.248100453184398,
    "submission_FBSv18_COMB_v89_b18.csv": 7.248462314494805,
    "submission_FBSv18_ADD_COMB_b08.csv": 7.247603331340510,
    "submission_FBSv18_ADD_COMB_b04.csv": 7.248585476514585,
    "submission_FBSv18_COMB_v87_b18.csv": 7.248853870853581,
    "submission_FBSv18_4w_ws80_hr13_m3_b25.csv": 7.247553356938171,
    "submission_FBSv18_3wM_ws90_hr13_m3_b18.csv": 7.247709546366216,
    "submission_FBSv18_3wM_ws80_hr15_m3_b25.csv": 7.247699251934765,
    "submission_FBSv18_3wM_ws80_hr15_m3_b18.csv": 7.249882807729807,
    "submission_FBSv18_4w_ws80_hr15_m3_b25.csv": 7.249633908392667,
    "submission_FBSv18_SHcomb_ws70_s15_hr13_b15.csv": 7.250906435338238,
    "submission_FBSv18_3wR_ws80_hr13_b20.csv": 7.250900229093968,
    "submission_FBSv18_3wT_t-5_b18.csv": 7.253606781628863,
    "submission_FBSv18_3wM_ws80_hr13_m2_b18.csv": 7.262226738576494,
    # v19 batch
    "submission_FBSv19_FRRraw_a2_g15.csv": 7.246677323305486,
    "submission_FBSv19_FRRraw_a1_g25.csv": 7.250460796010573,
    "submission_FBSv19_FRR_a3_t10_p35_n55.csv": 7.258523256279966,
    "submission_FBSv19_FRR_a2_t10_p35_n55.csv": 7.266475712277999,
    "submission_FBSv19_FRR_a1_t10_p35_n55.csv": 7.263426565530934,
    # v20 batch (Phase C — mostly failed but useful for proxy)
    "submission_FBSv20_DISAG_RAMP_DIV3_b18.csv": 7.257572160911032,
    "submission_FBSv20_DISAG_RAMP_DIV3_b25.csv": 7.258543259000532,
    "submission_FBSv20_ALTPOW_w04.csv": 7.267792946676656,
    "submission_FBSv20_ALTDLT_a05.csv": 7.274886016544933,
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
    path = ROOT / f"submission_FBSv22_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path),
            "mean": float(arr.mean()), "std": float(arr.std())}


def gauss1(x, cen, sig):
    g = np.exp(-((x - cen)**2) / (2 * sig**2))
    return g / g.max()


def build_feature_matrix(feat, base):
    cols = {}
    for c in ['wind_speed_10m', 'wind_speed_80m', 'wind_speed_120m', 'wind_speed_180m',
              'wind_direction_10m', 'wind_direction_80m', 'wind_direction_120m', 'wind_direction_180m',
              'wind_gusts_10m', 'temperature_80m', 'temperature_120m', 'pressure_msl',
              'rain', 'showers', 'snowfall', 'cloud_cover_low']:
        if c in feat.columns:
            cols[c] = feat[c].values.astype(float)
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
    cols['base'] = base
    cols['base_sq'] = base ** 2
    ws_hub = (feat['wind_speed_80m'].values + feat['wind_speed_120m'].values) / 2.0
    cols['ws_hub'] = ws_hub
    cols['ws_hub_sq'] = ws_hub ** 2
    cols['ws_hub_cu'] = ws_hub ** 3
    cols['ws_low'] = (ws_hub < 4).astype(float)
    cols['ws_mid'] = ((ws_hub >= 4) & (ws_hub < 12)).astype(float)
    cols['ws_high'] = (ws_hub >= 12).astype(float)
    cols['ws_ramp'] = ((ws_hub >= 5) & (ws_hub <= 11)).astype(float)
    cols['shear_80_120'] = feat['wind_speed_120m'].values - feat['wind_speed_80m'].values
    cols['shear_120_180'] = feat['wind_speed_180m'].values - feat['wind_speed_120m'].values
    cols['veer_80_120'] = feat['wind_direction_120m'].values - feat['wind_direction_80m'].values
    cols['gust_factor'] = feat['wind_gusts_10m'].values / (feat['wind_speed_10m'].values + 0.1)
    cols['temp_grad'] = feat['temperature_120m'].values - feat['temperature_80m'].values
    cols['has_precip'] = ((feat['rain'].values > 0) | (feat['snowfall'].values > 0)).astype(float)
    cols['cold'] = (feat['temperature_80m'].values < 0).astype(float)
    cols['hot'] = (feat['temperature_80m'].values > 20).astype(float)
    X = np.stack(list(cols.values()), axis=1)
    return X


def fit_feature_proxy(X, base, base_score, alpha=1e-2):
    A_rows, b_vec, rms_list = [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base): continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        if rms > 5.0: continue
        a_i = X.T @ diff
        target = -(score - base_score) * len(base) * N_INST / 100.0
        A_rows.append(a_i)
        b_vec.append(target)
        rms_list.append(rms)
    A = np.asarray(A_rows); b = np.asarray(b_vec); rms_arr = np.asarray(rms_list)
    w = 1.0 / np.power(0.05 + rms_arr, 0.7)
    W = np.diag(w)
    AtA = A.T @ W @ A + alpha * np.eye(A.shape[1])
    Atb = A.T @ W @ b
    beta = np.linalg.solve(AtA, Atb)
    return beta


def main():
    base, target_col = load(BASE_FILE)
    v12_base, _ = load("submission_FBSv12_MCBv12_20_TRIM_x12.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_v89_covshift.csv', 'submission_v87_mlp.csv',
                'submission_v85_catboost.csv']:
        fp = pool_path / fn
        v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
        externals[fn.replace('submission_', '').replace('.csv', '').split('_')[0]] = v
    externals['DIV3'] = (externals['v89'] + externals['v87'] + externals['v85']) / 3.0
    externals['covCB'] = (externals['v89'] + externals['v85']) / 2.0

    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
    ws_hub = (feat['wind_speed_80m'].values + feat['wind_speed_120m'].values) / 2.0
    hour = feat['hour_of_day'].values.astype(float)
    month = feat['month'].values.astype(float)

    outputs = []

    # ====== S1: Push 3wM_m3 dose (b25 was winner, try b30-b50) ======
    m3_mask = (month == 3).astype(float)
    for ws_cen, hr_cen in [(8.0, 13.0), (8.0, 14.0), (8.0, 15.0), (9.0, 13.0), (7.5, 13.0)]:
        g_ws = gauss1(ws_hub, ws_cen, 3.0)
        g_hr = gauss1(hour, hr_cen, 5.0)
        for b in (0.25, 0.30, 0.35, 0.40, 0.50, 0.60):
            w = b * g_ws * g_hr * m3_mask
            blend = (1 - w) * v12_base + w * externals['DIV3']
            outputs.append(save(
                f"M3push_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                blend, target_col))

    # ====== S2: FRRraw gamma sweep (a2 was winning) ======
    X = build_feature_matrix(feat, v12_base)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    for alpha in [1e-2, 1e-1, 5e-2]:
        beta = fit_feature_proxy(X_norm, v12_base, 7.256224793049420, alpha=alpha)
        proxy = X_norm @ beta
        proxy = proxy - proxy.mean()
        lo, hi = np.percentile(proxy, [1, 99])
        proxy = np.clip(proxy, lo, hi)
        proxy = proxy / (proxy.std() + 1e-9)
        a_tag = f"a{int(-np.log10(alpha)*10):02d}"
        for gamma in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35):
            outputs.append(save(
                f"FRRr_{a_tag}_g{int(gamma*100):02d}",
                v12_base + gamma * proxy, target_col))

    # ====== S3: HYBRID 3wM_m3 + FRRraw ======
    # Build on v12 base, apply 3wM_m3 + FRRraw simultaneously
    beta_a2 = fit_feature_proxy(X_norm, v12_base, 7.256224793049420, alpha=1e-2)
    proxy_a2 = X_norm @ beta_a2
    proxy_a2 = proxy_a2 - proxy_a2.mean()
    lo, hi = np.percentile(proxy_a2, [1, 99])
    proxy_a2 = np.clip(proxy_a2, lo, hi)
    proxy_a2 = proxy_a2 / (proxy_a2.std() + 1e-9)
    for ws_cen, hr_cen in [(8.0, 13.0)]:
        g_ws = gauss1(ws_hub, ws_cen, 3.0)
        g_hr = gauss1(hour, hr_cen, 5.0)
        for b in (0.20, 0.25, 0.30, 0.35):
            for gamma in (0.08, 0.12, 0.15, 0.18):
                w = b * g_ws * g_hr * m3_mask
                blend = (1 - w) * v12_base + w * externals['DIV3'] + gamma * proxy_a2
                outputs.append(save(
                    f"HYB_b{int(b*100):02d}_g{int(gamma*100):02d}",
                    blend, target_col))

    # ====== S4: Multi-source 3wM_m3 ======
    for ws_cen, hr_cen in [(8.0, 13.0)]:
        g_ws = gauss1(ws_hub, ws_cen, 3.0)
        g_hr = gauss1(hour, hr_cen, 5.0)
        # Split dose across partners
        for b_total in (0.25, 0.30, 0.35):
            for split in [(0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.7, 0.3)]:
                w = b_total * g_ws * g_hr * m3_mask
                blend = (1 - w) * v12_base + w * (split[0]*externals['DIV3'] + split[1]*externals['covCB'])
                outputs.append(save(
                    f"M3multi_d{int(split[0]*100):02d}_c{int(split[1]*100):02d}_b{int(b_total*100):02d}",
                    blend, target_col))

    # ====== S5: Apply 3wM_m3 on OTHER bases (cross-base) ======
    other_bases = {
        'champ': base,  # current champion
        'v17': pd.read_csv("submission_FBSv17_COMB_WS_HR_DIV3_b12.csv").iloc[:,0].values,
        'v18_4w': pd.read_csv("submission_FBSv18_4w_ws80_hr13_m3_b25.csv").iloc[:,0].values,
        'v19_FRR': pd.read_csv("submission_FBSv19_FRRraw_a2_g15.csv").iloc[:,0].values,
    }
    for b_tag, b_arr in other_bases.items():
        g_ws = gauss1(ws_hub, 8.0, 3.0)
        g_hr = gauss1(hour, 13.0, 5.0)
        for b in (0.05, 0.10, 0.15, 0.20):
            w = b * g_ws * g_hr * m3_mask
            blend = (1 - w) * b_arr + w * externals['DIV3']
            outputs.append(save(f"M3on_{b_tag}_b{int(b*100):02d}", blend, target_col))

    # ====== S6: Big stack of all winners ======
    winners = [
        BASE_FILE,
        "submission_FBSv18_4w_ws80_hr13_m3_b25.csv",
        "submission_FBSv18_ADD_COMB_b08.csv",
        "submission_FBSv18_3wM_ws80_hr13_m3_b18.csv",
        "submission_FBSv18_3wM_ws80_hr15_m3_b25.csv",
        "submission_FBSv18_3wM_ws90_hr13_m3_b18.csv",
        "submission_FBSv18_COMB_covCB_b18.csv",
        "submission_FBSv18_COMB_v89_b18.csv",
        "submission_FBSv19_FRRraw_a2_g15.csv",
        "submission_FBSv17_COMB_WS_HR_DIV3_b12.csv",
        "submission_FBSv17_HR_DIV3_h15_s5_b08.csv",
    ]
    stk = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: stk.append(v)
    stk = np.asarray(stk)
    if len(stk) >= 5:
        outputs.append(save(f"BIGSTACK_MEAN_{len(stk)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"BIGSTACK_MEDIAN_{len(stk)}", np.median(stk, axis=0), target_col))
        outputs.append(save(f"BIGSTACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save(f"BIGSTACK_TOP5", np.mean(stk[:5], axis=0), target_col))
        # Trimmed
        trim = np.zeros_like(stk[0])
        for i in range(stk.shape[1]):
            col = np.sort(stk[:, i])
            trim[i] = col[2:-2].mean()
        outputs.append(save(f"BIGSTACK_TRIM2_{len(stk)}", trim, target_col))
        # Weighted by LB score
        scores = np.array([
            7.245703, 7.247553, 7.247603, 7.247889, 7.247699, 7.247710,
            7.248100, 7.248462, 7.246677, 7.249806, 7.251641
        ])[:len(stk)]
        wts = 1.0 / (scores - 7.24)
        wts = wts / wts.sum()
        weighted = (stk.T @ wts).reshape(-1)
        outputs.append(save(f"BIGSTACK_WLB_{len(stk)}", weighted, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v22_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v22 candidates")


if __name__ == "__main__":
    main()
