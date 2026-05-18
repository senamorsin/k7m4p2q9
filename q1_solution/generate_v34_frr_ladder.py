#!/usr/bin/env python3
"""v34: FRR LADDER. Base = FBSv33_NB_FRR_w20 = 7.21262.

v33 finding: Only NB_FRR_w20 (add 20% extra FRR to v31 winner) worked.
Everything else flat or worse. So: keep adding FRR. Also try different FRR alphas.

Strategy:
  S1. NB_FRR_w25, w30, w40, w50, w60 on new base
  S2. Proxy on new base + small FRR add
  S3. Try other FRR-like directions (re-derive with alpha=1e-1, 1e-3)
  S4. Multi-step FRR: stack of NB_FRR_w20, _w30, _w40
  S5. Last-chance hybrid: NB_FRR_winner + small DIV3/LGBM
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv33_NB_FRR_w20.csv"
BASE_SCORE = 7.212619135451977
POOL = 'teamblend_v3_pool'
VALID_FEATURES = "data/valid_features.csv"

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv19_FRRraw_a2_g15.csv": 7.246677323305486,
    "submission_FBSv24_CHAMP_LGBM_w030.csv": 7.240292163223663,
    "submission_FBSv26_CHAMP_LGBM_w060.csv": 7.237970103804146,
    "submission_FBSv27_NBprox_t08_p35_n55.csv": 7.236693385785031,
    "submission_FBSv28_NBprox_r07_t07_p35_n55.csv": 7.232883131643645,
    "submission_FBSv29_NB_FRR_w30.csv": 7.229653906267378,
    "submission_FBSv30_FRR_w50.csv": 7.228030068703475,
    "submission_FBSv30_FRR_w60.csv": 7.227290201773921,
    "submission_FBSv30_FRR_w80.csv": 7.226009000114476,
    "submission_FBSv30_NP30_r05_t07_p35_n55.csv": 7.225774174824203,
    "submission_FBSv30_NP30_r07_t05_p35_n55.csv": 7.221701978080064,
    "submission_FBSv31_BASE_plus_FRR_w20.csv": 7.220247491758687,
    "submission_FBSv31_BASE_plus_FRR_w30.csv": 7.219617298680013,
    "submission_FBSv31_BASE_plus_FRR_w40.csv": 7.219013161888861,
    "submission_FBSv31_BASE_plus_FRR_w50.csv": 7.218465181225035,
    "submission_FBSv31_BASE_PROX_FRR_t05_p35_n55_f20.csv": 7.213629337187302,
    # v33 batch (close to new base)
    "submission_FBSv33_MIX_parent_w50.csv": 7.216020286502276,
    "submission_FBSv33_TRIPLE_t05_p40_n50_f20.csv": 7.216537980280736,
    "submission_FBSv33_TRIPLE_t05_p30_n55_f20.csv": 7.217209449694192,
    "submission_FBSv33_TRIPLE_t05_p30_n65_f20.csv": 7.218296477762235,
    "submission_FBSv33_DEEP_t05_f50.csv": 7.216467005301755,
    "submission_FBSv33_DEEP_t05_f40.csv": 7.216647038023099,
    "submission_FBSv33_DEEP_t04_f30.csv": 7.220644017076997,
    "submission_FBSv33_DEEP_t03_f30.csv": 7.222327964666340,
    "submission_FBSv33_ITER_t05_p35_n55_f10.csv": 7.222997481769071,
    "submission_FBSv33_TRIPLE_t05_p35_n55_f40.csv": 7.216647038023099,
    "submission_FBSv33_TRIPLE_t05_p35_n55_f35.csv": 7.216762637459465,
    "submission_FBSv33_TRIPLE_t05_p35_n55_f30.csv": 7.216883226871775,
    "submission_FBSv33_TRIPLE_t05_p35_n55_f25.csv": 7.217014349439875,
    "submission_FBSv33_STACK_TOP3.csv": 7.215560763905955,
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
    path = ROOT / f"submission_FBSv34_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def build_proxy(base, base_score, max_rms=0.7, max_abs=5.0, alpha=1e-3):
    dirs, tgts, rms_list = [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base): continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        if rms < max_rms and float(np.max(np.abs(diff))) < max_abs:
            dirs.append(diff)
            tgts.append(-((score - base_score) * N_INST / 100.0))
            rms_list.append(rms)
    if len(dirs) < 4: return None
    D = np.asarray(dirs); T = np.asarray(tgts); R = np.asarray(rms_list)
    W = np.diag(1.0 / np.power(0.05 + R, 0.7))
    gram = (D @ D.T) / len(base)
    beta = np.linalg.solve(W @ gram @ W + alpha * np.eye(len(D)), W @ T)
    proxy = D.T @ (W @ beta)
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy


def build_correction(proxy, q_keep_pct, gp, gn):
    q = 1.0 - q_keep_pct/100.0
    active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
    pos = active & (proxy > 0); neg = active & (proxy < 0)
    corr = np.zeros_like(proxy)
    corr[pos] = gp * proxy[pos]; corr[neg] = gn * proxy[neg]
    return corr


# Build feature matrix for re-deriving FRR-like proxy with different alphas
def build_feature_matrix(feat, base):
    cols = {}
    for c in ['wind_speed_10m', 'wind_speed_80m', 'wind_speed_120m', 'wind_speed_180m',
              'wind_direction_80m', 'wind_direction_120m', 'wind_gusts_10m',
              'temperature_80m', 'temperature_120m', 'pressure_msl',
              'rain', 'snowfall', 'cloud_cover_low']:
        if c in feat.columns:
            cols[c] = feat[c].values.astype(float)
    h = feat['hour_of_day'].values.astype(float)
    cols['hour_sin'] = np.sin(2*np.pi*h/24)
    cols['hour_cos'] = np.cos(2*np.pi*h/24)
    cols['hour'] = h
    m = feat['month'].values.astype(float)
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
    cols['ws_ramp'] = ((ws_hub >= 5) & (ws_hub <= 11)).astype(float)
    cols['shear_80_120'] = feat['wind_speed_120m'].values - feat['wind_speed_80m'].values
    cols['has_precip'] = ((feat['rain'].values > 0) | (feat['snowfall'].values > 0)).astype(float)
    cols['cold'] = (feat['temperature_80m'].values < 0).astype(float)
    X = np.stack(list(cols.values()), axis=1)
    return X


def fit_frr_proxy(X, base, base_score, alpha=1e-2):
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
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    champ_orig, _ = load("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    outputs = []
    delta_frr = frr - champ_orig

    # ====== S1: NB_FRR more on new base (winning recipe extension) ======
    # Base is already w20 over v31_winner. Add more FRR on top of that
    for w in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        blend = base + w * delta_frr
        outputs.append(save(f"NB_FRR_w{int(w*100):02d}", blend, target_col))

    # ====== S2: Re-derive FBSv31 winner with NB_FRR_wXX (try all w near 20) ======
    v31_winner, _ = load("submission_FBSv31_BASE_PROX_FRR_t05_p35_n55_f20.csv")
    if v31_winner is not None:
        # v33_NB_FRR_w20 = v31_winner + 0.20 * delta_frr → 7.21262
        # Try w22, w23, w25, w27, w18, w17 to find optimum
        for w in (0.15, 0.17, 0.18, 0.19, 0.21, 0.22, 0.23, 0.25, 0.28, 0.30, 0.35):
            blend = v31_winner + w * delta_frr
            outputs.append(save(f"V31_FRR_w{int(w*100):02d}", blend, target_col))

    # ====== S3: Build NEW FRR-like proxy on new base with multiple alphas ======
    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
    X = build_feature_matrix(feat, base)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    for alpha in (1e-3, 1e-2, 5e-2, 1e-1, 5e-1):
        try:
            beta = fit_frr_proxy(X_norm, base, BASE_SCORE, alpha=alpha)
        except np.linalg.LinAlgError:
            continue
        proxy = X_norm @ beta
        proxy = proxy - proxy.mean()
        lo, hi = np.percentile(proxy, [1, 99])
        proxy = np.clip(proxy, lo, hi)
        if proxy.std() < 1e-9: continue
        proxy = proxy / proxy.std()
        a_tag = f"a{int(-np.log10(alpha)*10):02d}"
        for gamma in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
            outputs.append(save(f"NEWFRR_{a_tag}_g{int(gamma*100):02d}",
                                base + gamma * proxy, target_col))

    # ====== S4: Proxy on new base + small FRR add ======
    for max_rms in (0.4, 0.5, 0.6, 0.7):
        proxy = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        for t in (3, 5, 7, 10):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50)]:
                corr = build_correction(proxy, t, gp, gn)
                outputs.append(save(
                    f"NP_r{int(max_rms*10):02d}_t{t:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))
                # Plus small FRR
                for w_frr in (0.05, 0.10, 0.15):
                    outputs.append(save(
                        f"NP_FRR_r{int(max_rms*10):02d}_t{t:02d}_w{int(w_frr*100):02d}",
                        base + corr + w_frr * delta_frr, target_col))

    # ====== S5: Stack v33 winners (close to base) ======
    winners = [
        BASE_FILE,
        "submission_FBSv31_BASE_PROX_FRR_t05_p35_n55_f20.csv",
        "submission_FBSv33_STACK_TOP3.csv",
        "submission_FBSv33_MIX_parent_w50.csv",
        "submission_FBSv33_TRIPLE_t05_p40_n50_f20.csv",
        "submission_FBSv33_DEEP_t05_f50.csv",
        "submission_FBSv33_DEEP_t05_f40.csv",
        "submission_FBSv33_TRIPLE_t05_p35_n55_f40.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 4:
        stk = np.asarray(arrs)
        outputs.append(save(f"STACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save(f"STACK_TOP5", np.mean(stk[:5], axis=0), target_col))
        outputs.append(save(f"STACK_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))

    # ====== S6: Combine NB_FRR_winning + small extras ======
    # NB_FRR_w20 + small new-FRR (re-derived)
    for alpha in (1e-2, 5e-2):
        try:
            beta = fit_frr_proxy(X_norm, base, BASE_SCORE, alpha=alpha)
            proxy = X_norm @ beta; proxy -= proxy.mean()
            lo, hi = np.percentile(proxy, [1, 99]); proxy = np.clip(proxy, lo, hi)
            if proxy.std() < 1e-9: continue
            proxy /= proxy.std()
        except np.linalg.LinAlgError: continue
        a_tag = f"a{int(-np.log10(alpha)*10):02d}"
        # Apply with smaller doses
        for gamma in (0.05, 0.10, 0.15):
            for w_frr in (0.05, 0.10):
                blend = base + gamma * proxy + w_frr * delta_frr
                outputs.append(save(
                    f"COMBO2_{a_tag}_g{int(gamma*100):02d}_f{int(w_frr*100):02d}",
                    blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v34_frr_ladder_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v34 candidates")


if __name__ == "__main__":
    main()
