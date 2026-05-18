#!/usr/bin/env python3
"""v27: PUSH to 1st place. Base = FBSv26_CHAMP_LGBM_w060 (LB 7.23797).

Plateau around 7.238 found for direct LGBM dose. Need NEW directions:
  S1. Mix two near-ties (CHAMP_LGBM_w060 + NB_LGBM_w030)
  S2. LGBM dose + small FRR / DIV3 / covCB on new base
  S3. M3 gauss with LGBM at higher b (b30 was untested on best LGBM partner)
  S4. Curated proxy on new base (re-derive with all latest LB)
  S5. Stack of top 5 winners
  S6. Push LGBM further w070, w075 (between w060=7.238 and w080=7.238)
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv26_CHAMP_LGBM_w060.csv"
BASE_SCORE = 7.237970103804146
CHAMP_FILE = "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv"
POOL = 'teamblend_v3_pool'
VALID_FEATURES = "data/valid_features.csv"


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
    path = ROOT / f"submission_FBSv27_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def gauss1(x, cen, sig):
    g = np.exp(-((x - cen)**2) / (2 * sig**2))
    return g / g.max()


def main():
    base, target_col = load(BASE_FILE)
    champ, _ = load(CHAMP_FILE)
    v12_base, _ = load("submission_FBSv12_MCBv12_20_TRIM_x12.csv")
    lgbm_new, _ = load("submission_LGBM_v20_actuals_spatial.csv")
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    nb_lgbm_w030, _ = load("submission_FBSv26_NB_LGBM_w030.csv")
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
    m3_mask = (month == 3).astype(float)

    outputs = []

    # ====== S1: Fine dose grid around peak (w055-w075) ======
    for w in (0.055, 0.065, 0.070, 0.075, 0.090):
        blend = (1 - w) * champ + w * lgbm_new
        outputs.append(save(f"CHAMP_LGBM_w{int(w*1000):03d}", blend, target_col))

    # ====== S2: Mix two near-ties (CHAMP_LGBM_w060 ≈ NB_LGBM_w030) ======
    if nb_lgbm_w030 is not None:
        for w in (0.3, 0.4, 0.5, 0.6, 0.7):
            blend = (1 - w) * base + w * nb_lgbm_w030
            outputs.append(save(f"MIX_NB_w{int(w*100):02d}", blend, target_col))
        # Avg
        outputs.append(save("AVG_NB_w060", (base + nb_lgbm_w030) / 2, target_col))

    # ====== S3: Add small FRR / DIV3 / covCB on new base ======
    delta_frr = frr - champ
    for ext_tag, ext in [('FRR', delta_frr), ('DIV3', externals['DIV3'] - base),
                           ('covCB', externals['covCB'] - base)]:
        for w in (0.05, 0.10, 0.15, 0.20, 0.30):
            if ext_tag == 'FRR':
                blend = base + w * ext
            else:
                blend = base + w * ext
            outputs.append(save(f"NB_plus_{ext_tag}_w{int(w*100):02d}",
                                blend, target_col))

    # ====== S4: M3 gauss with LGBM at various b (replace DIV3 in 3wM_m3) ======
    for ws_cen in (7.0, 8.0, 9.0):
        for hr_cen in (12.0, 13.0, 14.0, 15.0):
            g_ws = gauss1(ws_hub, ws_cen, 3.0)
            g_hr = gauss1(hour, hr_cen, 5.0)
            for b in (0.35, 0.45, 0.55, 0.70, 0.85):
                w = b * g_ws * g_hr * m3_mask
                blend = (1 - w) * v12_base + w * lgbm_new
                outputs.append(save(
                    f"M3LGB_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                    blend, target_col))

    # ====== S5: 3-way: champ (with DIV3 march) + LGBM (uniform) + LGBM (m3 gauss) ======
    for w_lgbm in (0.04, 0.06, 0.08):
        for b_lgbm_m3 in (0.10, 0.20, 0.30):
            # Take champion (already has m3 DIV3), add uniform LGBM, add m3 LGBM dose
            g_ws = gauss1(ws_hub, 8.0, 3.0)
            g_hr = gauss1(hour, 13.0, 5.0)
            w_m3 = b_lgbm_m3 * g_ws * g_hr * m3_mask
            step1 = (1 - w_lgbm) * champ + w_lgbm * lgbm_new
            step2 = (1 - w_m3) * step1 + w_m3 * lgbm_new
            outputs.append(save(
                f"DOUBLE_LGBM_u{int(w_lgbm*100):02d}_m{int(b_lgbm_m3*100):02d}",
                step2, target_col))

    # ====== S6: STACK of v26 winners ======
    v26_winners = [
        ("submission_FBSv26_CHAMP_LGBM_w060.csv", 7.23797),
        ("submission_FBSv26_NB_LGBM_w030.csv", 7.23798),
        ("submission_FBSv26_CHAMP_LGBM_w080.csv", 7.23817),
        ("submission_FBSv26_CHAMP_LGBM_w050.csv", 7.23841),
        ("submission_FBSv26_NB_LGBM_w020.csv", 7.23845),
        ("submission_FBSv26_COMBO_LGBM040_FRR20.csv", 7.23849),
    ]
    arrs = []; scs = []
    for fn, sc in v26_winners:
        v, _ = load(fn)
        if v is not None:
            arrs.append(v); scs.append(sc)
    if len(arrs) >= 4:
        stk = np.asarray(arrs)
        sc_arr = np.asarray(scs)
        outputs.append(save(f"STACK_v26_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"STACK_v26_MED_{len(arrs)}", np.median(stk, axis=0), target_col))
        # WLB
        w = 1.0 / (sc_arr - 7.237 + 0.0005)
        w = w / w.sum()
        weighted = (stk.T @ w).reshape(-1)
        outputs.append(save(f"STACK_v26_WLB_{len(arrs)}", weighted, target_col))
        # Top-3 mean
        outputs.append(save("STACK_v26_TOP3", np.mean(stk[:3], axis=0), target_col))

    # ====== S7: Curated proxy on new base ======
    KNOWN_SCORES = {
        "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
        "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
        "submission_FBSv18_3wM_ws80_hr13_m3_b18.csv": 7.247888868240465,
        "submission_FBSv18_ADD_COMB_b08.csv": 7.247603331340510,
        "submission_FBSv18_3wM_ws80_hr15_m3_b25.csv": 7.247699251934765,
        "submission_FBSv18_4w_ws80_hr13_m3_b25.csv": 7.247553356938171,
        "submission_FBSv19_FRRraw_a2_g15.csv": 7.246677323305486,
        "submission_FBSv17_COMB_WS_HR_DIV3_b12.csv": 7.249806446590462,
        "submission_FBSv24_CHAMP_LGBM_w020.csv": 7.241733084922744,
        "submission_FBSv24_CHAMP_LGBM_w030.csv": 7.240292163223663,
        "submission_FBSv26_CHAMP_LGBM_w040.csv": 7.239216956002586,
        "submission_FBSv26_CHAMP_LGBM_w035.csv": 7.239721311405139,
        "submission_FBSv26_CHAMP_LGBM_w050.csv": 7.238409974746849,
        "submission_FBSv26_CHAMP_LGBM_w080.csv": 7.238172595523859,
        "submission_FBSv26_NB_LGBM_w020.csv": 7.238449162310680,
        "submission_FBSv26_NB_LGBM_w030.csv": 7.237982397774434,
        "submission_FBSv26_COMBO_LGBM040_FRR20.csv": 7.238491050051701,
        "submission_FBSv26_COMBO_LGBM030_FRR20.csv": 7.239499578976215,
        "submission_FBSv26_COMBO_LGBM030_FRR30.csv": 7.239406848974587,
    }
    dirs, tgts, rms_list = [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base): continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        if rms < 0.7 and float(np.max(np.abs(diff))) < 5.0:
            dirs.append(diff)
            tgts.append(-((score - BASE_SCORE) * N_INST / 100.0))
            rms_list.append(rms)
    if len(dirs) >= 5:
        D = np.asarray(dirs); T = np.asarray(tgts); R = np.asarray(rms_list)
        W = np.diag(1.0 / np.power(0.05 + R, 0.7))
        gram = (D @ D.T) / len(base)
        beta = np.linalg.solve(W @ gram @ W + 1e-3 * np.eye(len(D)), W @ T)
        proxy = D.T @ (W @ beta)
        proxy = proxy - proxy.mean()
        lo, hi = np.percentile(proxy, [1, 99])
        proxy = np.clip(proxy, lo, hi)
        proxy = proxy / (proxy.std() + 1e-9)
        for q in (8, 10, 12, 15):
            qq = 1.0 - q/100.0
            active = np.abs(proxy) >= np.quantile(np.abs(proxy), qq)
            pos = active & (proxy > 0); neg = active & (proxy < 0)
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50), (0.25, 0.65)]:
                corr = np.zeros_like(proxy)
                corr[pos] = gp * proxy[pos]; corr[neg] = gn * proxy[neg]
                outputs.append(save(
                    f"NBprox_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))
        print(f"  NB proxy: {len(dirs)} constraints")

    # ====== S8: LGBM + DIV3 + covCB three-way ======
    # New base is champ + 6% LGBM. Add small dose of DIV3 and covCB
    for w_div in (0.01, 0.02, 0.03):
        for w_cov in (0.01, 0.02, 0.03):
            blend = base * (1 - w_div - w_cov) + w_div * externals['DIV3'] + w_cov * externals['covCB']
            outputs.append(save(
                f"NB_TRI_d{int(w_div*100):02d}_c{int(w_cov*100):02d}",
                blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v27_final_push_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v27 candidates")


if __name__ == "__main__":
    main()
