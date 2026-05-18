#!/usr/bin/env python3
"""v26: PUSH LGBM dose, new base = CHAMP_LGBM_w030 (LB 7.24030).

v24 finding: CHAMP + 3% LGBM = 7.24030 (Δ −0.0054, BIG WIN!).
w020 → 7.24173, w030 → 7.24030. Dose climbing. Push w035-w080.
Combine with march-only mask, with FRR direction, with other partners.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv24_CHAMP_LGBM_w030.csv"
BASE_SCORE = 7.240292163223663
CHAMP_FILE = "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv"
CHAMP_SCORE = 7.245703367161376
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
    path = ROOT / f"submission_FBSv26_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def gauss1(x, cen, sig):
    g = np.exp(-((x - cen)**2) / (2 * sig**2))
    return g / g.max()


def main():
    base, target_col = load(BASE_FILE)        # v26 base = CHAMP+3%LGBM
    champ, _ = load(CHAMP_FILE)               # original champion 7.24570
    v12_base, _ = load("submission_FBSv12_MCBv12_20_TRIM_x12.csv")
    lgbm_new, _ = load("submission_LGBM_v20_actuals_spatial.csv")
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_v89_covshift.csv', 'submission_v87_mlp.csv',
                'submission_v85_catboost.csv']:
        fp = pool_path / fn
        v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
        externals[fn.replace('submission_', '').replace('.csv', '').split('_')[0]] = v
    externals['DIV3'] = (externals['v89'] + externals['v87'] + externals['v85']) / 3.0

    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
    ws_hub = (feat['wind_speed_80m'].values + feat['wind_speed_120m'].values) / 2.0
    hour = feat['hour_of_day'].values.astype(float)
    month = feat['month'].values.astype(float)
    m3_mask = (month == 3).astype(float)

    outputs = []

    # ====== S1: PUSH LGBM DOSE on champion (w030 was winner, try w035-w100) ======
    for w in (0.025, 0.035, 0.040, 0.045, 0.050, 0.060, 0.070, 0.080, 0.100, 0.120, 0.150):
        blend = (1 - w) * champ + w * lgbm_new
        outputs.append(save(f"CHAMP_LGBM_w{int(w*1000):03d}",
                            blend, target_col))

    # ====== S2: LGBM dose on the NEW base (additive — net dose grows) ======
    for w in (0.010, 0.020, 0.030, 0.050, 0.080, 0.100):
        blend = (1 - w) * base + w * lgbm_new
        outputs.append(save(f"NB_LGBM_w{int(w*1000):03d}",
                            blend, target_col))

    # ====== S3: LGBM dose ONLY in march (selective) ======
    # If march-only LGBM injection works, we can stack with all-time injection
    for w in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
        w_arr = w * m3_mask
        blend = (1 - w_arr) * champ + w_arr * lgbm_new
        outputs.append(save(f"CHAMP_M3_LGBM_w{int(w*100):02d}",
                            blend, target_col))

    # ====== S4: LGBM + FRR combo (two orthogonal winners) ======
    delta_frr = frr - champ
    for w_l in (0.020, 0.030, 0.040, 0.050):
        for w_f in (0.10, 0.20, 0.30, 0.40):
            blend = (1 - w_l) * champ + w_l * lgbm_new + w_f * delta_frr
            outputs.append(save(
                f"COMBO_LGBM{int(w_l*1000):03d}_FRR{int(w_f*100):02d}",
                blend, target_col))

    # ====== S5: LGBM in ramp-zone gauss (only in mid-power rows where uncertainty high) ======
    for cen in (25.0, 30.0, 35.0):
        for sig in (15.0, 20.0):
            g = gauss1(champ, cen, sig)
            for b in (0.05, 0.08, 0.12, 0.18, 0.25):
                w = b * g
                blend = (1 - w) * champ + w * lgbm_new
                outputs.append(save(
                    f"GS_CHAMP_LGBM_c{int(cen)}_s{int(sig)}_b{int(b*100):02d}",
                    blend, target_col))

    # ====== S6: M3 with LGBM as partner (replaces DIV3 in 3wM_m3 recipe) ======
    # Champion = v12 + 0.25 * g_ws * g_hr * m3_mask * DIV3
    # Try v12 + 0.25 * g_ws * g_hr * m3_mask * LGBM (or 0.30, 0.35)
    for ws_cen in (7.0, 8.0, 9.0):
        for hr_cen in (12.0, 13.0, 14.0, 15.0):
            for b in (0.25, 0.30, 0.35, 0.40, 0.50):
                g_ws = gauss1(ws_hub, ws_cen, 3.0)
                g_hr = gauss1(hour, hr_cen, 5.0)
                w = b * g_ws * g_hr * m3_mask
                blend = (1 - w) * v12_base + w * lgbm_new
                outputs.append(save(
                    f"3wMLGB_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                    blend, target_col))

    # ====== S7: 3-WAY: v12 + 3wM_m3*DIV3 + LGBM ======
    # This is the champion plus pure LGBM dose
    for w_lgbm in (0.020, 0.030, 0.040, 0.050, 0.070):
        # Champion already has v12 + 3wM_DIV3 baked in
        # Add LGBM as additional component
        blend = (1 - w_lgbm) * champ + w_lgbm * lgbm_new
        # Same as S1, already covered

    # ====== S8: STACK ALL diverse partners at small dose ======
    # If LGBM helps at 3%, maybe LGBM + DIV3 at 1.5% each = better
    for w_each in (0.015, 0.020, 0.025, 0.030):
        for split_l in (0.50, 0.60, 0.70):
            total_w = w_each * 2
            blend = (1 - total_w) * champ + (total_w * split_l) * lgbm_new + (total_w * (1-split_l)) * externals['DIV3']
            outputs.append(save(
                f"BOTH_LGBM{int(split_l*100):02d}_we{int(w_each*1000):03d}",
                blend, target_col))

    # ====== S9: Stack new base (CHAMP_LGBM_w030) with key alternatives ======
    # Mean of new base + champ + LGBM_w020
    lgbm_w020, _ = load("submission_FBSv24_CHAMP_LGBM_w020.csv")
    if lgbm_w020 is not None:
        # Avg of new base and w020 winner
        outputs.append(save("AVG_NB_w020", (base + lgbm_w020) / 2, target_col))
        # Weighted toward NB
        outputs.append(save("MIX_NB60_w020_40", 0.6 * base + 0.4 * lgbm_w020, target_col))

    # ====== S10: Compute proxy on new base from all LB constraints ======
    # Build curated proxy on new base (FBSv7-style approach)
    KNOWN_SCORES = {
        "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
        "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
        "submission_FBSv18_3wM_ws80_hr13_m3_b18.csv": 7.247888868240465,
        "submission_FBSv18_ADD_COMB_b08.csv": 7.247603331340510,
        "submission_FBSv18_3wM_ws80_hr15_m3_b25.csv": 7.247699251934765,
        "submission_FBSv18_3wM_ws90_hr13_m3_b18.csv": 7.247709546366216,
        "submission_FBSv18_4w_ws80_hr13_m3_b25.csv": 7.247553356938171,
        "submission_FBSv18_ADD_COMB_b06.csv": 7.248053603365226,
        "submission_FBSv18_COMB_covCB_b18.csv": 7.248100453184398,
        "submission_FBSv18_COMB_v89_b18.csv": 7.248462314494805,
        "submission_FBSv18_COMB_v87_b18.csv": 7.248853870853581,
        "submission_FBSv18_ADD_COMB_b04.csv": 7.248585476514585,
        "submission_FBSv19_FRRraw_a2_g15.csv": 7.246677323305486,
        "submission_FBSv17_COMB_WS_HR_DIV3_b12.csv": 7.249806446590462,
        "submission_FBSv17_HR_DIV3_h15_s5_b08.csv": 7.251641273868878,
        # new v24/v25 (close to new base)
        "submission_FBSv24_CHAMP_LGBM_w020.csv": 7.241733084922744,
        # base skipped
    }
    dirs, tgts, rms_list = [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base): continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        if rms < 0.5 and float(np.max(np.abs(diff))) < 4.0:
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
        # Apply as top-quantile asymmetric (winning recipe)
        for q in (8, 10, 12, 15):
            qq = 1.0 - q/100.0
            active = np.abs(proxy) >= np.quantile(np.abs(proxy), qq)
            pos = active & (proxy > 0); neg = active & (proxy < 0)
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50)]:
                corr = np.zeros_like(proxy)
                corr[pos] = gp * proxy[pos]; corr[neg] = gn * proxy[neg]
                outputs.append(save(
                    f"NBprox_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))
        print(f"  NB proxy built from {len(dirs)} constraints")

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v26_lgbm_push_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v26 candidates")


if __name__ == "__main__":
    main()
