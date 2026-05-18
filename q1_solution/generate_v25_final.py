#!/usr/bin/env python3
"""v25: Final ensemble — integrates EVERYTHING.

Champion ladder:
  champ = FBSv18_3wM_ws80_hr13_m3_b25 (LB 7.24570)
  Phase B winner = FBSv19_FRRraw_a2_g15 (LB 7.24667)
  New partner = LGBM with --with-actuals --with-spatial (Phase D)

Strategies:
  S1. Champ × FRR × LGBM 3-way blend
  S2. Sequential: champ + small LGBM dose + small FRR-direction
  S3. v22 BIGSTACK with LGBM added as a 12th source
  S4. M3 gauss using BOTH DIV3 and LGBM at different proportions
  S5. Final safe ensemble — robust mean of top 15 winners across all phases
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
    path = ROOT / f"submission_FBSv25_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def gauss1(x, cen, sig):
    g = np.exp(-((x - cen)**2) / (2 * sig**2))
    return g / g.max()


def main():
    base, target_col = load(BASE_FILE)
    v12_base, _ = load("submission_FBSv12_MCBv12_20_TRIM_x12.csv")
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    lgbm_new, _ = load("submission_LGBM_v20_actuals_spatial.csv")
    print(f"BASE: {BASE_FILE}")
    print(f"FRR  : mean={frr.mean():.3f}")
    print(f"LGBM : mean={lgbm_new.mean():.3f}")

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

    # ====== S1: 3-way blend champ × FRR × LGBM ======
    for w_l, w_f in [(0.02, 0.10), (0.03, 0.15), (0.05, 0.20), (0.05, 0.30),
                     (0.08, 0.20), (0.05, 0.50), (0.10, 0.30), (0.03, 0.20),
                     (0.02, 0.30), (0.04, 0.40)]:
        delta_f = frr - base
        blend = (1 - w_l) * base + w_l * lgbm_new + w_f * delta_f
        outputs.append(save(
            f"3WAY_lgbm{int(w_l*100):02d}_frr{int(w_f*100):02d}",
            blend, target_col))

    # ====== S2: Sequential corrections — applying each on top of previous ======
    # champ → +LGBM dose → +FRR dose
    for w_l in (0.01, 0.02, 0.03, 0.05):
        for w_f in (0.10, 0.20, 0.30):
            step1 = (1 - w_l) * base + w_l * lgbm_new
            step2 = step1 + w_f * (frr - base)
            outputs.append(save(
                f"SEQ_l{int(w_l*100):02d}_f{int(w_f*100):02d}",
                step2, target_col))

    # ====== S3: M3 gauss with LGBM partner (3wM_m3 winner recipe but partner=LGBM) ======
    for ws_cen, hr_cen in [(8.0, 13.0), (8.0, 14.0), (8.0, 15.0), (9.0, 13.0)]:
        g_ws = gauss1(ws_hub, ws_cen, 3.0)
        g_hr = gauss1(hour, hr_cen, 5.0)
        for b in (0.10, 0.20, 0.30, 0.40, 0.50):
            w = b * g_ws * g_hr * m3_mask
            blend = (1 - w) * v12_base + w * lgbm_new
            outputs.append(save(
                f"M3lgbm_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                blend, target_col))

    # ====== S4: Mix DIV3 + LGBM in March gauss zone ======
    # Champion uses DIV3 at b=0.25. Add LGBM contribution.
    for ws_cen, hr_cen in [(8.0, 13.0)]:
        g_ws = gauss1(ws_hub, ws_cen, 3.0)
        g_hr = gauss1(hour, hr_cen, 5.0)
        for total_b in (0.25, 0.30, 0.35, 0.40):
            for lgbm_frac in (0.3, 0.5, 0.7):
                w = total_b * g_ws * g_hr * m3_mask
                mixed = lgbm_frac * lgbm_new + (1 - lgbm_frac) * externals['DIV3']
                blend = (1 - w) * v12_base + w * mixed
                outputs.append(save(
                    f"M3mix_LGBM{int(lgbm_frac*100):02d}_b{int(total_b*100):02d}",
                    blend, target_col))

    # ====== S5: Big robust ensemble of top 15 across all phases ======
    winners_with_scores = [
        ("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv", 7.24570),
        ("submission_FBSv19_FRRraw_a2_g15.csv", 7.24667),
        ("submission_FBSv18_4w_ws80_hr13_m3_b25.csv", 7.24755),
        ("submission_FBSv18_ADD_COMB_b08.csv", 7.24760),
        ("submission_FBSv18_3wM_ws80_hr15_m3_b25.csv", 7.24770),
        ("submission_FBSv18_3wM_ws90_hr13_m3_b18.csv", 7.24771),
        ("submission_FBSv18_3wM_ws80_hr13_m3_b18.csv", 7.24789),
        ("submission_FBSv18_ADD_COMB_b06.csv", 7.24805),
        ("submission_FBSv18_COMB_covCB_b18.csv", 7.24810),
        ("submission_FBSv18_COMB_v89_b18.csv", 7.24846),
        ("submission_FBSv18_ADD_COMB_b04.csv", 7.24858),
        ("submission_FBSv18_COMB_v87_b18.csv", 7.24885),
        ("submission_FBSv18_4w_ws80_hr15_m3_b25.csv", 7.24963),
        ("submission_FBSv17_COMB_WS_HR_DIV3_b12.csv", 7.24981),
        ("submission_FBSv18_3wM_ws80_hr15_m3_b18.csv", 7.24988),
    ]
    arrs = []
    scores = []
    for fn, sc in winners_with_scores:
        v, _ = load(fn)
        if v is not None:
            arrs.append(v)
            scores.append(sc)
    stk = np.asarray(arrs)
    sc_arr = np.asarray(scores)
    if len(arrs) >= 10:
        # ROBUST: trimmed mean (drop best/worst)
        trimmed = np.zeros_like(stk[0])
        for i in range(stk.shape[1]):
            col = np.sort(stk[:, i])
            trimmed[i] = col[2:-2].mean()
        outputs.append(save(f"ROBUST_TRIM2_{len(arrs)}", trimmed, target_col))

        # WLB weighted
        w = 1.0 / (sc_arr - 7.245 + 0.0005)
        w = w / w.sum()
        weighted = (stk.T @ w).reshape(-1)
        outputs.append(save(f"WLB_TIGHT_{len(arrs)}", weighted, target_col))

        # Top-3 only
        outputs.append(save("TOP3_MEAN", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save("TOP3_MEDIAN", np.median(stk[:3], axis=0), target_col))

        # Add LGBM to the stack
        stk_with_lgbm = np.vstack([stk, lgbm_new.reshape(1, -1)])
        outputs.append(save(f"BIGSTACK_with_LGBM_{len(arrs)+1}",
                            np.mean(stk_with_lgbm, axis=0), target_col))
        # Weighted with LGBM low weight (since LGBM untested at this score)
        sc_with_lgbm = np.append(sc_arr, 7.26)  # assume LGBM ~7.26 (conservative)
        w_lg = 1.0 / (sc_with_lgbm - 7.245 + 0.001)
        w_lg = w_lg / w_lg.sum()
        outputs.append(save(f"WLB_with_LGBM",
                            (stk_with_lgbm.T @ w_lg).reshape(-1), target_col))

    # ====== S6: ULTRA-CONSERVATIVE — champ + tiny LGBM in March only ======
    # Most safe: just nudge champion toward LGBM in March-only rows
    for w in (0.02, 0.04, 0.06, 0.08, 0.12, 0.15):
        nudge = m3_mask * w
        blend = (1 - nudge) * base + nudge * lgbm_new
        outputs.append(save(f"CHAMP_M3_LGBM_w{int(w*100):02d}",
                            blend, target_col))

    # ====== S7: All-month LGBM dose on champion (no march mask) ======
    for w in (0.01, 0.02, 0.03, 0.05, 0.07, 0.10):
        blend = (1 - w) * base + w * lgbm_new
        outputs.append(save(f"CHAMP_ALL_LGBM_w{int(w*100):02d}",
                            blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "lgbm_partner": "submission_LGBM_v20_actuals_spatial.csv",
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v25_final_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v25 candidates")


if __name__ == "__main__":
    main()
