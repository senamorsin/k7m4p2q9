#!/usr/bin/env python3
"""v24: New LGBM (--with-actuals --with-spatial) as orthogonal partner.

The fresh LGBM trained with --with-actuals --with-spatial flags is:
- corr=0.984 with champion (diverse, not redundant)
- |d|=3.0 from champion (meaningful displacement)
- mean=39.45 (vs champ 39.59 — slightly lower predictions)

This is comparable to teammate's diverse-model partners. Inject at varied doses.
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
LGBM_NEW = "submission_LGBM_v20_actuals_spatial.csv"
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
    path = ROOT / f"submission_FBSv24_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def gauss1(x, cen, sig):
    g = np.exp(-((x - cen)**2) / (2 * sig**2))
    return g / g.max()


def main():
    base, target_col = load(BASE_FILE)
    v12_base, _ = load("submission_FBSv12_MCBv12_20_TRIM_x12.csv")
    lgbm_new, _ = load(LGBM_NEW)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")
    print(f"LGBM partner: mean={lgbm_new.mean():.3f}, std={lgbm_new.std():.3f}")

    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_v89_covshift.csv', 'submission_v87_mlp.csv',
                'submission_v85_catboost.csv']:
        fp = pool_path / fn
        v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
        externals[fn.replace('submission_', '').replace('.csv', '').split('_')[0]] = v
    externals['DIV3'] = (externals['v89'] + externals['v87'] + externals['v85']) / 3.0
    externals['LGBM_AS'] = lgbm_new

    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
    ws_hub = (feat['wind_speed_80m'].values + feat['wind_speed_120m'].values) / 2.0
    hour = feat['hour_of_day'].values.astype(float)
    month = feat['month'].values.astype(float)

    outputs = []
    m3_mask = (month == 3).astype(float)

    # ====== S1: Uniform LGBM dose on v12 base ======
    for w in (0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.120, 0.150):
        blend = (1 - w) * v12_base + w * lgbm_new
        outputs.append(save(f"v12_LGBM_w{int(w*1000):03d}", blend, target_col))

    # ====== S2: Uniform LGBM dose on CHAMP base ======
    for w in (0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.120):
        blend = (1 - w) * base + w * lgbm_new
        outputs.append(save(f"CHAMP_LGBM_w{int(w*1000):03d}", blend, target_col))

    # ====== S3: 3wM_m3 gauss with LGBM as partner ======
    for ws_cen in (7.0, 8.0, 9.0):
        for hr_cen in (13.0, 14.0, 15.0):
            for b in (0.20, 0.30, 0.40, 0.50):
                g_ws = gauss1(ws_hub, ws_cen, 3.0)
                g_hr = gauss1(hour, hr_cen, 5.0)
                w = b * g_ws * g_hr * m3_mask
                blend = (1 - w) * v12_base + w * lgbm_new
                outputs.append(save(
                    f"M3_LGBM_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                    blend, target_col))

    # ====== S4: Multi-partner with LGBM ======
    # Mix DIV3 + LGBM at various ratios
    for ws_cen, hr_cen in [(8.0, 13.0), (8.0, 15.0)]:
        for total_b in (0.25, 0.30):
            for split_lgbm in (0.3, 0.5, 0.7):
                w = total_b * gauss1(ws_hub, ws_cen, 3.0) * gauss1(hour, hr_cen, 5.0) * m3_mask
                mixed_partner = split_lgbm * lgbm_new + (1 - split_lgbm) * externals['DIV3']
                blend = (1 - w) * v12_base + w * mixed_partner
                outputs.append(save(
                    f"M3_MIX_LGBM{int(split_lgbm*100):02d}_b{int(total_b*100):02d}_hr{int(hr_cen)}",
                    blend, target_col))

    # ====== S5: LGBM EXTgauss in power-space (like FBSv15 winning recipe but with LGBM) ======
    for b in (0.03, 0.05, 0.08, 0.12, 0.18, 0.25):
        for cen, sig in [(30.0, 20.0), (25.0, 15.0)]:
            g = gauss1(v12_base, cen, sig)
            w = b * g
            blend = (1 - w) * v12_base + w * lgbm_new
            outputs.append(save(
                f"GS_LGBM_c{int(cen)}_s{int(sig)}_b{int(b*100):02d}",
                blend, target_col))

    # ====== S6: Multi-CB with LGBM as one of the bases ======
    # Quick: just average champion and LGBM at various weights
    for w_l in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        blend = (1 - w_l) * base + w_l * lgbm_new
        outputs.append(save(f"CHAMP_LGBM_HEAVY_w{int(w_l*100):02d}", blend, target_col))

    # ====== S7: 3-way: champ + LGBM + DIV3 ======
    for w_l, w_d in [(0.05, 0.05), (0.08, 0.05), (0.10, 0.05), (0.05, 0.10),
                      (0.08, 0.08), (0.10, 0.10), (0.15, 0.05), (0.05, 0.15)]:
        w_c = 1 - w_l - w_d
        blend = w_c * base + w_l * lgbm_new + w_d * externals['DIV3']
        outputs.append(save(
            f"TRI_LGBM{int(w_l*100):02d}_DIV{int(w_d*100):02d}",
            blend, target_col))

    # ====== S8: LGBM on top of champion via M3 gauss only ======
    # Use champ as base, add small LGBM correction within March
    for ws_cen in (8.0,):
        for hr_cen in (13.0, 14.0, 15.0):
            for b in (0.10, 0.15, 0.20, 0.30):
                g_ws = gauss1(ws_hub, ws_cen, 3.0)
                g_hr = gauss1(hour, hr_cen, 5.0)
                w = b * g_ws * g_hr * m3_mask
                blend = (1 - w) * base + w * lgbm_new
                outputs.append(save(
                    f"M3on_CHAMP_LGBM_hr{int(hr_cen)}_b{int(b*100):02d}",
                    blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "lgbm_partner_file": LGBM_NEW,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v24_lgbm_partner_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v24 candidates")


if __name__ == "__main__":
    main()
