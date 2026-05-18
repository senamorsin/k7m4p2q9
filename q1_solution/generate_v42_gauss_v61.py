#!/usr/bin/env python3
"""v42: Feature-aware GAUSS injection of v61 (like v18 recipe but with v61).

v18 recipe: 0.25 * gauss(ws=8, hr=13, m=March) * DIV3
That gave Δ-0.005 jump. Apply same to v61 (which is now winning partner).

Base = FBSv40_v61_w050 = 7.20640
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv40_v61_w050.csv"
BASE_SCORE = 7.206400653558756
PARENT_FILE = "submission_FBSv35_NB_FRR_w20.csv"
PARENT_SCORE = 7.211396180429282
V18_BASE = "submission_FBSv12_MCBv12_20_TRIM_x12.csv"  # the v18 actually uses v12 base
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
    path = ROOT / f"submission_FBSv42_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def gauss1(x, cen, sig):
    g = np.exp(-((x - cen)**2) / (2 * sig**2))
    return g / g.max()


def main():
    base, target_col = load(BASE_FILE)
    parent, _ = load(PARENT_FILE)
    v61, _ = load("submission_v61_alone.csv")
    v18, _ = load(V18_BASE)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_v89_covshift.csv', 'submission_v87_mlp.csv',
                'submission_v85_catboost.csv', 'submission_TOP6_MEDIAN.csv',
                'submission_v71_sea_defaultmix_w25_dose35.csv']:
        fp = pool_path / fn
        v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
        externals[fn.replace('submission_', '').replace('.csv', '').split('_')[0]] = v
    if all(k in externals for k in ('v89', 'v87', 'v85')):
        externals['DIV3'] = (externals['v89'] + externals['v87'] + externals['v85']) / 3.0

    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
    ws_hub = (feat['wind_speed_80m'].values + feat['wind_speed_120m'].values) / 2.0
    hour = feat['hour_of_day'].values.astype(float)
    month = feat['month'].values.astype(float)
    m3_mask = (month == 3).astype(float)

    outputs = []

    # ====== S1: Feature-aware v61 gauss (like v18 but with v61) ======
    # Apply v61 only in (ws=8, hr=13, March) zone
    g_ws = gauss1(ws_hub, 8.0, 3.0)
    g_hr = gauss1(hour, 13.0, 5.0)
    g3 = g_ws * g_hr * m3_mask
    for b in (0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00):
        w = b * g3
        blend = (1 - w) * parent + w * v61
        outputs.append(save(f"M3v61_ws80_hr13_b{int(b*100):02d}",
                            blend, target_col))

    # ====== S2: Different gauss zones ======
    for ws_cen in (6.0, 7.0, 8.0, 9.0, 10.0):
        for hr_cen in (10.0, 12.0, 13.0, 14.0, 16.0):
            for b in (0.30, 0.50, 0.70):
                gw = gauss1(ws_hub, ws_cen, 3.0)
                gh = gauss1(hour, hr_cen, 5.0)
                w = b * gw * gh * m3_mask
                blend = (1 - w) * parent + w * v61
                outputs.append(save(
                    f"M3v61_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                    blend, target_col))

    # ====== S3: Q1 all months, gauss (without march-only constraint) ======
    for ws_cen in (8.0, 9.0):
        for hr_cen in (13.0, 15.0):
            gw = gauss1(ws_hub, ws_cen, 3.0)
            gh = gauss1(hour, hr_cen, 5.0)
            for b in (0.10, 0.20, 0.30, 0.40):
                w = b * gw * gh
                blend = (1 - w) * parent + w * v61
                outputs.append(save(
                    f"Q1v61_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                    blend, target_col))

    # ====== S4: v61 gauss applied on CURRENT BASE (additive layer) ======
    # base = parent + 5% v61. Add gauss-weighted MORE v61 on top
    for ws_cen in (8.0, 9.0):
        for hr_cen in (13.0, 15.0):
            gw = gauss1(ws_hub, ws_cen, 3.0)
            gh = gauss1(hour, hr_cen, 5.0)
            g3l = gw * gh * m3_mask
            for b in (0.10, 0.20, 0.30):
                w = b * g3l
                blend = (1 - w) * base + w * v61
                outputs.append(save(
                    f"NBv61_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                    blend, target_col))

    # ====== S5: BIG v61 dose only in extreme regime (ramp + March only) ======
    # Pure ramp zone (5-11 m/s) in March
    ramp_march = ((ws_hub >= 5) & (ws_hub <= 11)).astype(float) * m3_mask
    for b in (0.05, 0.10, 0.15, 0.20, 0.30):
        w = b * ramp_march
        blend = (1 - w) * parent + w * v61
        outputs.append(save(f"RAMPv61_b{int(b*100):02d}", blend, target_col))

    # ====== S6: Combined gauss: v61 in march × m3 mask, DIV3 in other ======
    # Champion v18 logic but extended
    g_v61_zone = g_ws * g_hr * m3_mask
    g_div3_zone = g_ws * g_hr * (1 - m3_mask)  # non-march
    for b_v61, b_div3 in [(0.30, 0.10), (0.50, 0.20), (0.20, 0.20), (0.40, 0.15),
                            (0.30, 0.20), (0.50, 0.10), (0.60, 0.15)]:
        w_v61 = b_v61 * g_v61_zone
        w_div3 = b_div3 * g_div3_zone
        blend = (1 - w_v61 - w_div3) * parent + w_v61 * v61 + w_div3 * externals['DIV3']
        outputs.append(save(
            f"DUALzone_v{int(b_v61*100):02d}_d{int(b_div3*100):02d}",
            blend, target_col))

    # ====== S7: v61 dose by wind-speed sector (low/mid/high split) ======
    # Apply v61 differently per ws sector
    ws_low = (ws_hub < 5).astype(float)
    ws_mid = ((ws_hub >= 5) & (ws_hub <= 11)).astype(float)
    ws_high = (ws_hub > 11).astype(float)
    for b_low, b_mid, b_high in [
        (0.00, 0.10, 0.00),
        (0.00, 0.15, 0.00),
        (0.05, 0.15, 0.00),
        (0.00, 0.20, 0.05),
        (0.05, 0.20, 0.05),
        (0.10, 0.20, 0.05),
        (0.00, 0.30, 0.00),
    ]:
        w = b_low * ws_low + b_mid * ws_mid + b_high * ws_high
        blend = (1 - w) * parent + w * v61
        outputs.append(save(
            f"SECTOR_l{int(b_low*100):02d}_m{int(b_mid*100):02d}_h{int(b_high*100):02d}",
            blend, target_col))

    # ====== S8: Apply v61 dose ONLY on rows where v61 is LOWER than parent (correct down) ======
    diff_v61 = v61 - parent  # negative = v61 thinks LOWER
    # On rows where v61<parent, dose 5-10%, otherwise 0
    for b in (0.05, 0.10, 0.15, 0.20, 0.30):
        mask = (diff_v61 < 0).astype(float)
        w = b * mask
        blend = (1 - w) * parent + w * v61
        outputs.append(save(f"v61dn_b{int(b*100):02d}", blend, target_col))
    # Where v61>parent (correct up)
    for b in (0.05, 0.10, 0.15, 0.20):
        mask = (diff_v61 > 0).astype(float)
        w = b * mask
        blend = (1 - w) * parent + w * v61
        outputs.append(save(f"v61up_b{int(b*100):02d}", blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v42_gauss_v61_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v42 candidates")


if __name__ == "__main__":
    main()
