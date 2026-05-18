#!/usr/bin/env python3
"""Feedback-sign v18 (Phase A): MULTI-FEATURE COMBO GAUSS.

v17 COMB_WS_HR_DIV3_b12 = 7.24981 won (Δ −0.003). Two features combined > one.
v18: push to 3-way and 4-way feature combinations.
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
    path = ROOT / f"submission_FBSv18_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path),
            "mean": float(arr.mean()), "std": float(arr.std())}


def gauss1(x, cen, sig):
    g = np.exp(-((x - cen)**2) / (2 * sig**2))
    return g / g.max()


def main():
    base, target_col = load(BASE_FILE)
    v12_base, _ = load("submission_FBSv12_MCBv12_20_TRIM_x12.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    # Load externals
    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_v89_covshift.csv', 'submission_v87_mlp.csv',
                'submission_v85_catboost.csv']:
        fp = pool_path / fn
        if fp.exists():
            v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
            if len(v) == len(base):
                externals[fn.replace('submission_', '').replace('.csv', '').split('_')[0]] = v
    externals['DIV3'] = (externals['v89'] + externals['v87'] + externals['v85']) / 3.0
    externals['covCB'] = (externals['v89'] + externals['v85']) / 2.0

    # Load features
    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
    ws_hub = (feat['wind_speed_80m'].values + feat['wind_speed_120m'].values) / 2.0
    ws80 = feat['wind_speed_80m'].values
    ws120 = feat['wind_speed_120m'].values
    ws180 = feat['wind_speed_180m'].values
    hour = feat['hour_of_day'].values.astype(float)
    month = feat['month'].values.astype(float)
    temp = feat['temperature_80m'].values
    repair = feat['Кол-во_ВЭУ_в_ремонте'].values.astype(float)
    rep_norm = repair / max(repair.max(), 1)
    wdir = feat['wind_direction_80m'].values
    pressure = feat['pressure_msl'].values

    outputs = []
    partner = externals['DIV3']
    base_for_apply = v12_base  # clean base, before any DIV3 baking

    # ====== 3-WAY combinations on v12 base ======
    # WS × HR × MONTH
    for ws_cen, ws_sig in [(7.0, 2.5), (8.0, 3.0), (9.0, 3.0)]:
        for hr_cen, hr_sig in [(12.0, 5.0), (13.0, 5.0), (14.0, 5.0), (15.0, 5.0)]:
            for m_target in (1, 2, 3):
                g_ws = gauss1(ws_hub, ws_cen, ws_sig)
                g_hr = gauss1(hour, hr_cen, hr_sig)
                m_mask = (month == m_target).astype(float)
                g3 = g_ws * g_hr * m_mask
                for b in (0.08, 0.12, 0.18, 0.25):
                    w = b * g3
                    blend = (1 - w) * base_for_apply + w * partner
                    outputs.append(save(
                        f"3wM_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_m{m_target}_b{int(b*100):02d}",
                        blend, target_col))

    # WS × HR × REPAIR (more repair = more uncertainty)
    for ws_cen, ws_sig in [(8.0, 3.0)]:
        for hr_cen, hr_sig in [(13.0, 5.0), (15.0, 5.0)]:
            g_ws = gauss1(ws_hub, ws_cen, ws_sig)
            g_hr = gauss1(hour, hr_cen, hr_sig)
            for b in (0.10, 0.15, 0.20, 0.30):
                w = b * g_ws * g_hr * rep_norm
                blend = (1 - w) * base_for_apply + w * partner
                outputs.append(save(
                    f"3wR_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                    blend, target_col))

    # WS × HR × TEMP (cold = different turbine behavior)
    for ws_cen, ws_sig in [(8.0, 3.0)]:
        for hr_cen, hr_sig in [(13.0, 5.0)]:
            g_ws = gauss1(ws_hub, ws_cen, ws_sig)
            g_hr = gauss1(hour, hr_cen, hr_sig)
            for t_cen, t_sig in [(0.0, 5.0), (5.0, 5.0), (-5.0, 5.0)]:
                g_t = gauss1(temp, t_cen, t_sig)
                for b in (0.10, 0.18, 0.25):
                    w = b * g_ws * g_hr * g_t
                    blend = (1 - w) * base_for_apply + w * partner
                    outputs.append(save(
                        f"3wT_t{int(t_cen):+d}_b{int(b*100):02d}",
                        blend, target_col))

    # WS × HR with DIFFERENT partners (DIV3 alone may not be optimal)
    for tag, p in [('covCB', externals['covCB']), ('v87', externals['v87']),
                    ('v89', externals['v89']), ('v85', externals['v85'])]:
        g_ws = gauss1(ws_hub, 8.0, 3.0)
        g_hr = gauss1(hour, 13.0, 5.0)
        g2 = g_ws * g_hr
        for b in (0.08, 0.12, 0.18, 0.25):
            w = b * g2
            blend = (1 - w) * base_for_apply + w * p
            outputs.append(save(f"COMB_{tag}_b{int(b*100):02d}",
                                blend, target_col))

    # ====== 4-WAY: WS × HR × MONTH × REPAIR ======
    for ws_cen, ws_sig in [(8.0, 3.0), (9.0, 3.0)]:
        for hr_cen, hr_sig in [(13.0, 5.0), (15.0, 5.0)]:
            for m_target in (1, 2, 3):
                g_ws = gauss1(ws_hub, ws_cen, ws_sig)
                g_hr = gauss1(hour, hr_cen, hr_sig)
                m_mask = (month == m_target).astype(float)
                g4 = g_ws * g_hr * m_mask * rep_norm
                for b in (0.15, 0.25, 0.40):
                    w = b * g4
                    blend = (1 - w) * base_for_apply + w * partner
                    outputs.append(save(
                        f"4w_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_m{m_target}_b{int(b*100):02d}",
                        blend, target_col))

    # ====== Sharper WS gauss (narrow ramp-up zone) ======
    for ws_cen in (6.0, 7.0, 8.0, 9.0, 10.0):
        for ws_sig in (1.5, 2.0, 2.5):
            for hr_cen in (12.0, 13.0, 14.0, 15.0):
                g_ws = gauss1(ws_hub, ws_cen, ws_sig)
                g_hr = gauss1(hour, hr_cen, 5.0)
                g2 = g_ws * g_hr
                for b in (0.10, 0.15, 0.20):
                    w = b * g2
                    blend = (1 - w) * base_for_apply + w * partner
                    outputs.append(save(
                        f"SHcomb_ws{int(ws_cen*10):02d}_s{int(ws_sig*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                        blend, target_col))

    # ====== ADD on existing champ (instead of v12 base) ======
    g_ws = gauss1(ws_hub, 8.0, 3.0)
    g_hr = gauss1(hour, 13.0, 5.0)
    g2 = g_ws * g_hr
    for b in (0.02, 0.04, 0.06, 0.08, 0.12):
        w = b * g2
        blend = (1 - w) * base + w * partner  # base is v17 winner
        outputs.append(save(f"ADD_COMB_b{int(b*100):02d}", blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "feedback_sign_v18_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v18 candidates (Phase A: multi-feature combo)")


if __name__ == "__main__":
    main()
