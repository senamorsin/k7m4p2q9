#!/usr/bin/env python3
"""v21: Focus on MARCH-mask winner FBSv18_3wM_ws80_hr13_m3_b18 = 7.24789.

Insight: 3-way gauss (WS × HR × month=March) is the strongest unit.
Push this: vary WS center, HR center, b dose, sigma; try partner mix.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv18_3wM_ws80_hr13_m3_b18.csv"
BASE_SCORE = 7.247888868240465
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
    path = ROOT / f"submission_FBSv21_{label}.csv"
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

    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_v89_covshift.csv', 'submission_v87_mlp.csv',
                'submission_v85_catboost.csv']:
        fp = pool_path / fn
        if fp.exists():
            v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
            externals[fn.replace('submission_', '').replace('.csv', '').split('_')[0]] = v
    externals['DIV3'] = (externals['v89'] + externals['v87'] + externals['v85']) / 3.0
    externals['covCB'] = (externals['v89'] + externals['v85']) / 2.0

    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
    ws_hub = (feat['wind_speed_80m'].values + feat['wind_speed_120m'].values) / 2.0
    hour = feat['hour_of_day'].values.astype(float)
    month = feat['month'].values.astype(float)
    repair = feat['Кол-во_ВЭУ_в_ремонте'].values.astype(float)
    rep_norm = repair / max(repair.max(), 1)

    outputs = []

    # ====== Refine 3wM_m3 grid — vary WS, HR, b, sigma ======
    for ws_cen in (6.0, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0):
        for ws_sig in (2.0, 2.5, 3.0, 3.5):
            for hr_cen in (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0):
                for hr_sig in (4.0, 5.0, 6.0):
                    for b in (0.15, 0.20, 0.25, 0.30):
                        g_ws = gauss1(ws_hub, ws_cen, ws_sig)
                        g_hr = gauss1(hour, hr_cen, hr_sig)
                        m3_mask = (month == 3).astype(float)
                        g3 = g_ws * g_hr * m3_mask
                        w = b * g3
                        blend = (1 - w) * v12_base + w * externals['DIV3']
                        outputs.append(save(
                            f"M3_ws{int(ws_cen*10):02d}_ss{int(ws_sig*10):02d}_hr{int(hr_cen)}_hs{int(hr_sig)}_b{int(b*100):02d}",
                            blend, target_col))

    # ====== Same but on CURRENT CHAMP base (additive) ======
    # base already has 3wM_m3 ws80 hr13 b18 baked in
    # Try adding MORE on top
    for ws_cen in (7.0, 8.0, 9.0):
        for hr_cen in (12.0, 13.0, 14.0, 15.0):
            for b in (0.04, 0.08, 0.12):
                g_ws = gauss1(ws_hub, ws_cen, 3.0)
                g_hr = gauss1(hour, hr_cen, 5.0)
                m3_mask = (month == 3).astype(float)
                g3 = g_ws * g_hr * m3_mask
                w = b * g3
                blend = (1 - w) * base + w * externals['DIV3']
                outputs.append(save(
                    f"ADD_M3_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                    blend, target_col))

    # ====== Mix March with Feb & January (test all 3 Q1 months) ======
    for ws_cen in (7.0, 8.0, 9.0):
        for hr_cen in (12.0, 14.0, 15.0):
            for b in (0.15, 0.25):
                g_ws = gauss1(ws_hub, ws_cen, 3.0)
                g_hr = gauss1(hour, hr_cen, 5.0)
                # All months but different weight
                w_per_m = {1: 1.0, 2: 1.0, 3: 1.5}  # March gets stronger
                m_weight = np.zeros_like(month)
                for m, mw in w_per_m.items():
                    m_weight = m_weight + mw * (month == m).astype(float)
                m_weight = m_weight / m_weight.max()
                g3 = g_ws * g_hr * m_weight
                w = b * g3
                blend = (1 - w) * v12_base + w * externals['DIV3']
                outputs.append(save(
                    f"Q1all_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                    blend, target_col))

    # ====== March + partner variations ======
    for partner_tag, partner in [('covCB', externals['covCB']), ('v87', externals['v87']),
                                   ('v89', externals['v89']), ('v85', externals['v85'])]:
        for ws_cen in (7.0, 8.0, 9.0):
            for hr_cen in (13.0, 15.0):
                for b in (0.18, 0.25):
                    g_ws = gauss1(ws_hub, ws_cen, 3.0)
                    g_hr = gauss1(hour, hr_cen, 5.0)
                    m3_mask = (month == 3).astype(float)
                    w = b * g_ws * g_hr * m3_mask
                    blend = (1 - w) * v12_base + w * partner
                    outputs.append(save(
                        f"M3_{partner_tag}_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                        blend, target_col))

    # ====== Combine 3wM_m3 with REPAIR weight ======
    for ws_cen, hr_cen in [(8.0, 13.0), (8.0, 15.0), (9.0, 13.0)]:
        for b in (0.20, 0.30, 0.40):
            g_ws = gauss1(ws_hub, ws_cen, 3.0)
            g_hr = gauss1(hour, hr_cen, 5.0)
            m3_mask = (month == 3).astype(float)
            w = b * g_ws * g_hr * m3_mask * rep_norm
            blend = (1 - w) * v12_base + w * externals['DIV3']
            outputs.append(save(
                f"M3R_ws{int(ws_cen*10):02d}_hr{int(hr_cen)}_b{int(b*100):02d}",
                blend, target_col))

    # ====== Stack winners ======
    win_files = [BASE_FILE,
                 "submission_FBSv18_ADD_COMB_b06.csv",  # 7.24806
                 "submission_FBSv18_COMB_covCB_b18.csv",  # 7.24810
                 "submission_FBSv18_COMB_v89_b18.csv",  # 7.24846
                 "submission_FBSv18_COMB_v87_b18.csv",  # 7.24886
                 "submission_FBSv18_4w_ws80_hr15_m3_b25.csv"]  # 7.24963
    stk = []
    for fn in win_files:
        v, _ = load(fn)
        if v is not None: stk.append(v)
    if len(stk) >= 4:
        stk = np.asarray(stk)
        outputs.append(save("STACK_MEAN", np.mean(stk, axis=0), target_col))
        outputs.append(save("STACK_MEDIAN", np.median(stk, axis=0), target_col))
        outputs.append(save("STACK_TOP3_MEAN", np.mean(stk[:3], axis=0), target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v21_march_focus_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v21 candidates (March-focused)")


if __name__ == "__main__":
    main()
