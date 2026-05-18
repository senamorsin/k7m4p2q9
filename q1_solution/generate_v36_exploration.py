#!/usr/bin/env python3
"""v36: Exploration — TOP6_MEDIAN inject + smoothing + multi-partner blends.

While user is away & can't upload (and LGBM_log_target trains in bg).

Base = FBSv34_NB_FRR_w30 = 7.21168 (current champion).

Ideas:
  S1. TOP6_MEDIAN micro-doses (teammate's distilled diverse partner)
  S2. Time-smoothing on champion (window 3, 5)
  S3. Multi-EXT (TOP6 + DIV3 + covCB at tiny each)
  S4. v71 (teammate's manual-tuned) at tiny dose
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv34_NB_FRR_w30.csv"
BASE_SCORE = 7.211678029707256
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
    path = ROOT / f"submission_FBSv36_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def main():
    base, target_col = load(BASE_FILE)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_v89_covshift.csv', 'submission_v87_mlp.csv',
                'submission_v85_catboost.csv', 'submission_TOP6_MEDIAN.csv',
                'submission_v77_capped.csv',
                'submission_v71_sea_defaultmix_w25_dose35.csv']:
        fp = pool_path / fn
        if fp.exists():
            v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
            if len(v) == len(base):
                key = fn.replace('submission_', '').replace('.csv', '').split('_')[0]
                externals[key] = v
    if all(k in externals for k in ('v89', 'v87', 'v85')):
        externals['DIV3'] = (externals['v89'] + externals['v87'] + externals['v85']) / 3.0

    outputs = []

    # ====== S1: TOP6_MEDIAN micro-dose (NEW partner — never tested at this stage) ======
    if 'TOP6' in externals:
        top6 = externals['TOP6']
        for w in (0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080, 0.100):
            blend = (1 - w) * base + w * top6
            outputs.append(save(f"TOP6_w{int(w*1000):03d}", blend, target_col))

    # ====== S2: v71 micro-dose ======
    if 'v71' in externals:
        v71 = externals['v71']
        for w in (0.005, 0.010, 0.020, 0.030, 0.050):
            blend = (1 - w) * base + w * v71
            outputs.append(save(f"v71_w{int(w*1000):03d}", blend, target_col))

    # ====== S3: v77 micro-dose ======
    if 'v77' in externals:
        v77 = externals['v77']
        for w in (0.005, 0.010, 0.020, 0.030):
            blend = (1 - w) * base + w * v77
            outputs.append(save(f"v77_w{int(w*1000):03d}", blend, target_col))

    # ====== S4: Multi-EXT (combine TOP6 + DIV3 + v71 at small doses) ======
    if all(k in externals for k in ('TOP6', 'DIV3', 'v71')):
        for w_top, w_div, w_v71 in [
            (0.01, 0.01, 0.005),
            (0.015, 0.01, 0.005),
            (0.02, 0.01, 0.005),
            (0.01, 0.02, 0.005),
            (0.02, 0.02, 0.01),
            (0.03, 0.01, 0.005),
        ]:
            total = w_top + w_div + w_v71
            blend = (1 - total) * base + w_top * externals['TOP6'] + w_div * externals['DIV3'] + w_v71 * externals['v71']
            outputs.append(save(
                f"MULTIEXT_t{int(w_top*1000):03d}_d{int(w_div*1000):03d}_v{int(w_v71*1000):03d}",
                blend, target_col))

    # ====== S5: Time-smoothing on champion (rolling mean) ======
    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
    if len(feat) == len(base):
        ts = pd.Series(base, index=feat['METEOFORECASTHOUR_OPENM_Datetime']).sort_index()
        for win in (3, 5, 7):
            smooth_full = ts.rolling(window=win, center=True, min_periods=1).mean()
            smooth = smooth_full.reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)
            # Mix with champion at various weights
            for w_smooth in (0.10, 0.20, 0.30, 0.50):
                blend = (1 - w_smooth) * base + w_smooth * smooth
                outputs.append(save(f"SMOOTH{win}_w{int(w_smooth*100):02d}",
                                    blend, target_col))

    # ====== S6: Median3/Median5 smoothing ======
    if len(feat) == len(base):
        ts = pd.Series(base, index=feat['METEOFORECASTHOUR_OPENM_Datetime']).sort_index()
        for win in (3, 5):
            med = ts.rolling(window=win, center=True, min_periods=1).median()
            med = med.reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)
            for w in (0.10, 0.20, 0.30):
                blend = (1 - w) * base + w * med
                outputs.append(save(f"MED{win}_w{int(w*100):02d}",
                                    blend, target_col))

    # ====== S7: Champion recipe (0.25 m3 + 0.30 m5 + 0.45 mean3) — from MORNING_REPORT ======
    if len(feat) == len(base):
        ts = pd.Series(base, index=feat['METEOFORECASTHOUR_OPENM_Datetime']).sort_index()
        m3 = ts.rolling(3, center=True, min_periods=1).median().reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)
        m5 = ts.rolling(5, center=True, min_periods=1).median().reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)
        mn3 = ts.rolling(3, center=True, min_periods=1).mean().reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)
        recipe = 0.25 * m3 + 0.30 * m5 + 0.45 * mn3
        for w in (0.05, 0.10, 0.20, 0.30, 0.50):
            blend = (1 - w) * base + w * recipe
            outputs.append(save(f"RECIPE_w{int(w*100):02d}", blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v36_exploration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v36 exploration candidates")


if __name__ == "__main__":
    main()
