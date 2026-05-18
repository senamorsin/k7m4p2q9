#!/usr/bin/env python3
"""v62 FIX: v61 had wrong base. v59 used NPstack_mean (6.92) as base, not v51 (7.058).

Correct recipe v59 SMSTACK_w50:
  base = FBSv58_NPstack_mean (6.92251)
  sm_mean = avg(smooth3 of 6 winners)
  result = (1 - 0.50) * base + 0.50 * sm_mean = 6.91866

Now vary ONLY the dose w around 50.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
# CORRECT base from v59:
BASE_FILE = "submission_FBSv58_NPstack_mean.csv"
VALID_FEATURES = "data/valid_features.csv"

V59_WINNERS = [
    "submission_FBSv57_NB_smooth3_w30.csv",
    "submission_FBSv58_NPstack_mean.csv",
    "submission_FBSv58_ITER_q07_q07.csv",
    "submission_FBSv57_NB_smooth3_w20.csv",
    "submission_FBSv57_ITER_q07_q07_r10.csv",
    "submission_FBSv57_ITER_q07_q05_r10.csv",
]


def load(name):
    path = ROOT / name
    if not path.exists(): return None, None
    df = pd.read_csv(path)
    return df.iloc[:, 0].to_numpy(dtype=float), df.columns[0]


def save(label, values, target_col):
    arr = np.clip(values, 0.0, N_INST)
    pd.DataFrame({target_col: arr}).to_csv(ROOT / f"submission_FBSv62_{label}.csv",
                                            index=False, encoding='utf-8')


def smooth_array(arr, feat, win, kind='mean'):
    ts = pd.Series(arr, index=feat['METEOFORECASTHOUR_OPENM_Datetime']).sort_index()
    if kind == 'mean':
        s = ts.rolling(window=win, center=True, min_periods=1).mean()
    else:
        s = ts.rolling(window=win, center=True, min_periods=1).median()
    return s.reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)


def main():
    base, target_col = load(BASE_FILE)
    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
    print(f"BASE: {BASE_FILE} (= NPstack_mean = 6.92251)")

    winners_raw = []
    for fn in V59_WINNERS:
        v, _ = load(fn)
        if v is not None: winners_raw.append(v)
    print(f"Loaded {len(winners_raw)} v59 winners")

    if len(winners_raw) < 6:
        print("ERROR")
        return

    # === STRATEGY 1: Exact replication then micro-vary w ===
    smoothed = [smooth_array(v, feat, 3, 'mean') for v in winners_raw]
    sm_mean = np.mean(smoothed, axis=0)
    # CORRECT recipe: (1-w) * base + w * sm_mean
    for w in (0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70):
        blend = (1 - w) * base + w * sm_mean
        save(f"SMSTACK_w{int(w*100):02d}", blend, target_col)

    # === STRATEGY 2: smooth5 windows ===
    smoothed_5 = [smooth_array(v, feat, 5, 'mean') for v in winners_raw]
    sm5_mean = np.mean(smoothed_5, axis=0)
    for w in (0.40, 0.50, 0.60, 0.70):
        blend = (1 - w) * base + w * sm5_mean
        save(f"SMSTACK_w5_w{int(w*100):02d}", blend, target_col)

    # === STRATEGY 3: smooth2 windows ===
    smoothed_2 = [smooth_array(v, feat, 2, 'mean') for v in winners_raw]
    sm2_mean = np.mean(smoothed_2, axis=0)
    for w in (0.40, 0.50, 0.60, 0.70):
        blend = (1 - w) * base + w * sm2_mean
        save(f"SMSTACK_w2_w{int(w*100):02d}", blend, target_col)

    # === STRATEGY 4: Median average instead of mean ===
    sm_med = np.median(smoothed, axis=0)
    for w in (0.40, 0.50, 0.60):
        blend = (1 - w) * base + w * sm_med
        save(f"SMSTACK_MED_w{int(w*100):02d}", blend, target_col)

    # === STRATEGY 5: include CURRENT champion (SMSTACK_w50 v59) in stack ===
    sm_w50_v59, _ = load("submission_FBSv59_SMSTACK_w50.csv")
    if sm_w50_v59 is not None:
        winners_plus = winners_raw + [sm_w50_v59]
        smoothed_plus = [smooth_array(v, feat, 3, 'mean') for v in winners_plus]
        sm_plus_mean = np.mean(smoothed_plus, axis=0)
        # Now blend with v59 champion as base (the current LB winner)
        for w in (0.30, 0.50, 0.70):
            blend = (1 - w) * sm_w50_v59 + w * sm_plus_mean
            save(f"PLUSch_w{int(w*100):02d}", blend, target_col)

    # === STRATEGY 6: blend current champion with smooth stack ===
    if sm_w50_v59 is not None:
        for w in (0.10, 0.20, 0.30, 0.50):
            blend = (1 - w) * sm_w50_v59 + w * sm_mean
            save(f"CHAMPmix_w{int(w*100):02d}", blend, target_col)

    print("Done")


if __name__ == "__main__":
    main()
