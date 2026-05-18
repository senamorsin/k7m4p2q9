#!/usr/bin/env python3
"""v63 FINAL: SMSTACK_w45 = 6.91751 winner. Curve has min around w40-w45.

Strategy:
  - Ultra-fine grid w38-w46
  - Try base = SMSTACK_w45 (new champion) + micro modifications
  - Different smoothing windows on new base
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv62_SMSTACK_w45.csv"  # NEW champion
VALID_FEATURES = "data/valid_features.csv"

NPSTACK = "submission_FBSv58_NPstack_mean.csv"

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
    pd.DataFrame({target_col: arr}).to_csv(ROOT / f"submission_FBSv63_{label}.csv",
                                            index=False, encoding='utf-8')


def smooth_array(arr, feat, win, kind='mean'):
    ts = pd.Series(arr, index=feat['METEOFORECASTHOUR_OPENM_Datetime']).sort_index()
    if kind == 'mean':
        s = ts.rolling(window=win, center=True, min_periods=1).mean()
    else:
        s = ts.rolling(window=win, center=True, min_periods=1).median()
    return s.reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)


def main():
    npstack, target_col = load(NPSTACK)
    new_champ, _ = load(BASE_FILE)
    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])

    winners_raw = []
    for fn in V59_WINNERS:
        v, _ = load(fn)
        if v is not None: winners_raw.append(v)

    smoothed = [smooth_array(v, feat, 3, 'mean') for v in winners_raw]
    sm_mean = np.mean(smoothed, axis=0)

    # ULTRA-fine grid around w45
    for w in (0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46):
        blend = (1 - w) * npstack + w * sm_mean
        save(f"SMSTACK_w{int(w*100):02d}", blend, target_col)

    # Mix new champion (SMSTACK_w45) with neighbors
    for w in (0.05, 0.10, 0.15, 0.20, 0.30):
        # Champion + small dose of sm_mean
        blend = (1 - w) * new_champ + w * sm_mean
        save(f"CHAMP_plus_sm_w{int(w*100):02d}", blend, target_col)
        # Champion + small dose of NPstack
        blend2 = (1 - w) * new_champ + w * npstack
        save(f"CHAMP_plus_NP_w{int(w*100):02d}", blend2, target_col)

    # Different smoothing windows
    for win in (2, 4, 5):
        sm_w = [smooth_array(v, feat, win, 'mean') for v in winners_raw]
        sm_w_mean = np.mean(sm_w, axis=0)
        for w in (0.30, 0.40, 0.45, 0.50):
            blend = (1 - w) * npstack + w * sm_w_mean
            save(f"SMSTACK_win{win}_w{int(w*100):02d}", blend, target_col)

    # Smooth the new champion itself
    sm_champ = smooth_array(new_champ, feat, 3, 'mean')
    for w in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
        blend = (1 - w) * new_champ + w * sm_champ
        save(f"CHAMPsm_w{int(w*100):02d}", blend, target_col)

    print("Done")


if __name__ == "__main__":
    main()
