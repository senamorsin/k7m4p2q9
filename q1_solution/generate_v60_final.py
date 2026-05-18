#!/usr/bin/env python3
"""v60 FINAL. Base = FBSv59_SMSTACK_w50 = 6.91866.

v59 SMSTACK won. Push it: MORE winners, MULTIPLE windows, HIGHER doses.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv59_SMSTACK_w50.csv"
BASE_SCORE = 6.918660714408993
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
    path = ROOT / f"submission_FBSv60_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')


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
    parent_v51, _ = load("submission_FBSv51_STACK_TOP5.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    # === BIGGER smoothed stack (ALL top winners across versions) ===
    winners_paths = [
        # v57-v58 top
        "submission_FBSv57_NB_smooth3_w30.csv",
        "submission_FBSv57_NB_smooth3_w20.csv",
        "submission_FBSv58_NPstack_mean.csv",
        "submission_FBSv58_ITER_q07_q07.csv",
        "submission_FBSv58_NPplus_SM_w20.csv",
        "submission_FBSv57_ITER_q07_q07_r10.csv",
        "submission_FBSv57_ITER_q07_q05_r10.csv",
        "submission_FBSv57_ITER_q07_q03_r10.csv",
        # v56
        "submission_FBSv56_ITER_q07_q07_r10.csv",
        "submission_FBSv56_ITER_q07_q05_r10.csv",
        "submission_FBSv56_NP_r10_t05_p35_n55.csv",
        # v55
        "submission_FBSv55_ITER_q07_q05_r10.csv",
    ]
    smoothed_arrs = []
    raw_arrs = []
    for fn in winners_paths:
        v, _ = load(fn)
        if v is not None:
            raw_arrs.append(v)
            smoothed_arrs.append(smooth_array(v, feat, 3, 'mean'))
    print(f"Loaded {len(smoothed_arrs)} winners for smoothing")

    if len(smoothed_arrs) >= 6:
        # Mean smoothed
        sm_mean = np.mean(smoothed_arrs, axis=0)
        raw_mean = np.mean(raw_arrs, axis=0)
        sm_med = np.median(smoothed_arrs, axis=0)

        # === CANDIDATE 1: BIG smoothed stack at various doses ===
        for w in (0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0):
            blend = (1 - w) * parent_v51 + w * sm_mean
            save(f"BIGSMSTACK_w{int(w*100):02d}", blend, target_col)

        # === CANDIDATE 2: Smoothed STACK median ===
        for w in (0.50, 0.70, 0.90, 1.0):
            blend = (1 - w) * parent_v51 + w * sm_med
            save(f"BIGSMSTACK_MED_w{int(w*100):02d}", blend, target_col)

        # === Multi-window smoothed stack ===
        # Apply smooth3 + smooth5 + smooth7 each to raw_arrs, then average
        sm3_all = [smooth_array(v, feat, 3, 'mean') for v in raw_arrs]
        sm5_all = [smooth_array(v, feat, 5, 'mean') for v in raw_arrs]
        sm7_all = [smooth_array(v, feat, 7, 'mean') for v in raw_arrs]
        sm3_mean = np.mean(sm3_all, axis=0)
        sm5_mean = np.mean(sm5_all, axis=0)
        sm7_mean = np.mean(sm7_all, axis=0)
        # Average of 3 windows
        multi_win = (sm3_mean + sm5_mean + sm7_mean) / 3.0
        for w in (0.50, 0.70, 0.90, 1.0):
            blend = (1 - w) * parent_v51 + w * multi_win
            save(f"MULTIWIN_w{int(w*100):02d}", blend, target_col)

    # === CANDIDATE 3: Smooth on current base (BASE = SMSTACK_w50) ===
    sm_base = smooth_array(base, feat, 3, 'mean')
    for w in (0.10, 0.20, 0.30, 0.50):
        blend = (1 - w) * base + w * sm_base
        save(f"NBsm_w{int(w*100):02d}", blend, target_col)

    # === CANDIDATE 4: Smooth NPstack (v58 champion) HEAVILY ===
    npstack, _ = load("submission_FBSv58_NPstack_mean.csv")
    if npstack is not None:
        sm_np = smooth_array(npstack, feat, 3, 'mean')
        for w in (0.30, 0.50, 0.70, 0.90, 1.0):
            blend = (1 - w) * npstack + w * sm_np
            save(f"SMnpstack_w{int(w*100):02d}", blend, target_col)
        # Also median smooth
        med_np = smooth_array(npstack, feat, 3, 'median')
        for w in (0.30, 0.50, 0.70, 1.0):
            blend = (1 - w) * npstack + w * med_np
            save(f"MEDnpstack_w{int(w*100):02d}", blend, target_col)

    # === CANDIDATE 5: STACK ALL top winners (raw, no smoothing) at higher weights ===
    if len(raw_arrs) >= 6:
        raw_mean_full = np.mean(raw_arrs, axis=0)
        save("RAWSTACK_FULL", raw_mean_full, target_col)
        raw_med = np.median(raw_arrs, axis=0)
        save("RAWSTACK_MED", raw_med, target_col)

    print("Done")


if __name__ == "__main__":
    main()
