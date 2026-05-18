#!/usr/bin/env python3
"""v61 FOCUS: Mirror v59 winning recipe exactly.

v59 SMSTACK_w50 winning recipe:
  - 6 specific winners (v55-v58 era)
  - smooth3 of each
  - average them
  - blend 50% with parent v51

Vary ONLY one parameter at a time around that exact recipe.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
PARENT_FILE = "submission_FBSv51_STACK_TOP5.csv"  # The v59 parent
POOL = 'teamblend_v3_pool'
VALID_FEATURES = "data/valid_features.csv"

# The EXACT 6 winners v59 used (from generate_v59_final.py):
V59_WINNERS = [
    "submission_FBSv57_NB_smooth3_w30.csv",
    "submission_FBSv58_NPstack_mean.csv",
    "submission_FBSv58_ITER_q07_q07.csv",
    "submission_FBSv57_NB_smooth3_w20.csv",
    "submission_FBSv57_ITER_q07_q07_r10.csv",
    "submission_FBSv57_ITER_q07_q05_r10.csv",
]


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    path = ROOT / name
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    vals = df.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(vals).all(): return None, None
    return vals, df.columns[0]


def save(label, values, target_col):
    arr = np.clip(values, 0.0, N_INST)
    path = ROOT / f"submission_FBSv61_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')


def smooth_array(arr, feat, win, kind='mean'):
    ts = pd.Series(arr, index=feat['METEOFORECASTHOUR_OPENM_Datetime']).sort_index()
    if kind == 'mean':
        s = ts.rolling(window=win, center=True, min_periods=1).mean()
    else:
        s = ts.rolling(window=win, center=True, min_periods=1).median()
    return s.reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)


def main():
    parent, target_col = load(PARENT_FILE)
    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
    print(f"PARENT: {PARENT_FILE}")

    # Load the EXACT 6 v59 winners
    winners_raw = []
    for fn in V59_WINNERS:
        v, _ = load(fn)
        if v is not None:
            winners_raw.append(v)
    print(f"Loaded {len(winners_raw)} v59 winners")

    if len(winners_raw) < 6:
        print("ERROR: missing winners")
        return

    # === STRATEGY 1: vary dose around winning w50 ===
    smoothed = [smooth_array(v, feat, 3, 'mean') for v in winners_raw]
    sm_mean = np.mean(smoothed, axis=0)
    for w in (0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65):
        blend = (1 - w) * parent + w * sm_mean
        save(f"SMSTACK_w{int(w*100):02d}", blend, target_col)

    # === STRATEGY 2: swap 1 winner from v59 with newer ones (v58/v59 champs) ===
    # The v59 winner was 6 winners. Swap one with NPstack_mean (latest champion-tier)
    npstack, _ = load("submission_FBSv58_NPstack_mean.csv")  # already in winners
    sm_w50, _ = load("submission_FBSv59_SMSTACK_w50.csv")  # current champion

    # Replace the worst v59 winner with SMSTACK_w50 (current champion)
    if sm_w50 is not None:
        # Mix current champ as 1 of 6
        winners_alt = winners_raw[:-1] + [sm_w50]
        smoothed_alt = [smooth_array(v, feat, 3, 'mean') for v in winners_alt]
        sm_mean_alt = np.mean(smoothed_alt, axis=0)
        for w in (0.45, 0.50, 0.55):
            blend = (1 - w) * parent + w * sm_mean_alt
            save(f"SMSTACK_alt_w{int(w*100):02d}", blend, target_col)

    # === STRATEGY 3: smooth5 (different window) — exact same procedure ===
    smoothed_5 = [smooth_array(v, feat, 5, 'mean') for v in winners_raw]
    sm5_mean = np.mean(smoothed_5, axis=0)
    for w in (0.45, 0.50, 0.55, 0.60):
        blend = (1 - w) * parent + w * sm5_mean
        save(f"SMSTACK_win5_w{int(w*100):02d}", blend, target_col)

    # === STRATEGY 4: smooth2 (narrower) ===
    smoothed_2 = [smooth_array(v, feat, 2, 'mean') for v in winners_raw]
    sm2_mean = np.mean(smoothed_2, axis=0)
    for w in (0.50, 0.60, 0.70):
        blend = (1 - w) * parent + w * sm2_mean
        save(f"SMSTACK_win2_w{int(w*100):02d}", blend, target_col)

    # === STRATEGY 5: Mix current champion (SMSTACK_w50) with newer top winners (NO parent blend) ===
    # The idea: instead of using parent v51, use the current champion as base
    if sm_w50 is not None:
        # Smoothed top recent winners
        top_recent = [
            "submission_FBSv57_NB_smooth3_w30.csv",
            "submission_FBSv58_NPstack_mean.csv",
            "submission_FBSv58_ITER_q07_q07.csv",
        ]
        recent_arrs = []
        for fn in top_recent:
            v, _ = load(fn)
            if v is not None: recent_arrs.append(v)
        if len(recent_arrs) >= 2:
            smoothed_top = [smooth_array(v, feat, 3, 'mean') for v in recent_arrs]
            top_mean = np.mean(smoothed_top, axis=0)
            # Blend current champion with smoothed top
            for w in (0.10, 0.20, 0.30, 0.50):
                blend = (1 - w) * sm_w50 + w * top_mean
                save(f"CHAMPplus_w{int(w*100):02d}", blend, target_col)

    # === STRATEGY 6: Smooth the current champion itself ===
    if sm_w50 is not None:
        sm_sm = smooth_array(sm_w50, feat, 3, 'mean')
        for w in (0.05, 0.10, 0.15, 0.20, 0.30):
            blend = (1 - w) * sm_w50 + w * sm_sm
            save(f"CHAMPsm_w{int(w*100):02d}", blend, target_col)

    print("Done")


if __name__ == "__main__":
    main()
