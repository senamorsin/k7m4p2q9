#!/usr/bin/env python3
"""v64 FINAL: SMSTACK_w36 = 6.91615 winner. Push lower w (30-38)."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
NPSTACK = "submission_FBSv58_NPstack_mean.csv"  # CORRECT base from v59 winning recipe
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
    pd.DataFrame({target_col: arr}).to_csv(ROOT / f"submission_FBSv64_{label}.csv",
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
    new_champ, _ = load("submission_FBSv63_SMSTACK_w36.csv")
    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])

    winners_raw = []
    for fn in V59_WINNERS:
        v, _ = load(fn)
        if v is not None: winners_raw.append(v)

    smoothed = [smooth_array(v, feat, 3, 'mean') for v in winners_raw]
    sm_mean = np.mean(smoothed, axis=0)

    # ULTRA-narrow search around w36 (current champ)
    # Curve: w36=6.91615, w38=6.91635, w40=6.91667 - very flat near min
    for w in (0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.35, 0.36, 0.37):
        blend = (1 - w) * npstack + w * sm_mean
        save(f"SMSTACK_w{int(w*100):02d}", blend, target_col)

    # Smooth current champion w36 slightly
    if new_champ is not None:
        sm_ch = smooth_array(new_champ, feat, 3, 'mean')
        for w in (0.02, 0.03, 0.05, 0.08, 0.10):
            blend = (1 - w) * new_champ + w * sm_ch
            save(f"CHAMP36sm_w{int(w*100):02d}", blend, target_col)

    print("Done")


if __name__ == "__main__":
    main()
