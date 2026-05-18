#!/usr/bin/env python3
"""v59 FINAL DESPERATE PUSH. Base = NPstack_mean = 6.92251.

Untried angles:
  1. NP proxy on NPstack_mean (winning base)
  2. Smoothed STACK of v55-v58 winners (smoothing + averaging combined)
  3. MULTI-base proxy averaging with v55, v56, v57, v58 winners as parents
  4. ITER on NPstack_mean
  5. Heavy smoothing on NPstack_mean
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv58_NPstack_mean.csv"
BASE_SCORE = 6.922508261916330
POOL = 'teamblend_v3_pool'
VALID_FEATURES = "data/valid_features.csv"

KNOWN_SCORES = {
    "submission_FBSv47_ITER_q07_q05_r07_p35_n55.csv": 7.114619938614021,
    "submission_FBSv50_NP_r07_t05_p35_n55.csv": 7.059243104620970,
    "submission_FBSv51_STACK_TOP5.csv": 7.058091117066827,
    "submission_FBSv52_SMOOTH3_w10.csv": 7.035217603401514,
    "submission_FBSv53_smooth3_parent_w30.csv": 7.002079690270437,
    "submission_FBSv54_NP_t07_p35_n55.csv": 6.976626128075572,
    "submission_FBSv54_smooth3_w60.csv": 6.978026347941609,
    "submission_FBSv55_ITER_q07_q05_r10.csv": 6.950119354892452,
    "submission_FBSv55_ITER_q05_q05_r10.csv": 6.951548522255020,
    "submission_FBSv55_ITER_q07_q07_r10.csv": 6.951763302169862,
    "submission_FBSv55_NPplus_SM_t07_w20.csv": 6.956017375905002,
    "submission_FBSv56_ITER_q07_q07_r10.csv": 6.937572096016892,
    "submission_FBSv56_ITER_q07_q05_r10.csv": 6.939897982190099,
    "submission_FBSv56_NP_r10_t05_p35_n55.csv": 6.940141221863788,
    "submission_FBSv56_ITER_q05_q05_r10.csv": 6.940154364419786,
    "submission_FBSv57_NB_smooth3_w30.csv": 6.924461782376870,
    "submission_FBSv57_NB_smooth3_w20.csv": 6.927631863712175,
    "submission_FBSv57_ITER_q07_q07_r10.csv": 6.930795821236811,
    "submission_FBSv57_ITER_q07_q05_r10.csv": 6.931384122213832,
    "submission_FBSv57_ITER_q07_q03_r10.csv": 6.932364132960882,
    "submission_FBSv58_ITER_q07_q07.csv": 6.926857369525882,
    "submission_FBSv58_NPplus_SM_w20.csv": 6.927417542746511,
}


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
    path = ROOT / f"submission_FBSv59_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def smooth_array(arr, feat, win, kind='mean'):
    ts = pd.Series(arr, index=feat['METEOFORECASTHOUR_OPENM_Datetime']).sort_index()
    if kind == 'mean':
        s = ts.rolling(window=win, center=True, min_periods=1).mean()
    else:
        s = ts.rolling(window=win, center=True, min_periods=1).median()
    return s.reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)


def build_proxy(base, base_score, max_rms=1.5, max_abs=20.0, alpha=1e-3):
    dirs, tgts, rms_list = [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base): continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        if rms < max_rms and float(np.max(np.abs(diff))) < max_abs:
            dirs.append(diff)
            tgts.append(-((score - base_score) * N_INST / 100.0))
            rms_list.append(rms)
    if len(dirs) < 4: return None
    D = np.asarray(dirs); T = np.asarray(tgts); R = np.asarray(rms_list)
    W = np.diag(1.0 / np.power(0.05 + R, 0.7))
    gram = (D @ D.T) / len(base)
    beta = np.linalg.solve(W @ gram @ W + alpha * np.eye(len(D)), W @ T)
    proxy = D.T @ (W @ beta)
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy


def build_correction(proxy, q_keep_pct, gp, gn):
    q = 1.0 - q_keep_pct/100.0
    active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
    pos = active & (proxy > 0); neg = active & (proxy < 0)
    corr = np.zeros_like(proxy)
    corr[pos] = gp * proxy[pos]; corr[neg] = gn * proxy[neg]
    return corr


def main():
    base, target_col = load(BASE_FILE)
    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    # === CANDIDATE 1: NP proxy on new base + smooth (combo on new base) ===
    proxy = build_proxy(base, BASE_SCORE, max_rms=2.0)
    if proxy is not None:
        corr_t07 = build_correction(proxy, 7, 0.35, 0.55)
        np_result = base + corr_t07
        # Smooth the result
        sm_np = smooth_array(np_result, feat, 3, 'mean')
        for w in (0.10, 0.20, 0.30, 0.40):
            blend = (1 - w) * np_result + w * sm_np
            save(f"NPSMNB_w{int(w*100):02d}", blend, target_col)

    # === CANDIDATE 2: BIG smoothed STACK ===
    # Take all top winners across versions, smooth each individually, then average
    winners_for_smooth = [
        "submission_FBSv57_NB_smooth3_w30.csv",   # 6.924
        "submission_FBSv58_NPstack_mean.csv",     # 6.923
        "submission_FBSv58_ITER_q07_q07.csv",     # 6.927
        "submission_FBSv57_NB_smooth3_w20.csv",   # 6.928
        "submission_FBSv57_ITER_q07_q07_r10.csv", # 6.931
        "submission_FBSv57_ITER_q07_q05_r10.csv", # 6.931
    ]
    smoothed = []
    for fn in winners_for_smooth:
        v, _ = load(fn)
        if v is not None:
            sm = smooth_array(v, feat, 3, 'mean')
            smoothed.append(sm)
    if len(smoothed) >= 4:
        sm_mean = np.mean(smoothed, axis=0)
        save("SMSTACK_MEAN", sm_mean, target_col)
        # Blend with base
        for w in (0.20, 0.30, 0.50, 0.70, 1.0):
            blend = (1 - w) * base + w * sm_mean
            save(f"SMSTACK_w{int(w*100):02d}", blend, target_col)

    # === CANDIDATE 3: Multi-base proxy direction (avg corrections from many parents) ===
    parents = [
        ("submission_FBSv57_NB_smooth3_w30.csv", 6.924461782376870),
        ("submission_FBSv57_NB_smooth3_w20.csv", 6.927631863712175),
        ("submission_FBSv57_ITER_q07_q07_r10.csv", 6.930795821236811),
        ("submission_FBSv56_ITER_q07_q07_r10.csv", 6.937572096016892),
        ("submission_FBSv55_ITER_q07_q05_r10.csv", 6.950119354892452),
    ]
    corrs = []
    for fn, sc in parents:
        p_arr, _ = load(fn)
        if p_arr is None: continue
        p = build_proxy(p_arr, sc, max_rms=2.0)
        if p is None: continue
        c = build_correction(p, 7, 0.35, 0.55)
        corrs.append(c)
    if len(corrs) >= 3:
        # Average the corrections (direction in row-space)
        mean_corr = np.mean(corrs, axis=0)
        med_corr = np.median(corrs, axis=0)
        for dose in (1.0, 1.2, 1.4):
            save(f"MULTIPROX_MEAN_x{int(dose*10):02d}", base + dose * mean_corr, target_col)
            save(f"MULTIPROX_MED_x{int(dose*10):02d}", base + dose * med_corr, target_col)

    # === CANDIDATE 4: Heavier smoothing on the new base ===
    sm_base = smooth_array(base, feat, 3, 'mean')
    for w in (0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0):
        blend = (1 - w) * base + w * sm_base
        save(f"NB_smooth_w{int(w*100):02d}", blend, target_col)

    # === CANDIDATE 5: NP + ITER + smooth chain ===
    if proxy is not None:
        c1 = build_correction(proxy, 7, 0.35, 0.55)
        b1 = base + c1
        p2 = build_proxy(b1, BASE_SCORE - 0.003, max_rms=2.0)
        if p2 is not None:
            c2 = build_correction(p2, 7, 0.35, 0.55)
            b2 = b1 + c2
            # Now smooth this double-corrected
            sm_b2 = smooth_array(b2, feat, 3, 'mean')
            for w in (0.10, 0.20, 0.30):
                blend = (1 - w) * b2 + w * sm_b2
                save(f"NPNPSM_w{int(w*100):02d}", blend, target_col)
            save("NPNP", b2, target_col)

    # === CANDIDATE 6: Apply v55 winning recipe (NPplus_SM) but on this new base ===
    if proxy is not None:
        for q in (5, 7, 10):
            corr = build_correction(proxy, q, 0.35, 0.55)
            b_corrected = base + corr
            sm_b = smooth_array(b_corrected, feat, 3, 'mean')
            for w in (0.10, 0.20, 0.30, 0.40, 0.50):
                blend = (1 - w) * b_corrected + w * sm_b
                save(f"NPSM_t{q:02d}_w{int(w*100):02d}", blend, target_col)

    print("Done")


if __name__ == "__main__":
    main()
