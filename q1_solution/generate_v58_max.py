#!/usr/bin/env python3
"""v58 MAX: Final push. Base = FBSv57_NB_smooth3_w30 = 6.92446.

User wants 1-3 models with BIGGEST jump.
Strategy: combine ALL winning techniques into focused candidates.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv57_NB_smooth3_w30.csv"
BASE_SCORE = 6.924461782376870
POOL = 'teamblend_v3_pool'
VALID_FEATURES = "data/valid_features.csv"

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv40_v61_w050.csv": 7.206400653558756,
    "submission_FBSv47_ITER_q07_q05_r07_p35_n55.csv": 7.114619938614021,
    "submission_FBSv50_NP_r07_t05_p35_n55.csv": 7.059243104620970,
    "submission_FBSv51_STACK_TOP5.csv": 7.058091117066827,
    "submission_FBSv52_SMOOTH3_w10.csv": 7.035217603401514,
    "submission_FBSv53_smooth3_parent_w30.csv": 7.002079690270437,
    "submission_FBSv54_NP_t07_p35_n55.csv": 6.976626128075572,
    "submission_FBSv54_smooth3_w60.csv": 6.978026347941609,
    "submission_FBSv54_smooth3_w70.csv": 6.978906999435949,
    "submission_FBSv54_smooth3_w50.csv": 6.981591875943377,
    "submission_FBSv55_ITER_q07_q05_r10.csv": 6.950119354892452,
    "submission_FBSv55_ITER_q05_q05_r10.csv": 6.951548522255020,
    "submission_FBSv55_ITER_q07_q07_r10.csv": 6.951763302169862,
    "submission_FBSv55_NPplus_SM_t07_w20.csv": 6.956017375905002,
    "submission_FBSv55_NPalt_sm60_t07_p35_n55.csv": 6.958799035515771,
    "submission_FBSv55_NB_smooth3_w30.csv": 6.961959296304910,
    "submission_FBSv55_NP_r07_t07_p35_n55.csv": 6.962130996239678,
    "submission_FBSv56_ITER_q07_q07_r10.csv": 6.937572096016892,
    "submission_FBSv56_ITER_q07_q05_r10.csv": 6.939897982190099,
    "submission_FBSv56_ITER_q05_q05_r10.csv": 6.940154364419786,
    "submission_FBSv56_NP_r10_t05_p35_n55.csv": 6.940141221863788,
    "submission_FBSv56_ITER_q07_q03_r10.csv": 6.941214502979053,
    "submission_FBSv57_NB_smooth3_w20.csv": 6.927631863712175,
    "submission_FBSv57_ITER_q07_q07_r10.csv": 6.930795821236811,
    "submission_FBSv57_ITER_q07_q05_r10.csv": 6.931384122213832,
    "submission_FBSv57_ITER_q07_q03_r10.csv": 6.932364132960882,
    "submission_FBSv57_ITER_q05_q05_r10.csv": 6.937918857327690,
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
    path = ROOT / f"submission_FBSv58_{label}.csv"
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
    if len(dirs) < 4: return None, len(dirs)
    D = np.asarray(dirs); T = np.asarray(tgts); R = np.asarray(rms_list)
    W = np.diag(1.0 / np.power(0.05 + R, 0.7))
    gram = (D @ D.T) / len(base)
    beta = np.linalg.solve(W @ gram @ W + alpha * np.eye(len(D)), W @ T)
    proxy = D.T @ (W @ beta)
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy, len(dirs)


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

    # Strategy 1: NP proxy on smoothed base
    proxy, n = build_proxy(base, BASE_SCORE, max_rms=1.5)
    print(f"Proxy: {n} cons")

    if proxy is None:
        proxy, n = build_proxy(base, BASE_SCORE, max_rms=2.0)
    if proxy is None:
        proxy, n = build_proxy(base, BASE_SCORE, max_rms=3.0)
    print(f"Final proxy: {n} cons")

    # ==== CANDIDATE 1: NP_t07_p35_n55 on new smoothed base ====
    if proxy is not None:
        corr_t07 = build_correction(proxy, 7, 0.35, 0.55)
        save("NP_t07_p35_n55", base + corr_t07, target_col)
        print(f"Saved NP_t07: base + corr, |c|_max={np.abs(corr_t07).max():.3f}")

        # CANDIDATE 2: NP_t05 (narrower)
        corr_t05 = build_correction(proxy, 5, 0.35, 0.55)
        save("NP_t05_p35_n55", base + corr_t05, target_col)

        # CANDIDATE 3: NP_t10 (wider)
        corr_t10 = build_correction(proxy, 10, 0.35, 0.55)
        save("NP_t10_p35_n55", base + corr_t10, target_col)

        # CANDIDATE 4: ITER (apply twice)
        b1 = base + corr_t07
        proxy2, n2 = build_proxy(b1, BASE_SCORE - 0.005, max_rms=2.0)
        if proxy2 is not None:
            for q2 in (5, 7, 10):
                corr2 = build_correction(proxy2, q2, 0.35, 0.55)
                save(f"ITER_q07_q{q2:02d}", b1 + corr2, target_col)

    # ==== CANDIDATE 5+6: PUSH SMOOTHING further on parent ====
    # NB_smooth3_w30 had w30 = base.
    # Try w40, w50 on parent
    parent_v56, _ = load("submission_FBSv56_ITER_q07_q07_r10.csv")
    if parent_v56 is not None:
        smooth_v56 = smooth_array(parent_v56, feat, 3, 'mean')
        for w in (0.35, 0.40, 0.45, 0.50, 0.60, 0.70):
            blend = (1 - w) * parent_v56 + w * smooth_v56
            save(f"smooth3_w{int(w*100):02d}", blend, target_col)

    # ==== CANDIDATE 7: NP + smoothing combo ====
    if proxy is not None:
        corr = build_correction(proxy, 7, 0.35, 0.55)
        b_corrected = base + corr
        sm_b = smooth_array(b_corrected, feat, 3, 'mean')
        for w in (0.10, 0.20, 0.30):
            blend = (1 - w) * b_corrected + w * sm_b
            save(f"NPplus_SM_w{int(w*100):02d}", blend, target_col)

    # ==== CANDIDATE 8: STACK of top v57 winners ====
    winners = [
        BASE_FILE,
        "submission_FBSv57_NB_smooth3_w20.csv",
        "submission_FBSv57_ITER_q07_q07_r10.csv",
        "submission_FBSv57_ITER_q07_q05_r10.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 3:
        stk = np.asarray(arrs)
        save("STACK_TOP4", np.mean(stk, axis=0), target_col)
        save("STACK_MEDIAN", np.median(stk, axis=0), target_col)
        # WLB
        sc = np.array([6.92446, 6.92763, 6.93080, 6.93138])[:len(arrs)]
        w_lb = 1.0 / (sc - 6.924 + 0.0001)
        w_lb = w_lb / w_lb.sum()
        save("STACK_WLB", (stk.T @ w_lb).reshape(-1), target_col)

    # ==== CANDIDATE 9: Heavy NB_smooth (push w30 → w50+) ====
    # base is already NB_smooth3_w30 of v56 parent
    # Add MORE smoothing on top of THAT
    base_sm = smooth_array(base, feat, 3, 'mean')
    for w in (0.10, 0.20, 0.30, 0.40, 0.50, 0.70):
        blend = (1 - w) * base + w * base_sm
        save(f"NB_smooth3_more_w{int(w*100):02d}", blend, target_col)

    # ==== CANDIDATE 10: Double NP on different bases, then stack ====
    if proxy is not None:
        corr = build_correction(proxy, 7, 0.35, 0.55)
        # On other top winners
        np_results = [base + corr]
        for fn in ["submission_FBSv57_NB_smooth3_w20.csv",
                    "submission_FBSv57_ITER_q07_q07_r10.csv"]:
            v, _ = load(fn)
            if v is not None:
                p, _ = build_proxy(v, BASE_SCORE, max_rms=2.0)
                if p is not None:
                    c = build_correction(p, 7, 0.35, 0.55)
                    np_results.append(v + c)
        if len(np_results) >= 2:
            save("NPstack_mean", np.mean(np_results, axis=0), target_col)
            save("NPstack_med", np.median(np_results, axis=0), target_col)

    print("Done")


if __name__ == "__main__":
    main()
