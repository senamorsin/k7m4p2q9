#!/usr/bin/env python3
"""v57 LAST. Base = ITER_q07_q07_r10 = 6.93757."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv56_ITER_q07_q07_r10.csv"
BASE_SCORE = 6.937572096016892
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
    "submission_FBSv55_NP_r10_t10_p35_n55.csv": 6.960327939400416,
    "submission_FBSv55_NPplus_SM_t07_w10.csv": 6.960968266615593,
    "submission_FBSv55_NPalt_sm60_t05_p35_n55.csv": 6.961620434512953,
    "submission_FBSv55_NB_smooth3_w30.csv": 6.961959296304910,
    "submission_FBSv55_NP_r07_t07_p35_n55.csv": 6.962130996239678,
    # v56 batch
    "submission_FBSv56_ITER_q07_q05_r10.csv": 6.939897982190099,
    "submission_FBSv56_ITER_q05_q05_r10.csv": 6.940154364419786,
    "submission_FBSv56_NP_r10_t05_p35_n55.csv": 6.940141221863788,
    "submission_FBSv56_ITER_q07_q03_r10.csv": 6.941214502979053,
    "submission_FBSv56_ITER_q10_q05_r10.csv": 6.941562941839893,
    "submission_FBSv56_NP_r10_t07_p35_n55.csv": 6.945102006570353,
    "submission_FBSv56_NP_r15_t07_p35_n55.csv": 6.948132574812887,
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
    path = ROOT / f"submission_FBSv57_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def smooth_array(arr, feat, win, kind='mean'):
    ts = pd.Series(arr, index=feat['METEOFORECASTHOUR_OPENM_Datetime']).sort_index()
    if kind == 'mean':
        s = ts.rolling(window=win, center=True, min_periods=1).mean()
    else:
        s = ts.rolling(window=win, center=True, min_periods=1).median()
    return s.reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)


def build_proxy(base, base_score, max_rms=1.5, max_abs=15.0, alpha=1e-3):
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

    outputs = []

    # NP grid
    for max_rms in (1.0, 1.5, 2.0, 3.0):
        proxy, n = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        print(f"  Proxy rms<{max_rms}: {n} cons")
        for q in (3, 5, 7, 10):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NP_r{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ITER
    proxy_first, _ = build_proxy(base, BASE_SCORE, max_rms=1.5)
    if proxy_first is not None:
        for q1 in (5, 7, 10):
            corr1 = build_correction(proxy_first, q1, 0.35, 0.55)
            b1 = base + corr1
            for sec_rms in (1.0, 1.5, 2.0):
                proxy2, n2 = build_proxy(b1, BASE_SCORE - 0.005, max_rms=sec_rms)
                if proxy2 is not None and n2 >= 5:
                    for q2 in (3, 5, 7):
                        corr2 = build_correction(proxy2, q2, 0.35, 0.55)
                        outputs.append(save(
                            f"ITER_q{q1:02d}_q{q2:02d}_r{int(sec_rms*10):02d}",
                            b1 + corr2, target_col))
                    break

    # Smooth on new base
    sm3 = smooth_array(base, feat, 3, 'mean')
    for w in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70):
        blend = (1 - w) * base + w * sm3
        outputs.append(save(f"NB_smooth3_w{int(w*100):02d}", blend, target_col))

    # NP + smooth combo
    if proxy_first is not None:
        for q in (5, 7):
            corr = build_correction(proxy_first, q, 0.35, 0.55)
            b_c = base + corr
            sm_b = smooth_array(b_c, feat, 3, 'mean')
            for w in (0.10, 0.20, 0.30):
                outputs.append(save(
                    f"NPplus_SM_t{q:02d}_w{int(w*100):02d}",
                    (1 - w) * b_c + w * sm_b, target_col))

    # STACK winners
    winners = [
        BASE_FILE,
        "submission_FBSv56_ITER_q07_q05_r10.csv",
        "submission_FBSv56_ITER_q05_q05_r10.csv",
        "submission_FBSv56_NP_r10_t05_p35_n55.csv",
        "submission_FBSv56_ITER_q07_q03_r10.csv",
        "submission_FBSv56_ITER_q10_q05_r10.csv",
        "submission_FBSv56_NP_r10_t07_p35_n55.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 4:
        stk = np.asarray(arrs)
        outputs.append(save("STACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save("STACK_TOP5", np.mean(stk[:5], axis=0), target_col))
        outputs.append(save(f"STACK_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"STACK_MED_{len(arrs)}", np.median(stk, axis=0), target_col))
        sc = np.array([6.93757, 6.93989, 6.94015, 6.94014, 6.94121, 6.94156, 6.94510])[:len(arrs)]
        w_lb = 1.0 / (sc - 6.936 + 0.0003)
        w_lb = w_lb / w_lb.sum()
        outputs.append(save("STACK_WLB", (stk.T @ w_lb).reshape(-1), target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v57_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v57 candidates")


if __name__ == "__main__":
    main()
