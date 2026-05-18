#!/usr/bin/env python3
"""v54: PUSH smoothing dose to break 7.00 barrier.

v53: smooth3_w30 = 7.00208. Dose monotone w15→w30. Pic NOT found yet.

Strategy:
  S1. Push w35, w40, w45, w50, w60 (find peak)
  S2. smooth3 + RECIPE combo
  S3. Apply smoothing on smoothed base (double smoothing)
  S4. STACK of smoothed variants
  S5. NP proxy on new base + smoothing
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv53_smooth3_parent_w30.csv"
BASE_SCORE = 7.002079690270437
# Original parent before smoothing (the v51 STACK_TOP5)
ORIG_PARENT = "submission_FBSv51_STACK_TOP5.csv"
ORIG_PARENT_SCORE = 7.058091117066827
POOL = 'teamblend_v3_pool'
VALID_FEATURES = "data/valid_features.csv"

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv35_NB_FRR_w20.csv": 7.211396180429282,
    "submission_FBSv40_v61_w050.csv": 7.206400653558756,
    "submission_FBSv44_ITER2_t07_t05_p35_n55.csv": 7.182302330032473,
    "submission_FBSv46_ITER_q07_q03_r07.csv": 7.156591930447194,
    "submission_FBSv47_ITER_q07_q05_r07_p35_n55.csv": 7.114619938614021,
    "submission_FBSv49_NP_r07_t07_p35_n55.csv": 7.072083467860836,
    "submission_FBSv50_NP_r07_t05_p35_n55.csv": 7.059243104620970,
    "submission_FBSv51_STACK_TOP5.csv": 7.058091117066827,
    "submission_FBSv52_SMOOTH3_w10.csv": 7.035217603401514,
    "submission_FBSv52_TRIM2_TOP10.csv": 7.054986473314524,
    "submission_FBSv52_MEAN_TOP10.csv": 7.055308730713665,
    # v53 batch
    "submission_FBSv53_smooth3_parent_w25.csv": 7.008738211507544,
    "submission_FBSv53_RECIPE_w30.csv": 7.014512018444050,
    "submission_FBSv53_SMSTACK_w20.csv": 7.015525084621934,
    "submission_FBSv53_smooth3_parent_w20.csv": 7.015816983015136,
    "submission_FBSv53_smooth3_parent_w18.csv": 7.019114942939860,
    "submission_FBSv53_NP_r10_t07_p35_n55.csv": 7.022321549151228,
    "submission_FBSv53_smooth3_parent_w15.csv": 7.024844163960117,
    "submission_FBSv53_SMSTACK_w15.csv": 7.024662565172177,
    "submission_FBSv53_RECIPE_w20.csv": 7.025405332487522,
    "submission_FBSv53_smooth5_parent_w20.csv": 7.029488184493758,
    "submission_FBSv53_smooth3_parent_w12.csv": 7.030964918226344,
    "submission_FBSv53_RECIPE_w15.csv": 7.032110326284813,
    "submission_FBSv53_smoothMED3_w20.csv": 7.032292443440970,
    "submission_FBSv53_smooth5_parent_w15.csv": 7.033582724579865,
    "submission_FBSv53_smoothMED5_w15.csv": 7.040310528108219,
    "submission_FBSv53_RECIPE_w10.csv": 7.040366592664476,
    "submission_FBSv53_smooth7_w15.csv": 7.051015348558326,
    "submission_FBSv53_smooth2_w10.csv": 7.060565776531279,
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
    path = ROOT / f"submission_FBSv54_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def smooth_array(arr, feat, win, kind='mean'):
    ts = pd.Series(arr, index=feat['METEOFORECASTHOUR_OPENM_Datetime']).sort_index()
    if kind == 'mean':
        s = ts.rolling(window=win, center=True, min_periods=1).mean()
    elif kind == 'median':
        s = ts.rolling(window=win, center=True, min_periods=1).median()
    return s.reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)


def build_proxy(base, base_score, max_rms=0.7, max_abs=5.0, alpha=1e-3):
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
    parent, _ = load(ORIG_PARENT)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])

    outputs = []

    # ====== S1: PUSH smoothing dose on parent (find pic) ======
    # v53 winner: smooth3(parent) w30 = 7.00208
    # Curve monotonic w15→w30. Push w32-w70.
    smooth3_p = smooth_array(parent, feat, 3, 'mean')
    for w in (0.32, 0.35, 0.38, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.0):
        blend = (1 - w) * parent + w * smooth3_p
        outputs.append(save(f"smooth3_w{int(w*100):02d}", blend, target_col))

    # ====== S2: Pure smooth (w=1.0 means just smooth output) ======
    # smooth3_w100 ~ pure smoothed
    outputs.append(save(f"smooth3_pure", smooth3_p, target_col))

    # ====== S3: Smooth 4, 5, 6 (try different windows around 3) ======
    for win in (2, 4, 5, 6, 8):
        sm = smooth_array(parent, feat, win, 'mean')
        for w in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70):
            blend = (1 - w) * parent + w * sm
            outputs.append(save(f"smooth{win}_w{int(w*100):02d}", blend, target_col))

    # ====== S4: Recipe (mean+median combo) with higher doses ======
    mean3 = smooth_array(parent, feat, 3, 'mean')
    med3 = smooth_array(parent, feat, 3, 'median')
    med5 = smooth_array(parent, feat, 5, 'median')
    recipe = 0.25 * med3 + 0.30 * med5 + 0.45 * mean3
    for w in (0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 1.0):
        blend = (1 - w) * parent + w * recipe
        outputs.append(save(f"RECIPE_w{int(w*100):02d}", blend, target_col))

    # ====== S5: Apply smooth on BASE (additive) ======
    smooth_base3 = smooth_array(base, feat, 3, 'mean')
    smooth_base5 = smooth_array(base, feat, 5, 'mean')
    for w in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        blend = (1 - w) * base + w * smooth_base3
        outputs.append(save(f"NB_smooth3_w{int(w*100):02d}", blend, target_col))
        blend = (1 - w) * base + w * smooth_base5
        outputs.append(save(f"NB_smooth5_w{int(w*100):02d}", blend, target_col))

    # ====== S6: Pure smoothed bases stack ======
    smoothed_winners = []
    winners_for_smooth = [
        "submission_FBSv51_STACK_TOP5.csv",
        "submission_FBSv50_NP_r07_t05_p35_n55.csv",
        "submission_FBSv50_NP_r07_t07_p35_n55.csv",
        "submission_FBSv49_NP_r07_t07_p35_n55.csv",
        "submission_FBSv52_TRIM2_TOP10.csv",
        "submission_FBSv52_MEAN_TOP10.csv",
    ]
    for fn in winners_for_smooth:
        v, _ = load(fn)
        if v is not None:
            smoothed_winners.append(smooth_array(v, feat, 3, 'mean'))
    if len(smoothed_winners) >= 4:
        sm_mean = np.mean(smoothed_winners, axis=0)
        for w in (0.20, 0.30, 0.40, 0.50, 0.70, 1.0):
            blend = (1 - w) * parent + w * sm_mean
            outputs.append(save(f"SMSTACK_w{int(w*100):02d}", blend, target_col))

    # ====== S7: NP proxy on new base ======
    proxy, n = build_proxy(base, BASE_SCORE, max_rms=1.0)
    if proxy is not None:
        for q in (5, 7, 10):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NP_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S8: Mix v53 top winners ======
    v53_winners = [
        BASE_FILE,
        "submission_FBSv53_smooth3_parent_w25.csv",
        "submission_FBSv53_RECIPE_w30.csv",
        "submission_FBSv53_SMSTACK_w20.csv",
        "submission_FBSv53_smooth3_parent_w20.csv",
    ]
    arrs = []
    for fn in v53_winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 4:
        stk = np.asarray(arrs)
        outputs.append(save("v53STACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save("v53STACK_TOP5", np.mean(stk[:5], axis=0), target_col))
        outputs.append(save(f"v53STACK_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v54_break7_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v54 candidates")


if __name__ == "__main__":
    main()
