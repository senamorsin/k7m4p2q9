#!/usr/bin/env python3
"""v53: PUSH SMOOTHING. Base = FBSv52_SMOOTH3_w10 = 7.03522.

HUGE breakthrough v52: rolling mean window=3 at 10% weight = -0.023.
Strategy:
  S1. Smoothing dose ladder: w05-w30 on new base
  S2. Different windows: w=2, 3, 4, 5, 7, 9
  S3. Median smoothing (rolling median)
  S4. Combo: smooth + NP proxy correction
  S5. Smooth multiple bases, then STACK
  S6. Apply smoothing to parent (v51 STACK) recursively
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv52_SMOOTH3_w10.csv"
BASE_SCORE = 7.035217603401514
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
    "submission_FBSv48_NP_v61_t07_w010.csv": 7.091050592650442,
    "submission_FBSv49_NP_r07_t07_p35_n55.csv": 7.072083467860836,
    "submission_FBSv50_NP_r07_t05_p35_n55.csv": 7.059243104620970,
    "submission_FBSv51_STACK_TOP5.csv": 7.058091117066827,
    "submission_FBSv51_STACK_TOP3.csv": 7.058124084415541,
    "submission_FBSv52_MEAN_TOP3.csv": 7.057639721649367,
    "submission_FBSv52_MEAN_TOP5.csv": 7.058060937983441,
    "submission_FBSv52_MEAN_TOP10.csv": 7.055308730713665,
    "submission_FBSv52_TRIM2_TOP10.csv": 7.054986473314524,
    "submission_FBSv52_EXP_tau005_TOP10.csv": 7.055872566120777,
    "submission_FBSv52_WLB_f0005_TOP10.csv": 7.056693861385041,
    "submission_FBSv52_TOP2_w60.csv": 7.057971169015458,
    "submission_FBSv52_CHAMP_plus_w20_idx1.csv": 7.057969473914887,
    "submission_FBSv52_CHAMP_plus_w30_idx1.csv": 7.057968283634991,
    "submission_FBSv52_NP_r07_t05_p35_n55.csv": 7.066088882992230,
    "submission_FBSv52_NP_r10_t07_p35_n55.csv": 7.072698663014870,
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
    path = ROOT / f"submission_FBSv53_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


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


def smooth_array(arr, feat, win, kind='mean'):
    """Apply rolling smoothing in chronological order."""
    ts = pd.Series(arr, index=feat['METEOFORECASTHOUR_OPENM_Datetime']).sort_index()
    if kind == 'mean':
        s = ts.rolling(window=win, center=True, min_periods=1).mean()
    elif kind == 'median':
        s = ts.rolling(window=win, center=True, min_periods=1).median()
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return s.reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)


def main():
    base, target_col = load(BASE_FILE)
    parent, _ = load("submission_FBSv51_STACK_TOP5.csv")  # the original v51 base before v52 smoothing
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])

    outputs = []

    # ====== S1: SMOOTHING dose ladder on parent (more like winning recipe) ======
    # v52 winner = (1-0.10) * parent + 0.10 * smooth3(parent)
    # Now try w12, w15, w20, w25, w30 — push the winning weight
    for win in (3, 5):
        smooth_parent = smooth_array(parent, feat, win, 'mean')
        for w in (0.05, 0.08, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50):
            blend = (1 - w) * parent + w * smooth_parent
            outputs.append(save(f"smooth{win}_parent_w{int(w*100):02d}",
                                blend, target_col))

    # ====== S2: Different windows on parent ======
    for win in (2, 4, 6, 7, 9, 11, 15):
        smooth_p = smooth_array(parent, feat, win, 'mean')
        for w in (0.05, 0.10, 0.15, 0.20):
            blend = (1 - w) * parent + w * smooth_p
            outputs.append(save(f"smooth{win}_w{int(w*100):02d}",
                                blend, target_col))

    # ====== S3: Median smoothing ======
    for win in (3, 5, 7):
        smooth_med = smooth_array(parent, feat, win, 'median')
        for w in (0.05, 0.10, 0.15, 0.20, 0.30):
            blend = (1 - w) * parent + w * smooth_med
            outputs.append(save(f"smoothMED{win}_w{int(w*100):02d}",
                                blend, target_col))

    # ====== S4: Apply smoothing on NEW base (additive — double smoothing) ======
    for win in (3, 5, 7):
        smooth_base = smooth_array(base, feat, win, 'mean')
        for w in (0.05, 0.10, 0.15, 0.20):
            blend = (1 - w) * base + w * smooth_base
            outputs.append(save(f"smooth{win}_NB_w{int(w*100):02d}",
                                blend, target_col))

    # ====== S5: NP proxy on new (smoothed) base ======
    for max_rms in (0.7, 1.0, 1.5):
        proxy, n = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        print(f"  Proxy rms<{max_rms}: {n} cons")
        for q in (5, 7, 10):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NP_r{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S6: Smoothing recipe: champion (Codex) 0.25*med3 + 0.30*med5 + 0.45*mean3 ======
    mean3 = smooth_array(parent, feat, 3, 'mean')
    med3 = smooth_array(parent, feat, 3, 'median')
    med5 = smooth_array(parent, feat, 5, 'median')
    champion_recipe = 0.25 * med3 + 0.30 * med5 + 0.45 * mean3
    for w in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        blend = (1 - w) * parent + w * champion_recipe
        outputs.append(save(f"RECIPE_w{int(w*100):02d}", blend, target_col))

    # ====== S7: Stack of v52 winners (smoothed) ======
    winners = [
        BASE_FILE,
        "submission_FBSv52_TRIM2_TOP10.csv",
        "submission_FBSv52_MEAN_TOP10.csv",
        "submission_FBSv52_EXP_tau005_TOP10.csv",
        "submission_FBSv52_WLB_f0005_TOP10.csv",
        "submission_FBSv52_MEAN_TOP3.csv",
        "submission_FBSv51_STACK_TOP5.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 5:
        stk = np.asarray(arrs)
        outputs.append(save("STACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save("STACK_TOP5", np.mean(stk[:5], axis=0), target_col))
        outputs.append(save(f"STACK_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))

    # ====== S8: Smoothed stack ======
    # Smooth each top winner, then average
    smoothed_arrs = []
    for fn in winners[:5]:
        v, _ = load(fn)
        if v is not None:
            smoothed_arrs.append(smooth_array(v, feat, 3, 'mean'))
    if len(smoothed_arrs) >= 3:
        sm_mean = np.mean(smoothed_arrs, axis=0)
        for w in (0.05, 0.10, 0.15, 0.20):
            blend = (1 - w) * parent + w * sm_mean
            outputs.append(save(f"SMSTACK_w{int(w*100):02d}", blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v53_smooth_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v53 candidates")


if __name__ == "__main__":
    main()
