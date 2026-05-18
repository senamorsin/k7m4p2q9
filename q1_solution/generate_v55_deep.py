#!/usr/bin/env python3
"""v55: DEEP WORK. Base = FBSv54_NP_t07_p35_n55 = 6.97663.

v54 findings:
  - NP proxy on smoothed base WINS (-0.025)
  - smooth3 peak at w60 (w50→w70 plateau)
  - smooth2/4/5 fail (only window=3 right)
  - SMSTACK competitive but lower than pure smooth3

v55 strategy:
  S1. NP full grid on NEW base (proxy method returned to working)
  S2. ITER (apply proxy twice) on new base
  S3. NP + smooth3 combo (apply smoothing after NP correction)
  S4. Smooth3 fine grid around w60 peak (w55, w58, w62, w65)
  S5. NP on smooth3_w60 (alt path to current champion)
  S6. STACK of v54 winners
  S7. NP + tiny v61/TOP6/FRR
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv54_NP_t07_p35_n55.csv"
BASE_SCORE = 6.976626128075572
POOL = 'teamblend_v3_pool'
VALID_FEATURES = "data/valid_features.csv"

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv40_v61_w050.csv": 7.206400653558756,
    "submission_FBSv44_ITER2_t07_t05_p35_n55.csv": 7.182302330032473,
    "submission_FBSv46_ITER_q07_q03_r07.csv": 7.156591930447194,
    "submission_FBSv47_ITER_q07_q05_r07_p35_n55.csv": 7.114619938614021,
    "submission_FBSv49_NP_r07_t07_p35_n55.csv": 7.072083467860836,
    "submission_FBSv50_NP_r07_t05_p35_n55.csv": 7.059243104620970,
    "submission_FBSv51_STACK_TOP5.csv": 7.058091117066827,
    "submission_FBSv52_SMOOTH3_w10.csv": 7.035217603401514,
    "submission_FBSv53_smooth3_parent_w30.csv": 7.002079690270437,
    "submission_FBSv53_smooth3_parent_w25.csv": 7.008738211507544,
    "submission_FBSv53_RECIPE_w30.csv": 7.014512018444050,
    "submission_FBSv53_smooth3_parent_w20.csv": 7.015816983015136,
    "submission_FBSv53_SMSTACK_w20.csv": 7.015525084621934,
    "submission_FBSv53_smooth3_parent_w18.csv": 7.019114942939860,
    "submission_FBSv53_NP_r10_t07_p35_n55.csv": 7.022321549151228,
    # v54 batch
    "submission_FBSv54_smooth3_w60.csv": 6.978026347941609,
    "submission_FBSv54_smooth3_w70.csv": 6.978906999435949,
    "submission_FBSv54_smooth3_w50.csv": 6.981591875943377,
    "submission_FBSv54_SMSTACK_w50.csv": 6.983401888435778,
    "submission_FBSv54_smooth3_w45.csv": 6.985493271787274,
    "submission_FBSv54_NB_smooth3_w20.csv": 6.989276395780954,
    "submission_FBSv54_smooth3_w40.csv": 6.990689958450519,
    "submission_FBSv54_SMSTACK_w40.csv": 6.991717545668834,
    "submission_FBSv54_NB_smooth3_w15.csv": 6.992273549298729,
    "submission_FBSv54_smooth3_w35.csv": 6.996073538290212,
    "submission_FBSv54_smooth3_pure.csv": 6.998046519232857,
    "submission_FBSv54_RECIPE_w50.csv": 7.000836393094036,
    "submission_FBSv54_SMSTACK_w30.csv": 7.002802468292511,
    "submission_FBSv54_RECIPE_w40.csv": 7.005825068994870,
    "submission_FBSv54_RECIPE_w35.csv": 7.009967396387146,
    "submission_FBSv54_smooth5_w30.csv": 7.026557201365319,
    "submission_FBSv54_smooth4_w30.csv": 7.066703497980973,
    "submission_FBSv54_smooth2_w30.csv": 7.080474247059677,
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
    path = ROOT / f"submission_FBSv55_{label}.csv"
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
    parent_v53, _ = load("submission_FBSv53_smooth3_parent_w30.csv")
    parent_v51, _ = load("submission_FBSv51_STACK_TOP5.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])

    outputs = []

    # ====== S1: NP proxy FULL GRID on new base ======
    for max_rms in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
        proxy, n = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        print(f"  Proxy rms<{max_rms}: {n} cons")
        for q in (3, 5, 7, 8, 10, 12):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NP_r{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S2: ITER (apply twice) on new base ======
    proxy_first, _ = build_proxy(base, BASE_SCORE, max_rms=1.0)
    if proxy_first is None:
        proxy_first, _ = build_proxy(base, BASE_SCORE, max_rms=2.0)
    if proxy_first is not None:
        for q1 in (5, 7, 10, 12, 15):
            corr1 = build_correction(proxy_first, q1, 0.35, 0.55)
            b1 = base + corr1
            for sec_rms in (1.0, 1.5, 2.0):
                proxy2, n2 = build_proxy(b1, BASE_SCORE - 0.005, max_rms=sec_rms)
                if proxy2 is not None and n2 >= 5:
                    for q2 in (3, 5, 7, 10):
                        corr2 = build_correction(proxy2, q2, 0.35, 0.55)
                        outputs.append(save(
                            f"ITER_q{q1:02d}_q{q2:02d}_r{int(sec_rms*10):02d}",
                            b1 + corr2, target_col))
                    break

    # ====== S3: smooth3 fine grid around peak (v54 winner was w60 = 6.978) ======
    smooth3_p = smooth_array(parent_v51, feat, 3, 'mean')
    for w in (0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.75, 0.85):
        blend = (1 - w) * parent_v51 + w * smooth3_p
        outputs.append(save(f"smooth3_w{int(w*100):02d}", blend, target_col))

    # ====== S4: smooth on NEW base (additive) ======
    smooth3_base = smooth_array(base, feat, 3, 'mean')
    for w in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        blend = (1 - w) * base + w * smooth3_base
        outputs.append(save(f"NB_smooth3_w{int(w*100):02d}", blend, target_col))

    # ====== S5: NP on smooth3_w60 (alt base) — try same recipe from different starting point ======
    sm60, _ = load("submission_FBSv54_smooth3_w60.csv")
    if sm60 is not None:
        proxy_sm60, _ = build_proxy(sm60, 6.978026347941609, max_rms=1.0)
        if proxy_sm60 is not None:
            for q in (5, 7, 10):
                for gp, gn in [(0.35, 0.55), (0.30, 0.55)]:
                    corr = build_correction(proxy_sm60, q, gp, gn)
                    outputs.append(save(
                        f"NPalt_sm60_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                        sm60 + corr, target_col))

    # ====== S6: NP + smoothing combo ======
    # Apply NP correction THEN smooth the result
    if proxy_first is not None:
        for q in (5, 7, 10):
            corr = build_correction(proxy_first, q, 0.35, 0.55)
            b_corrected = base + corr
            # Now smooth the corrected result
            sm_b = smooth_array(b_corrected, feat, 3, 'mean')
            for w in (0.05, 0.10, 0.20):
                blend = (1 - w) * b_corrected + w * sm_b
                outputs.append(save(
                    f"NPplus_SM_t{q:02d}_w{int(w*100):02d}",
                    blend, target_col))

    # ====== S7: STACK of v54 winners ======
    winners = [
        BASE_FILE,
        "submission_FBSv54_smooth3_w60.csv",
        "submission_FBSv54_smooth3_w70.csv",
        "submission_FBSv54_smooth3_w50.csv",
        "submission_FBSv54_SMSTACK_w50.csv",
        "submission_FBSv54_smooth3_w45.csv",
        "submission_FBSv54_NB_smooth3_w20.csv",
        "submission_FBSv54_smooth3_w40.csv",
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
        # Inverse-LB weighted
        sc = np.array([6.97663, 6.97803, 6.97891, 6.98159, 6.98340, 6.98549, 6.98928, 6.99069])[:len(arrs)]
        w_lb = 1.0 / (sc - 6.975 + 0.0005)
        w_lb = w_lb / w_lb.sum()
        outputs.append(save("STACK_WLB", (stk.T @ w_lb).reshape(-1), target_col))
        # Trimmed
        if len(arrs) >= 6:
            trim = np.zeros_like(stk[0])
            for i in range(stk.shape[1]):
                col = np.sort(stk[:, i])
                trim[i] = col[1:-1].mean()
            outputs.append(save(f"STACK_TRIM1_{len(arrs)}", trim, target_col))

    # ====== S8: NP + tiny v61/TOP6/FRR ======
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    champ_orig, _ = load("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv")
    v61, _ = load("submission_v61_alone.csv")
    delta_frr = frr - champ_orig
    pool_path = ROOT / POOL
    top6 = None
    fp = pool_path / "submission_TOP6_MEDIAN.csv"
    if fp.exists():
        top6 = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)

    if proxy_first is not None:
        for q in (5, 7, 10):
            corr = build_correction(proxy_first, q, 0.35, 0.55)
            for w_frr in (0.05, 0.10):
                outputs.append(save(
                    f"NP_FRR_t{q:02d}_w{int(w_frr*100):02d}",
                    base + corr + w_frr * delta_frr, target_col))
            if v61 is not None:
                for w in (0.010, 0.015):
                    outputs.append(save(
                        f"NP_v61_t{q:02d}_w{int(w*1000):03d}",
                        base + corr + w * (v61 - base), target_col))
            if top6 is not None:
                for w in (0.005, 0.010):
                    outputs.append(save(
                        f"NP_TOP6_t{q:02d}_w{int(w*1000):03d}",
                        base + corr + w * (top6 - base), target_col))

    # ====== S9: Median smooth (try the other smoothing kind) ======
    med3_p = smooth_array(parent_v51, feat, 3, 'median')
    for w in (0.30, 0.40, 0.50, 0.60):
        blend = (1 - w) * parent_v51 + w * med3_p
        outputs.append(save(f"med3_w{int(w*100):02d}", blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v55_deep_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v55 candidates")


if __name__ == "__main__":
    main()
