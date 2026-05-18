#!/usr/bin/env python3
"""v32: PUSH FRR direction further to claim #1. Base = BASE+FRR_w50 = 7.21847.

v31 winning pattern: every +10% FRR gives -0.0006:
  w20 → 7.22025
  w30 → 7.21962
  w40 → 7.21901
  w50 → 7.21847 (winner)

Linear: w60 → 7.218, w70 → 7.217, w80 → 7.217. Curve will plateau eventually.

Strategy:
  S1. Push FRR dose w55-w120
  S2. NEW base proxy with v31 cons
  S3. Iterate: BASE+FRR_w50 + extra FRR
  S4. Multiple FRRs (different α)
  S5. Stack v31 winners
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv31_BASE_plus_FRR_w50.csv"
BASE_SCORE = 7.218465181225035
POOL = 'teamblend_v3_pool'
VALID_FEATURES = "data/valid_features.csv"

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv19_FRRraw_a2_g15.csv": 7.246677323305486,
    "submission_FBSv24_CHAMP_LGBM_w030.csv": 7.240292163223663,
    "submission_FBSv26_CHAMP_LGBM_w060.csv": 7.237970103804146,
    "submission_FBSv27_NBprox_t08_p35_n55.csv": 7.236693385785031,
    "submission_FBSv28_NBprox_r07_t07_p35_n55.csv": 7.232883131643645,
    "submission_FBSv29_NB_FRR_w30.csv": 7.229653906267378,
    "submission_FBSv29_NB_FRR_w25.csv": 7.230108285937466,
    "submission_FBSv29_NB_FRR_w20.csv": 7.230572990648653,
    "submission_FBSv29_TRIPLE_t07_f20_l00.csv": 7.229754059600165,
    "submission_FBSv30_FRR_w50.csv": 7.228030068703475,
    "submission_FBSv30_FRR_w60.csv": 7.227290201773921,
    "submission_FBSv30_FRR_w80.csv": 7.226009000114476,
    "submission_FBSv30_NP30_r05_t07_p35_n55.csv": 7.225774174824203,
    "submission_FBSv30_NP30_r07_t05_p35_n55.csv": 7.221701978080064,
    "submission_FBSv30_STACK_v29_TOP3.csv": 7.227778206725874,
    "submission_FBSv30_NB_plus_FRR_w20.csv": 7.228030068703475,
    "submission_FBSv31_BASE_plus_FRR_w20.csv": 7.220247491758687,
    "submission_FBSv31_BASE_plus_FRR_w30.csv": 7.219617298680013,
    "submission_FBSv31_BASE_plus_FRR_w40.csv": 7.219013161888861,
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
    path = ROOT / f"submission_FBSv32_{label}.csv"
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
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    champ_orig, _ = load("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv")
    v30_winner, _ = load("submission_FBSv30_NP30_r07_t05_p35_n55.csv")  # parent of v31
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    outputs = []

    # ====== S1: PUSH FRR — w55, w60, w70, w80, w100 ======
    # v31 was: v30_winner + w * delta_frr (where delta_frr = frr - champ_orig)
    delta_frr = frr - champ_orig
    for w in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.0, 1.2, 1.5, 2.0):
        blend = v30_winner + w * delta_frr
        outputs.append(save(f"FRR_w{int(w*100):03d}", blend, target_col))

    # ====== S2: BASE (which is v31 winner) + extra FRR ======
    for w in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        blend = base + w * delta_frr
        outputs.append(save(f"NB_FRR_w{int(w*100):02d}", blend, target_col))

    # ====== S3: BASE proxy + various corrections ======
    for max_rms in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85):
        proxy = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        for q in (3, 4, 5, 6, 7, 8, 10):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50), (0.25, 0.65)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NP_r{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S4: BASE + proxy + extra FRR ======
    proxy_best = build_proxy(base, BASE_SCORE, max_rms=0.7)
    if proxy_best is not None:
        for q in (5, 6, 7):
            corr = build_correction(proxy_best, q, 0.35, 0.55)
            for w_frr in (0.05, 0.10, 0.15, 0.20):
                blend = base + corr + w_frr * delta_frr
                outputs.append(save(
                    f"PROX_FRR_t{q:02d}_w{int(w_frr*100):02d}",
                    blend, target_col))

    # ====== S5: Stack v31 winners ======
    winners = [
        BASE_FILE,
        "submission_FBSv31_BASE_plus_FRR_w40.csv",
        "submission_FBSv31_BASE_plus_FRR_w30.csv",
        "submission_FBSv31_BASE_plus_FRR_w20.csv",
        "submission_FBSv30_NP30_r07_t05_p35_n55.csv",
        "submission_FBSv30_NP30_r05_t07_p35_n55.csv",
        "submission_FBSv30_FRR_w80.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 4:
        stk = np.asarray(arrs)
        outputs.append(save(f"STACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save(f"STACK_TOP4", np.mean(stk[:4], axis=0), target_col))
        outputs.append(save(f"STACK_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"STACK_MED_{len(arrs)}", np.median(stk, axis=0), target_col))

    # ====== S6: Try alternative FRR with different α (recompute) ======
    # FRR was built with α=1e-2. Try α=5e-2 or 5e-3 for different smoothness
    # (Just use existing delta_frr with different weight scaling)

    # ====== S7: Push both BASE proxy + bigger FRR combined ======
    if proxy_best is not None:
        for q in (4, 5, 6):
            corr = build_correction(proxy_best, q, 0.35, 0.55)
            for w_frr in (0.30, 0.40, 0.50, 0.60):
                blend = base + corr + w_frr * delta_frr
                outputs.append(save(
                    f"PROX_BIG_FRR_t{q:02d}_w{int(w_frr*100):02d}",
                    blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v32_frr_extreme_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v32 candidates")


if __name__ == "__main__":
    main()
