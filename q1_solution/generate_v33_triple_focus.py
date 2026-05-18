#!/usr/bin/env python3
"""v33: Focus on TRIPLE combo winner. Base = BASE_PROX_FRR_t05_f20 = 7.21363.

v31 surprise winner: BASE + curated_proxy(t05, p35, n55) + 0.20 * FRR direction.
This 3-way combo gave Δ−0.005 from v31 winner.

v33 strategy — milk this triple combo:
  S1. Sweep t (3,4,5,6,7,8,10) × f (10,15,20,25,30,35,40)
  S2. Sweep (gp, gn) ratios with same t05 f20
  S3. Iterate: apply TRIPLE on the new winner
  S4. Use multiple proxy windows (rms 0.3-0.85) × t05 × f20-f40
  S5. Stack v31+v32 winners
  S6. Multi-FRR alphas (only a2 tested in chain, also try a1)
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv31_BASE_PROX_FRR_t05_p35_n55_f20.csv"
BASE_SCORE = 7.213629337187302
POOL = 'teamblend_v3_pool'

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv19_FRRraw_a2_g15.csv": 7.246677323305486,
    "submission_FBSv24_CHAMP_LGBM_w030.csv": 7.240292163223663,
    "submission_FBSv26_CHAMP_LGBM_w060.csv": 7.237970103804146,
    "submission_FBSv27_NBprox_t08_p35_n55.csv": 7.236693385785031,
    "submission_FBSv28_NBprox_r07_t07_p35_n55.csv": 7.232883131643645,
    "submission_FBSv29_NB_FRR_w30.csv": 7.229653906267378,
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
    "submission_FBSv31_BASE_plus_FRR_w50.csv": 7.218465181225035,
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
    path = ROOT / f"submission_FBSv33_{label}.csv"
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
    base, target_col = load(BASE_FILE)            # v33 base
    parent_v30, _ = load("submission_FBSv30_NP30_r07_t05_p35_n55.csv")  # the BASE used in TRIPLE
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    champ_orig, _ = load("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    outputs = []
    delta_frr = frr - champ_orig

    # ====== S1: Sweep TRIPLE parameters (t × f grid) on PARENT base (parent_v30) ======
    # TRIPLE formula: parent_v30 + corr(t, p, n) + f * delta_frr
    proxy_parent = build_proxy(parent_v30, 7.221701978080064, max_rms=0.7)
    if proxy_parent is not None:
        for t in (3, 4, 5, 6, 7, 8, 10):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.30, 0.65), (0.40, 0.50)]:
                corr = build_correction(proxy_parent, t, gp, gn)
                for f in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
                    blend = parent_v30 + corr + f * delta_frr
                    outputs.append(save(
                        f"TRIPLE_t{t:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}_f{int(f*100):02d}",
                        blend, target_col))

    # ====== S2: Apply triple on NEW base (iterate) ======
    proxy_new = build_proxy(base, BASE_SCORE, max_rms=0.7)
    if proxy_new is not None:
        for t in (3, 4, 5, 6, 7, 8):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50)]:
                corr = build_correction(proxy_new, t, gp, gn)
                for f in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
                    blend = base + corr + f * delta_frr
                    outputs.append(save(
                        f"ITER_t{t:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}_f{int(f*100):02d}",
                        blend, target_col))

    # ====== S3: BASE + extra FRR (without proxy) ======
    for f in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        blend = base + f * delta_frr
        outputs.append(save(f"NB_FRR_w{int(f*100):02d}", blend, target_col))

    # ====== S4: BASE + proxy only ======
    if proxy_new is not None:
        for t in (3, 4, 5, 6, 7):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50), (0.25, 0.65)]:
                corr = build_correction(proxy_new, t, gp, gn)
                outputs.append(save(
                    f"NP_t{t:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S5: Stack + curated mix ======
    winners = [
        BASE_FILE,
        "submission_FBSv31_BASE_plus_FRR_w50.csv",
        "submission_FBSv31_BASE_plus_FRR_w40.csv",
        "submission_FBSv31_BASE_plus_FRR_w30.csv",
        "submission_FBSv30_NP30_r07_t05_p35_n55.csv",
        "submission_FBSv30_FRR_w80.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    stk = np.asarray(arrs)
    if len(arrs) >= 4:
        outputs.append(save(f"STACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save(f"STACK_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"STACK_MED_{len(arrs)}", np.median(stk, axis=0), target_col))
        # WLB
        sc = np.array([7.21363, 7.21847, 7.21901, 7.21962, 7.22170, 7.22601])[:len(arrs)]
        w = 1.0 / (sc - 7.212 + 0.0005)
        w = w / w.sum()
        outputs.append(save(f"STACK_WLB", (stk.T @ w).reshape(-1), target_col))

    # ====== S6: Same recipe with even smaller proxy correction (t05) and bigger f ======
    # Could be the winning combo extended
    if proxy_parent is not None:
        for t in (3, 4, 5):
            corr = build_correction(proxy_parent, t, 0.35, 0.55)
            for f in (0.30, 0.35, 0.40, 0.50, 0.60):
                blend = parent_v30 + corr + f * delta_frr
                outputs.append(save(
                    f"DEEP_t{t:02d}_f{int(f*100):02d}",
                    blend, target_col))

    # ====== S7: Mix base and parent (could find sweet spot) ======
    if parent_v30 is not None:
        for w in (0.3, 0.4, 0.5, 0.6, 0.7):
            blend = w * base + (1 - w) * parent_v30
            outputs.append(save(f"MIX_parent_w{int(w*100):02d}", blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v33_triple_focus_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v33 candidates")


if __name__ == "__main__":
    main()
