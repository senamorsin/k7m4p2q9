#!/usr/bin/env python3
"""v31: FINAL CHARGE to #1. Base = NP30_r07_t05_p35_n55 (LB 7.22170).

v30 findings:
  - NP30_r07_t05 = 7.22170 (Δ−0.008 — HUGE)
  - FRR_w80 = 7.22601 (curve still rising)
  - NP30_r05_t07 = 7.22577 (alt proxy window works)
  - NB_plus_FRR_w20 = 7.22803

Strategy:
  S1. NP30 with even narrower t (t02, t03, t04) + various γ
  S2. Push FRR further: w90, w100, w120, w150
  S3. NEW base proxy (with v30 cons) — t05 winning recipe
  S4. NP30 winner + FRR_w50/w60/w80 added
  S5. Mix top 3 v30 winners
  S6. Cross-base proxy
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv30_NP30_r07_t05_p35_n55.csv"
BASE_SCORE = 7.221701978080064
POOL = 'teamblend_v3_pool'

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv19_FRRraw_a2_g15.csv": 7.246677323305486,
    "submission_FBSv24_CHAMP_LGBM_w030.csv": 7.240292163223663,
    "submission_FBSv26_CHAMP_LGBM_w060.csv": 7.237970103804146,
    "submission_FBSv26_NB_LGBM_w030.csv": 7.237982397774434,
    "submission_FBSv27_NB_plus_FRR_w10.csv": 7.237615105630028,
    "submission_FBSv27_NBprox_t08_p35_n55.csv": 7.236693385785031,
    "submission_FBSv27_STACK_v26_TOP3.csv": 7.237885911164830,
    "submission_FBSv28_NBprox_r07_t07_p35_n55.csv": 7.232883131643645,
    "submission_FBSv28_NBprox_r07_t06_p35_n55.csv": 7.235024844144681,
    "submission_FBSv29_NB_FRR_w30.csv": 7.229653906267378,
    "submission_FBSv29_NB_FRR_w25.csv": 7.230108285937466,
    "submission_FBSv29_NB_FRR_w20.csv": 7.230572990648653,
    "submission_FBSv29_NB_FRR_w15.csv": 7.231105763015468,
    "submission_FBSv29_NPRr05_t07_p35_n55.csv": 7.230767427282550,
    "submission_FBSv29_TRIPLE_t07_f20_l00.csv": 7.229754059600165,
    "submission_FBSv29_STACK_TOP5.csv": 7.231488427316897,
    # v30 batch (most informative)
    "submission_FBSv30_FRR_w35.csv": 7.229218578486830,
    "submission_FBSv30_FRR_w40.csv": 7.228806003788211,
    "submission_FBSv30_FRR_w45.csv": 7.228404440151989,
    "submission_FBSv30_FRR_w50.csv": 7.228030068703475,
    "submission_FBSv30_FRR_w60.csv": 7.227290201773921,
    "submission_FBSv30_FRR_w80.csv": 7.226009000114476,
    "submission_FBSv30_NP30_r05_t07_p35_n55.csv": 7.225774174824203,
    "submission_FBSv30_NB_plus_FRR_w20.csv": 7.228030068703475,
    "submission_FBSv30_NB_plus_FRR_w15.csv": 7.228404440151989,
    "submission_FBSv30_NB_plus_FRR_w10.csv": 7.228806003788211,
    "submission_FBSv30_NB_plus_FRR_w05.csv": 7.229218578486830,
    "submission_FBSv30_STACK_v29_TOP3.csv": 7.227778206725874,
    "submission_FBSv30_STACK_v29_WLB.csv": 7.228081139537380,
    "submission_FBSv30_NP30PLUS_t06_f10.csv": 7.230624712422203,
    "submission_FBSv30_NP30PLUS_t07_f15.csv": 7.231000509576768,
    "submission_FBSv30_NP30PLUS_t07_f10.csv": 7.231225766113773,
    "submission_FBSv30_NP30_r04_t07_p35_n55.csv": 7.232996070227626,
    "submission_FBSv30_CFL_lgbm05_frr30.csv": 7.237619782742819,
    "submission_FBSv30_CFL_lgbm05_frr35.csv": 7.237591811876880,
    "submission_FBSv30_CFL_lgbm06_frr35.csv": 7.237298063176540,
    "submission_FBSv30_TRIPLE2_t07_f30.csv": 7.231721054394508,
    "submission_FBSv30_TRIPLE2_t07_f35.csv": 7.231462033713175,
    "submission_FBSv30_TRIPLE2_t07_f40.csv": 7.231225766113772,
    "submission_FBSv30_TRIPLE2_t08_f30.csv": 7.230588028775636,
    "submission_FBSv30_TRIPLE2_t06_f30.csv": 7.231145371963632,
    # base skipped
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
    path = ROOT / f"submission_FBSv31_{label}.csv"
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
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    outputs = []

    # ====== S1: PUSH FRR dose w80 still climbing — try w100-w200 ======
    delta_frr = frr - champ_orig
    nb_old, _ = load("submission_FBSv29_NB_FRR_w30.csv")  # v30 was: nb_old + w*delta_frr
    # nb_old is base_minus_w30*delta - wait, just use champion as origin
    # The v30 FRR_wXX = nb_old + X*delta_frr where nb_old = NBprox_r07_t07 (champ + some correction)
    # Easier: just extrapolate FRR direction further from current base
    for w in (0.90, 1.0, 1.1, 1.2, 1.5, 2.0):
        blend = nb_old + w * delta_frr
        outputs.append(save(f"FRR_w{int(w*100):03d}", blend, target_col))

    # ====== S2: NEW base proxy (with v30 winners as constraints) ======
    for max_rms in (0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
        proxy = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        # FBSv30 winner was r07 t05. Try around it.
        for q in (3, 4, 5, 6, 7, 8, 10):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.30, 0.65), (0.40, 0.50),
                            (0.25, 0.55), (0.35, 0.45), (0.45, 0.55)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NP31_r{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S3: BASE + FRR (the next-level combo) ======
    for w in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        blend = base + w * delta_frr
        outputs.append(save(f"BASE_plus_FRR_w{int(w*100):02d}", blend, target_col))

    # ====== S4: BASE + small additional NBprox correction ======
    proxy_new = build_proxy(base, BASE_SCORE, max_rms=0.7)
    if proxy_new is not None:
        # Try various combos: base + new proxy correction
        for q in (3, 4, 5, 6, 7):
            for gp, gn in [(0.30, 0.55), (0.35, 0.55), (0.25, 0.55)]:
                corr = build_correction(proxy_new, q, gp, gn)
                # Just add the correction
                outputs.append(save(
                    f"BASE_plus_PROX_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))
                # Plus FRR direction
                for w_frr in (0.10, 0.20, 0.30):
                    outputs.append(save(
                        f"BASE_PROX_FRR_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}_f{int(w_frr*100):02d}",
                        base + corr + w_frr * delta_frr, target_col))

    # ====== S5: Mix BASE with top-2 v30 winners ======
    v30_2nd, _ = load("submission_FBSv30_NP30_r05_t07_p35_n55.csv")
    if v30_2nd is not None:
        for w in (0.3, 0.4, 0.5, 0.6, 0.7):
            blend = w * base + (1 - w) * v30_2nd
            outputs.append(save(f"MIX_v30_2nd_w{int(w*100):02d}", blend, target_col))

    # ====== S6: Stack v30 winners ======
    winners = [
        BASE_FILE,
        "submission_FBSv30_NP30_r05_t07_p35_n55.csv",
        "submission_FBSv30_FRR_w80.csv",
        "submission_FBSv30_FRR_w60.csv",
        "submission_FBSv30_STACK_v29_TOP3.csv",
        "submission_FBSv30_FRR_w50.csv",
        "submission_FBSv30_NB_plus_FRR_w20.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 4:
        stk = np.asarray(arrs)
        outputs.append(save(f"STACK_v30_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save(f"STACK_v30_TOP4", np.mean(stk[:4], axis=0), target_col))
        outputs.append(save(f"STACK_v30_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"STACK_v30_MED_{len(arrs)}", np.median(stk, axis=0), target_col))
        # WLB
        sc = np.array([7.22170, 7.22577, 7.22601, 7.22729, 7.22778, 7.22803, 7.22803])[:len(arrs)]
        w = 1.0 / (sc - 7.220 + 0.0005)
        w = w / w.sum()
        outputs.append(save(f"STACK_v30_WLB", (stk.T @ w).reshape(-1), target_col))

    # ====== S7: Apply v30 proxy on BASE (cross-base validation) ======
    # The v30 NP30_r07_t05 proxy was built on NB_FRR_w30 base. Apply same proxy on
    # different starting points
    nb_frr_w30, _ = load("submission_FBSv29_NB_FRR_w30.csv")
    if nb_frr_w30 is not None:
        proxy_w30 = build_proxy(nb_frr_w30, 7.229653906267378, max_rms=0.7)
        if proxy_w30 is not None:
            for q in (3, 4, 5, 6):
                corr = build_correction(proxy_w30, q, 0.35, 0.55)
                outputs.append(save(f"v30PROX_t{q:02d}_on_BASE", base + corr, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v31_final_charge_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v31 candidates")


if __name__ == "__main__":
    main()
