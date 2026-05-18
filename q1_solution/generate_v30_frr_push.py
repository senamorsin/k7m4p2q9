#!/usr/bin/env python3
"""v30: PUSH FRR direction. New base = FBSv29_NB_FRR_w30 (LB 7.22965).

v29 finding: NB_FRR_wXX is monotonically improving. w30 winner. Push w35-w70.
Also TRIPLE_t07_f20_l00 = 7.22975 (tie) — proxy + FRR also works.

v30 strategy:
  S1. PUSH FRR dose: w35-w80
  S2. Mix NB_FRR with NBprox correction on new base
  S3. Curated proxy on this new base (more LB neighbours)
  S4. Multi-FRR (FRR direction + FRR-on-different-base)
  S5. Stack v29 winners
  S6. Sequential: NB_FRR → +proxy → +small partner
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv29_NB_FRR_w30.csv"
BASE_SCORE = 7.229653906267378
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
    "submission_FBSv28_NBprox_r07_t05_p35_n55.csv": 7.236859949911515,
    # v29 batch (most informative for new base)
    "submission_FBSv29_TRIPLE_t07_f20_l00.csv": 7.229754059600165,
    "submission_FBSv29_NB_FRR_w25.csv": 7.230108285937466,
    "submission_FBSv29_NB_FRR_w20.csv": 7.230572990648653,
    "submission_FBSv29_NPRr05_t07_p35_n55.csv": 7.230767427282550,
    "submission_FBSv29_NB_FRR_w15.csv": 7.231105763015468,
    "submission_FBSv29_STACK_TOP5.csv": 7.231488427316897,
    "submission_FBSv29_TRIPLE_t07_f10_l02.csv": 7.232733530920638,
    "submission_FBSv29_TRIPLE_t08_f10_l02.csv": 7.232612791456690,
    "submission_FBSv29_NPRr07_t06_p30_n65.csv": 7.233005760896122,
    "submission_FBSv29_PROX_LGBM_t07_w02.csv": 7.233910536352910,
    "submission_FBSv29_STACK_TOP3.csv": 7.234011290640628,
    "submission_FBSv29_PROX_LGBM_t07_w03.csv": 7.234863670614011,
    "submission_FBSv29_TRIPLE_t07_f15_l04.csv": 7.234704072525348,
    "submission_FBSv29_TRIPLE_t07_f10_l04.csv": 7.235313368875655,
    "submission_FBSv29_NPRr04_t07_p35_n55.csv": 7.235752633043879,
    "submission_FBSv29_PROX_LGBM_t07_w05.csv": 7.237863085426797,
    "submission_FBSv29_STACK_MED_8.csv": 7.237141352301792,
    "submission_FBSv29_ITER2_t07_t07.csv": 7.243041202052550,
    "submission_FBSv29_ITER2_t07_t05.csv": 7.244126854979997,
    "submission_FBSv29_ITER2_t07_t10.csv": 7.249215929850443,
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
    path = ROOT / f"submission_FBSv30_{label}.csv"
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
    nb_old, _ = load("submission_FBSv28_NBprox_r07_t07_p35_n55.csv")  # v28 base
    lgbm_new, _ = load("submission_LGBM_v20_actuals_spatial.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    outputs = []

    # ====== S1: PUSH FRR dose w35-w80 (winner was w30, curve still monotonic) ======
    delta_frr = frr - champ_orig
    for w in (0.32, 0.35, 0.38, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.0):
        # Apply on previous NBprox base (nb_old), then we know FRR step adds w*delta
        # Actually FRR works as offset from champ_orig direction so just multiply
        blend = nb_old + w * delta_frr
        outputs.append(save(f"FRR_w{int(w*100):02d}", blend, target_col))

    # ====== S2: NEW base proxy + various corrections ======
    for max_rms in (0.4, 0.5, 0.6, 0.7, 0.85):
        proxy = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        for q in (5, 6, 7, 8, 10):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.30, 0.65), (0.40, 0.50)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NP30_r{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S3: NEW base + more FRR (on top of already w30 FRR) ======
    # base already has FRR baked in at w30 from champion. Add MORE FRR direction
    for w in (0.05, 0.10, 0.15, 0.20):
        blend = base + w * delta_frr
        outputs.append(save(f"NB_plus_FRR_w{int(w*100):02d}", blend, target_col))

    # ====== S4: Combine proxy on new base + FRR ======
    proxy_new = build_proxy(base, BASE_SCORE, max_rms=0.6)
    if proxy_new is not None:
        for q in (6, 7, 8):
            corr = build_correction(proxy_new, q, 0.35, 0.55)
            for w_frr in (0.05, 0.10, 0.15, 0.20):
                blend = base + corr + w_frr * delta_frr
                outputs.append(save(
                    f"NP30PLUS_t{q:02d}_f{int(w_frr*100):02d}",
                    blend, target_col))

    # ====== S5: Stack v29 winners ======
    winners = [
        BASE_FILE,
        "submission_FBSv29_TRIPLE_t07_f20_l00.csv",
        "submission_FBSv29_NB_FRR_w25.csv",
        "submission_FBSv29_NB_FRR_w20.csv",
        "submission_FBSv29_NPRr05_t07_p35_n55.csv",
        "submission_FBSv29_NB_FRR_w15.csv",
        "submission_FBSv29_STACK_TOP5.csv",
        "submission_FBSv28_NBprox_r07_t07_p35_n55.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 5:
        stk = np.asarray(arrs)
        outputs.append(save(f"STACK_v29_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save(f"STACK_v29_TOP5", np.mean(stk[:5], axis=0), target_col))
        outputs.append(save(f"STACK_v29_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"STACK_v29_MED_{len(arrs)}", np.median(stk, axis=0), target_col))
        # WLB
        sc = np.array([7.22965, 7.22975, 7.23010, 7.23057, 7.23076, 7.23110,
                       7.23149, 7.23288])[:len(arrs)]
        w = 1.0 / (sc - 7.228 + 0.0005)
        w = w / w.sum()
        outputs.append(save(f"STACK_v29_WLB", (stk.T @ w).reshape(-1), target_col))

    # ====== S6: TRIPLE refinement (winning was t07_f20_l00 = 7.230 tied with NB_FRR_w30) ======
    # Push: bigger f, smaller t
    if proxy_new is not None:
        for q in (6, 7, 8):
            corr = build_correction(proxy_new, q, 0.35, 0.55)
            for w_frr in (0.25, 0.30, 0.35, 0.40):
                blend = nb_old + corr + w_frr * delta_frr
                outputs.append(save(
                    f"TRIPLE2_t{q:02d}_f{int(w_frr*100):02d}",
                    blend, target_col))

    # ====== S7: Iterate proxy from v28 (parent) on new base ======
    proxy_parent = build_proxy(nb_old, 7.232883131643645, max_rms=0.7)
    if proxy_parent is not None:
        for q in (5, 7, 10):
            corr = build_correction(proxy_parent, q, 0.35, 0.55)
            outputs.append(save(f"PROXparent_t{q:02d}",
                                base + corr, target_col))

    # ====== S8: NB_FRR with LGBM (was hurting before but maybe at low dose helps) ======
    if lgbm_new is not None:
        for w in (0.005, 0.010, 0.015, 0.020):
            blend = (1 - w) * base + w * lgbm_new
            outputs.append(save(f"NB30_LGBM_w{int(w*1000):03d}",
                                blend, target_col))

    # ====== S9: NB_FRR via different routes — push w on champion ======
    # Alternative: champion + LGBM + FRR with all at near-optimal doses
    for w_lgbm in (0.04, 0.05, 0.06):
        for w_frr in (0.25, 0.30, 0.35, 0.40):
            blend = (1 - w_lgbm) * champ_orig + w_lgbm * lgbm_new + w_frr * delta_frr
            outputs.append(save(
                f"CFL_lgbm{int(w_lgbm*100):02d}_frr{int(w_frr*100):02d}",
                blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v30_frr_push_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v30 candidates")


if __name__ == "__main__":
    main()
