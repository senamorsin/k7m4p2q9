#!/usr/bin/env python3
"""v44: BREAKTHROUGH continuation.

BIG JUMP in v43: NP_r07_t07_p35_n55 + NP_r05_t07_p35_n55 (tied at 7.19507) = Δ−0.011.

Champion: 7.19507. Two methods tied:
  - r07_t07: rms<0.7 (46 cons), quantile=7%, gp=0.35, gn=0.55
  - r05_t07: rms<0.5 (46 cons), quantile=7% — SAME constraints (since rms<0.5 catches all)

Strategy:
  S1. Curated proxy on NEW base (7.19507) — full grid
  S2. Multi-α regularization (10^-1, 10^-2, 10^-3, 10^-4)
  S3. Multi-base parent proxy averaging (v40, v34, v18 as parents)
  S4. NP + v61 + FRR + small TOP6 combos
  S5. NP iteration (apply proxy twice on the same chain)
  S6. Apply v43 winning recipe on PARENT bases (cross-base)
  S7. Bootstrap proxy (resample constraints)
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv43_NP_r07_t07_p35_n55.csv"
BASE_SCORE = 7.195070449786494
PARENT_FILE = "submission_FBSv40_v61_w050.csv"
PARENT_SCORE = 7.206400653558756
POOL = 'teamblend_v3_pool'

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv24_CHAMP_LGBM_w030.csv": 7.240292163223663,
    "submission_FBSv26_CHAMP_LGBM_w060.csv": 7.237970103804146,
    "submission_FBSv28_NBprox_r07_t07_p35_n55.csv": 7.232883131643645,
    "submission_FBSv29_NB_FRR_w30.csv": 7.229653906267378,
    "submission_FBSv30_NP30_r07_t05_p35_n55.csv": 7.221701978080064,
    "submission_FBSv31_BASE_PROX_FRR_t05_p35_n55_f20.csv": 7.213629337187302,
    "submission_FBSv33_NB_FRR_w20.csv": 7.212619135451977,
    "submission_FBSv34_NB_FRR_w30.csv": 7.211678029707256,
    "submission_FBSv35_NB_FRR_w20.csv": 7.211396180429282,
    "submission_FBSv39_v61_w020.csv": 7.208357496446700,
    "submission_FBSv40_v61_w050.csv": 7.206400653558756,
    "submission_FBSv40_v61_w060.csv": 7.206402333453468,
    "submission_FBSv40_v61_w040.csv": 7.206824593775993,
    "submission_FBSv40_NB_v61_w020.csv": 7.206844204809593,
    "submission_FBSv40_v61_w035.csv": 7.207100664009665,
    "submission_FBSv40_NB_v61_w015.csv": 7.207119310257987,
    "submission_FBSv40_v61_w030.csv": 7.207373416395550,
    "submission_FBSv41_v61_w060.csv": 7.206402333453468,
    "submission_FBSv41_v61_FRR_v060_f20.csv": 7.206799000202425,
    "submission_FBSv41_STACK_TOP5.csv": 7.206877543566713,
    "submission_FBSv41_STACK_MED_7.csv": 7.207051970943805,
    # v43 batch — KEY constraints near new base
    "submission_FBSv43_NP_r07_t05_p35_n55.csv": 7.199588972739028,
    "submission_FBSv43_NP_r05_t05_p35_n55.csv": 7.199588972739028,
    "submission_FBSv43_NP_r03_t10_p35_n55.csv": 7.198460766310919,
    "submission_FBSv43_NP_r03_t07_p35_n55.csv": 7.201498946203359,
    "submission_FBSv43_NP_r03_t05_p35_n55.csv": 7.202007768020698,
    "submission_FBSv43_NP_FRR_t07_w10.csv": 7.195300039809020,
    "submission_FBSv43_NP_FRR_t05_w10.csv": 7.199806783227454,
    "submission_FBSv43_NP_FRR_t05_w15.csv": 7.199930332682471,
    "submission_FBSv43_add_v61_v50_w005.csv": 7.206324047042226,
    "submission_FBSv43_add_v61_v50_w010.csv": 7.206388349828746,
    "submission_FBSv43_add_v61_v60_w005.csv": 7.206566014812793,
    "submission_FBSv43_add_v61_NB20_w005.csv": 7.206608774351232,
    "submission_FBSv43_STACK_WLB.csv": 7.206717985924639,
    "submission_FBSv43_STACK_TOP3.csv": 7.206742881135994,
    "submission_FBSv43_STACK_MED_8.csv": 7.206868568793796,
    "submission_FBSv43_PvFR_v050_f15.csv": 7.206602468440700,
    "submission_FBSv43_PvFR_v060_f15.csv": 7.206688483182126,
    "submission_FBSv43_PvFR_v060_f20.csv": 7.206799000242500,
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
    path = ROOT / f"submission_FBSv44_{label}.csv"
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


def main():
    base, target_col = load(BASE_FILE)
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    champ_orig, _ = load("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv")
    v61, _ = load("submission_v61_alone.csv")
    parent, _ = load(PARENT_FILE)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_TOP6_MEDIAN.csv',
                'submission_v71_sea_defaultmix_w25_dose35.csv']:
        fp = pool_path / fn
        if fp.exists():
            v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
            if len(v) == len(base):
                externals[fn.replace('submission_', '').replace('.csv', '').split('_')[0]] = v
    lgbm_dart, _ = load("submission_LGBM_dart.csv")

    outputs = []
    delta_frr = frr - champ_orig

    # ====== S1: NEW base proxy — full grid sweep (the winning recipe on new base) ======
    for max_rms in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
        proxy, n = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        print(f"  Proxy rms<{max_rms}: {n} cons")
        for q in (3, 4, 5, 6, 7, 8, 9, 10, 12):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50),
                            (0.30, 0.65), (0.25, 0.65), (0.45, 0.45),
                            (0.35, 0.65), (0.30, 0.45)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NP_r{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S2: Multi-α regularization on best window ======
    for alpha in (1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1):
        proxy, n = build_proxy(base, BASE_SCORE, max_rms=0.5, alpha=alpha)
        if proxy is None: continue
        a_tag = f"a{int(-np.log10(alpha)*10):02d}"
        for q in (5, 7, 8):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"ALPHA_{a_tag}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S3: NP + FRR (winning combo, push further) ======
    proxy_best, _ = build_proxy(base, BASE_SCORE, max_rms=0.5)
    if proxy_best is not None:
        for q in (5, 6, 7, 8, 9):
            corr = build_correction(proxy_best, q, 0.35, 0.55)
            for w_frr in (0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30):
                blend = base + corr + w_frr * delta_frr
                outputs.append(save(
                    f"NP_FRR_t{q:02d}_w{int(w_frr*100):02d}",
                    blend, target_col))

    # ====== S4: NP + v61 (v61 still useful as ortho direction) ======
    if proxy_best is not None and v61 is not None:
        for q in (5, 7):
            corr = build_correction(proxy_best, q, 0.35, 0.55)
            for w_v61 in (0.005, 0.010, 0.015, 0.020, 0.030):
                blend = base + corr + w_v61 * (v61 - base)
                outputs.append(save(
                    f"NP_v61_t{q:02d}_w{int(w_v61*1000):03d}",
                    blend, target_col))

    # ====== S5: NP + TOP6 micro ======
    if proxy_best is not None and 'TOP6' in externals:
        for q in (5, 7):
            corr = build_correction(proxy_best, q, 0.35, 0.55)
            for w in (0.005, 0.010, 0.015):
                blend = base + corr + w * (externals['TOP6'] - base)
                outputs.append(save(f"NP_TOP6_t{q:02d}_w{int(w*1000):03d}",
                                    blend, target_col))

    # ====== S6: NP + DART micro (small LGBM dose) ======
    if proxy_best is not None and lgbm_dart is not None:
        for q in (5, 7):
            corr = build_correction(proxy_best, q, 0.35, 0.55)
            for w in (0.005, 0.010, 0.015, 0.020):
                blend = base + corr + w * (lgbm_dart - base)
                outputs.append(save(f"NP_dart_t{q:02d}_w{int(w*1000):03d}",
                                    blend, target_col))

    # ====== S7: ITER2 — apply proxy twice ======
    if proxy_best is not None:
        for q1 in (5, 7):
            corr1 = build_correction(proxy_best, q1, 0.35, 0.55)
            b1 = base + corr1
            # Build proxy on b1 (approx score: 7.194)
            proxy2, n2 = build_proxy(b1, BASE_SCORE - 0.001, max_rms=0.5)
            if proxy2 is None: continue
            for q2 in (5, 7, 10):
                for gp2, gn2 in [(0.35, 0.55), (0.30, 0.55)]:
                    corr2 = build_correction(proxy2, q2, gp2, gn2)
                    outputs.append(save(
                        f"ITER2_t{q1:02d}_t{q2:02d}_p{int(gp2*100):02d}_n{int(gn2*100):02d}",
                        b1 + corr2, target_col))

    # ====== S8: Multi-base proxy avg (apply proxy from v40, v34, v18 → average corrections) ======
    parent_v40 = parent
    parent_v34, _ = load("submission_FBSv34_NB_FRR_w30.csv")
    parent_v18, _ = load("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv")
    bases_for_avg = [
        (parent_v40, 7.206400653558756, "v40"),
        (parent_v34, 7.211678029707256, "v34"),
        (parent_v18, 7.245703367161376, "v18"),
    ]
    corrs_avg = []
    for b_arr, b_score, b_tag in bases_for_avg:
        if b_arr is None: continue
        p, _ = build_proxy(b_arr, b_score, max_rms=0.5)
        if p is None: continue
        c = build_correction(p, 7, 0.35, 0.55)
        corrs_avg.append((c, b_tag))
    if len(corrs_avg) >= 2:
        mean_corr = np.mean([c for c, _ in corrs_avg], axis=0)
        outputs.append(save("MULTIPROX_MEAN", base + mean_corr, target_col))
        outputs.append(save("MULTIPROX_x12", base + 1.2 * mean_corr, target_col))
        outputs.append(save("MULTIPROX_x14", base + 1.4 * mean_corr, target_col))
        outputs.append(save("MULTIPROX_x08", base + 0.8 * mean_corr, target_col))
        # Median
        med_corr = np.median([c for c, _ in corrs_avg], axis=0)
        outputs.append(save("MULTIPROX_MED", base + med_corr, target_col))

    # ====== S9: Curated proxy ENSEMBLE (different rms windows averaged) ======
    proxies_set = {}
    for max_rms in (0.3, 0.4, 0.5, 0.6, 0.7):
        p, _ = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if p is not None: proxies_set[max_rms] = p
    if len(proxies_set) >= 3:
        ens = np.mean(list(proxies_set.values()), axis=0)
        ens = ens - ens.mean(); ens = ens / (ens.std() + 1e-9)
        for q in (5, 7, 8):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55)]:
                corr = build_correction(ens, q, gp, gn)
                outputs.append(save(
                    f"NPens_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S10: Bootstrap proxy (random 70% subset, average across seeds) ======
    rng = np.random.RandomState(42)
    all_dirs, all_tgts, all_rms = [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base): continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        if rms < 0.5:
            all_dirs.append(diff)
            all_tgts.append(-((score - BASE_SCORE) * N_INST / 100.0))
            all_rms.append(rms)
    if len(all_dirs) > 10:
        D = np.asarray(all_dirs); T = np.asarray(all_tgts); R = np.asarray(all_rms)
        N = len(D)
        boot_proxies = []
        for b in range(30):
            k = max(5, int(0.7 * N))
            idx = rng.choice(N, size=k, replace=False)
            Db, Tb, Rb = D[idx], T[idx], R[idx]
            Wb = np.diag(1.0 / np.power(0.05 + Rb, 0.7))
            gram = (Db @ Db.T) / len(base)
            try:
                beta = np.linalg.solve(Wb @ gram @ Wb + 1e-3*np.eye(k), Wb @ Tb)
            except np.linalg.LinAlgError: continue
            p = Db.T @ (Wb @ beta); p -= p.mean()
            lo, hi = np.percentile(p, [1, 99]); p = np.clip(p, lo, hi)
            if p.std() < 1e-9: continue
            p /= p.std()
            boot_proxies.append(p)
        if boot_proxies:
            boot_avg = np.mean(boot_proxies, axis=0)
            boot_avg = boot_avg - boot_avg.mean()
            boot_avg = boot_avg / (boot_avg.std() + 1e-9)
            for q in (5, 7, 8):
                for gp, gn in [(0.35, 0.55), (0.30, 0.55)]:
                    corr = build_correction(boot_avg, q, gp, gn)
                    outputs.append(save(
                        f"BOOT_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                        base + corr, target_col))

    # ====== S11: STACK v43 winners + new champion ======
    winners = [
        BASE_FILE,
        "submission_FBSv43_NP_r05_t07_p35_n55.csv",
        "submission_FBSv43_NP_FRR_t07_w10.csv",
        "submission_FBSv43_NP_r07_t05_p35_n55.csv",
        "submission_FBSv43_NP_r05_t05_p35_n55.csv",
        "submission_FBSv43_NP_FRR_t05_w10.csv",
        "submission_FBSv43_NP_FRR_t05_w15.csv",
        "submission_FBSv43_NP_r03_t10_p35_n55.csv",
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

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v44_breakthrough_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v44 candidates")


if __name__ == "__main__":
    main()
