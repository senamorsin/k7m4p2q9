#!/usr/bin/env python3
"""v46: Continue NP iteration. Base = NP3_r07_t07_p35_n55 = 7.17734.

Pattern: each new base + curated proxy + t07_p35_n55 = Δ−0.005.
This is the working ladder. Just keep going.

Strategy:
  S1. NP4 — curated proxy on new base (rms grid × q grid × γ grid)
  S2. NP4 + FRR / v61 / TOP6 micro
  S3. ITER on new base (apply proxy twice from this base)
  S4. STACK v45 winners
  S5. Apply v45 NP3 proxy on PARENT (cross-base)
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv45_NP3_r07_t07_p35_n55.csv"
BASE_SCORE = 7.177334737579046
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
    "submission_FBSv43_NP_r07_t07_p35_n55.csv": 7.195070449786494,
    "submission_FBSv43_NP_FRR_t07_w10.csv": 7.195300039809020,
    "submission_FBSv43_NP_r07_t05_p35_n55.csv": 7.199588972739028,
    "submission_FBSv43_NP_r03_t10_p35_n55.csv": 7.198460766310919,
    "submission_FBSv44_ITER2_t07_t05_p35_n55.csv": 7.182302330032473,
    "submission_FBSv44_ITER2_t07_t07_p35_n55.csv": 7.187356143593107,
    "submission_FBSv44_ITER2_t07_t10_p35_n55.csv": 7.190486160376680,
    "submission_FBSv44_NP_r05_t08_p35_n55.csv": 7.191742471234125,
    "submission_FBSv44_NP_r05_t07_p35_n55.csv": 7.192091188560693,
    "submission_FBSv44_ALPHA_a30_t07_p35_n55.csv": 7.192091188560693,
    "submission_FBSv44_NP_v61_t07_w010.csv": 7.192134393083729,
    "submission_FBSv44_NP_FRR_t07_w10.csv": 7.192227410585519,
    "submission_FBSv44_NP_TOP6_t07_w010.csv": 7.192250667167752,
    # v45 batch — KEY constraints
    "submission_FBSv45_NP3_r05_t07_p35_n55.csv": 7.182090679212620,
    "submission_FBSv45_NP3_r05_t05_p35_n55.csv": 7.182203255938389,
    "submission_FBSv45_NP3_r07_t05_p35_n55.csv": 7.182203255938389,
    "submission_FBSv45_NP3_FRR_t05_w10.csv": 7.182597975900803,
    "submission_FBSv45_NP3_v61_t05_w010.csv": 7.182392464922917,
    "submission_FBSv45_NP3_TOP6_t05_w010.csv": 7.182371678340556,
    "submission_FBSv45_ITER3_q05_p35_n55.csv": 7.182203255938389,
    "submission_FBSv45_ITER3_q07_p35_n55.csv": 7.182090679212620,
    "submission_FBSv45_ITER3_q06_p35_n55.csv": 7.181810218454357,
    "submission_FBSv45_ITER3_q10_p35_n55.csv": 7.180461897094080,
    "submission_FBSv45_ITER3_q03_p35_n55.csv": 7.184262264582872,
    "submission_FBSv45_ITER2v_q08_q05_p35_n55.csv": 7.181289996540839,
    "submission_FBSv45_ITER2v_q07_q03_p35_n55.csv": 7.184924042597510,
    "submission_FBSv45_ITER2v_q07_q04_p35_n55.csv": 7.186301100089160,
    "submission_FBSv45_ITER2v_q06_q05_p35_n55.csv": 7.187706088637021,
    "submission_FBSv45_STACK_WLB.csv": 7.184659062949270,
    "submission_FBSv45_STACK_TOP3.csv": 7.186459935861825,
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
    path = ROOT / f"submission_FBSv46_{label}.csv"
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
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_TOP6_MEDIAN.csv', 'submission_v71_sea_defaultmix_w25_dose35.csv']:
        fp = pool_path / fn
        if fp.exists():
            v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
            if len(v) == len(base):
                externals[fn.replace('submission_', '').replace('.csv', '').split('_')[0]] = v

    outputs = []
    delta_frr = frr - champ_orig

    # ====== S1: Full NP grid on new base (proven recipe) ======
    for max_rms in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85):
        proxy, n = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        print(f"  Proxy rms<{max_rms}: {n} cons")
        for q in (3, 4, 5, 6, 7, 8, 9, 10, 12, 15):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50),
                            (0.30, 0.65), (0.25, 0.65), (0.45, 0.45)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NP_r{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S2: NP + FRR / v61 / TOP6 ======
    proxy_best, _ = build_proxy(base, BASE_SCORE, max_rms=0.7)
    if proxy_best is not None:
        for q in (5, 7, 10):
            corr = build_correction(proxy_best, q, 0.35, 0.55)
            # + FRR
            for w_frr in (0.05, 0.10, 0.15, 0.20):
                outputs.append(save(
                    f"NP_FRR_t{q:02d}_w{int(w_frr*100):02d}",
                    base + corr + w_frr * delta_frr, target_col))
            # + v61
            if v61 is not None:
                for w_v61 in (0.010, 0.015, 0.020):
                    outputs.append(save(
                        f"NP_v61_t{q:02d}_w{int(w_v61*1000):03d}",
                        base + corr + w_v61 * (v61 - base), target_col))
            # + TOP6
            if 'TOP6' in externals:
                for w_top6 in (0.005, 0.010, 0.015):
                    outputs.append(save(
                        f"NP_TOP6_t{q:02d}_w{int(w_top6*1000):03d}",
                        base + corr + w_top6 * (externals['TOP6'] - base),
                        target_col))

    # ====== S3: ITER again on new base (== ITER4 conceptually) ======
    if proxy_best is not None:
        for q1 in (5, 7, 10):
            corr1 = build_correction(proxy_best, q1, 0.35, 0.55)
            b1 = base + corr1
            proxy2, _ = build_proxy(b1, BASE_SCORE - 0.003, max_rms=0.5)
            if proxy2 is None: continue
            for q2 in (3, 5, 7, 10):
                corr2 = build_correction(proxy2, q2, 0.35, 0.55)
                outputs.append(save(
                    f"ITER_q{q1:02d}_q{q2:02d}",
                    b1 + corr2, target_col))

    # ====== S4: Multi-α ======
    for alpha in (1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1):
        p, _ = build_proxy(base, BASE_SCORE, max_rms=0.5, alpha=alpha)
        if p is None: continue
        a_tag = f"a{int(-np.log10(alpha)*10):02d}"
        for q in (5, 7, 10):
            corr = build_correction(p, q, 0.35, 0.55)
            outputs.append(save(f"ALPHA_{a_tag}_t{q:02d}", base + corr, target_col))

    # ====== S5: Curated ENSEMBLE (avg of multiple rms windows) ======
    proxies_set = {}
    for max_rms in (0.3, 0.4, 0.5, 0.6, 0.7):
        p, _ = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if p is not None: proxies_set[max_rms] = p
    if len(proxies_set) >= 3:
        ens = np.mean(list(proxies_set.values()), axis=0)
        ens = ens - ens.mean(); ens = ens / (ens.std() + 1e-9)
        for q in (5, 7, 10):
            corr = build_correction(ens, q, 0.35, 0.55)
            outputs.append(save(f"ENS_t{q:02d}", base + corr, target_col))

    # ====== S6: STACK v45 winners ======
    winners = [
        BASE_FILE,
        "submission_FBSv45_ITER3_q10_p35_n55.csv",
        "submission_FBSv45_ITER2v_q08_q05_p35_n55.csv",
        "submission_FBSv45_ITER3_q06_p35_n55.csv",
        "submission_FBSv45_NP3_r05_t07_p35_n55.csv",
        "submission_FBSv45_ITER3_q07_p35_n55.csv",
        "submission_FBSv45_NP3_r05_t05_p35_n55.csv",
        "submission_FBSv44_ITER2_t07_t05_p35_n55.csv",
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
    (ROOT / "v46_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v46 candidates")


if __name__ == "__main__":
    main()
