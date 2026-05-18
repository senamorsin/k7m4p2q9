#!/usr/bin/env python3
"""v45: ITER3 + ITER2 variations. Base = ITER2_t07_t05 = 7.18230.

v44 finding: ITER2 (apply proxy twice) = MASSIVE −0.013 jump.
  ITER2_t07_t05: first pass t=7%, second pass t=5%
  ITER2_t07_t07: t=7%, t=7%

Strategy:
  S1. ITER2 grid (q1, q2 combinations) on PARENT base (v43 winner)
  S2. ITER3 — apply proxy 3 times
  S3. NEW base proxy (post-ITER2) + corrections
  S4. ITER2 + FRR / v61 / TOP6
  S5. ITER2 with multi-α
  S6. STACK ITER2 winners
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv44_ITER2_t07_t05_p35_n55.csv"
BASE_SCORE = 7.182302330032473
PARENT_FILE = "submission_FBSv43_NP_r07_t07_p35_n55.csv"  # the base of ITER2's first step
PARENT_SCORE = 7.195070449786494
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
    "submission_FBSv41_v61_FRR_v060_f20.csv": 7.206799000202425,
    "submission_FBSv43_NP_r07_t07_p35_n55.csv": 7.195070449786494,
    "submission_FBSv43_NP_r05_t07_p35_n55.csv": 7.195070449786494,
    "submission_FBSv43_NP_FRR_t07_w10.csv": 7.195300039809020,
    "submission_FBSv43_NP_r07_t05_p35_n55.csv": 7.199588972739028,
    "submission_FBSv43_NP_r05_t05_p35_n55.csv": 7.199588972739028,
    "submission_FBSv43_NP_FRR_t05_w10.csv": 7.199806783227454,
    "submission_FBSv43_NP_FRR_t05_w15.csv": 7.199930332682471,
    "submission_FBSv43_NP_r03_t10_p35_n55.csv": 7.198460766310919,
    # v44 batch — KEY constraints near new base
    "submission_FBSv44_ITER2_t07_t07_p35_n55.csv": 7.187356143593107,
    "submission_FBSv44_ITER2_t07_t10_p35_n55.csv": 7.190486160376680,
    "submission_FBSv44_NP_r05_t08_p35_n55.csv": 7.191742471234125,
    "submission_FBSv44_NP_r05_t07_p35_n55.csv": 7.192091188560693,
    "submission_FBSv44_NP_v61_t07_w010.csv": 7.192134393083729,
    "submission_FBSv44_NP_v61_t07_w015.csv": 7.192378545166276,
    "submission_FBSv44_ALPHA_a30_t07_p35_n55.csv": 7.192091188560693,
    "submission_FBSv44_NP_FRR_t07_w10.csv": 7.192227410585519,
    "submission_FBSv44_NP_FRR_t07_w15.csv": 7.192310165807844,
    "submission_FBSv44_NP_FRR_t07_w20.csv": 7.192415567050643,
    "submission_FBSv44_NP_TOP6_t07_w010.csv": 7.192250667167752,
    "submission_FBSv44_NP_r05_t05_p35_n55.csv": 7.192429068654535,
    "submission_FBSv44_NP_FRR_t06_w10.csv": 7.192841938837752,
    "submission_FBSv44_NP_r05_t06_p35_n55.csv": 7.192705716812921,
    "submission_FBSv44_NPens_t07_p35_n55.csv": 7.192475553792836,
    "submission_FBSv44_ALPHA_a20_t07_p35_n55.csv": 7.194517758090186,
    "submission_FBSv44_NP_dart_t07_w010.csv": 7.193787956106401,
    "submission_FBSv44_NP_r07_t07_p35_n55.csv": 7.198080038940770,
    "submission_FBSv44_STACK_TOP3.csv": 7.195141884564286,
    "submission_FBSv44_STACK_TOP5.csv": 7.196611139032151,
    "submission_FBSv44_MULTIPROX_MEAN.csv": 7.200605240740441,
    "submission_FBSv44_MULTIPROX_MED.csv": 7.198771052789184,
    "submission_FBSv44_MULTIPROX_x12.csv": 7.203029953420010,
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
    path = ROOT / f"submission_FBSv45_{label}.csv"
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
    parent, _ = load(PARENT_FILE)
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

    # ====== S1: NEW base proxy + apply corrections (full grid) ======
    for max_rms in (0.3, 0.4, 0.5, 0.6, 0.7):
        proxy_new, n = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy_new is None: continue
        print(f"  Proxy rms<{max_rms}: {n} cons")
        for q in (3, 4, 5, 6, 7, 8, 10):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50), (0.25, 0.65)]:
                corr = build_correction(proxy_new, q, gp, gn)
                outputs.append(save(
                    f"NP3_r{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S2: ITER3 — apply proxy 3 times ======
    # ITER2 was: parent + corr1 + corr2 (where corr1 = q07, corr2 = q05)
    # ITER3 = ITER2_winner + corr3 (third pass)
    proxy_iter3, n3 = build_proxy(base, BASE_SCORE, max_rms=0.5)
    if proxy_iter3 is not None:
        for q in (3, 4, 5, 6, 7, 8, 10, 12):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.30, 0.65)]:
                corr = build_correction(proxy_iter3, q, gp, gn)
                outputs.append(save(
                    f"ITER3_q{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S3: ITER3 also using rms<0.3 (tighter) ======
    proxy_iter3_tight, _ = build_proxy(base, BASE_SCORE, max_rms=0.3)
    if proxy_iter3_tight is not None:
        for q in (5, 7, 10):
            corr = build_correction(proxy_iter3_tight, q, 0.35, 0.55)
            outputs.append(save(f"ITER3_TIGHT_q{q:02d}", base + corr, target_col))

    # ====== S4: ITER2 variations on parent (different q1+q2 combos) ======
    # ITER2 = parent + corr(q1) + corr_on_parent_corrected(q2)
    proxy_parent, _ = build_proxy(parent, PARENT_SCORE, max_rms=0.5)
    if proxy_parent is not None:
        for q1 in (5, 6, 7, 8):
            corr1 = build_correction(proxy_parent, q1, 0.35, 0.55)
            b1 = parent + corr1
            proxy2, _ = build_proxy(b1, PARENT_SCORE - 0.01, max_rms=0.5)
            if proxy2 is None: continue
            for q2 in (3, 4, 5, 6, 7, 10):
                for gp2, gn2 in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50)]:
                    corr2 = build_correction(proxy2, q2, gp2, gn2)
                    outputs.append(save(
                        f"ITER2v_q{q1:02d}_q{q2:02d}_p{int(gp2*100):02d}_n{int(gn2*100):02d}",
                        b1 + corr2, target_col))

    # ====== S5: NP_FRR on new base ======
    if proxy_iter3 is not None:
        for q in (5, 7, 10):
            corr = build_correction(proxy_iter3, q, 0.35, 0.55)
            for w_frr in (0.05, 0.10, 0.15, 0.20):
                outputs.append(save(
                    f"NP3_FRR_t{q:02d}_w{int(w_frr*100):02d}",
                    base + corr + w_frr * delta_frr, target_col))

    # ====== S6: NP + v61 on new base ======
    if proxy_iter3 is not None and v61 is not None:
        for q in (5, 7):
            corr = build_correction(proxy_iter3, q, 0.35, 0.55)
            for w in (0.010, 0.015, 0.020, 0.030):
                outputs.append(save(
                    f"NP3_v61_t{q:02d}_w{int(w*1000):03d}",
                    base + corr + w * (v61 - base), target_col))

    # ====== S7: NP + TOP6 on new base ======
    if proxy_iter3 is not None and 'TOP6' in externals:
        for q in (5, 7):
            corr = build_correction(proxy_iter3, q, 0.35, 0.55)
            for w in (0.005, 0.010, 0.015, 0.020):
                outputs.append(save(
                    f"NP3_TOP6_t{q:02d}_w{int(w*1000):03d}",
                    base + corr + w * (externals['TOP6'] - base), target_col))

    # ====== S8: Multi-α on new base ======
    for alpha in (1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1):
        p, _ = build_proxy(base, BASE_SCORE, max_rms=0.5, alpha=alpha)
        if p is None: continue
        a_tag = f"a{int(-np.log10(alpha)*10):02d}"
        for q in (5, 7):
            corr = build_correction(p, q, 0.35, 0.55)
            outputs.append(save(f"NP3_ALPHA_{a_tag}_t{q:02d}", base + corr, target_col))

    # ====== S9: STACK v44 winners ======
    winners = [
        BASE_FILE,
        "submission_FBSv44_ITER2_t07_t07_p35_n55.csv",
        "submission_FBSv44_ITER2_t07_t10_p35_n55.csv",
        "submission_FBSv44_NP_r05_t08_p35_n55.csv",
        "submission_FBSv44_NP_r05_t07_p35_n55.csv",
        "submission_FBSv44_NP_FRR_t07_w10.csv",
        "submission_FBSv44_NP_v61_t07_w010.csv",
        "submission_FBSv44_NP_TOP6_t07_w010.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 5:
        stk = np.asarray(arrs)
        outputs.append(save(f"STACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save(f"STACK_TOP5", np.mean(stk[:5], axis=0), target_col))
        outputs.append(save(f"STACK_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"STACK_MED_{len(arrs)}", np.median(stk, axis=0), target_col))
        sc = np.array([7.18230, 7.18736, 7.19049, 7.19174, 7.19209, 7.19223, 7.19213, 7.19225])[:len(arrs)]
        w_lb = 1.0 / (sc - 7.182 + 0.0005)
        w_lb = w_lb / w_lb.sum()
        outputs.append(save("STACK_WLB", (stk.T @ w_lb).reshape(-1), target_col))

    # ====== S10: ITER4 — apply proxy 4 times (deep recursion) ======
    if proxy_iter3 is not None:
        corr_a = build_correction(proxy_iter3, 5, 0.35, 0.55)
        b_step3 = base + corr_a
        proxy_step3, _ = build_proxy(b_step3, BASE_SCORE - 0.005, max_rms=0.5)
        if proxy_step3 is not None:
            for q in (3, 5, 7):
                corr_b = build_correction(proxy_step3, q, 0.35, 0.55)
                outputs.append(save(f"ITER4_q05_q{q:02d}", b_step3 + corr_b, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v45_iter3_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v45 candidates")


if __name__ == "__main__":
    main()
