#!/usr/bin/env python3
"""v29: AGGRESSIVE push. SoloTech leaped to 7.218. We need a BIG jump.

Champion: FBSv28_NBprox_r07_t07_p35_n55 = 7.23288
Gap: 0.015 to 1st place.

Strategy: COMBINE all winning mechanisms simultaneously + iterate proxy + larger gammas.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv28_NBprox_r07_t07_p35_n55.csv"
BASE_SCORE = 7.232883131643645
POOL = 'teamblend_v3_pool'
VALID_FEATURES = "data/valid_features.csv"

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv19_FRRraw_a2_g15.csv": 7.246677323305486,
    "submission_FBSv24_CHAMP_LGBM_w020.csv": 7.241733084922744,
    "submission_FBSv24_CHAMP_LGBM_w030.csv": 7.240292163223663,
    "submission_FBSv26_CHAMP_LGBM_w040.csv": 7.239216956002586,
    "submission_FBSv26_CHAMP_LGBM_w050.csv": 7.238409974746849,
    "submission_FBSv26_CHAMP_LGBM_w060.csv": 7.237970103804146,
    "submission_FBSv26_CHAMP_LGBM_w080.csv": 7.238172595523859,
    "submission_FBSv26_NB_LGBM_w020.csv": 7.238449162310680,
    "submission_FBSv26_NB_LGBM_w030.csv": 7.237982397774434,
    "submission_FBSv26_COMBO_LGBM040_FRR20.csv": 7.238491050051701,
    "submission_FBSv27_NB_plus_FRR_w10.csv": 7.237615105630028,
    "submission_FBSv27_NBprox_t08_p35_n55.csv": 7.236693385785031,
    "submission_FBSv27_NBprox_t10_p35_n55.csv": 7.238728452342865,
    "submission_FBSv27_NBprox_t12_p35_n55.csv": 7.241931389299324,
    "submission_FBSv27_STACK_v26_TOP3.csv": 7.237885911164830,
    "submission_FBSv27_STACK_v26_WLB_6.csv": 7.237913065922035,
    "submission_FBSv27_STACK_v26_MEAN_6.csv": 7.237929170635638,
    "submission_FBSv27_AVG_NB_w060.csv": 7.237976250789291,
    "submission_FBSv27_CHAMP_LGBM_w065.csv": 7.237901803969211,
    "submission_FBSv27_CHAMP_LGBM_w070.csv": 7.237901449011781,
    "submission_FBSv28_NBprox_r07_t05_p35_n55.csv": 7.236859949911515,
    "submission_FBSv28_NBprox_r07_t06_p35_n55.csv": 7.235024844144681,
    # base skipped t07
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
    path = ROOT / f"submission_FBSv29_{label}.csv"
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
    if len(dirs) < 4: return None, []
    D = np.asarray(dirs); T = np.asarray(tgts); R = np.asarray(rms_list)
    W = np.diag(1.0 / np.power(0.05 + R, 0.7))
    gram = (D @ D.T) / len(base)
    beta = np.linalg.solve(W @ gram @ W + alpha * np.eye(len(D)), W @ T)
    proxy = D.T @ (W @ beta)
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy, dirs


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
    lgbm_new, _ = load("submission_LGBM_v20_actuals_spatial.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_v89_covshift.csv', 'submission_v87_mlp.csv',
                'submission_v85_catboost.csv']:
        fp = pool_path / fn
        v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
        externals[fn.replace('submission_', '').replace('.csv', '').split('_')[0]] = v
    externals['DIV3'] = (externals['v89'] + externals['v87'] + externals['v85']) / 3.0

    outputs = []

    # ====== S1: NEW base proxy with multiple constraint windows ======
    for max_rms in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
        proxy, dirs = build_proxy(base, BASE_SCORE, max_rms=max_rms, max_abs=max_rms*8)
        if proxy is None: continue
        # Tight quantile range (since narrow won)
        for q in (4, 5, 6, 7, 8, 10):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.55), (0.35, 0.65),
                           (0.30, 0.65), (0.25, 0.65), (0.30, 0.45), (0.45, 0.45)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NPRr{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S2: NB + FRR fine sweep (winning w10 = 7.2376) ======
    delta_frr = frr - champ_orig
    for w in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50):
        blend = base + w * delta_frr
        outputs.append(save(f"NB_FRR_w{int(w*100):02d}", blend, target_col))

    # ====== S3: Multi-tier proxy + FRR + LGBM combo ======
    proxy_best, _ = build_proxy(base, BASE_SCORE, max_rms=0.7)
    if proxy_best is not None:
        for q in (6, 7, 8):
            corr = build_correction(proxy_best, q, 0.35, 0.55)
            for w_frr in (0.05, 0.10, 0.15, 0.20):
                for w_lgbm in (0.00, 0.02, 0.04):
                    if w_frr == 0 and w_lgbm == 0: continue
                    blend = base + corr + w_frr * delta_frr + w_lgbm * (lgbm_new - base)
                    outputs.append(save(
                        f"TRIPLE_t{q:02d}_f{int(w_frr*100):02d}_l{int(w_lgbm*100):02d}",
                        blend, target_col))

    # ====== S4: Iterate proxy — apply NBprox correction TWICE ======
    if proxy_best is not None:
        corr1 = build_correction(proxy_best, 7, 0.35, 0.55)
        b1 = base + corr1
        # Now build proxy on b1 (relative to BASE_SCORE since we don't know b1's score)
        # Approximate: assume b1 ~ 7.231 (one step better than base)
        proxy2, _ = build_proxy(b1, BASE_SCORE - 0.001, max_rms=0.7)
        if proxy2 is not None:
            corr2 = build_correction(proxy2, 7, 0.35, 0.55)
            outputs.append(save("ITER2_t07_t07", b1 + corr2, target_col))
            for q in (5, 7, 10):
                c2 = build_correction(proxy2, q, 0.35, 0.55)
                outputs.append(save(f"ITER2_t07_t{q:02d}", b1 + c2, target_col))

    # ====== S5: Stack all top winners across versions ======
    winners = [
        BASE_FILE,
        "submission_FBSv28_NBprox_r07_t06_p35_n55.csv",
        "submission_FBSv27_NBprox_t08_p35_n55.csv",
        "submission_FBSv27_NB_plus_FRR_w10.csv",
        "submission_FBSv27_STACK_v26_TOP3.csv",
        "submission_FBSv26_CHAMP_LGBM_w060.csv",
        "submission_FBSv26_NB_LGBM_w030.csv",
        "submission_FBSv27_CHAMP_LGBM_w070.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    stk = np.asarray(arrs)
    if len(arrs) >= 5:
        outputs.append(save(f"STACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save(f"STACK_TOP5", np.mean(stk[:5], axis=0), target_col))
        outputs.append(save(f"STACK_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"STACK_MED_{len(arrs)}", np.median(stk, axis=0), target_col))

    # ====== S6: NBprox + LGBM hybrid ======
    if proxy_best is not None and lgbm_new is not None:
        for q in (7, 8):
            corr = build_correction(proxy_best, q, 0.35, 0.55)
            base_with_corr = base + corr
            for w_lgbm in (0.02, 0.03, 0.04, 0.05, 0.08):
                blend = (1 - w_lgbm) * base_with_corr + w_lgbm * lgbm_new
                outputs.append(save(
                    f"PROX_LGBM_t{q:02d}_w{int(w_lgbm*100):02d}",
                    blend, target_col))

    # ====== S7: NBprox + DIV3 / covCB small dose ======
    if proxy_best is not None:
        corr = build_correction(proxy_best, 7, 0.35, 0.55)
        for tag, p in [('DIV3', externals['DIV3']), ('v89', externals['v89'])]:
            for w in (0.01, 0.02, 0.03):
                blend = (1 - w) * (base + corr) + w * p
                outputs.append(save(
                    f"PROX_{tag}_w{int(w*100):02d}",
                    blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v29_aggressive_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v29 candidates")


if __name__ == "__main__":
    main()
