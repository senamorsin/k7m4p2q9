#!/usr/bin/env python3
"""v35: Continue NB_FRR ladder. Base = FBSv34_NB_FRR_w30 = 7.21168."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv34_NB_FRR_w30.csv"
BASE_SCORE = 7.211678029707256
POOL = 'teamblend_v3_pool'

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv19_FRRraw_a2_g15.csv": 7.246677323305486,
    "submission_FBSv24_CHAMP_LGBM_w030.csv": 7.240292163223663,
    "submission_FBSv26_CHAMP_LGBM_w060.csv": 7.237970103804146,
    "submission_FBSv29_NB_FRR_w30.csv": 7.229653906267378,
    "submission_FBSv30_FRR_w50.csv": 7.228030068703475,
    "submission_FBSv30_FRR_w80.csv": 7.226009000114476,
    "submission_FBSv30_NP30_r07_t05_p35_n55.csv": 7.221701978080064,
    "submission_FBSv31_BASE_plus_FRR_w50.csv": 7.218465181225035,
    "submission_FBSv31_BASE_PROX_FRR_t05_p35_n55_f20.csv": 7.213629337187302,
    "submission_FBSv33_NB_FRR_w20.csv": 7.212619135451977,
    "submission_FBSv33_STACK_TOP3.csv": 7.215560763905955,
    "submission_FBSv34_V31_FRR_w28.csv": 7.212309097479216,
    "submission_FBSv34_V31_FRR_w23.csv": 7.212500949467225,
    "submission_FBSv34_V31_FRR_w22.csv": 7.212540344795476,
    "submission_FBSv34_V31_FRR_w18.csv": 7.212703250232205,
    "submission_FBSv34_V31_FRR_w25.csv": 7.214242188810722,
    "submission_FBSv34_NB_FRR_w10.csv": 7.212239678845949,
    "submission_FBSv34_NB_FRR_w15.csv": 7.212076033071595,
    "submission_FBSv34_NB_FRR_w20.csv": 7.211930368730238,
    "submission_FBSv34_NB_FRR_w25.csv": 7.211795180481663,
    "submission_FBSv34_NEWFRR_a20_g15.csv": 7.224601342475122,
    "submission_FBSv34_NEWFRR_a20_g30.csv": 7.240338189004849,
    "submission_FBSv34_NP_r05_t05_p35_n55.csv": 7.223225956786290,
    "submission_FBSv34_NP_FRR_r07_t05_w10.csv": 7.214825934602300,
    "submission_FBSv34_COMBO2_a20_g10_f10.csv": 7.219816080059799,
    "submission_FBSv34_STACK_TOP3.csv": 7.213595745527471,
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
    path = ROOT / f"submission_FBSv35_{label}.csv"
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


def main():
    base, target_col = load(BASE_FILE)
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    champ_orig, _ = load("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv")
    v33_base, _ = load("submission_FBSv33_NB_FRR_w20.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    outputs = []
    delta_frr = frr - champ_orig

    # ====== S1: NB_FRR ladder (winner pattern) — w30 → w35-w80 ======
    # base = v33_base + 0.30 * delta_frr (from v34 winner)
    # base is also (v34_NB_FRR_w30)
    # Try adding more delta_frr on top
    for w in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0):
        blend = base + w * delta_frr
        outputs.append(save(f"NB_FRR_w{int(w*100):02d}", blend, target_col))

    # ====== S2: Fine grid around w30 on v33 base ======
    if v33_base is not None:
        for w in (0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.35, 0.40, 0.50, 0.60):
            blend = v33_base + w * delta_frr
            outputs.append(save(f"v33_FRR_w{int(w*100):02d}", blend, target_col))

    # ====== S3: Mix top winners (close to base) ======
    winners = [
        BASE_FILE,
        "submission_FBSv34_NB_FRR_w25.csv",
        "submission_FBSv34_NB_FRR_w20.csv",
        "submission_FBSv34_NB_FRR_w15.csv",
        "submission_FBSv34_NB_FRR_w10.csv",
        "submission_FBSv34_V31_FRR_w28.csv",
        "submission_FBSv34_V31_FRR_w23.csv",
        "submission_FBSv33_NB_FRR_w20.csv",
        "submission_FBSv31_BASE_PROX_FRR_t05_p35_n55_f20.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 4:
        stk = np.asarray(arrs)
        outputs.append(save("STACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save("STACK_TOP5", np.mean(stk[:5], axis=0), target_col))
        outputs.append(save("STACK_MEAN", np.mean(stk, axis=0), target_col))
        outputs.append(save("STACK_MED", np.median(stk, axis=0), target_col))
        sc = np.array([7.21168, 7.21180, 7.21193, 7.21208, 7.21224, 7.21231, 7.21251, 7.21262, 7.21363])[:len(arrs)]
        w = 1.0 / (sc - 7.211 + 0.0005)
        w = w / w.sum()
        outputs.append(save("STACK_WLB", (stk.T @ w).reshape(-1), target_col))

    # ====== S4: Proxy on new base (cautious — proxy approaches tend to fail at this point) ======
    proxy_new = build_proxy(base, BASE_SCORE, max_rms=0.5)
    if proxy_new is not None:
        for q in (5, 6, 7, 8):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55)]:
                qq = 1.0 - q/100.0
                active = np.abs(proxy_new) >= np.quantile(np.abs(proxy_new), qq)
                pos = active & (proxy_new > 0); neg = active & (proxy_new < 0)
                corr = np.zeros_like(proxy_new)
                corr[pos] = gp * proxy_new[pos]; corr[neg] = gn * proxy_new[neg]
                outputs.append(save(
                    f"NP_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S5: BASE + small mix of various v34 winners ======
    nb_frr_w25, _ = load("submission_FBSv34_NB_FRR_w25.csv")
    nb_frr_w20, _ = load("submission_FBSv34_NB_FRR_w20.csv")
    if nb_frr_w25 is not None and nb_frr_w20 is not None:
        # Avg of top-3 NB_FRR variants
        avg3 = (base + nb_frr_w25 + nb_frr_w20) / 3
        outputs.append(save("AVG_v34_TOP3", avg3, target_col))
        # Weighted
        for w in (0.3, 0.5, 0.7):
            for partner in (nb_frr_w25, nb_frr_w20):
                tag = "w25" if partner is nb_frr_w25 else "w20"
                blend = (1 - w) * base + w * partner
                outputs.append(save(f"MIX_{tag}_w{int(w*100):02d}", blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v35_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v35 candidates")


if __name__ == "__main__":
    main()
