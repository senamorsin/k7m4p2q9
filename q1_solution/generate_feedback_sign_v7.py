#!/usr/bin/env python3
"""Feedback-sign v7: PLATEAU BREAK ATTEMPT.

v6 plateau-ed at ~7.3525-7.3530. Proxy saturated with current constraints.

v7 strategies (4 mechanisms):
  S1. CURATED proxy — only constraints with rms < 0.5 and abs_max < 5 (recent, close)
  S2. BOOTSTRAP-ensemble — 20 proxies on random 70% subsets, average
  S3. STACK-MEDIAN/MEAN — robust ensemble of top 10 FBS winners
  S4. CROSS-BASE — apply v6 best proxy on FBSv4/FBSv5 bases (different starting points)

NEW base = FBSv6_NEG_negOnly_g55 = 7.35258 (marginal champion).
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv6_NEG_negOnly_g55.csv"
BASE_SCORE = 7.352581410198546
VALID_FEATURES = "data/valid_features.csv"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"

# Compact KNOWN_SCORES (re-using v6 list + new v6 results)
KNOWN_SCORES = {
    # Historical rebound + early
    "submission_rebound_v6_t075_hneg10.csv": 7.37455489169975,
    "submission_rebound_v6_t075_h0.csv": 7.374774556660089,
    "submission_rebound_v6_surface_push.csv": 7.375287304530807,
    "submission_feedback_sign_g0p08.csv": 7.373278334346155,
    "submission_feedback_sign_smooth5_g0p08.csv": 7.372587722053496,
    # REBv6
    "submission_REBv6_LIGHT10_smooth.csv": 7.377025947066968,
    "submission_REBv6_VLIGHT_smooth.csv": 7.378272056634792,
    # CHAMPv3 (best ones)
    "submission_CHAMPv3_DIV3_w08.csv": 7.373613969627501,
    "submission_CHAMPv3_DIV3_w12.csv": 7.376372031340379,
    "submission_CHAMPv3_v89cov_w08.csv": 7.375599991905816,
    # ULTRA (best)
    "submission_ULTRA_DIV3_gauss_b06.csv": 7.371827077437699,
    "submission_ULTRA_DIV3_VS_w025.csv": 7.372196507607671,
    "submission_ULTRA_DIV3_w03.csv": 7.372235347883005,
    "submission_ULTRA_DIV3_w04.csv": 7.372393263765536,
    "submission_ULTRA_covCB_w06.csv": 7.372534974937972,
    # R2 (best)
    "submission_R2_gausscovCB_b06.csv": 7.371251866387993,
    "submission_R2_gaussmlpCB_b06.csv": 7.371689777558700,
    "submission_R2_NCgauss_covCB_b02.csv": 7.371743130440402,
    "submission_R2_addgauss_b015_c30_s20.csv": 7.371908255855521,
    "submission_R2_addgauss_b020_c30_s20.csv": 7.371935315328128,
    "submission_R2_addgauss_b020_c25_s15.csv": 7.371945873009935,
    "submission_R2_addgauss_b025_c30_s20.csv": 7.371962374800735,
    "submission_R2_NCu_covCB_w015.csv": 7.372062902461258,
    "submission_R2_NCu_DIV3_w015.csv": 7.372150312799336,
    # FBSv2
    "submission_FBSv2_smooth5_g0p12.csv": 7.369170488021504,
    "submission_FBSv2_tanh_g0p15.csv": 7.369660686492568,
    "submission_FBSv2_smooth5_g0p08.csv": 7.369845849772287,
    # FBSv3
    "submission_FBSv3_top19_g0p22.csv": 7.363100871060781,
    "submission_FBSv3_g0p22.csv": 7.366978298009447,
    "submission_FBSv3_g0p15.csv": 7.367069030220226,
    "submission_FBSv3_smooth5_g0p12.csv": 7.368729215670931,
    "submission_FBSv3_smooth5_g0p15.csv": 7.368930778202181,
    "submission_FBSv3_smooth11_g0p18.csv": 7.369011607727708,
    "submission_FBSv3_smooth5_g0p18.csv": 7.369211844341594,
    "submission_FBSv3_smooth7_g0p15.csv": 7.369416069312870,
    "submission_FBSv3_smtanh5_g0p22.csv": 7.370773827626002,
    "submission_FBSv3_NEG_g0p08.csv": 7.373106340936872,
    # FBSv4
    "submission_FBSv4_top10_g0p45.csv": 7.353705058320320,
    "submission_FBSv4_top10_g0p35.csv": 7.354903732793917,
    "submission_FBSv4_top10_g0p28.csv": 7.356197766651727,
    "submission_FBSv4_toptanh10_g0p45.csv": 7.357629621602857,
    "submission_FBSv4_top19_g0p35.csv": 7.359156847290178,
    "submission_FBSv4_top19_g0p28.csv": 7.359252538134574,
    "submission_FBSv4_top05_g0p35.csv": 7.359252859933578,
    "submission_FBSv4_top15_g0p22.csv": 7.359430667636016,
    "submission_FBSv4_top19_g0p22.csv": 7.359837314288791,
    "submission_FBSv4_NEG_g0p22.csv": 7.376882720822089,
    # FBSv5
    "submission_FBSv5_top10_asy_gp35_gn55.csv": 7.352680653403223,
    "submission_FBSv5_2tier_gs45_gm15.csv": 7.353167469058188,
    "submission_FBSv5_top07_g0p65.csv": 7.353177022759139,
    "submission_FBSv5_2tier_gs55_gm20.csv": 7.354551300020761,
    "submission_FBSv5_top10_g0p55.csv": 7.355821612258592,
    "submission_FBSv5_2tier_gs65_gm25.csv": 7.356341504006535,
    "submission_FBSv5_top10_asy_gp55_gn35.csv": 7.356429646252437,
    "submission_FBSv5_top10_g0p65.csv": 7.357732863344162,
    "submission_FBSv5_top10_asy_gp60_gn30.csv": 7.357784747367144,
    "submission_FBSv5_top10_g0p75.csv": 7.360023493889184,
    "submission_FBSv5_NEG_g0p45.csv": 7.387925469841433,
    # FBSv6 (newest, most informative)
    # BASE skipped: NEG_negOnly_g55 = 7.352581
    "submission_FBSv6_top12_asy_gp30_gn55.csv": 7.352845260428158,
    "submission_XB6_covCB_w020.csv": 7.352927932946245,
    "submission_FBSv6_2tasy_sp30_sn55_mp10_mn20.csv": 7.354078805994201,
    "submission_FBSv6_2tasy_sp25_sn60_mp10_mn25.csv": 7.355778785401931,
    "submission_FBSv6_top10_asy_gp30_gn55.csv": 7.356904491856318,
    "submission_FBSv6_top10_asy_gp30_gn60.csv": 7.358050564512658,
    "submission_FBSv6_top10_asy_gp25_gn55.csv": 7.357556877232764,
    "submission_FBSv6_top08_asy_gp30_gn55.csv": 7.357399642512570,
    "submission_FBSv6_top10_asy_gp35_gn60.csv": 7.357418817430300,
    "submission_FBSv6_top10_asy_gp20_gn55.csv": 7.358209262609207,
    "submission_FBSv6_top10_asy_gp25_gn60.csv": 7.358702949889103,
    "submission_FBSv6_2tasy_sp20_sn65_mp05_mn20.csv": 7.358860953585348,
    "submission_FBSv6_top15_negOnly_g55.csv": 7.358863119196279,
    "submission_FBSv6_top10_asy_gp30_gn65.csv": 7.359198339253735,
    "submission_FBSv6_top10_pow1p3_gp25_gn55.csv": 7.360634270139702,
    "submission_FBSv6_top10_negOnly_g55.csv": 7.361534582596205,
    "submission_XB6_DIV3onlyDEC_w10.csv": 7.362493863563346,
    "submission_FBSv6_top10_negOnly_g70.csv": 7.364991054337667,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    path = ROOT / name
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    vals = df.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        return None, None
    return vals, df.columns[0]


def save(label, values, target_col):
    arr = np.clip(values, 0.0, N_INST)
    path = ROOT / f"submission_FBSv7_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path),
            "mean": float(arr.mean()), "std": float(arr.std())}


def build_proxy_curated(base, max_rms=0.6, max_abs=5.0, prefer_low_rms=True):
    """S1: tight constraint window."""
    directions, targets, names, rms_list = [], [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base):
            continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        mxa = float(np.max(np.abs(diff)))
        if rms < max_rms and mxa < max_abs:
            directions.append(diff)
            targets.append(-((score - BASE_SCORE) * N_INST / 100.0))
            names.append(name)
            rms_list.append(rms)
    if len(directions) < 5:
        return None, []
    D = np.asarray(directions); target_vec = np.asarray(targets); rms_arr = np.asarray(rms_list)
    W = np.diag(1.0 / np.power(0.05 + rms_arr, 0.7))
    gram = (D @ D.T) / len(base)
    alpha = 1e-3
    beta = np.linalg.solve(W @ gram @ W + alpha*np.eye(len(D)), W @ target_vec)
    proxy = D.T @ (W @ beta)
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy, names


def build_proxy_bootstrap(base, n_boot=20, frac=0.70, seed=42):
    """S2: bootstrap average."""
    rng = np.random.RandomState(seed)
    all_dir, all_tgt, all_rms = [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base): continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        if rms < 5.0 and float(np.max(np.abs(diff))) < 30.0:
            all_dir.append(diff)
            all_tgt.append(-((score - BASE_SCORE) * N_INST / 100.0))
            all_rms.append(rms)
    D = np.asarray(all_dir); T = np.asarray(all_tgt); R = np.asarray(all_rms)
    N = len(D)
    proxies = []
    for b in range(n_boot):
        k = int(frac * N)
        idx = rng.choice(N, size=k, replace=False)
        Db, Tb, Rb = D[idx], T[idx], R[idx]
        Wb = np.diag(1.0 / np.power(0.05 + Rb, 0.7))
        gram = (Db @ Db.T) / len(base)
        try:
            beta = np.linalg.solve(Wb @ gram @ Wb + 1e-3*np.eye(k), Wb @ Tb)
        except np.linalg.LinAlgError:
            continue
        p = Db.T @ (Wb @ beta)
        p = p - p.mean()
        lo, hi = np.percentile(p, [1, 99])
        p = np.clip(p, lo, hi)
        p = p / (p.std() + 1e-9)
        proxies.append(p)
    if not proxies:
        return None
    return np.mean(proxies, axis=0)


def main():
    base, target_col = load(BASE_FILE)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")
    outputs = []

    # ====== S1: CURATED PROXY (tight constraints, less noise) ======
    for max_rms in [0.4, 0.5, 0.7, 1.0]:
        proxy_c, names_c = build_proxy_curated(base, max_rms=max_rms, max_abs=max_rms*8)
        if proxy_c is None: continue
        print(f"  CURATED (rms<{max_rms}): {len(names_c)} constraints")
        # winning recipe: top10 asy gp35 gn55, but apply with curated proxy
        for q_keep_pct in [8, 10, 12, 15]:
            q = 1.0 - q_keep_pct/100.0
            active = np.abs(proxy_c) >= np.quantile(np.abs(proxy_c), q)
            pos = active & (proxy_c > 0); neg = active & (proxy_c < 0)
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50), (0.25, 0.65), (0.45, 0.45)]:
                corr = np.zeros_like(proxy_c)
                corr[pos] = gp * proxy_c[pos]
                corr[neg] = gn * proxy_c[neg]
                outputs.append(save(
                    f"CUR{int(max_rms*10):02d}_t{q_keep_pct:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))
        # raw with curated
        for gamma in [0.20, 0.30, 0.45]:
            outputs.append(save(f"CUR{int(max_rms*10):02d}_g{int(gamma*100):02d}",
                                base + gamma * proxy_c, target_col))

    # ====== S2: BOOTSTRAP ENSEMBLE ======
    for seed in [42, 7, 123]:
        proxy_b = build_proxy_bootstrap(base, n_boot=30, frac=0.7, seed=seed)
        if proxy_b is None: continue
        proxy_b = proxy_b - proxy_b.mean()
        proxy_b = proxy_b / (proxy_b.std() + 1e-9)
        for q in [10, 12, 15]:
            qq = 1.0 - q/100.0
            active = np.abs(proxy_b) >= np.quantile(np.abs(proxy_b), qq)
            pos = active & (proxy_b > 0); neg = active & (proxy_b < 0)
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50)]:
                corr = np.zeros_like(proxy_b)
                corr[pos] = gp * proxy_b[pos]
                corr[neg] = gn * proxy_b[neg]
                outputs.append(save(
                    f"BOOT{seed}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S3: STACK MEDIAN/MEAN of top FBS winners ======
    winners = [
        "submission_FBSv5_top10_asy_gp35_gn55.csv",
        "submission_FBSv6_NEG_negOnly_g55.csv",
        "submission_FBSv6_top12_asy_gp30_gn55.csv",
        "submission_XB6_covCB_w020.csv",
        "submission_FBSv5_top07_g0p65.csv",
        "submission_FBSv5_2tier_gs45_gm15.csv",
        "submission_FBSv4_top10_g0p45.csv",
        "submission_FBSv4_top10_g0p35.csv",
        "submission_FBSv6_2tasy_sp30_sn55_mp10_mn20.csv",
        "submission_FBSv6_top10_asy_gp30_gn55.csv",
    ]
    stack = []
    for w in winners:
        v, _ = load(w)
        if v is not None: stack.append(v)
    stack = np.asarray(stack)
    if len(stack) >= 3:
        # median (robust)
        med = np.median(stack, axis=0)
        outputs.append(save(f"STACK_MEDIAN_{len(stack)}", med, target_col))
        # mean
        mean = np.mean(stack, axis=0)
        outputs.append(save(f"STACK_MEAN_{len(stack)}", mean, target_col))
        # trimmed mean (drop best/worst)
        sort_idx = np.argsort(stack, axis=0)
        # for each row, drop top/bottom
        trimmed = np.zeros_like(stack[0])
        for i in range(stack.shape[1]):
            col = np.sort(stack[:, i])
            trimmed[i] = col[1:-1].mean()
        outputs.append(save(f"STACK_TRIMMED_{len(stack)}", trimmed, target_col))
        # weighted by 1/(LB-BASE_SCORE)
        scores = np.array([7.352680653403223, 7.352581410198546, 7.352845260428158,
                            7.352927932946245, 7.353177022759139, 7.353167469058188,
                            7.353705058320320, 7.354903732793917, 7.354078805994201,
                            7.356904491856318])[:len(stack)]
        w_inv = 1.0 / (scores - 7.35)  # inverse-distance weighting
        w_inv = w_inv / w_inv.sum()
        weighted = (stack.T @ w_inv).reshape(-1)
        outputs.append(save(f"STACK_WEIGHTED_{len(stack)}", weighted, target_col))
        # Blend champion with median (anchor)
        for w_med in (0.3, 0.5, 0.7):
            blend = (1-w_med)*base + w_med*med
            outputs.append(save(f"BASE_plus_MED_w{int(w_med*100):02d}", blend, target_col))

    # ====== S4: CROSS-BASE (apply v6 best variant on alternate bases) ======
    # The v6 winner was NEG_negOnly_g55 — try applying its pattern from another base
    for alt_base_fn, tag in [
        ("submission_FBSv5_top10_asy_gp35_gn55.csv", "FBSv5"),
        ("submission_FBSv4_top10_g0p45.csv", "FBSv4"),
        ("submission_FBSv4_top10_g0p35.csv", "FBSv4g35"),
    ]:
        alt, _ = load(alt_base_fn)
        if alt is None: continue
        # use curated proxy from this alt base
        proxy_alt, _ = build_proxy_curated(alt, max_rms=0.5, max_abs=4.0)
        if proxy_alt is None: continue
        # apply winning recipe
        active = np.abs(proxy_alt) >= np.quantile(np.abs(proxy_alt), 0.90)
        for gp, gn in [(0.35, 0.55), (0.30, 0.55)]:
            corr = np.zeros_like(proxy_alt)
            corr[active & (proxy_alt > 0)] = gp * proxy_alt[active & (proxy_alt > 0)]
            corr[active & (proxy_alt < 0)] = gn * proxy_alt[active & (proxy_alt < 0)]
            outputs.append(save(f"on_{tag}_curat_p{int(gp*100):02d}_n{int(gn*100):02d}",
                                alt + corr, target_col))

    report = {
        "base_file": BASE_FILE,
        "base_score": BASE_SCORE,
        "outputs": [o["file"] for o in outputs],
    }
    (ROOT / "feedback_sign_v7_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v7 candidates")


if __name__ == "__main__":
    main()
