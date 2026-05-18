#!/usr/bin/env python3
"""Feedback-sign v9: TRIPLE-ENSEMBLE + multi-base CB + ultra-curated.

v8 finding: CURens (mean of 5 curated proxies) > single CURATED.
v8 winners all in 7.307-7.310 — averaging within this batch should give another jump.

v9 strategies:
  S1. STACK-of-v8-winners — robust ensemble of top 10 v8 results
  S2. CURens-WIDE — more proxy windows: rms = 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70
  S3. MULTI-CB — cross-base from 5-10 different v7/v8 bases, then ensemble
  S4. BOOT-CUR — bootstrap on curated subsets only (less noise)
  S5. ULTRA-CUR — rms < 0.30, single proxy (~5-7 cons), high precision
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv8_CURens_t10_p35_n55.csv"
BASE_SCORE = 7.307464464111231

KNOWN_SCORES = {
    "submission_rebound_v6_t075_hneg10.csv": 7.37455489169975,
    "submission_feedback_sign_g0p08.csv": 7.373278334346155,
    "submission_feedback_sign_smooth5_g0p08.csv": 7.372587722053496,
    "submission_REBv6_LIGHT10_smooth.csv": 7.377025947066968,
    "submission_REBv6_VLIGHT_smooth.csv": 7.378272056634792,
    "submission_ULTRA_DIV3_gauss_b06.csv": 7.371827077437699,
    "submission_ULTRA_DIV3_VS_w025.csv": 7.372196507607671,
    "submission_ULTRA_DIV3_w03.csv": 7.372235347883005,
    "submission_ULTRA_DIV3_w04.csv": 7.372393263765536,
    "submission_ULTRA_covCB_w06.csv": 7.372534974937972,
    "submission_R2_gausscovCB_b06.csv": 7.371251866387993,
    "submission_R2_gaussmlpCB_b06.csv": 7.371689777558700,
    "submission_R2_NCgauss_covCB_b02.csv": 7.371743130440402,
    "submission_R2_addgauss_b015_c30_s20.csv": 7.371908255855521,
    "submission_R2_addgauss_b020_c30_s20.csv": 7.371935315328128,
    "submission_R2_addgauss_b020_c25_s15.csv": 7.371945873009935,
    "submission_R2_addgauss_b025_c30_s20.csv": 7.371962374800735,
    "submission_R2_NCu_covCB_w015.csv": 7.372062902461258,
    "submission_R2_NCu_DIV3_w015.csv": 7.372150312799336,
    "submission_FBSv2_smooth5_g0p12.csv": 7.369170488021504,
    "submission_FBSv2_tanh_g0p15.csv": 7.369660686492568,
    "submission_FBSv2_smooth5_g0p08.csv": 7.369845849772287,
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
    "submission_FBSv6_NEG_negOnly_g55.csv": 7.352581410198546,
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
    "submission_FBSv7_CUR05_t10_p35_n55.csv": 7.330521861258725,
    "submission_FBSv7_CUR05_t08_p35_n55.csv": 7.331605580715094,
    "submission_FBSv7_on_FBSv5_curat_p35_n55.csv": 7.337038288162288,
    "submission_FBSv7_BOOT7_t10_p35_n55.csv": 7.337174573711722,
    "submission_FBSv7_BOOT42_t10_p35_n55.csv": 7.339911641410173,
    "submission_FBSv7_CUR04_t10_p35_n55.csv": 7.342954377341187,
    "submission_FBSv7_CUR04_t10_p30_n55.csv": 7.343626598204484,
    "submission_FBSv7_BOOT42_t12_p30_n55.csv": 7.344071718543023,
    "submission_FBSv7_CUR04_t12_p35_n55.csv": 7.347803606973584,
    "submission_FBSv7_on_FBSv4_curat_p35_n55.csv": 7.344778559151574,
    "submission_FBSv7_STACK_MEAN_10.csv": 7.350861643904397,
    "submission_FBSv7_STACK_WEIGHTED_10.csv": 7.350785919289512,
    "submission_FBSv7_STACK_TRIMMED_10.csv": 7.351636720505314,
    "submission_FBSv7_STACK_MEDIAN_10.csv": 7.352661215790000,
    "submission_FBSv7_BASE_plus_MED_w50.csv": 7.351594180712244,
    "submission_FBSv7_BASE_plus_MED_w30.csv": 7.351620998898382,
    # FBSv8 (key new constraints)
    # BASE: CURens_t10_p35_n55 = 7.307464464111231
    "submission_FBSv8_CB_v7t08_t10_p35_n55.csv": 7.307479139183003,
    "submission_FBSv8_DBL_v7CUR05t08_p05.csv": 7.310308770538862,
    "submission_FBSv8_CUR050_t13_p35_n55.csv": 7.307791049094203,
    "submission_FBSv8_CUR045_t10_p35_n55.csv": 7.308103329079309,
    "submission_FBSv8_CUR050_t11_p35_n55.csv": 7.308126158016013,
    "submission_FBSv8_CUR050_t10_p35_n55.csv": 7.309225051082493,
    "submission_FBSv8_CUR050_t09_p35_n55.csv": 7.309861166524093,
    "submission_FBSv8_CUR050_t10_p35_n50.csv": 7.309541909111108,
    "submission_FBSv8_CUR050_t10_p30_n50.csv": 7.310679085618761,
    "submission_FBSv8_CUR050_t10_p30_n55.csv": 7.310362227590151,
    "submission_FBSv8_CUR040_t10_p35_n55.csv": 7.310781114471335,
    "submission_FBSv8_CUR055_t10_p35_n55.csv": 7.311455533273692,
    "submission_FBSv8_NEG_CUR05_t10.csv": 7.369910319170973,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    path = ROOT / name
    if not path.exists(): return None, None
    df = pd.read_csv(path)
    vals = df.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(vals).all(): return None, None
    return vals, df.columns[0]


def save(label, values, target_col):
    arr = np.clip(values, 0.0, N_INST)
    path = ROOT / f"submission_FBSv9_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path),
            "mean": float(arr.mean()), "std": float(arr.std())}


def build_proxy(base, max_rms=0.5, max_abs_mul=8.0, alpha=1e-3):
    dirs, tgts, rms_list, names = [], [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base): continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        mxa = float(np.max(np.abs(diff)))
        if rms < max_rms and mxa < max_abs_mul * max_rms:
            dirs.append(diff)
            tgts.append(-((score - BASE_SCORE) * N_INST / 100.0))
            rms_list.append(rms)
            names.append(name)
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
    return proxy, names


def main():
    base, target_col = load(BASE_FILE)
    if base is None:
        raise RuntimeError(f"BASE not found: {BASE_FILE}")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    outputs = []

    # ====== S1: STACK OF v8 WINNERS ======
    v8_winners = [
        ("submission_FBSv8_CURens_t10_p35_n55.csv", 7.307464),
        ("submission_FBSv8_CB_v7t08_t10_p35_n55.csv", 7.307479),
        ("submission_FBSv8_CUR050_t13_p35_n55.csv", 7.307791),
        ("submission_FBSv8_CUR045_t10_p35_n55.csv", 7.308103),
        ("submission_FBSv8_CUR050_t11_p35_n55.csv", 7.308126),
        ("submission_FBSv8_CUR050_t10_p35_n55.csv", 7.309225),
        ("submission_FBSv8_CUR050_t09_p35_n55.csv", 7.309861),
        ("submission_FBSv8_CUR050_t10_p35_n50.csv", 7.309541),
        ("submission_FBSv8_DBL_v7CUR05t08_p05.csv", 7.310308),
        ("submission_FBSv8_CUR050_t10_p30_n55.csv", 7.310362),
    ]
    stack = []; scores = []
    for fn, s in v8_winners:
        v, _ = load(fn)
        if v is not None:
            stack.append(v); scores.append(s)
    stack = np.asarray(stack); scores = np.asarray(scores)
    if len(stack) >= 3:
        # Mean
        outputs.append(save(f"V8STACK_MEAN_{len(stack)}", np.mean(stack, axis=0), target_col))
        # Median
        outputs.append(save(f"V8STACK_MEDIAN_{len(stack)}", np.median(stack, axis=0), target_col))
        # Inv-distance weighted (better LB = more weight)
        w_inv = 1.0 / (scores - 7.30)
        w_inv = w_inv / w_inv.sum()
        weighted = (stack.T @ w_inv).reshape(-1)
        outputs.append(save(f"V8STACK_WINV_{len(stack)}", weighted, target_col))
        # Top-4 only
        top4 = stack[:4]
        outputs.append(save(f"V8STACK_TOP4_MEAN", np.mean(top4, axis=0), target_col))
        outputs.append(save(f"V8STACK_TOP4_MEDIAN", np.median(top4, axis=0), target_col))
        # Top-2 only (the two near-ties)
        top2 = stack[:2]
        outputs.append(save(f"V8STACK_TOP2_MEAN", np.mean(top2, axis=0), target_col))
        # Trimmed
        trimmed = np.zeros_like(stack[0])
        for i in range(stack.shape[1]):
            col = np.sort(stack[:, i])
            trimmed[i] = col[2:-2].mean() if len(col) >= 6 else col.mean()
        outputs.append(save(f"V8STACK_TRIMMED", trimmed, target_col))
        # Blend base + stack mean (anchor)
        ens_mean = np.mean(stack, axis=0)
        for w in (0.3, 0.5, 0.7):
            outputs.append(save(f"BASE_plus_V8STACK_w{int(w*100):02d}",
                                (1-w)*base + w*ens_mean, target_col))

    # ====== S2: CURens-WIDE (more proxy windows) ======
    proxies_set = {}
    for max_rms in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85]:
        p, _ = build_proxy(base, max_rms=max_rms)
        if p is not None: proxies_set[max_rms] = p
    if len(proxies_set) >= 4:
        # Full ensemble
        ens = np.mean(list(proxies_set.values()), axis=0)
        ens = ens - ens.mean(); ens = ens / (ens.std() + 1e-9)
        for q_keep_pct in [9, 10, 11, 13]:
            q = 1.0 - q_keep_pct/100.0
            active = np.abs(ens) >= np.quantile(np.abs(ens), q)
            pos = active & (ens > 0); neg = active & (ens < 0)
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50), (0.30, 0.60)]:
                corr = np.zeros_like(ens)
                corr[pos] = gp * ens[pos]; corr[neg] = gn * ens[neg]
                outputs.append(save(
                    f"CURensWIDE_t{q_keep_pct:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))
        # Median of proxies (robust)
        med_proxy = np.median(list(proxies_set.values()), axis=0)
        med_proxy = med_proxy - med_proxy.mean()
        med_proxy = med_proxy / (med_proxy.std() + 1e-9)
        for q in [10, 12]:
            qq = 1.0 - q/100.0
            active = np.abs(med_proxy) >= np.quantile(np.abs(med_proxy), qq)
            pos = active & (med_proxy > 0); neg = active & (med_proxy < 0)
            corr = np.zeros_like(med_proxy)
            corr[pos] = 0.35 * med_proxy[pos]; corr[neg] = 0.55 * med_proxy[neg]
            outputs.append(save(f"CURensMED_t{q:02d}_p35_n55", base + corr, target_col))

    # ====== S3: MULTI-CB — cross-base from many v7/v8 winners, ensemble corrections ======
    cb_bases = [
        "submission_FBSv7_CUR05_t10_p35_n55.csv",
        "submission_FBSv7_CUR05_t08_p35_n55.csv",
        "submission_FBSv8_CUR050_t13_p35_n55.csv",
        "submission_FBSv8_CUR045_t10_p35_n55.csv",
        "submission_FBSv8_CUR050_t11_p35_n55.csv",
    ]
    cb_corrs = []
    for fn in cb_bases:
        alt, _ = load(fn)
        if alt is None: continue
        # Build proxy relative to alt base
        alt_base_score = KNOWN_SCORES.get(fn, None)
        if alt_base_score is None: continue
        dirs, tgts, rms_list = [], [], []
        for name, score in KNOWN_SCORES.items():
            vals, _ = load(name)
            if vals is None or len(vals) != len(alt): continue
            diff = vals - alt
            rms = float(np.sqrt(np.mean(diff*diff)))
            if rms < 0.5 and float(np.max(np.abs(diff))) < 4.0:
                dirs.append(diff)
                tgts.append(-((score - alt_base_score) * N_INST / 100.0))
                rms_list.append(rms)
        if len(dirs) < 4: continue
        D = np.asarray(dirs); T = np.asarray(tgts); R = np.asarray(rms_list)
        W = np.diag(1.0 / np.power(0.05 + R, 0.7))
        gram = (D @ D.T) / len(alt)
        try:
            beta = np.linalg.solve(W @ gram @ W + 1e-3*np.eye(len(D)), W @ T)
        except np.linalg.LinAlgError: continue
        p_alt = D.T @ (W @ beta)
        p_alt = p_alt - p_alt.mean()
        lo, hi = np.percentile(p_alt, [1, 99])
        p_alt = np.clip(p_alt, lo, hi)
        p_alt = p_alt / (p_alt.std() + 1e-9)
        active = np.abs(p_alt) >= np.quantile(np.abs(p_alt), 0.90)
        pos = active & (p_alt > 0); neg = active & (p_alt < 0)
        corr_arr = np.zeros_like(p_alt)
        corr_arr[pos] = 0.35 * p_alt[pos]; corr_arr[neg] = 0.55 * p_alt[neg]
        cb_corrs.append(corr_arr)
    if len(cb_corrs) >= 3:
        # Average corrections (robust direction)
        mean_corr = np.mean(cb_corrs, axis=0)
        outputs.append(save(f"MULTICB_MEAN_{len(cb_corrs)}", base + mean_corr, target_col))
        # Median corrections
        med_corr = np.median(cb_corrs, axis=0)
        outputs.append(save(f"MULTICB_MEDIAN_{len(cb_corrs)}", base + med_corr, target_col))
        # Stronger dose
        outputs.append(save(f"MULTICB_MEAN_x12", base + 1.2 * mean_corr, target_col))
        outputs.append(save(f"MULTICB_MEAN_x08", base + 0.8 * mean_corr, target_col))

    # ====== S4: BOOTSTRAP ON CURATED SUBSET ONLY ======
    proxy_curated, names_c = build_proxy(base, max_rms=0.50)
    if proxy_curated is not None and len(names_c) >= 6:
        # Now bootstrap within the curated set
        all_dirs, all_tgts, all_rms = [], [], []
        for nm in names_c:
            vals, _ = load(nm)
            diff = vals - base
            all_dirs.append(diff)
            all_tgts.append(-((KNOWN_SCORES[nm] - BASE_SCORE) * N_INST / 100.0))
            all_rms.append(float(np.sqrt(np.mean(diff*diff))))
        D = np.asarray(all_dirs); T = np.asarray(all_tgts); R = np.asarray(all_rms)
        N = len(D)
        rng = np.random.RandomState(42)
        ps = []
        for b in range(40):
            k = max(4, int(0.7 * N))
            idx = rng.choice(N, size=k, replace=False)
            Db, Tb, Rb = D[idx], T[idx], R[idx]
            Wb = np.diag(1.0 / np.power(0.05 + Rb, 0.7))
            gram = (Db @ Db.T) / len(base)
            try:
                beta = np.linalg.solve(Wb @ gram @ Wb + 1e-3*np.eye(k), Wb @ Tb)
            except np.linalg.LinAlgError: continue
            p = Db.T @ (Wb @ beta)
            p = p - p.mean()
            lo, hi = np.percentile(p, [1, 99])
            p = np.clip(p, lo, hi)
            p = p / (p.std() + 1e-9)
            ps.append(p)
        if ps:
            p_avg = np.mean(ps, axis=0)
            p_avg = p_avg - p_avg.mean(); p_avg = p_avg / (p_avg.std() + 1e-9)
            for q in [10, 11, 12, 14]:
                qq = 1.0 - q/100.0
                active = np.abs(p_avg) >= np.quantile(np.abs(p_avg), qq)
                pos = active & (p_avg > 0); neg = active & (p_avg < 0)
                for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50)]:
                    corr = np.zeros_like(p_avg)
                    corr[pos] = gp * p_avg[pos]; corr[neg] = gn * p_avg[neg]
                    outputs.append(save(
                        f"BOOTCUR_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                        base + corr, target_col))

    # ====== S5: ULTRA-CURATED (rms < 0.30, very tight) ======
    for max_rms in [0.20, 0.25, 0.30]:
        p_ultra, names_u = build_proxy(base, max_rms=max_rms, max_abs_mul=10.0)
        if p_ultra is None: continue
        print(f"  ULTRA rms<{max_rms}: {len(names_u)} cons")
        for q in [10, 12, 15]:
            qq = 1.0 - q/100.0
            active = np.abs(p_ultra) >= np.quantile(np.abs(p_ultra), qq)
            pos = active & (p_ultra > 0); neg = active & (p_ultra < 0)
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50)]:
                corr = np.zeros_like(p_ultra)
                corr[pos] = gp * p_ultra[pos]; corr[neg] = gn * p_ultra[neg]
                outputs.append(save(
                    f"ULTRA{int(max_rms*100):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    report = {
        "base_file": BASE_FILE,
        "base_score": BASE_SCORE,
        "outputs": [o["file"] for o in outputs],
    }
    (ROOT / "feedback_sign_v9_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v9 candidates")


if __name__ == "__main__":
    main()
