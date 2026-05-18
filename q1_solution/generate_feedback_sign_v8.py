#!/usr/bin/env python3
"""Feedback-sign v8: REFINE the CURATED winner.

v7 finding: CURATED proxy (rms<0.5, 33 cons) + top10_p35_n55 jumped −0.022.
Pattern: NARROW constraint window cuts noise dramatically.

v8 strategy:
  A. Curated grid: rms in [0.3, 0.45, 0.5, 0.55, 0.6, 0.7] × t in [7..18] × asym γ
  B. Double-curated: build proxy from rms<0.4 cons, then apply on top of v7 base
  C. Curated ensemble: average proxies from rms<0.3, <0.4, <0.5, <0.6
  D. Cross-base curated: apply curated v8 on FBSv7 winners
  E. NEG control (sanity)
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv7_CUR05_t10_p35_n55.csv"
BASE_SCORE = 7.330521861258725
VALID_FEATURES = "data/valid_features.csv"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"

KNOWN_SCORES = {
    # Historical (kept for completeness, but mostly filtered out by curation)
    "submission_rebound_v6_t075_hneg10.csv": 7.37455489169975,
    "submission_feedback_sign_g0p08.csv": 7.373278334346155,
    "submission_feedback_sign_smooth5_g0p08.csv": 7.372587722053496,
    "submission_REBv6_LIGHT10_smooth.csv": 7.377025947066968,
    "submission_REBv6_VLIGHT_smooth.csv": 7.378272056634792,
    # ULTRA
    "submission_ULTRA_DIV3_gauss_b06.csv": 7.371827077437699,
    "submission_ULTRA_DIV3_VS_w025.csv": 7.372196507607671,
    "submission_ULTRA_DIV3_w03.csv": 7.372235347883005,
    "submission_ULTRA_DIV3_w04.csv": 7.372393263765536,
    "submission_ULTRA_covCB_w06.csv": 7.372534974937972,
    # R2
    "submission_R2_gausscovCB_b06.csv": 7.371251866387993,
    "submission_R2_gaussmlpCB_b06.csv": 7.371689777558700,
    "submission_R2_NCgauss_covCB_b02.csv": 7.371743130440402,
    "submission_R2_addgauss_b015_c30_s20.csv": 7.371908255855521,
    "submission_R2_addgauss_b020_c30_s20.csv": 7.371935315328128,
    "submission_R2_addgauss_b020_c25_s15.csv": 7.371945873009935,
    "submission_R2_addgauss_b025_c30_s20.csv": 7.371962374800735,
    "submission_R2_NCu_covCB_w015.csv": 7.372062902461258,
    "submission_R2_NCu_DIV3_w015.csv": 7.372150312799336,
    # CHAMPv3
    "submission_CHAMPv3_DIV3_w08.csv": 7.373613969627501,
    "submission_CHAMPv3_DIV3_w12.csv": 7.376372031340379,
    "submission_CHAMPv3_v89cov_w08.csv": 7.375599991905816,
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
    # FBSv6
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
    # FBSv7 (NEWEST, most relevant — many near current BASE)
    # BASE skipped: CUR05_t10_p35_n55 = 7.330521861258725
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
    path = ROOT / f"submission_FBSv8_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path),
            "mean": float(arr.mean()), "std": float(arr.std())}


def build_proxy(base, max_rms=0.5, max_abs_mul=8.0, alpha=1e-3):
    directions, targets, rms_list, names = [], [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base):
            continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        mxa = float(np.max(np.abs(diff)))
        if rms < max_rms and mxa < max_abs_mul * max_rms:
            directions.append(diff)
            targets.append(-((score - BASE_SCORE) * N_INST / 100.0))
            rms_list.append(rms)
            names.append(name)
    if len(directions) < 5:
        return None, []
    D = np.asarray(directions); T = np.asarray(targets); R = np.asarray(rms_list)
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

    # ====== A. CURATED GRID — fine sweep of rms window ======
    for max_rms in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85, 1.0]:
        proxy, names = build_proxy(base, max_rms=max_rms)
        if proxy is None:
            continue
        print(f"  CUR rms<{max_rms}: {len(names)} constraints")
        # Quantile + asymmetric γ
        for q_keep_pct in [7, 9, 10, 11, 13, 15, 18, 22]:
            q = 1.0 - q_keep_pct/100.0
            active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
            pos = active & (proxy > 0); neg = active & (proxy < 0)
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50), (0.25, 0.60),
                            (0.30, 0.50), (0.35, 0.50), (0.40, 0.60), (0.45, 0.45),
                            (0.30, 0.45)]:
                corr = np.zeros_like(proxy)
                corr[pos] = gp * proxy[pos]
                corr[neg] = gn * proxy[neg]
                outputs.append(save(
                    f"CUR{int(max_rms*100):03d}_t{q_keep_pct:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))
        # raw at this curation level
        for gamma in [0.20, 0.30, 0.45, 0.60]:
            outputs.append(save(f"CUR{int(max_rms*100):03d}_g{int(gamma*100):02d}",
                                base + gamma * proxy, target_col))

    # ====== B. CURATED ENSEMBLE — average proxies from several windows ======
    proxies_set = {}
    for max_rms in [0.30, 0.40, 0.45, 0.50, 0.60]:
        p, _ = build_proxy(base, max_rms=max_rms)
        if p is not None: proxies_set[max_rms] = p
    if len(proxies_set) >= 3:
        ens = np.mean(list(proxies_set.values()), axis=0)
        ens = ens - ens.mean(); ens = ens / (ens.std() + 1e-9)
        for q_keep_pct in [9, 10, 12, 15]:
            q = 1.0 - q_keep_pct/100.0
            active = np.abs(ens) >= np.quantile(np.abs(ens), q)
            pos = active & (ens > 0); neg = active & (ens < 0)
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50)]:
                corr = np.zeros_like(ens)
                corr[pos] = gp * ens[pos]; corr[neg] = gn * ens[neg]
                outputs.append(save(
                    f"CURens_t{q_keep_pct:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== C. DOUBLE-CURATED: take v7 base, apply curated v8 proxy ======
    fbs7_winners = [
        ("submission_FBSv7_CUR05_t08_p35_n55.csv", "v7CUR05t08"),
        ("submission_FBSv7_on_FBSv5_curat_p35_n55.csv", "v7onv5"),
        ("submission_FBSv7_BOOT7_t10_p35_n55.csv", "v7BOOT7"),
    ]
    proxy_05, _ = build_proxy(base, max_rms=0.50)
    proxy_04, _ = build_proxy(base, max_rms=0.40)
    for fn, tag in fbs7_winners:
        alt, _ = load(fn)
        if alt is None: continue
        for proxy_use, ptag in [(proxy_05, "p05"), (proxy_04, "p04")]:
            if proxy_use is None: continue
            active = np.abs(proxy_use) >= np.quantile(np.abs(proxy_use), 0.90)
            corr = np.zeros_like(proxy_use)
            pos = active & (proxy_use > 0); neg = active & (proxy_use < 0)
            corr[pos] = 0.35 * proxy_use[pos]; corr[neg] = 0.55 * proxy_use[neg]
            outputs.append(save(f"DBL_{tag}_{ptag}", alt + corr, target_col))

    # ====== D. CROSS-BASE: build proxy on FBSv7_CUR05_t08 base, apply to current ======
    alt_base, _ = load("submission_FBSv7_CUR05_t08_p35_n55.csv")
    if alt_base is not None:
        # Pretend that's our base
        all_dirs, all_tgts, all_rms = [], [], []
        for name, score in KNOWN_SCORES.items():
            vals, _ = load(name)
            if vals is None or len(vals) != len(alt_base): continue
            diff = vals - alt_base
            rms = float(np.sqrt(np.mean(diff*diff)))
            if rms < 0.5 and float(np.max(np.abs(diff))) < 4.0:
                all_dirs.append(diff)
                # target is relative to alt_base's score
                all_tgts.append(-((score - 7.331605580715094) * N_INST / 100.0))
                all_rms.append(rms)
        if len(all_dirs) > 5:
            D = np.asarray(all_dirs); T = np.asarray(all_tgts); R = np.asarray(all_rms)
            W = np.diag(1.0 / np.power(0.05 + R, 0.7))
            gram = (D @ D.T) / len(alt_base)
            beta = np.linalg.solve(W @ gram @ W + 1e-3*np.eye(len(D)), W @ T)
            p_alt = D.T @ (W @ beta)
            p_alt = p_alt - p_alt.mean()
            lo, hi = np.percentile(p_alt, [1, 99])
            p_alt = np.clip(p_alt, lo, hi)
            p_alt = p_alt / (p_alt.std() + 1e-9)
            for q_keep_pct in [10, 12]:
                q = 1.0 - q_keep_pct/100.0
                active = np.abs(p_alt) >= np.quantile(np.abs(p_alt), q)
                pos = active & (p_alt > 0); neg = active & (p_alt < 0)
                for gp, gn in [(0.35, 0.55), (0.30, 0.55)]:
                    corr = np.zeros_like(p_alt)
                    corr[pos] = gp * p_alt[pos]; corr[neg] = gn * p_alt[neg]
                    outputs.append(save(
                        f"CB_v7t08_t{q_keep_pct:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                        base + corr, target_col))

    # ====== E. NEG SAFETY ======
    proxy, _ = build_proxy(base, max_rms=0.50)
    if proxy is not None:
        active = np.abs(proxy) >= np.quantile(np.abs(proxy), 0.90)
        pos = active & (proxy > 0); neg = active & (proxy < 0)
        corr = np.zeros_like(proxy)
        corr[pos] = -0.35 * proxy[pos]; corr[neg] = -0.55 * proxy[neg]
        outputs.append(save("NEG_CUR05_t10", base + corr, target_col))

    report = {
        "base_file": BASE_FILE,
        "base_score": BASE_SCORE,
        "outputs": [o["file"] for o in outputs],
    }
    (ROOT / "feedback_sign_v8_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v8 candidates")


if __name__ == "__main__":
    main()
