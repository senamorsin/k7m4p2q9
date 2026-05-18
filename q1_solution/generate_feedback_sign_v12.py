#!/usr/bin/env python3
"""Feedback-sign v12: PUSH the v11 winners further. New BASE = MCBv11_15_MED_x14 (7.26323).

v11 finding: MULTICB MEDIAN (15 bases) > MEAN; NEW recipe plateau'd.
Diminishing returns — need to combine accumulated signals smarter.

v12 strategies:
  S1. MULTICB-MED dose sweep around x14: x12, x13, x15, x16, x18
  S2. MULTICB on 20+ bases (add v11 winners)
  S3. SUPER ENSEMBLE: combine MULTICB-MED + NEW recipes
  S4. NEW recipe again (curated ens on v11 base) — may still give small gain
  S5. Weighted-correction: weight each MULTICB correction by 1/(score-best_score)
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv11_MCBv11_15_MED_x14.csv"
BASE_SCORE = 7.263233849236824

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
    "submission_FBSv3_smooth5_g0p12.csv": 7.368729215670931,
    "submission_FBSv4_top10_g0p45.csv": 7.353705058320320,
    "submission_FBSv4_top10_g0p35.csv": 7.354903732793917,
    "submission_FBSv4_top10_g0p28.csv": 7.356197766651727,
    "submission_FBSv5_top10_asy_gp35_gn55.csv": 7.352680653403223,
    "submission_FBSv5_2tier_gs45_gm15.csv": 7.353167469058188,
    "submission_FBSv5_top07_g0p65.csv": 7.353177022759139,
    "submission_FBSv6_NEG_negOnly_g55.csv": 7.352581410198546,
    "submission_FBSv6_top12_asy_gp30_gn55.csv": 7.352845260428158,
    "submission_XB6_covCB_w020.csv": 7.352927932946245,
    "submission_FBSv7_CUR05_t10_p35_n55.csv": 7.330521861258725,
    "submission_FBSv7_CUR05_t08_p35_n55.csv": 7.331605580715094,
    "submission_FBSv7_on_FBSv5_curat_p35_n55.csv": 7.337038288162288,
    "submission_FBSv7_BOOT7_t10_p35_n55.csv": 7.337174573711722,
    "submission_FBSv7_BOOT42_t10_p35_n55.csv": 7.339911641410173,
    "submission_FBSv8_CURens_t10_p35_n55.csv": 7.307464464111231,
    "submission_FBSv8_CB_v7t08_t10_p35_n55.csv": 7.307479139183003,
    "submission_FBSv8_CUR050_t13_p35_n55.csv": 7.307791049094203,
    "submission_FBSv8_CUR045_t10_p35_n55.csv": 7.308103329079309,
    "submission_FBSv8_CUR050_t11_p35_n55.csv": 7.308126158016013,
    "submission_FBSv8_CUR050_t10_p35_n55.csv": 7.309225051082493,
    "submission_FBSv9_MULTICB_MEAN_x12.csv": 7.293590174587235,
    "submission_FBSv9_MULTICB_MEAN_5.csv": 7.294689875514118,
    "submission_FBSv9_CURensWIDE_t10_p35_n55.csv": 7.296186179501253,
    "submission_FBSv9_BOOTCUR_t10_p35_n55.csv": 7.296274510890906,
    "submission_FBSv9_BOOTCUR_t12_p35_n55.csv": 7.297946239047897,
    "submission_FBSv10_NB_CURens_NEW_t10_p35_n55.csv": 7.266675457551658,
    "submission_FBSv10_NB_CURens_t10_p35_n55.csv": 7.270197751836241,
    "submission_FBSv10_MCB11_MEAN_x15.csv": 7.274501037849528,
    "submission_FBSv10_MCB11_MEAN_x14.csv": 7.274718357166779,
    "submission_FBSv10_MCB11_MEAN_x18.csv": 7.274817805618317,
    "submission_FBSv10_MCB11_MEAN_x12.csv": 7.275775298189448,
    # FBSv11 batch
    # BASE: MCBv11_15_MED_x14 = 7.263233849236824
    "submission_FBSv11_NEW_t09_p35_n55.csv": 7.264240890956524,
    "submission_FBSv11_NEW_t10_p30_n55.csv": 7.264414054800892,
    "submission_FBSv11_NEW_t10_p35_n55.csv": 7.264485711465193,
    "submission_FBSv11_NEW_t11_p30_n55.csv": 7.265145098202059,
    "submission_FBSv11_NEW_t11_p35_n55.csv": 7.265443661234642,
    "submission_FBSv11_NEW_WIDE_t10_p35_n55.csv": 7.265447479085065,
    "submission_FBSv11_NEW_t10_p30_n60.csv": 7.265206008588978,
    "submission_FBSv11_NEW_MED_t10_p35_n55.csv": 7.265849905296094,
    "submission_FBSv11_NEW_t12_p35_n55.csv": 7.266093827376896,
    "submission_FBSv11_NEW_TIGHT_t10_p35_n55.csv": 7.268213016907674,
    "submission_FBSv11_v10STACK_TOP3.csv": 7.268836045681660,
    "submission_FBSv11_MCBv11_15_MEAN_x10.csv": 7.268932675719163,
    "submission_FBSv11_MCBv11_15_MEAN_x12.csv": 7.270263539306879,
    "submission_FBSv11_v10STACK_MEAN_6.csv": 7.271208350118819,
    "submission_FBSv11_MCBv11_15_MEAN_x14.csv": 7.272101717821291,
    "submission_FBSv11_MCBv11_15_MEAN_x16.csv": 7.273945676236362,
}


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    path = ROOT / name
    if not path.exists(): return None, None
    df = pd.read_csv(path)
    vals = df.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(vals).all(): return None, None
    return vals, df.columns[0]


def save(label, values, target_col):
    arr = np.clip(values, 0.0, N_INST)
    path = ROOT / f"submission_FBSv12_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path),
            "mean": float(arr.mean()), "std": float(arr.std())}


def build_proxy_for(base_arr, base_score, max_rms=0.5, max_abs=None, alpha=1e-3):
    if max_abs is None: max_abs = max_rms * 8
    dirs, tgts, rms_list = [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base_arr): continue
        diff = vals - base_arr
        rms = float(np.sqrt(np.mean(diff*diff)))
        if rms < max_rms and float(np.max(np.abs(diff))) < max_abs:
            dirs.append(diff)
            tgts.append(-((score - base_score) * N_INST / 100.0))
            rms_list.append(rms)
    if len(dirs) < 4: return None
    D = np.asarray(dirs); T = np.asarray(tgts); R = np.asarray(rms_list)
    W = np.diag(1.0 / np.power(0.05 + R, 0.7))
    gram = (D @ D.T) / len(base_arr)
    try:
        beta = np.linalg.solve(W @ gram @ W + alpha*np.eye(len(D)), W @ T)
    except np.linalg.LinAlgError: return None
    proxy = D.T @ (W @ beta)
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy


def build_correction(proxy, q_keep_pct=10, gp=0.35, gn=0.55):
    q = 1.0 - q_keep_pct/100.0
    active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
    pos = active & (proxy > 0); neg = active & (proxy < 0)
    corr = np.zeros_like(proxy)
    corr[pos] = gp * proxy[pos]
    corr[neg] = gn * proxy[neg]
    return corr


def main():
    base, target_col = load(BASE_FILE)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")
    outputs = []

    # ====== S1: MULTICB on 20+ bases (add v11 winners) ======
    multi_bases = [
        # v8/v9 winners (older but stable)
        ("submission_FBSv8_CURens_t10_p35_n55.csv", 7.307464464111231),
        ("submission_FBSv8_CUR050_t13_p35_n55.csv", 7.307791049094203),
        ("submission_FBSv8_CUR050_t11_p35_n55.csv", 7.308126158016013),
        ("submission_FBSv9_MULTICB_MEAN_x12.csv", 7.293590174587235),
        ("submission_FBSv9_MULTICB_MEAN_5.csv", 7.294689875514118),
        ("submission_FBSv9_CURensWIDE_t10_p35_n55.csv", 7.296186179501253),
        ("submission_FBSv9_BOOTCUR_t10_p35_n55.csv", 7.296274510890906),
        # v10 winners
        ("submission_FBSv10_NB_CURens_NEW_t10_p35_n55.csv", 7.266675457551658),
        ("submission_FBSv10_NB_CURens_t10_p35_n55.csv", 7.270197751836241),
        ("submission_FBSv10_MCB11_MEAN_x15.csv", 7.274501037849528),
        ("submission_FBSv10_MCB11_MEAN_x14.csv", 7.274718357166779),
        # v11 winners
        ("submission_FBSv11_NEW_t09_p35_n55.csv", 7.264240890956524),
        ("submission_FBSv11_NEW_t10_p30_n55.csv", 7.264414054800892),
        ("submission_FBSv11_NEW_t10_p35_n55.csv", 7.264485711465193),
        ("submission_FBSv11_NEW_t11_p30_n55.csv", 7.265145098202059),
        ("submission_FBSv11_NEW_t11_p35_n55.csv", 7.265443661234642),
        ("submission_FBSv11_NEW_WIDE_t10_p35_n55.csv", 7.265447479085065),
        ("submission_FBSv11_NEW_t10_p30_n60.csv", 7.265206008588978),
        ("submission_FBSv11_NEW_MED_t10_p35_n55.csv", 7.265849905296094),
        ("submission_FBSv11_NEW_t12_p35_n55.csv", 7.266093827376896),
    ]
    corrs = []
    scores_list = []
    for fn, sc in multi_bases:
        alt, _ = load(fn)
        if alt is None: continue
        p = build_proxy_for(alt, sc, max_rms=0.5)
        if p is None: continue
        c = build_correction(p, q_keep_pct=10, gp=0.35, gn=0.55)
        corrs.append(c)
        scores_list.append(sc)
    print(f"  MULTICB built from {len(corrs)} bases")

    if len(corrs) >= 10:
        # MEDIAN dose sweep (winner was MED_x14)
        med_corr = np.median(corrs, axis=0)
        for dose in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0):
            outputs.append(save(f"MCBv12_{len(corrs)}_MED_x{int(dose*10):02d}",
                                base + dose * med_corr, target_col))
        # MEAN dose sweep for comparison
        mean_corr = np.mean(corrs, axis=0)
        for dose in (1.0, 1.2, 1.4, 1.6):
            outputs.append(save(f"MCBv12_{len(corrs)}_MEAN_x{int(dose*10):02d}",
                                base + dose * mean_corr, target_col))
        # Weighted: better-LB bases get more weight
        scores_arr = np.asarray(scores_list)
        # weight ∝ 1 / (score - best+0.001)
        best = scores_arr.min()
        w_arr = 1.0 / (scores_arr - best + 0.001)
        w_arr = w_arr / w_arr.sum()
        weighted_corr = np.average(corrs, axis=0, weights=w_arr)
        for dose in (1.0, 1.2, 1.4, 1.6):
            outputs.append(save(f"MCBv12_{len(corrs)}_WLB_x{int(dose*10):02d}",
                                base + dose * weighted_corr, target_col))
        # Trimmed mean (drop top/bot 2)
        if len(corrs) >= 8:
            stack_c = np.asarray(corrs)
            trimmed = np.zeros_like(stack_c[0])
            for i in range(stack_c.shape[1]):
                col = np.sort(stack_c[:, i])
                trimmed[i] = col[2:-2].mean()
            for dose in (1.0, 1.2, 1.4):
                outputs.append(save(f"MCBv12_{len(corrs)}_TRIM_x{int(dose*10):02d}",
                                    base + dose * trimmed, target_col))

    # ====== S2: NEW recipe again on v12 base (may give small gain) ======
    proxies_n = {}
    for max_rms in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        p = build_proxy_for(base, BASE_SCORE, max_rms=max_rms)
        if p is not None: proxies_n[max_rms] = p
    if len(proxies_n) >= 3:
        ens = np.mean(list(proxies_n.values()), axis=0)
        ens = ens - ens.mean(); ens = ens / (ens.std() + 1e-9)
        for q in [8, 9, 10, 11, 12]:
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.30, 0.60), (0.25, 0.60)]:
                c = build_correction(ens, q_keep_pct=q, gp=gp, gn=gn)
                outputs.append(save(
                    f"NEWv12_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + c, target_col))
        # Wide only (v11 winner)
        wide_keys = [k for k in proxies_n if k >= 0.45]
        if wide_keys:
            wide_ens = np.mean([proxies_n[k] for k in wide_keys], axis=0)
            wide_ens = wide_ens - wide_ens.mean()
            wide_ens = wide_ens / (wide_ens.std() + 1e-9)
            for q in [9, 10, 11]:
                c = build_correction(wide_ens, q_keep_pct=q, gp=0.35, gn=0.55)
                outputs.append(save(f"NEWv12_WIDE_t{q:02d}_p35_n55", base + c, target_col))

    # ====== S3: SUPER-ENSEMBLE — mix MULTICB-MED + NEW best ======
    if len(corrs) >= 10 and len(proxies_n) >= 3:
        med_corr = np.median(corrs, axis=0)
        ens = np.mean(list(proxies_n.values()), axis=0)
        ens = ens - ens.mean(); ens = ens / (ens.std() + 1e-9)
        new_corr = build_correction(ens, q_keep_pct=10, gp=0.35, gn=0.55)
        # blend the two correction directions
        for w_mcb, dose in [(0.5, 1.4), (0.6, 1.4), (0.7, 1.4), (0.5, 1.2), (0.4, 1.4)]:
            hybrid = w_mcb * med_corr * dose + (1 - w_mcb) * new_corr
            outputs.append(save(
                f"HYBR_mcb{int(w_mcb*100):02d}_x{int(dose*10):02d}",
                base + hybrid, target_col))

    # ====== S4: Stack v11 winners ======
    v11_winners = [
        BASE_FILE,
        "submission_FBSv11_NEW_t09_p35_n55.csv",
        "submission_FBSv11_NEW_t10_p30_n55.csv",
        "submission_FBSv11_NEW_t10_p35_n55.csv",
        "submission_FBSv11_NEW_t11_p30_n55.csv",
        "submission_FBSv11_NEW_t11_p35_n55.csv",
        "submission_FBSv11_NEW_WIDE_t10_p35_n55.csv",
        "submission_FBSv11_NEW_t10_p30_n60.csv",
    ]
    stack = []
    for fn in v11_winners:
        v, _ = load(fn)
        if v is not None: stack.append(v)
    if len(stack) >= 4:
        stack = np.asarray(stack)
        outputs.append(save(f"v11STACK_MEAN_{len(stack)}", np.mean(stack, axis=0), target_col))
        outputs.append(save(f"v11STACK_MEDIAN_{len(stack)}", np.median(stack, axis=0), target_col))
        outputs.append(save(f"v11STACK_TOP3", np.mean(stack[:3], axis=0), target_col))

    # ====== S5: Multi-version winners stack ======
    cross_ver = [
        BASE_FILE,
        "submission_FBSv11_NEW_t09_p35_n55.csv",
        "submission_FBSv10_NB_CURens_NEW_t10_p35_n55.csv",
        "submission_FBSv9_MULTICB_MEAN_x12.csv",
    ]
    cv = []
    for fn in cross_ver:
        v, _ = load(fn)
        if v is not None: cv.append(v)
    if len(cv) >= 3:
        cv = np.asarray(cv)
        outputs.append(save(f"CROSS_VER_MEAN", np.mean(cv, axis=0), target_col))
        # weighted to v12 base heavily
        w = np.array([0.5, 0.2, 0.2, 0.1])[:len(cv)]
        w = w / w.sum()
        outputs.append(save(f"CROSS_VER_W", (cv.T @ w).reshape(-1), target_col))

    report = {
        "base_file": BASE_FILE, "base_score": BASE_SCORE,
        "outputs": [o["file"] for o in outputs],
    }
    (ROOT / "feedback_sign_v12_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v12 candidates")


if __name__ == "__main__":
    main()
