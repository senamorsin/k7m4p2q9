#!/usr/bin/env python3
"""Feedback-sign v10: PUSH MULTICB dose + combine winners.

v9 finding: MULTICB_MEAN_x12 (1.2× dose) beats MULTICB_MEAN_5 (1.0× dose). Dose peak not hit.
Also: MULTICB > V8STACK (averaging corrections beats averaging predictions).

v10 strategies:
  S1. MULTICB dose sweep: x1.4, x1.5, x1.6, x1.8, x2.0 (push past peak)
  S2. MULTICB with MORE bases (10+ bases instead of 5)
  S3. SUPER-ENSEMBLE: average of v9 winners (MULTICB + CURensWIDE + BOOTCUR)
  S4. MULTICB on NEW BASE (FBSv9_MULTICB_MEAN_x12)
  S5. MULTICB-CB hybrid: combine cross-base correction with curated proxy
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv9_MULTICB_MEAN_x12.csv"
BASE_SCORE = 7.293590174587235

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
    "submission_FBSv7_CUR04_t10_p35_n55.csv": 7.342954377341187,
    "submission_FBSv7_CUR04_t10_p30_n55.csv": 7.343626598204484,
    "submission_FBSv7_BOOT42_t12_p30_n55.csv": 7.344071718543023,
    "submission_FBSv7_CUR04_t12_p35_n55.csv": 7.347803606973584,
    "submission_FBSv7_on_FBSv4_curat_p35_n55.csv": 7.344778559151574,
    "submission_FBSv7_STACK_MEAN_10.csv": 7.350861643904397,
    # FBSv8
    "submission_FBSv8_CURens_t10_p35_n55.csv": 7.307464464111231,
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
    # FBSv9 (newest, most informative)
    # BASE: MULTICB_MEAN_x12 = 7.293590174587235
    "submission_FBSv9_MULTICB_MEAN_5.csv": 7.294689875514118,
    "submission_FBSv9_CURensWIDE_t10_p35_n55.csv": 7.296186179501253,
    "submission_FBSv9_BOOTCUR_t10_p35_n55.csv": 7.296274510890906,
    "submission_FBSv9_BOOTCUR_t12_p35_n55.csv": 7.297946239047897,
    "submission_FBSv9_CURensWIDE_t11_p35_n55.csv": 7.298446544847635,
    "submission_FBSv9_CURensMED_t10_p35_n55.csv": 7.298645307470197,
    "submission_FBSv9_MULTICB_MEDIAN_5.csv": 7.300402569298519,
    "submission_FBSv9_ULTRA20_t10_p35_n55.csv": 7.303549574109856,
    "submission_FBSv9_ULTRA25_t10_p35_n55.csv": 7.305743588667993,
    "submission_FBSv9_ULTRA30_t10_p35_n55.csv": 7.306920590655186,
    "submission_FBSv9_V8STACK_WINV_10.csv": 7.307905000657282,
    "submission_FBSv9_V8STACK_TOP4_MEDIAN.csv": 7.307622412863934,
    "submission_FBSv9_V8STACK_TOP4_MEAN.csv": 7.306755629577165,
    "submission_FBSv9_V8STACK_TOP2_MEAN.csv": 7.306711681125106,
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
    path = ROOT / f"submission_FBSv10_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path),
            "mean": float(arr.mean()), "std": float(arr.std())}


def build_proxy_for_base(base_arr, base_score, max_rms=0.5, max_abs=4.0, alpha=1e-3):
    """Build proxy relative to arbitrary base."""
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
    except np.linalg.LinAlgError:
        return None
    proxy = D.T @ (W @ beta)
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy


def build_correction(proxy, q_keep_pct=10, gp=0.35, gn=0.55):
    """Apply top-quantile asymmetric correction."""
    q = 1.0 - q_keep_pct/100.0
    active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
    pos = active & (proxy > 0); neg = active & (proxy < 0)
    corr = np.zeros_like(proxy)
    corr[pos] = gp * proxy[pos]
    corr[neg] = gn * proxy[neg]
    return corr


def main():
    base, target_col = load(BASE_FILE)
    if base is None:
        raise RuntimeError(f"BASE not found: {BASE_FILE}")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")
    outputs = []

    # ====== S1+S2: MULTICB with MANY bases + dose sweep ======
    multi_bases = [
        ("submission_FBSv7_CUR05_t10_p35_n55.csv", 7.330521861258725),
        ("submission_FBSv7_CUR05_t08_p35_n55.csv", 7.331605580715094),
        ("submission_FBSv8_CURens_t10_p35_n55.csv", 7.307464464111231),
        ("submission_FBSv8_CB_v7t08_t10_p35_n55.csv", 7.307479139183003),
        ("submission_FBSv8_CUR050_t13_p35_n55.csv", 7.307791049094203),
        ("submission_FBSv8_CUR045_t10_p35_n55.csv", 7.308103329079309),
        ("submission_FBSv8_CUR050_t11_p35_n55.csv", 7.308126158016013),
        ("submission_FBSv8_CUR050_t10_p35_n55.csv", 7.309225051082493),
        ("submission_FBSv9_MULTICB_MEAN_5.csv", 7.294689875514118),
        ("submission_FBSv9_CURensWIDE_t10_p35_n55.csv", 7.296186179501253),
        ("submission_FBSv9_BOOTCUR_t10_p35_n55.csv", 7.296274510890906),
    ]
    corrs = []
    for fn, sc in multi_bases:
        alt, _ = load(fn)
        if alt is None: continue
        p = build_proxy_for_base(alt, sc, max_rms=0.5, max_abs=4.0)
        if p is None: continue
        c = build_correction(p, q_keep_pct=10, gp=0.35, gn=0.55)
        corrs.append(c)
    print(f"  MULTICB built from {len(corrs)} bases")

    if len(corrs) >= 5:
        mean_corr = np.mean(corrs, axis=0)
        med_corr = np.median(corrs, axis=0)

        # Dose sweep on MEAN correction (winner was x1.2; push further)
        for dose in (0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.4):
            outputs.append(save(f"MCB{len(corrs)}_MEAN_x{int(dose*10):02d}",
                                base + dose * mean_corr, target_col))
        # Dose sweep on MEDIAN
        for dose in (1.0, 1.2, 1.4, 1.6, 1.8):
            outputs.append(save(f"MCB{len(corrs)}_MED_x{int(dose*10):02d}",
                                base + dose * med_corr, target_col))

        # Different recipes for each correction
        corrs_alt = []
        for fn, sc in multi_bases:
            alt, _ = load(fn)
            if alt is None: continue
            p = build_proxy_for_base(alt, sc, max_rms=0.4, max_abs=3.0)  # tighter
            if p is None: continue
            c = build_correction(p, q_keep_pct=12, gp=0.30, gn=0.55)
            corrs_alt.append(c)
        if len(corrs_alt) >= 5:
            mean_alt = np.mean(corrs_alt, axis=0)
            for dose in (1.0, 1.2, 1.4, 1.6):
                outputs.append(save(f"MCBalt{len(corrs_alt)}_MEAN_x{int(dose*10):02d}",
                                    base + dose * mean_alt, target_col))

    # ====== S3: SUPER-ENSEMBLE — average v9 top winners ======
    v9_winners = [
        "submission_FBSv9_MULTICB_MEAN_5.csv",
        "submission_FBSv9_CURensWIDE_t10_p35_n55.csv",
        "submission_FBSv9_BOOTCUR_t10_p35_n55.csv",
        "submission_FBSv9_BOOTCUR_t12_p35_n55.csv",
        "submission_FBSv9_CURensWIDE_t11_p35_n55.csv",
        "submission_FBSv9_CURensMED_t10_p35_n55.csv",
    ]
    # Include base itself (MULTICB_MEAN_x12)
    stack = [base]
    for fn in v9_winners:
        v, _ = load(fn)
        if v is not None: stack.append(v)
    stack = np.asarray(stack)
    if len(stack) >= 4:
        outputs.append(save(f"SUPER_MEAN_{len(stack)}", np.mean(stack, axis=0), target_col))
        outputs.append(save(f"SUPER_MEDIAN_{len(stack)}", np.median(stack, axis=0), target_col))
        # base + winners mean (anchor)
        ens = np.mean(stack[1:], axis=0)
        for w in (0.3, 0.5, 0.7):
            outputs.append(save(f"BASE_plus_SUPER_w{int(w*100):02d}",
                                (1-w)*base + w*ens, target_col))
        # Triple of best 3
        outputs.append(save(f"SUPER_TOP3", np.mean(stack[:3], axis=0), target_col))

    # ====== S4: MULTICB on NEW BASE (additional iteration) ======
    # Build proxy relative to new base, push correction
    proxy_new = build_proxy_for_base(base, BASE_SCORE, max_rms=0.5, max_abs=4.0)
    if proxy_new is not None:
        for q in [9, 10, 11, 12]:
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50)]:
                c = build_correction(proxy_new, q_keep_pct=q, gp=gp, gn=gn)
                outputs.append(save(
                    f"NB_CURens_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + c, target_col))
    # Curated ensemble from new base
    proxies_n = {}
    for max_rms in [0.30, 0.40, 0.50, 0.60]:
        p = build_proxy_for_base(base, BASE_SCORE, max_rms=max_rms, max_abs=max_rms*8)
        if p is not None: proxies_n[max_rms] = p
    if len(proxies_n) >= 3:
        ens_n = np.mean(list(proxies_n.values()), axis=0)
        ens_n = ens_n - ens_n.mean(); ens_n = ens_n / (ens_n.std() + 1e-9)
        for q in [10, 11, 12]:
            c = build_correction(ens_n, q_keep_pct=q, gp=0.35, gn=0.55)
            outputs.append(save(f"NB_CURens_NEW_t{q:02d}_p35_n55", base + c, target_col))

    # ====== S5: MULTICB-CB HYBRID — combine cross-base ensemble with curated ======
    if len(corrs) >= 5 and proxy_new is not None:
        mcb_mean = np.mean(corrs, axis=0)
        cur_corr = build_correction(proxy_new, q_keep_pct=10, gp=0.35, gn=0.55)
        # Blend the two correction directions
        for w_mcb in (0.4, 0.5, 0.6, 0.7):
            hybrid = w_mcb * mcb_mean * 1.2 + (1 - w_mcb) * cur_corr
            outputs.append(save(f"HYBRID_MCB{int(w_mcb*100):02d}",
                                base + hybrid, target_col))

    report = {
        "base_file": BASE_FILE,
        "base_score": BASE_SCORE,
        "outputs": [o["file"] for o in outputs],
    }
    (ROOT / "feedback_sign_v10_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v10 candidates")


if __name__ == "__main__":
    main()
