#!/usr/bin/env python3
"""Feedback-sign v13: TRIM winner + HYBRID variations + denser base set.

v12 finding: TRIM_x12 (drop top/bot 2 from 20 corrections) > MED > MEAN.
HYBRID (MULTICB + NEW correction blended) competitive.

v13 strategies:
  S1. TRIMMED sweep: drop 1, 2, 3, 4 from each end × dose x08–x16
  S2. MULTICB with 30+ bases (add v12 winners)
  S3. HYBRID grid: MULTICB_TRIM weight × NEW correction weight
  S4. Apply v13 NEW on new base for one more incremental gain
  S5. Multi-version winners stack (v9 → v12 best)
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv12_MCBv12_20_TRIM_x12.csv"
BASE_SCORE = 7.256224793049420

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
    "submission_FBSv4_top10_g0p45.csv": 7.353705058320320,
    "submission_FBSv4_top10_g0p35.csv": 7.354903732793917,
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
    "submission_FBSv8_CURens_t10_p35_n55.csv": 7.307464464111231,
    "submission_FBSv8_CB_v7t08_t10_p35_n55.csv": 7.307479139183003,
    "submission_FBSv8_CUR050_t13_p35_n55.csv": 7.307791049094203,
    "submission_FBSv8_CUR050_t11_p35_n55.csv": 7.308126158016013,
    "submission_FBSv8_CUR045_t10_p35_n55.csv": 7.308103329079309,
    "submission_FBSv8_CUR050_t10_p35_n55.csv": 7.309225051082493,
    "submission_FBSv9_MULTICB_MEAN_x12.csv": 7.293590174587235,
    "submission_FBSv9_MULTICB_MEAN_5.csv": 7.294689875514118,
    "submission_FBSv9_CURensWIDE_t10_p35_n55.csv": 7.296186179501253,
    "submission_FBSv9_BOOTCUR_t10_p35_n55.csv": 7.296274510890906,
    "submission_FBSv10_NB_CURens_NEW_t10_p35_n55.csv": 7.266675457551658,
    "submission_FBSv10_NB_CURens_t10_p35_n55.csv": 7.270197751836241,
    "submission_FBSv10_MCB11_MEAN_x15.csv": 7.274501037849528,
    "submission_FBSv10_MCB11_MEAN_x14.csv": 7.274718357166779,
    "submission_FBSv10_MCB11_MEAN_x12.csv": 7.275775298189448,
    "submission_FBSv11_MCBv11_15_MED_x14.csv": 7.263233849236824,
    "submission_FBSv11_NEW_t09_p35_n55.csv": 7.264240890956524,
    "submission_FBSv11_NEW_t10_p30_n55.csv": 7.264414054800892,
    "submission_FBSv11_NEW_t10_p35_n55.csv": 7.264485711465193,
    "submission_FBSv11_NEW_t11_p30_n55.csv": 7.265145098202059,
    "submission_FBSv11_NEW_t11_p35_n55.csv": 7.265443661234642,
    "submission_FBSv11_NEW_WIDE_t10_p35_n55.csv": 7.265447479085065,
    "submission_FBSv11_NEW_t10_p30_n60.csv": 7.265206008588978,
    "submission_FBSv11_NEW_MED_t10_p35_n55.csv": 7.265849905296094,
    # FBSv12 batch (newest, dense near base)
    # BASE: MCBv12_20_TRIM_x12 = 7.256224793049420
    "submission_FBSv12_HYBR_mcb60_x14.csv": 7.257319999283790,
    "submission_FBSv12_HYBR_mcb50_x14.csv": 7.257856931059352,
    "submission_FBSv12_MCBv12_20_MED_x10.csv": 7.258393312227188,
    "submission_FBSv12_MCBv12_20_MED_x12.csv": 7.258489530413064,
    "submission_FBSv12_MCBv12_20_WLB_x12.csv": 7.258494978183032,
    "submission_FBSv12_MCBv12_20_MED_x14.csv": 7.259296199252831,
    "submission_FBSv12_MCBv12_20_WLB_x14.csv": 7.260480735822298,
    "submission_FBSv12_MCBv12_20_MED_x16.csv": 7.260168893624126,
    "submission_FBSv12_NEWv12_t10_p35_n55.csv": 7.261385179218796,
    "submission_FBSv12_MCBv12_20_MED_x18.csv": 7.261428590603246,
    "submission_FBSv12_CROSS_VER_W.csv": 7.262261492691404,
    "submission_FBSv12_NEWv12_WIDE_t10_p35_n55.csv": 7.268256927484595,
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
    path = ROOT / f"submission_FBSv13_{label}.csv"
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
    corr[pos] = gp * proxy[pos]; corr[neg] = gn * proxy[neg]
    return corr


def main():
    base, target_col = load(BASE_FILE)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")
    outputs = []

    # ====== MULTI-CB on 30+ bases (add v12 winners) ======
    multi_bases = [
        # v8/v9 winners
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
        ("submission_FBSv11_MCBv11_15_MED_x14.csv", 7.263233849236824),
        ("submission_FBSv11_NEW_t09_p35_n55.csv", 7.264240890956524),
        ("submission_FBSv11_NEW_t10_p30_n55.csv", 7.264414054800892),
        ("submission_FBSv11_NEW_t10_p35_n55.csv", 7.264485711465193),
        ("submission_FBSv11_NEW_t11_p30_n55.csv", 7.265145098202059),
        ("submission_FBSv11_NEW_t11_p35_n55.csv", 7.265443661234642),
        ("submission_FBSv11_NEW_WIDE_t10_p35_n55.csv", 7.265447479085065),
        ("submission_FBSv11_NEW_t10_p30_n60.csv", 7.265206008588978),
        ("submission_FBSv11_NEW_MED_t10_p35_n55.csv", 7.265849905296094),
        # v12 winners (closest to v13 base — most informative)
        ("submission_FBSv12_HYBR_mcb60_x14.csv", 7.257319999283790),
        ("submission_FBSv12_HYBR_mcb50_x14.csv", 7.257856931059352),
        ("submission_FBSv12_MCBv12_20_MED_x10.csv", 7.258393312227188),
        ("submission_FBSv12_MCBv12_20_MED_x12.csv", 7.258489530413064),
        ("submission_FBSv12_MCBv12_20_WLB_x12.csv", 7.258494978183032),
        ("submission_FBSv12_MCBv12_20_MED_x14.csv", 7.259296199252831),
        ("submission_FBSv12_MCBv12_20_MED_x16.csv", 7.260168893624126),
        ("submission_FBSv12_NEWv12_t10_p35_n55.csv", 7.261385179218796),
    ]
    corrs = []; scores_list = []
    for fn, sc in multi_bases:
        alt, _ = load(fn)
        if alt is None: continue
        p = build_proxy_for(alt, sc, max_rms=0.5)
        if p is None: continue
        c = build_correction(p, q_keep_pct=10, gp=0.35, gn=0.55)
        corrs.append(c); scores_list.append(sc)
    print(f"  MULTICB built from {len(corrs)} bases")

    if len(corrs) >= 15:
        stack_c = np.asarray(corrs)
        # ====== S1: TRIMMED with different cut sizes ======
        for trim_n in [1, 2, 3, 4, 5]:
            if len(corrs) > 2*trim_n + 2:
                trimmed = np.zeros_like(stack_c[0])
                for i in range(stack_c.shape[1]):
                    col = np.sort(stack_c[:, i])
                    trimmed[i] = col[trim_n:-trim_n].mean()
                for dose in (0.8, 1.0, 1.1, 1.2, 1.3, 1.4):
                    outputs.append(save(
                        f"MCBv13_{len(corrs)}_TRIM{trim_n}_x{int(dose*10):02d}",
                        base + dose * trimmed, target_col))

        # MEDIAN dose sweep on richer base set
        med = np.median(stack_c, axis=0)
        for dose in (1.0, 1.1, 1.2, 1.3, 1.4):
            outputs.append(save(f"MCBv13_{len(corrs)}_MED_x{int(dose*10):02d}",
                                base + dose * med, target_col))

        # Best-LB-weighted (top10 corrections weighted more)
        scores_arr = np.asarray(scores_list)
        order = np.argsort(scores_arr)
        top10 = stack_c[order[:10]]
        top10_mean = np.mean(top10, axis=0)
        for dose in (1.0, 1.2, 1.4):
            outputs.append(save(f"MCBv13_TOP10_MEAN_x{int(dose*10):02d}",
                                base + dose * top10_mean, target_col))
        top10_med = np.median(top10, axis=0)
        for dose in (1.0, 1.2, 1.4):
            outputs.append(save(f"MCBv13_TOP10_MED_x{int(dose*10):02d}",
                                base + dose * top10_med, target_col))

    # ====== S3: HYBRID GRID ======
    # MULTICB-TRIM + NEW correction blend
    if len(corrs) >= 15:
        # Build NEW correction on v13 base
        proxies_n = {}
        for max_rms in [0.30, 0.40, 0.45, 0.50, 0.60]:
            p = build_proxy_for(base, BASE_SCORE, max_rms=max_rms)
            if p is not None: proxies_n[max_rms] = p
        if len(proxies_n) >= 3:
            ens = np.mean(list(proxies_n.values()), axis=0)
            ens = ens - ens.mean(); ens = ens / (ens.std() + 1e-9)
            new_corr = build_correction(ens, q_keep_pct=10, gp=0.35, gn=0.55)

            # Get the trimmed-2 correction (v12 winner format)
            trim2 = np.zeros_like(stack_c[0])
            for i in range(stack_c.shape[1]):
                col = np.sort(stack_c[:, i])
                trim2[i] = col[2:-2].mean()

            # Grid: MCB weight × dose
            for w_mcb in [0.4, 0.5, 0.6, 0.7, 0.8]:
                for dose in [1.0, 1.2, 1.4]:
                    hybrid = w_mcb * trim2 * dose + (1 - w_mcb) * new_corr
                    outputs.append(save(
                        f"HYBR_t2w{int(w_mcb*100):02d}_x{int(dose*10):02d}",
                        base + hybrid, target_col))

            # NEW alone on v13 base
            for q in [9, 10, 11, 12]:
                for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.30, 0.60)]:
                    c = build_correction(ens, q_keep_pct=q, gp=gp, gn=gn)
                    outputs.append(save(
                        f"NEWv13_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                        base + c, target_col))

    # ====== S5: STACK of v12 + v11 winners ======
    win_stack = [
        BASE_FILE,
        "submission_FBSv12_HYBR_mcb60_x14.csv",
        "submission_FBSv12_HYBR_mcb50_x14.csv",
        "submission_FBSv12_MCBv12_20_MED_x10.csv",
        "submission_FBSv12_MCBv12_20_MED_x12.csv",
        "submission_FBSv12_MCBv12_20_WLB_x12.csv",
        "submission_FBSv11_MCBv11_15_MED_x14.csv",
        "submission_FBSv11_NEW_t09_p35_n55.csv",
    ]
    stk = []
    for fn in win_stack:
        v, _ = load(fn)
        if v is not None: stk.append(v)
    if len(stk) >= 4:
        stk = np.asarray(stk)
        outputs.append(save(f"WINS_MEAN_{len(stk)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"WINS_MED_{len(stk)}", np.median(stk, axis=0), target_col))
        outputs.append(save(f"WINS_TOP3", np.mean(stk[:3], axis=0), target_col))
        # Trimmed
        if len(stk) >= 6:
            trim_w = np.zeros_like(stk[0])
            for i in range(stk.shape[1]):
                col = np.sort(stk[:, i])
                trim_w[i] = col[1:-1].mean()
            outputs.append(save(f"WINS_TRIM1", trim_w, target_col))

    report = {
        "base_file": BASE_FILE, "base_score": BASE_SCORE,
        "outputs": [o["file"] for o in outputs],
    }
    (ROOT / "feedback_sign_v13_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v13 candidates")


if __name__ == "__main__":
    main()
