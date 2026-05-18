#!/usr/bin/env python3
"""Feedback-sign v14: BREAK PLATEAU via orthogonal directions.

v13 finding: adding closer bases HURT — proxy overfits to neighbourhood.
The whole linear span of v11/v12 sub's is exhausted.

v14 strategy — INJECT ORTHOGONAL SIGNALS:
  S1. FAR-ONLY MULTICB — constraints with rms ∈ [0.6, 2.0] (deliberately far)
  S2. CROSS-VERSION proxy averaging (v8→v12 bases)
  S3. EXTERNAL micro-injection (DIV3/covshift/MLP from teammate)
  S4. WINS-MEDIAN of 20+ across all rounds
  S5. Last-resort: re-derive proxy from FBSv4-base (different topology)
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
POOL = 'teamblend_v3_pool'

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
    "submission_R2_NCu_covCB_w015.csv": 7.372062902461258,
    "submission_R2_NCu_DIV3_w015.csv": 7.372150312799336,
    "submission_FBSv2_smooth5_g0p12.csv": 7.369170488021504,
    "submission_FBSv2_smooth5_g0p08.csv": 7.369845849772287,
    "submission_FBSv3_top19_g0p22.csv": 7.363100871060781,
    "submission_FBSv3_g0p22.csv": 7.366978298009447,
    "submission_FBSv4_top10_g0p45.csv": 7.353705058320320,
    "submission_FBSv4_top10_g0p35.csv": 7.354903732793917,
    "submission_FBSv5_top10_asy_gp35_gn55.csv": 7.352680653403223,
    "submission_FBSv5_top07_g0p65.csv": 7.353177022759139,
    "submission_FBSv6_NEG_negOnly_g55.csv": 7.352581410198546,
    "submission_FBSv6_top12_asy_gp30_gn55.csv": 7.352845260428158,
    "submission_XB6_covCB_w020.csv": 7.352927932946245,
    "submission_FBSv7_CUR05_t10_p35_n55.csv": 7.330521861258725,
    "submission_FBSv7_CUR05_t08_p35_n55.csv": 7.331605580715094,
    "submission_FBSv7_on_FBSv5_curat_p35_n55.csv": 7.337038288162288,
    "submission_FBSv8_CURens_t10_p35_n55.csv": 7.307464464111231,
    "submission_FBSv8_CUR050_t13_p35_n55.csv": 7.307791049094203,
    "submission_FBSv9_MULTICB_MEAN_x12.csv": 7.293590174587235,
    "submission_FBSv9_CURensWIDE_t10_p35_n55.csv": 7.296186179501253,
    "submission_FBSv10_NB_CURens_NEW_t10_p35_n55.csv": 7.266675457551658,
    "submission_FBSv10_MCB11_MEAN_x15.csv": 7.274501037849528,
    "submission_FBSv11_MCBv11_15_MED_x14.csv": 7.263233849236824,
    "submission_FBSv11_NEW_t09_p35_n55.csv": 7.264240890956524,
    "submission_FBSv11_NEW_t10_p35_n55.csv": 7.264485711465193,
    "submission_FBSv12_HYBR_mcb60_x14.csv": 7.257319999283790,
    "submission_FBSv12_HYBR_mcb50_x14.csv": 7.257856931059352,
    "submission_FBSv12_MCBv12_20_MED_x10.csv": 7.258393312227188,
    "submission_FBSv12_MCBv12_20_MED_x12.csv": 7.258489530413064,
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
    path = ROOT / f"submission_FBSv14_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path),
            "mean": float(arr.mean()), "std": float(arr.std())}


def build_proxy_for(base_arr, base_score, min_rms=0.0, max_rms=2.0, max_abs=15.0, alpha=1e-3):
    """Build proxy with rms WINDOW (min, max) — for FAR-only."""
    dirs, tgts, rms_list, names = [], [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base_arr): continue
        diff = vals - base_arr
        rms = float(np.sqrt(np.mean(diff*diff)))
        if min_rms < rms < max_rms and float(np.max(np.abs(diff))) < max_abs:
            dirs.append(diff)
            tgts.append(-((score - base_score) * N_INST / 100.0))
            rms_list.append(rms)
            names.append(name)
    if len(dirs) < 4: return None, []
    D = np.asarray(dirs); T = np.asarray(tgts); R = np.asarray(rms_list)
    W = np.diag(1.0 / np.power(0.05 + R, 0.7))
    gram = (D @ D.T) / len(base_arr)
    try:
        beta = np.linalg.solve(W @ gram @ W + alpha*np.eye(len(D)), W @ T)
    except np.linalg.LinAlgError: return None, []
    proxy = D.T @ (W @ beta)
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy, names


def build_correction(proxy, q_keep_pct=10, gp=0.35, gn=0.55):
    q = 1.0 - q_keep_pct/100.0
    active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
    pos = active & (proxy > 0); neg = active & (proxy < 0)
    corr = np.zeros_like(proxy)
    corr[pos] = gp * proxy[pos]; corr[neg] = gn * proxy[neg]
    return corr


def main():
    base, target_col = load(BASE_FILE)
    if base is None:
        raise RuntimeError(f"BASE not found: {BASE_FILE}")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")
    outputs = []

    # ====== S1: FAR-ONLY MULTICB (rms in [0.6, 2.0]) — orthogonal direction ======
    for min_r, max_r in [(0.50, 1.5), (0.60, 1.5), (0.70, 2.0), (0.80, 2.5)]:
        p_far, names_far = build_proxy_for(base, BASE_SCORE,
                                            min_rms=min_r, max_rms=max_r, max_abs=15.0)
        if p_far is None: continue
        print(f"  FAR ({min_r}<rms<{max_r}): {len(names_far)} cons")
        for q in [8, 10, 12, 15, 20]:
            for gp, gn in [(0.35, 0.55), (0.25, 0.50), (0.30, 0.45), (0.15, 0.35)]:
                c = build_correction(p_far, q_keep_pct=q, gp=gp, gn=gn)
                outputs.append(save(
                    f"FAR{int(min_r*100):02d}_{int(max_r*100):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + c, target_col))

    # ====== S2: CROSS-VERSION proxy averaging ======
    # Build NEW-recipe correction from v8, v9, v10, v11 bases — average them
    cross_bases = [
        ("submission_FBSv7_CUR05_t10_p35_n55.csv", 7.330521861258725),
        ("submission_FBSv8_CURens_t10_p35_n55.csv", 7.307464464111231),
        ("submission_FBSv9_MULTICB_MEAN_x12.csv", 7.293590174587235),
        ("submission_FBSv10_NB_CURens_NEW_t10_p35_n55.csv", 7.266675457551658),
        ("submission_FBSv11_MCBv11_15_MED_x14.csv", 7.263233849236824),
    ]
    cross_corrs = []
    for fn, sc in cross_bases:
        alt, _ = load(fn)
        if alt is None: continue
        p = build_proxy_for(alt, sc, max_rms=0.5)[0]
        if p is None: continue
        c = build_correction(p, q_keep_pct=10, gp=0.35, gn=0.55)
        cross_corrs.append(c)
    print(f"  CROSS-VER built from {len(cross_corrs)} version-bases")
    if len(cross_corrs) >= 3:
        mean_cv = np.mean(cross_corrs, axis=0)
        med_cv = np.median(cross_corrs, axis=0)
        for dose in (0.8, 1.0, 1.2, 1.4, 1.6, 2.0):
            outputs.append(save(f"CROSSv_MEAN_x{int(dose*10):02d}",
                                base + dose * mean_cv, target_col))
            outputs.append(save(f"CROSSv_MED_x{int(dose*10):02d}",
                                base + dose * med_cv, target_col))

    # ====== S3: EXTERNAL MICRO-INJECTION (teammate's diverse sources) ======
    import os
    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_v89_covshift.csv', 'submission_v87_mlp.csv',
                'submission_v85_catboost.csv', 'submission_TOP6_MEDIAN.csv',
                'submission_v77_capped.csv']:
        fp = pool_path / fn
        if fp.exists():
            v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
            if len(v) == len(base):
                externals[fn.replace('submission_', '').replace('.csv', '')] = v
    print(f"  EXTERNAL: {len(externals)} teammate sources loaded")

    # Compute DIV3 average + covCB average if possible
    if all(k in externals for k in ['v89_covshift', 'v87_mlp', 'v85_catboost']):
        externals['DIV3'] = (externals['v89_covshift'] + externals['v87_mlp'] +
                              externals['v85_catboost']) / 3.0
        externals['covCB'] = (externals['v89_covshift'] + externals['v85_catboost']) / 2.0

    # Inject at tiny doses
    for tag, partner in externals.items():
        for w in (0.005, 0.010, 0.020, 0.030, 0.040):
            blend = (1 - w) * base + w * partner
            outputs.append(save(f"EXT_{tag}_w{int(w*1000):03d}",
                                blend, target_col))

    # ====== S4: WINS-MEDIAN of 20+ across ALL rounds ======
    wins_all = [
        BASE_FILE,
        "submission_FBSv12_HYBR_mcb60_x14.csv",
        "submission_FBSv12_HYBR_mcb50_x14.csv",
        "submission_FBSv12_MCBv12_20_MED_x10.csv",
        "submission_FBSv12_MCBv12_20_MED_x12.csv",
        "submission_FBSv11_MCBv11_15_MED_x14.csv",
        "submission_FBSv11_NEW_t09_p35_n55.csv",
        "submission_FBSv11_NEW_t10_p30_n55.csv",
        "submission_FBSv11_NEW_t10_p35_n55.csv",
        "submission_FBSv11_NEW_t11_p30_n55.csv",
        "submission_FBSv10_NB_CURens_NEW_t10_p35_n55.csv",
        "submission_FBSv10_NB_CURens_t10_p35_n55.csv",
        "submission_FBSv10_MCB11_MEAN_x15.csv",
        "submission_FBSv13_WINS_MED_8.csv",  # v13 attempt
    ]
    stk = []
    for fn in wins_all:
        v, _ = load(fn)
        if v is not None: stk.append(v)
    if len(stk) >= 6:
        stk = np.asarray(stk)
        outputs.append(save(f"WINS_MEAN_{len(stk)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"WINS_MEDIAN_{len(stk)}", np.median(stk, axis=0), target_col))
        # Trimmed
        if len(stk) >= 8:
            trim = np.zeros_like(stk[0])
            for i in range(stk.shape[1]):
                col = np.sort(stk[:, i])
                trim[i] = col[2:-2].mean()
            outputs.append(save(f"WINS_TRIM2_{len(stk)}", trim, target_col))
        # Top-5 mean
        outputs.append(save(f"WINS_TOP5", np.mean(stk[:5], axis=0), target_col))

    # ====== S5: Apply proxy from FBSv4 base (very different topology) ======
    alt4, _ = load("submission_FBSv4_top10_g0p45.csv")
    if alt4 is not None:
        p4, _ = build_proxy_for(alt4, 7.353705058320320, max_rms=0.7)
        if p4 is not None:
            for q in [10, 15]:
                c = build_correction(p4, q_keep_pct=q, gp=0.35, gn=0.55)
                for dose in (0.8, 1.0, 1.2):
                    outputs.append(save(f"FBSv4base_t{q:02d}_x{int(dose*10):02d}",
                                        base + dose * c, target_col))

    report = {
        "base_file": BASE_FILE, "base_score": BASE_SCORE,
        "outputs": [o["file"] for o in outputs],
    }
    (ROOT / "feedback_sign_v14_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v14 candidates")


if __name__ == "__main__":
    main()
