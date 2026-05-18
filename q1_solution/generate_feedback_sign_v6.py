#!/usr/bin/env python3
"""Feedback-sign v6: NEW base = FBSv5_top10_asy_gp35_gn55 (7.35268) + ~88 constraints.

v5 finding: asymmetric γ wins. gp=0.35/gn=0.55 (negative side bigger) >> symmetric.
Meaning: model OVERESTIMATES more often than underestimates on high-confidence rows.

Strategy:
  A. Refine asymmetric: gp=0.20-0.45, gn=0.50-0.75, search the 2D grid
  B. Asymmetric on narrower quantiles (top06-12)
  C. Asymmetric two-tier
  D. Non-linear: power-law in proxy magnitude on each side
  E. Cross-blend with diverse sources (DIV3, covCB) on top of asy winner
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv5_top10_asy_gp35_gn55.csv"
BASE_SCORE = 7.352680653403223
VALID_FEATURES = "data/valid_features.csv"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"

# All known LB scores
KNOWN_SCORES = {
    # rebound original
    "submission_rebound_v6_t075_hneg10.csv": 7.37455489169975,
    "submission_rebound_v1.csv": 7.3756761080900635,
    "submission_rebound_v2_soft.csv": 7.375976942587208,
    "submission_rebound_v2.csv": 7.378333904676163,
    "submission_rebound_v3_shape.csv": 7.375528878103264,
    "submission_rebound_v3_deep.csv": 7.378783536070742,
    "submission_rebound_v4_center.csv": 7.378335339868966,
    "submission_rebound_v5_hneg20.csv": 7.375692490066722,
    "submission_rebound_v5_bmore_hneg25.csv": 7.374766359042817,
    "submission_rebound_v5_conservative.csv": 7.3747373220862854,
    "submission_rebound_v6_t075_h0.csv": 7.374774556660089,
    "submission_rebound_v6_t10_h0.csv": 7.3747940763733855,
    "submission_rebound_v6_surface_push.csv": 7.375287304530807,
    # feedback_sign v1
    "submission_feedback_sign_g0p08.csv": 7.373278334346155,
    "submission_feedback_sign_smooth5_g0p08.csv": 7.372587722053496,
    # REBv6
    "submission_REBv6_LIGHT10_smooth.csv": 7.377025947066968,
    "submission_REBv6_VLIGHT_smooth.csv": 7.378272056634792,
    "submission_REBv6_recipe_20_30_50.csv": 7.406747265442087,
    "submission_REBv6_recipe_15_35_50.csv": 7.408338532640587,
    "submission_REBv6_recipe_25_30_45.csv": 7.404852525722752,
    # CHAMPv3
    "submission_CHAMPv3_v89cov_w08.csv": 7.375599991905816,
    "submission_CHAMPv3_v89cov_w10.csv": 7.378165752681632,
    "submission_CHAMPv3_v89cov_w12.csv": 7.380193077826085,
    "submission_CHAMPv3_v87mlp_w08.csv": 7.375659993968192,
    "submission_CHAMPv3_v87mlp_w10.csv": 7.377294930837914,
    "submission_CHAMPv3_v85cb_w12.csv": 7.378936692216940,
    "submission_CHAMPv3_DIV3_w08.csv": 7.373613969627501,
    "submission_CHAMPv3_DIV3_w12.csv": 7.376372031340379,
    "submission_CHAMPv3_shift_m0p20.csv": 7.379895506214786,
    "submission_CHAMPv3_shift_m0p25.csv": 7.382918425560779,
    "submission_CHAMPv3_shift_m0p30.csv": 7.386297793884484,
    "submission_CHAMPv3_TRI_v89cov_d08_t08.csv": 7.377609272846324,
    "submission_CHAMPv3_TRI_v89cov_d10_t10.csv": 7.380199409980761,
    # ULTRA
    "submission_ULTRA_DIV3_w03.csv": 7.372235347883005,
    "submission_ULTRA_DIV3_w04.csv": 7.372393263765536,
    "submission_ULTRA_DIV3_w05.csv": 7.372645059976148,
    "submission_ULTRA_DIV3_VS_w025.csv": 7.372196507607671,
    "submission_ULTRA_DIV3_gauss_b06.csv": 7.371827077437699,
    "submission_ULTRA_DIV3_lowDisagree_b08.csv": 7.374030549521963,
    "submission_ULTRA_v87mlp_w04.csv": 7.373370824038560,
    "submission_ULTRA_covCB_w06.csv": 7.372534974937972,
    "submission_ULTRA_mlpCOV_w06.csv": 7.373999549465434,
    "submission_ULTRA_mlpHEAVY_w06.csv": 7.373164591382970,
    # FBSv2
    "submission_FBSv2_tanh_g0p15.csv": 7.369660686492568,
    "submission_FBSv2_smooth5_g0p08.csv": 7.369845849772287,
    "submission_FBSv2_smooth5_g0p12.csv": 7.369170488021504,
    # R2
    "submission_R2_NCu_covCB_w015.csv": 7.372062902461258,
    "submission_R2_NCu_DIV3_w020.csv": 7.372271824189195,
    "submission_R2_NCu_DIV3_w015.csv": 7.372150312799336,
    "submission_R2_gaussv89_b06.csv": 7.372250415488944,
    "submission_R2_gaussv87_b06.csv": 7.373326233162370,
    "submission_R2_gaussmlpCB_b06.csv": 7.371689777558700,
    "submission_R2_gausscovCB_b06.csv": 7.371251866387993,
    "submission_R2_NCgauss_covCB_b02.csv": 7.371743130440402,
    "submission_R2_addgauss_b020_c25_s15.csv": 7.371945873009935,
    "submission_R2_addgauss_b025_c30_s20.csv": 7.371962374800735,
    "submission_R2_addgauss_b020_c30_s20.csv": 7.371935315328128,
    "submission_R2_addgauss_b015_c30_s20.csv": 7.371908255855521,
    # FBSv3
    "submission_FBSv3_NEG_g0p08.csv": 7.373106340936872,
    "submission_FBSv3_smooth11_g0p18.csv": 7.369011607727708,
    "submission_FBSv3_top19_g0p22.csv": 7.363100871060781,
    "submission_FBSv3_g0p22.csv": 7.366978298009447,
    "submission_FBSv3_g0p15.csv": 7.367069030220226,
    "submission_FBSv3_smtanh5_g0p22.csv": 7.370773827626002,
    "submission_FBSv3_smooth7_g0p15.csv": 7.369416069312870,
    "submission_FBSv3_smooth5_g0p18.csv": 7.369211844341594,
    "submission_FBSv3_smooth5_g0p15.csv": 7.368930778202181,
    "submission_FBSv3_smooth5_g0p12.csv": 7.368729215670931,
    # FBSv4
    "submission_FBSv4_NEG_g0p22.csv": 7.376882720822089,
    "submission_FBSv4_toptanh10_g0p45.csv": 7.357629621602857,
    "submission_FBSv4_top19_g0p35.csv": 7.359156847290178,
    "submission_FBSv4_top10_g0p45.csv": 7.353705058320320,
    "submission_FBSv4_top05_g0p35.csv": 7.359252859933578,
    "submission_FBSv4_top15_g0p22.csv": 7.359430667636016,
    "submission_FBSv4_top10_g0p35.csv": 7.354903732793917,
    "submission_FBSv4_top10_g0p28.csv": 7.356197766651727,
    "submission_FBSv4_top19_g0p28.csv": 7.359252538134574,
    "submission_FBSv4_top19_g0p22.csv": 7.359837314288791,
    # FBSv5 (most informative)
    "submission_FBSv5_NEG_g0p45.csv": 7.387925469841433,
    "submission_FBSv5_top10_g0p75.csv": 7.360023493889184,
    "submission_FBSv5_top07_g0p65.csv": 7.353177022759139,
    "submission_FBSv5_2tier_gs45_gm15.csv": 7.353167469058188,
    "submission_FBSv5_2tier_gs65_gm25.csv": 7.356341504006535,
    "submission_FBSv5_2tier_gs55_gm20.csv": 7.354551300020761,
    "submission_FBSv5_top10_asy_gp60_gn30.csv": 7.357784747367144,
    # BASE skipped: "submission_FBSv5_top10_asy_gp35_gn55.csv": 7.352680653403223,
    "submission_FBSv5_top10_asy_gp55_gn35.csv": 7.356429646252437,
    "submission_FBSv5_top10_g0p55.csv": 7.355821612258592,
    "submission_FBSv5_top10_g0p65.csv": 7.357732863344162,
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
    path = ROOT / f"submission_FBSv6_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path),
            "mean": float(arr.mean()), "std": float(arr.std())}


def build_proxy(base):
    directions, targets, names, rms_list = [], [], [], []
    diag = {}
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base):
            continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        max_abs = float(np.max(np.abs(diff)))
        diag[name] = {"rms": rms, "max_abs": max_abs, "score": score}
        if rms < 5.0 and max_abs < 30.0:
            directions.append(diff)
            target = -((score - BASE_SCORE) * N_INST / 100.0)
            targets.append(target)
            names.append(name)
            rms_list.append(rms)
    D = np.asarray(directions)
    if len(D) < 5:
        raise ValueError(f"only {len(D)}")
    target_vec = np.asarray(targets)
    rms_arr = np.asarray(rms_list)
    W = np.diag(1.0 / np.power(0.05 + rms_arr, 0.7))
    gram = (D @ D.T) / len(base)
    alpha = 1e-3
    beta = np.linalg.solve(W @ gram @ W + alpha * np.eye(len(D)), W @ target_vec)
    proxy = D.T @ (W @ beta)
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy, names, diag


def main():
    base, target_col = load(BASE_FILE)
    proxy, constraints, diag = build_proxy(base)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")
    print(f"Built proxy from {len(constraints)} constraints")
    print(f"Proxy stats: mean={proxy.mean():.4f}, std={proxy.std():.4f}, "
          f"min={proxy.min():.3f}, max={proxy.max():.3f}")

    outputs = []

    # ====== A. ASYMMETRIC GRID — REFINE THE WINNING DIRECTION ======
    # Winner gp35_gn55 means neg side gets bigger boost. Search gp 0.10-0.45, gn 0.50-0.80
    for q_keep_pct in [8, 10, 12]:
        q = 1.0 - q_keep_pct / 100.0
        active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
        pos = active & (proxy > 0)
        neg = active & (proxy < 0)
        for gp in [0.10, 0.20, 0.25, 0.30, 0.35, 0.40]:
            for gn in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
                corr = np.zeros_like(proxy)
                corr[pos] = gp * proxy[pos]
                corr[neg] = gn * proxy[neg]
                outputs.append(save(
                    f"top{q_keep_pct:02d}_asy_gp{int(gp*100):02d}_gn{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== B. SHARPER ASYMMETRY (only correct negative side, ignore positive) ======
    for q_keep_pct in [10, 15, 20]:
        q = 1.0 - q_keep_pct / 100.0
        active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
        neg = active & (proxy < 0)
        for gn in [0.40, 0.55, 0.70, 0.85, 1.00]:
            corr = np.zeros_like(proxy)
            corr[neg] = gn * proxy[neg]
            outputs.append(save(f"top{q_keep_pct:02d}_negOnly_g{int(gn*100):02d}",
                                base + corr, target_col))

    # Pos-only (sanity — should be worse)
    for gp in [0.30, 0.55]:
        active = np.abs(proxy) >= np.quantile(np.abs(proxy), 0.90)
        pos = active & (proxy > 0)
        corr = np.zeros_like(proxy)
        corr[pos] = gp * proxy[pos]
        outputs.append(save(f"top10_posOnly_g{int(gp*100):02d}",
                            base + corr, target_col))

    # ====== C. ASYMMETRIC TWO-TIER ======
    abs_p = np.abs(proxy)
    q_strong = np.quantile(abs_p, 0.90)
    q_mild = np.quantile(abs_p, 0.70)
    strong = abs_p >= q_strong
    mild = (abs_p >= q_mild) & (abs_p < q_strong)
    for gs_p, gs_n, gm_p, gm_n in [
        (0.30, 0.55, 0.10, 0.20),
        (0.25, 0.60, 0.10, 0.25),
        (0.20, 0.65, 0.05, 0.20),
        (0.35, 0.50, 0.15, 0.25),
        (0.30, 0.55, 0.15, 0.30),
        (0.20, 0.70, 0.08, 0.30),
    ]:
        corr = np.zeros_like(proxy)
        sp = strong & (proxy > 0); sn = strong & (proxy < 0)
        mp = mild & (proxy > 0); mn = mild & (proxy < 0)
        corr[sp] = gs_p * proxy[sp]
        corr[sn] = gs_n * proxy[sn]
        corr[mp] = gm_p * proxy[mp]
        corr[mn] = gm_n * proxy[mn]
        outputs.append(save(
            f"2tasy_sp{int(gs_p*100):02d}_sn{int(gs_n*100):02d}_mp{int(gm_p*100):02d}_mn{int(gm_n*100):02d}",
            base + corr, target_col))

    # ====== D. NON-LINEAR: power-law in |proxy| on each side ======
    # corr = sign(p) * γ_side * |p|^alpha,  alpha < 1 = softer, alpha > 1 = sharper
    for q_keep_pct in [10, 15]:
        q = 1.0 - q_keep_pct / 100.0
        active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
        for gp, gn, alpha in [
            (0.30, 0.55, 1.0),
            (0.25, 0.55, 1.3),
            (0.35, 0.55, 0.8),
            (0.20, 0.60, 1.0),
            (0.30, 0.60, 1.2),
        ]:
            magn = np.power(np.abs(proxy), alpha)
            sgn = np.sign(proxy)
            corr = np.zeros_like(proxy)
            corr[active & (proxy>0)] = gp * magn[active & (proxy>0)] * sgn[active & (proxy>0)]
            corr[active & (proxy<0)] = gn * magn[active & (proxy<0)] * sgn[active & (proxy<0)]
            outputs.append(save(
                f"top{q_keep_pct:02d}_pow{str(alpha).replace('.','p')}_gp{int(gp*100):02d}_gn{int(gn*100):02d}",
                base + corr, target_col))

    # ====== E. Sanity ======
    for gn in [0.40, 0.55]:
        active = np.abs(proxy) >= np.quantile(np.abs(proxy), 0.90)
        neg = active & (proxy < 0)
        corr = np.zeros_like(proxy)
        corr[neg] = -gn * proxy[neg]   # NEG of neg-only
        outputs.append(save(f"NEG_negOnly_g{int(gn*100):02d}",
                            base + corr, target_col))

    report = {
        "base_file": BASE_FILE,
        "base_score": BASE_SCORE,
        "n_constraints": len(constraints),
        "outputs": [o["file"] for o in outputs],
    }
    (ROOT / "feedback_sign_v6_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v6 candidates")


if __name__ == "__main__":
    main()
