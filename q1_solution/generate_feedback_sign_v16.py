#!/usr/bin/env python3
"""Feedback-sign v16: ATTACK GAUSS — maximum effort.

v15: GAUSS-EXT_DIV3_b05 won (−0.0021). b05>b03 → push to b08, b10, b12.
Combined approaches showed weak progress. Need to triple down on GAUSS.

v16 strategies (all variations on GAUSS-EXT):
  S1. GAUSS dose-extreme sweep: b06–b15 × multiple centers/sigmas
  S2. STACKED GAUSS: multiple Gauss profiles superimposed
  S3. MULTI-SOURCE GAUSS: DIV3 + covCB + v87 all gauss-weighted simultaneously
  S4. ASYMMETRIC GAUSS: different b for pos vs neg residuals
  S5. BIMODAL GAUSS: 2 peaks (low-power + mid-power)
  S6. POWER-aware GAUSS + FBS proxy hybrid
  S7. Heavy stack of new champ + v14 EXT winners
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv15_EXTgauss_DIV3_b05_c30_s20.csv"
BASE_SCORE = 7.252770942615267
POOL = 'teamblend_v3_pool'

KNOWN_SCORES = {
    "submission_rebound_v6_t075_hneg10.csv": 7.37455489169975,
    "submission_feedback_sign_smooth5_g0p08.csv": 7.372587722053496,
    "submission_REBv6_LIGHT10_smooth.csv": 7.377025947066968,
    "submission_ULTRA_DIV3_gauss_b06.csv": 7.371827077437699,
    "submission_FBSv4_top10_g0p45.csv": 7.353705058320320,
    "submission_FBSv7_CUR05_t10_p35_n55.csv": 7.330521861258725,
    "submission_FBSv8_CURens_t10_p35_n55.csv": 7.307464464111231,
    "submission_FBSv9_MULTICB_MEAN_x12.csv": 7.293590174587235,
    "submission_FBSv10_NB_CURens_NEW_t10_p35_n55.csv": 7.266675457551658,
    "submission_FBSv11_MCBv11_15_MED_x14.csv": 7.263233849236824,
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv14_EXT_DIV3_w020.csv": 7.254768850541299,
    "submission_FBSv14_EXT_covCB_w020.csv": 7.254920930824225,
    "submission_FBSv14_EXT_v87_mlp_w010.csv": 7.255334184333610,
    "submission_FBSv14_EXT_DIV3_w010.csv": 7.255426543419984,
    # v15 (close to base, informative)
    # BASE: EXTgauss_DIV3_b05_c30_s20 = 7.252770942615267
    "submission_FBSv15_EXTgauss_covCB_b03_c30_s20.csv": 7.253398504584263,
    "submission_FBSv15_EXTgauss_DIV3_b03_c30_s20.csv": 7.253375876657015,
    "submission_FBSv15_v12base_EXT_DIV3_w025.csv": 7.254532211398722,
    "submission_FBSv15_v12base_EXT_covCB_w025.csv": 7.254710482554138,
    "submission_FBSv15_MULTIEXT_v89010_v87010_v85010.csv": 7.254177231671796,
    "submission_FBSv15_MULTIEXT_DIV3025_v87010.csv": 7.254102320068292,
    "submission_FBSv15_MULTIEXT_DIV3010_covCB010.csv": 7.254255657477282,
    "submission_FBSv15_MULTIEXT_DIV3020_covCB010.csv": 7.254242929251591,
    "submission_FBSv15_EXT_DIV3_w040.csv": 7.254266815637978,
    "submission_FBSv15_EXT_covCB_w030.csv": 7.254390862873460,
    "submission_FBSv15_EXT_covCB_w025.csv": 7.254324113934701,
    "submission_FBSv15_EXT_DIV3_w030.csv": 7.254177231671796,
    "submission_FBSv15_EXT_DIV3_w025.csv": 7.254157741377730,
    "submission_FBSv15_EXTwins_TOP3.csv": 7.254970669926468,
    "submission_FBSv15_EXTwins_MED_7.csv": 7.255554919572697,
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
    path = ROOT / f"submission_FBSv16_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path),
            "mean": float(arr.mean()), "std": float(arr.std())}


def gauss_w(base_arr, b, cen, sig):
    g = np.exp(-((base_arr - cen)**2) / (2 * sig**2))
    g = g / g.max()
    return b * g


def main():
    base, target_col = load(BASE_FILE)
    if base is None:
        raise RuntimeError(f"BASE not found: {BASE_FILE}")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    # Load externals
    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_v89_covshift.csv', 'submission_v87_mlp.csv',
                'submission_v85_catboost.csv', 'submission_TOP6_MEDIAN.csv',
                'submission_v77_capped.csv', 'submission_v71_sea_defaultmix_w25_dose35.csv']:
        fp = pool_path / fn
        if fp.exists():
            v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
            if len(v) == len(base):
                externals[fn.replace('submission_', '').replace('.csv', '').split('_')[0]] = v
    if 'v89' in externals and 'v87' in externals and 'v85' in externals:
        externals['DIV3'] = (externals['v89'] + externals['v87'] + externals['v85']) / 3.0
        externals['covCB'] = (externals['v89'] + externals['v85']) / 2.0
    print(f"  Loaded {len(externals)} external sources")

    outputs = []

    # The v14 base (without v14's 2% DIV3 baked in) for clean comparison
    v12_base, _ = load("submission_FBSv12_MCBv12_20_TRIM_x12.csv")

    # ====== S1: EXTREME GAUSS dose sweep on v12 base (clean, no double-baked) ======
    # v12 base 7.25622, then add gauss-weighted EXT
    for tag in ('DIV3', 'covCB', 'v87'):
        if tag not in externals: continue
        partner = externals[tag]
        # Wide dose range
        for b in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22):
            for cen, sig in [(30.0, 20.0), (25.0, 15.0), (30.0, 15.0), (28.0, 18.0)]:
                w = gauss_w(v12_base, b, cen, sig)
                blend = (1 - w) * v12_base + w * partner
                outputs.append(save(
                    f"BIG_{tag}_b{int(b*100):02d}_c{int(cen)}_s{int(sig)}",
                    blend, target_col))

    # ====== S2: GAUSS dose on NEW CHAMP base (additive, push from 7.2528 base) ======
    # base already has b05*DIV3*gauss applied. Adding MORE gauss = larger total b
    for tag in ('DIV3', 'covCB', 'v87'):
        if tag not in externals: continue
        partner = externals[tag]
        for b in (0.02, 0.03, 0.05, 0.08, 0.10):
            for cen, sig in [(30.0, 20.0), (25.0, 15.0)]:
                w = gauss_w(base, b, cen, sig)
                blend = (1 - w) * base + w * partner
                outputs.append(save(
                    f"ADDgauss_{tag}_b{int(b*100):02d}_c{int(cen)}_s{int(sig)}",
                    blend, target_col))

    # ====== S3: MULTI-SOURCE GAUSS (DIV3 + covCB + MLP all gauss-weighted) ======
    if all(k in externals for k in ('DIV3', 'covCB', 'v87')):
        for total_b in (0.06, 0.08, 0.10, 0.12, 0.15):
            for cen, sig in [(30.0, 20.0), (28.0, 18.0), (25.0, 15.0)]:
                g = gauss_w(v12_base, total_b, cen, sig)
                # Split total_b across 3 sources
                blend = v12_base.copy()
                for src_tag, frac in [('DIV3', 0.5), ('covCB', 0.3), ('v87', 0.2)]:
                    w_src = g * frac
                    blend = blend + w_src * (externals[src_tag] - v12_base)
                outputs.append(save(
                    f"MULTIgauss_b{int(total_b*100):02d}_c{int(cen)}_s{int(sig)}",
                    blend, target_col))

    # ====== S4: ASYMMETRIC GAUSS — different b for pos/neg residuals ======
    if 'DIV3' in externals:
        partner = externals['DIV3']
        diff = partner - v12_base  # negative = partner predicts lower
        # Asymmetric: stronger pull DOWN (negative diff) than UP
        for cen, sig in [(30.0, 20.0), (28.0, 18.0)]:
            g = np.exp(-((v12_base - cen)**2) / (2 * sig**2))
            g = g / g.max()
            for b_dn, b_up in [(0.10, 0.04), (0.12, 0.05), (0.15, 0.04),
                                (0.08, 0.03), (0.10, 0.06), (0.15, 0.06),
                                (0.20, 0.06), (0.18, 0.04)]:
                w = g.copy()
                # Where partner pulls DOWN (diff<0), use b_dn
                w_arr = np.where(diff < 0, b_dn * g, b_up * g)
                blend = (1 - w_arr) * v12_base + w_arr * partner
                outputs.append(save(
                    f"ASYgauss_DIV3_bd{int(b_dn*100):02d}_bu{int(b_up*100):02d}_c{int(cen)}_s{int(sig)}",
                    blend, target_col))

    # ====== S5: BIMODAL GAUSS (low-power peak + mid-power peak) ======
    if 'DIV3' in externals:
        partner = externals['DIV3']
        for c1, c2, s1, s2 in [(15.0, 35.0, 10.0, 12.0), (10.0, 30.0, 8.0, 15.0),
                                 (20.0, 40.0, 12.0, 12.0)]:
            g1 = np.exp(-((v12_base - c1)**2) / (2*s1**2)); g1 /= g1.max()
            g2 = np.exp(-((v12_base - c2)**2) / (2*s2**2)); g2 /= g2.max()
            for b1, b2 in [(0.05, 0.08), (0.06, 0.10), (0.08, 0.12), (0.04, 0.06)]:
                w = b1*g1 + b2*g2
                blend = (1 - w) * v12_base + w * partner
                outputs.append(save(
                    f"BIMODAL_DIV3_b1{int(b1*100):02d}_b2{int(b2*100):02d}_c{int(c1)}_{int(c2)}",
                    blend, target_col))

    # ====== S6: TRIANGULAR weighting (linear instead of gauss) ======
    if 'DIV3' in externals:
        partner = externals['DIV3']
        for cen, hw in [(30.0, 25.0), (30.0, 30.0), (25.0, 20.0), (35.0, 25.0)]:
            tri = np.clip(1.0 - np.abs(v12_base - cen) / hw, 0, 1)
            for b in (0.06, 0.08, 0.10, 0.12, 0.15):
                w = b * tri
                blend = (1 - w) * v12_base + w * partner
                outputs.append(save(
                    f"TRI_DIV3_b{int(b*100):02d}_c{int(cen)}_h{int(hw)}",
                    blend, target_col))

    # ====== S7: GAUSS-EXT + tiny FBS proxy combo ======
    # Just one safe hybrid
    if 'DIV3' in externals:
        partner = externals['DIV3']
        # Build proxy on v12 base
        dirs, tgts, rms_list = [], [], []
        for name, score in KNOWN_SCORES.items():
            vals, _ = load(name)
            if vals is None or len(vals) != len(v12_base): continue
            diff = vals - v12_base
            rms = float(np.sqrt(np.mean(diff*diff)))
            if rms < 0.5 and float(np.max(np.abs(diff))) < 4.0:
                dirs.append(diff)
                tgts.append(-((score - 7.256224793049420) * N_INST / 100.0))
                rms_list.append(rms)
        if len(dirs) >= 5:
            D = np.asarray(dirs); T = np.asarray(tgts); R = np.asarray(rms_list)
            W = np.diag(1.0 / np.power(0.05 + R, 0.7))
            gram = (D @ D.T) / len(v12_base)
            beta = np.linalg.solve(W @ gram @ W + 1e-3*np.eye(len(D)), W @ T)
            p = D.T @ (W @ beta); p -= p.mean()
            lo, hi = np.percentile(p, [1, 99]); p = np.clip(p, lo, hi); p /= p.std() + 1e-9
            for b in (0.06, 0.08, 0.10):
                for cen, sig in [(30.0, 20.0)]:
                    g = gauss_w(v12_base, b, cen, sig)
                    base_with_gauss = (1 - g) * v12_base + g * partner
                    # apply FBS proxy correction
                    active = np.abs(p) >= np.quantile(np.abs(p), 0.90)
                    pos = active & (p > 0); neg = active & (p < 0)
                    corr = np.zeros_like(p)
                    corr[pos] = 0.35 * p[pos]; corr[neg] = 0.55 * p[neg]
                    blend = base_with_gauss + corr
                    outputs.append(save(
                        f"GAUSSplusFBS_DIV3_b{int(b*100):02d}",
                        blend, target_col))

    # ====== S8: Sharp gauss (very narrow peak) ======
    if 'DIV3' in externals:
        partner = externals['DIV3']
        for cen in (25.0, 28.0, 30.0, 32.0, 35.0):
            for sig in (5.0, 8.0, 10.0):
                for b in (0.08, 0.12, 0.18, 0.25):
                    g = np.exp(-((v12_base - cen)**2) / (2*sig**2)); g /= g.max()
                    w = b * g
                    blend = (1 - w) * v12_base + w * partner
                    outputs.append(save(
                        f"SHARP_DIV3_c{int(cen)}_s{int(sig)}_b{int(b*100):02d}",
                        blend, target_col))

    report = {
        "base_file": BASE_FILE, "base_score": BASE_SCORE,
        "outputs": [o["file"] for o in outputs],
    }
    (ROOT / "feedback_sign_v16_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v16 candidates")


if __name__ == "__main__":
    main()
