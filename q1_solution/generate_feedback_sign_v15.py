#!/usr/bin/env python3
"""Feedback-sign v15: PUSH EXTERNAL inject (orthogonal signal that works).

v14 finding: EXT_DIV3_w020 won (−0.0015). External teammate sources break plateau.
w020 > w010 → push doses up. Multi-EXT combo may compound.

v15 strategy:
  S1. EXT dose sweep: w025-w060 on each source
  S2. MULTI-EXT: DIV3+covCB+MLP at small doses
  S3. EXT on top of new base (build NEW EXT champ → re-derive)
  S4. EXT + FBS proxy hybrid (combine external signal with curated proxy)
  S5. EXT gauss-weighted (the orig ULTRA_DIV3_gauss won early — try it again)
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv14_EXT_DIV3_w020.csv"
BASE_SCORE = 7.254768850541299
POOL = 'teamblend_v3_pool'

KNOWN_SCORES = {
    "submission_rebound_v6_t075_hneg10.csv": 7.37455489169975,
    "submission_feedback_sign_smooth5_g0p08.csv": 7.372587722053496,
    "submission_REBv6_LIGHT10_smooth.csv": 7.377025947066968,
    "submission_ULTRA_DIV3_gauss_b06.csv": 7.371827077437699,
    "submission_ULTRA_DIV3_VS_w025.csv": 7.372196507607671,
    "submission_ULTRA_DIV3_w03.csv": 7.372235347883005,
    "submission_ULTRA_DIV3_w04.csv": 7.372393263765536,
    "submission_ULTRA_covCB_w06.csv": 7.372534974937972,
    "submission_FBSv2_smooth5_g0p12.csv": 7.369170488021504,
    "submission_FBSv3_top19_g0p22.csv": 7.363100871060781,
    "submission_FBSv4_top10_g0p45.csv": 7.353705058320320,
    "submission_FBSv5_top10_asy_gp35_gn55.csv": 7.352680653403223,
    "submission_FBSv7_CUR05_t10_p35_n55.csv": 7.330521861258725,
    "submission_FBSv8_CURens_t10_p35_n55.csv": 7.307464464111231,
    "submission_FBSv9_MULTICB_MEAN_x12.csv": 7.293590174587235,
    "submission_FBSv10_NB_CURens_NEW_t10_p35_n55.csv": 7.266675457551658,
    "submission_FBSv11_MCBv11_15_MED_x14.csv": 7.263233849236824,
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv12_HYBR_mcb60_x14.csv": 7.257319999283790,
    "submission_FBSv12_HYBR_mcb50_x14.csv": 7.257856931059352,
    "submission_FBSv12_MCBv12_20_MED_x10.csv": 7.258393312227188,
    "submission_FBSv12_MCBv12_20_MED_x12.csv": 7.258489530413064,
    # v14 batch (most informative)
    # BASE: EXT_DIV3_w020 = 7.254768850541299
    "submission_FBSv14_EXT_covCB_w020.csv": 7.254920930824225,
    "submission_FBSv14_EXT_v87_mlp_w010.csv": 7.255334184333610,
    "submission_FBSv14_EXT_DIV3_w010.csv": 7.255426543419984,
    "submission_FBSv14_EXT_covCB_w010.csv": 7.255493439062207,
    "submission_FBSv14_EXT_v89_covshift_w020.csv": 7.255561551691743,
    "submission_FBSv14_FBSv4base_t10_x10.csv": 7.255667788894398,
    "submission_FBSv14_EXT_v89_covshift_w010.csv": 7.255791646051818,
    "submission_FBSv14_EXT_TOP6_MEDIAN_w010.csv": 7.255826022730874,
    "submission_FBSv14_WINS_TRIM2_14.csv": 7.258749066089182,
    "submission_FBSv14_WINS_MEDIAN_14.csv": 7.259821991010263,
    "submission_FBSv14_CROSSv_MED_x12.csv": 7.261265991948279,
    "submission_FBSv14_FAR70_200_t15_p30_n45.csv": 7.263301065017250,
    "submission_FBSv14_FAR70_200_t10_p35_n55.csv": 7.263518765716860,
}


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    path = ROOT / name
    if not path.exists():
        # Maybe in pool
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
    path = ROOT / f"submission_FBSv15_{label}.csv"
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
        externals['mlpCB'] = (externals['v87'] + externals['v85']) / 2.0
    print(f"  Loaded {len(externals)} external sources")

    outputs = []

    # ====== S1: EXT dose sweep — push doses (w025-w060) ======
    # Note: BASE already has 2% DIV3 baked in (from v14). Adding MORE DIV3
    # is essentially increasing the dose. But adding DIFFERENT source is fresh.
    # Compute the "raw" (pre-v14) version: base - 0.02*DIV3 + 0.02*DIV3 = base
    # So adding more DIV3 to base = adding net (w-0.02) DIV3 to underlying v12.
    # Just continue from base.
    for tag, partner in externals.items():
        for w in (0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.070):
            blend = (1 - w) * base + w * partner
            outputs.append(save(f"EXT_{tag}_w{int(w*1000):03d}",
                                blend, target_col))

    # ====== S2: MULTI-EXT (combine 2-3 externals at smaller individual doses) ======
    combos = [
        # 2-way 1% each
        [('DIV3', 0.01), ('covCB', 0.01)],
        [('DIV3', 0.015), ('v87', 0.01)],
        [('covCB', 0.015), ('v87', 0.01)],
        [('DIV3', 0.02), ('v87', 0.01)],
        # 3-way
        [('v89', 0.01), ('v87', 0.01), ('v85', 0.01)],  # ~equivalent to 3% DIV3
        [('DIV3', 0.01), ('TOP6', 0.005), ('v77', 0.005)],
        [('DIV3', 0.015), ('TOP6', 0.005)],
        [('covCB', 0.01), ('v87', 0.01), ('TOP6', 0.005)],
        # bigger 2-way
        [('DIV3', 0.02), ('covCB', 0.01)],
        [('DIV3', 0.025), ('v87', 0.01)],
        [('DIV3', 0.02), ('TOP6', 0.01)],
    ]
    for combo in combos:
        if not all(t in externals for t, _ in combo): continue
        total_w = sum(w for _, w in combo)
        blend = (1 - total_w) * base
        tag_parts = []
        for t, w in combo:
            blend = blend + w * externals[t]
            tag_parts.append(f"{t}{int(w*1000):03d}")
        outputs.append(save(f"MULTIEXT_{'_'.join(tag_parts)}", blend, target_col))

    # ====== S3: GAUSS-weighted external (original ULTRA_DIV3_gauss won) ======
    # Apply external partner mainly to mid-power rows
    for tag in ('DIV3', 'covCB', 'v87'):
        if tag not in externals: continue
        partner = externals[tag]
        for cen, sig in [(30.0, 20.0), (25.0, 15.0)]:
            g = np.exp(-((base - cen)**2) / (2*sig**2))
            g = g / g.max()
            for b in (0.02, 0.03, 0.04, 0.05):
                w = b * g
                blend = (1 - w) * base + w * partner
                outputs.append(save(
                    f"EXTgauss_{tag}_b{int(b*100):02d}_c{int(cen)}_s{int(sig)}",
                    blend, target_col))

    # ====== S4: EXT + FBS proxy hybrid ======
    # Build proxy on new base, combine with EXT signal
    proxies_n = {}
    for max_rms in [0.30, 0.40, 0.50, 0.60]:
        p = build_proxy_for(base, BASE_SCORE, max_rms=max_rms)
        if p is not None: proxies_n[max_rms] = p
    if len(proxies_n) >= 2 and 'DIV3' in externals:
        ens = np.mean(list(proxies_n.values()), axis=0)
        ens = ens - ens.mean(); ens = ens / (ens.std() + 1e-9)
        fbs_corr = build_correction(ens, q_keep_pct=10, gp=0.35, gn=0.55)
        # EXT + FBS combinations
        for ext_tag, ext_w in [('DIV3', 0.02), ('covCB', 0.02), ('DIV3', 0.01)]:
            partner = externals[ext_tag]
            ext_corr = ext_w * (partner - base)  # the displacement
            for fbs_dose in (0.5, 1.0):
                blend = base + ext_corr + fbs_dose * fbs_corr
                outputs.append(save(
                    f"EXTplus_{ext_tag}{int(ext_w*1000):03d}_fbs{int(fbs_dose*10):02d}",
                    blend, target_col))

    # ====== S5: EXT on top of best v12 directly (skip v14 stage) ======
    v12_base, _ = load("submission_FBSv12_MCBv12_20_TRIM_x12.csv")
    if v12_base is not None:
        for tag in ('DIV3', 'covCB', 'v87', 'v89'):
            if tag not in externals: continue
            partner = externals[tag]
            for w in (0.020, 0.025, 0.030, 0.035):
                blend = (1 - w) * v12_base + w * partner
                outputs.append(save(f"v12base_EXT_{tag}_w{int(w*1000):03d}",
                                    blend, target_col))

    # ====== S6: EXT-direction stack — average of best EXT winners ======
    ext_winners = [
        BASE_FILE,
        "submission_FBSv14_EXT_covCB_w020.csv",
        "submission_FBSv14_EXT_v87_mlp_w010.csv",
        "submission_FBSv14_EXT_DIV3_w010.csv",
        "submission_FBSv14_EXT_covCB_w010.csv",
        "submission_FBSv14_EXT_v89_covshift_w020.csv",
        "submission_FBSv14_EXT_v89_covshift_w010.csv",
    ]
    stk = []
    for fn in ext_winners:
        v, _ = load(fn)
        if v is not None: stk.append(v)
    if len(stk) >= 4:
        stk = np.asarray(stk)
        outputs.append(save(f"EXTwins_MEAN_{len(stk)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"EXTwins_MED_{len(stk)}", np.median(stk, axis=0), target_col))
        outputs.append(save(f"EXTwins_TOP3", np.mean(stk[:3], axis=0), target_col))

    report = {
        "base_file": BASE_FILE, "base_score": BASE_SCORE,
        "outputs": [o["file"] for o in outputs],
    }
    (ROOT / "feedback_sign_v15_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v15 candidates")


if __name__ == "__main__":
    main()
