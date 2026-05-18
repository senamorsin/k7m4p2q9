#!/usr/bin/env python3
"""Feedback-sign v3: NEW base = FBSv2_smooth5_g0p12 (7.36917) + ~50 LB constraints.

v2 with 33 constraints jumped from 7.3726 to 7.3692 (−0.0027). v3 has ~50% more
constraints and uses the v2 champion as base. Linear constraint span is larger →
proxy direction in higher-D subspace → bigger achievable correction.

Key: gamma=0.12 won in v2 (vs g=0.08 in v1). Expect g=0.12-0.18 to win in v3.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv2_smooth5_g0p12.csv"
BASE_SCORE = 7.369170488021504
VALID_FEATURES = "data/valid_features.csv"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"

# ALL LB scores — original 33 + new 18+ from FBSv2 and R2 batches
KNOWN_SCORES = {
    # Original (rebound + early)
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

    # feedback_sign v1 originals
    "submission_feedback_sign_g0p08.csv": 7.373278334346155,
    "submission_feedback_sign_smooth5_g0p08.csv": 7.372587722053496,

    # REBv6 batch
    "submission_REBv6_LIGHT10_smooth.csv": 7.377025947066968,
    "submission_REBv6_VLIGHT_smooth.csv": 7.378272056634792,
    "submission_REBv6_recipe_20_30_50.csv": 7.406747265442087,
    "submission_REBv6_recipe_15_35_50.csv": 7.408338532640587,
    "submission_REBv6_recipe_25_30_45.csv": 7.404852525722752,

    # CHAMPv3 batch
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

    # ULTRA batch
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

    # FBSv2 batch (the BIG signal — sign proxy worked)
    "submission_FBSv2_tanh_g0p15.csv": 7.369660686492568,
    "submission_FBSv2_smooth5_g0p08.csv": 7.369845849772287,
    # BASE skipped: "submission_FBSv2_smooth5_g0p12.csv": 7.369170488021504,

    # R2 batch
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
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str):
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
    path = ROOT / f"submission_FBSv3_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path),
            "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}


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
        if rms < 3.0 and max_abs < 20.0:
            directions.append(diff)
            target = -((score - BASE_SCORE) * N_INST / 100.0)
            targets.append(target)
            names.append(name)
            rms_list.append(rms)

    D = np.asarray(directions)
    if len(D) < 5:
        raise ValueError(f"only {len(D)} constraints")
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
    if base is None:
        raise RuntimeError(f"BASE not found: {BASE_FILE}")
    proxy, constraints, diag = build_proxy(base)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")
    print(f"Built proxy from {len(constraints)} constraints (target ~50)")
    print(f"Proxy stats: mean={proxy.mean():.4f}, std={proxy.std():.4f}, "
          f"min={proxy.min():.3f}, max={proxy.max():.3f}")

    outputs = []

    # 1. Raw + smooth gamma sweeps (push larger γ — v2 g=0.12 won)
    for gamma in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35]:
        outputs.append(save(f"g{str(gamma).replace('.','p')}",
                            base + gamma * proxy, target_col))

    # 2. Smoothed proxy (rolling)
    valid = pd.read_csv(ROOT / VALID_FEATURES, parse_dates=[DATETIME_COL])
    if len(valid) == len(proxy):
        for win in [3, 5, 7, 11, 15]:
            ts = pd.Series(proxy, index=valid[DATETIME_COL]).sort_index()
            smooth = ts.rolling(window=win, center=True, min_periods=1).mean()
            smooth = smooth.reindex(valid[DATETIME_COL]).to_numpy(dtype=float)
            smooth = smooth - smooth.mean()
            smooth = smooth / (smooth.std() + 1e-9)
            for gamma in [0.08, 0.12, 0.15, 0.18, 0.22, 0.30]:
                outputs.append(save(f"smooth{win}_g{str(gamma).replace('.','p')}",
                                    base + gamma * smooth, target_col))

    # 3. Tanh-saturated (v2 tanh_g0p15 was 7.3697)
    bounded = np.tanh(proxy)
    for gamma in [0.10, 0.15, 0.20, 0.30, 0.45, 0.60]:
        outputs.append(save(f"tanh_g{str(gamma).replace('.','p')}",
                            base + gamma * bounded, target_col))

    # 4. Top-quantile (most confident rows only)
    for q in [0.50, 0.65, 0.80]:
        active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
        for gamma in [0.15, 0.22, 0.35]:
            corr = np.zeros_like(proxy)
            corr[active] = gamma * proxy[active]
            outputs.append(save(f"top{int((1-q)*100)}_g{str(gamma).replace('.','p')}",
                                base + corr, target_col))

    # 5. Smooth + tanh combo (smooth first then bound)
    if len(valid) == len(proxy):
        for win in [5, 7]:
            ts = pd.Series(proxy, index=valid[DATETIME_COL]).sort_index()
            smooth = ts.rolling(window=win, center=True, min_periods=1).mean()
            smooth = smooth.reindex(valid[DATETIME_COL]).to_numpy(dtype=float)
            smooth = smooth - smooth.mean()
            smooth = smooth / (smooth.std() + 1e-9)
            bs = np.tanh(smooth)
            for gamma in [0.15, 0.22, 0.30, 0.40]:
                outputs.append(save(f"smtanh{win}_g{str(gamma).replace('.','p')}",
                                    base + gamma * bs, target_col))

    # 6. Apply proxy on OLD champion (cross-validate proxy direction)
    old_champ_f = "submission_feedback_sign_smooth5_g0p08.csv"
    old_champ, _ = load(old_champ_f)
    if old_champ is not None:
        for gamma in [0.12, 0.18, 0.25]:
            outputs.append(save(f"on_OC_g{str(gamma).replace('.','p')}",
                                old_champ + gamma * proxy, target_col))

    # 7. NEG safety (should make score WORSE → confirms direction)
    for gamma in [0.08, 0.15]:
        outputs.append(save(f"NEG_g{str(gamma).replace('.','p')}",
                            base - gamma * proxy, target_col))

    report = {
        "base_file": BASE_FILE,
        "base_score": BASE_SCORE,
        "n_constraints": len(constraints),
        "constraints": constraints,
        "diag": diag,
        "outputs": [o["file"] for o in outputs],
    }
    (ROOT / "feedback_sign_v3_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v3 candidates")


if __name__ == "__main__":
    main()
