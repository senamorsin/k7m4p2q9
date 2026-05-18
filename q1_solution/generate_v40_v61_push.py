#!/usr/bin/env python3
"""v40: PUSH v61_alone — the surprise winner.

v61_alone at 2% gave Δ−0.003 jump. v61 = forensic-recovered teammate submission,
corr=0.975 with champion. Unique orthogonal direction.

Strategy:
  S1. v61 dose sweep w10-w50
  S2. v61 + FRR direction
  S3. v61 + champion winners (across versions)
  S4. v61 + tiny extras (TOP6, DIV3 micro)
  S5. New base = v61_w020 = 7.20836, push from there

Base = FBSv39_v61_w020 = 7.20836
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv39_v61_w020.csv"
BASE_SCORE = 7.208357496446700
PARENT_FILE = "submission_FBSv35_NB_FRR_w20.csv"  # parent — where v61 was applied
PARENT_SCORE = 7.211396180429282
POOL = 'teamblend_v3_pool'

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv24_CHAMP_LGBM_w030.csv": 7.240292163223663,
    "submission_FBSv26_CHAMP_LGBM_w060.csv": 7.237970103804146,
    "submission_FBSv27_NBprox_t08_p35_n55.csv": 7.236693385785031,
    "submission_FBSv28_NBprox_r07_t07_p35_n55.csv": 7.232883131643645,
    "submission_FBSv29_NB_FRR_w30.csv": 7.229653906267378,
    "submission_FBSv30_NP30_r07_t05_p35_n55.csv": 7.221701978080064,
    "submission_FBSv31_BASE_PROX_FRR_t05_p35_n55_f20.csv": 7.213629337187302,
    "submission_FBSv33_NB_FRR_w20.csv": 7.212619135451977,
    "submission_FBSv34_NB_FRR_w30.csv": 7.211678029707256,
    "submission_FBSv35_NB_FRR_w20.csv": 7.211396180429282,
    "submission_FBSv35_v33_FRR_w50.csv": 7.211396180429282,
    "submission_FBSv38_NN4avg_w020.csv": 7.222538891065310,
    "submission_FBSv39_DUO_v22_ml010_v61_al010.csv": 7.216010931052730,
    "submission_FBSv39_TRI3_v22010_v61010_v60005.csv": 7.218883416152318,
    "submission_FBSv39_v22m_w020.csv": 7.224783126802958,
    "submission_FBSv39_v60w_w015.csv": 7.219498737656290,
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
    path = ROOT / f"submission_FBSv40_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def main():
    base, target_col = load(BASE_FILE)
    parent, _ = load(PARENT_FILE)
    v61, _ = load("submission_v61_alone.csv")
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    champ_orig, _ = load("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")
    print(f"PARENT: {PARENT_FILE} (LB={PARENT_SCORE})")
    print(f"v61_alone: mean={v61.mean():.3f}, std={v61.std():.3f}")

    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_v89_covshift.csv', 'submission_v87_mlp.csv',
                'submission_v85_catboost.csv', 'submission_TOP6_MEDIAN.csv',
                'submission_v71_sea_defaultmix_w25_dose35.csv',
                'submission_v77_capped.csv']:
        fp = pool_path / fn
        if fp.exists():
            v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
            if len(v) == len(base):
                key = fn.replace('submission_', '').replace('.csv', '').split('_')[0]
                externals[key] = v
    if all(k in externals for k in ('v89', 'v87', 'v85')):
        externals['DIV3'] = (externals['v89'] + externals['v87'] + externals['v85']) / 3.0

    outputs = []
    delta_frr = frr - champ_orig

    # ====== S1: v61 dose sweep (peak unknown — push wide range) ======
    # PARENT_v61_wXX = (1-X) * parent + X * v61
    for w in (0.005, 0.010, 0.015, 0.020, 0.022, 0.025, 0.027, 0.030, 0.035, 0.040, 0.050, 0.060, 0.080, 0.100, 0.120, 0.150):
        blend = (1 - w) * parent + w * v61
        outputs.append(save(f"v61_w{int(w*1000):03d}", blend, target_col))

    # ====== S2: NEW base + extra v61 (additive) ======
    # base already has 2% v61 — add more
    for w in (0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080):
        blend = (1 - w) * base + w * v61
        outputs.append(save(f"NB_v61_w{int(w*1000):03d}", blend, target_col))

    # ====== S3: NEW base + FRR direction (combine v61 + FRR) ======
    for w in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        blend = base + w * delta_frr
        outputs.append(save(f"NB_FRR_w{int(w*100):02d}", blend, target_col))

    # ====== S4: PARENT + v61 + FRR (TRIPLE direction) ======
    for w_v61 in (0.015, 0.020, 0.025, 0.030):
        for w_frr in (0.05, 0.10, 0.15, 0.20, 0.30):
            blend = (1 - w_v61) * parent + w_v61 * v61 + w_frr * delta_frr
            outputs.append(save(
                f"TRI_v61{int(w_v61*1000):03d}_FRR{int(w_frr*100):02d}",
                blend, target_col))

    # ====== S5: v61 applied on OTHER bases (cross-base) ======
    other_bases = {
        'v34_w30': load("submission_FBSv34_NB_FRR_w30.csv")[0],
        'v34_w25': load("submission_FBSv34_NB_FRR_w25.csv")[0],
        'v35_v33_FRR_w50': load("submission_FBSv35_v33_FRR_w50.csv")[0],
        'v33_NB_w20': load("submission_FBSv33_NB_FRR_w20.csv")[0],
        'v31_TRIPLE': load("submission_FBSv31_BASE_PROX_FRR_t05_p35_n55_f20.csv")[0],
        'v18_orig': champ_orig,
    }
    for tag, other in other_bases.items():
        if other is None: continue
        for w in (0.010, 0.020, 0.030, 0.050):
            blend = (1 - w) * other + w * v61
            outputs.append(save(f"v61_on_{tag}_w{int(w*1000):03d}",
                                blend, target_col))

    # ====== S6: v61 + small TOP6 / DIV3 ======
    if 'TOP6' in externals:
        for w_v61, w_t6 in [(0.020, 0.005), (0.020, 0.010), (0.025, 0.010),
                              (0.015, 0.010), (0.020, 0.015), (0.030, 0.010)]:
            blend = (1 - w_v61 - w_t6) * parent + w_v61 * v61 + w_t6 * externals['TOP6']
            outputs.append(save(
                f"v61_TOP6_v{int(w_v61*1000):03d}_t{int(w_t6*1000):03d}",
                blend, target_col))

    # ====== S7: Stack of v61-enhanced subs ======
    # Best 3 v61 doses + parent + base = robust avg
    v61_w015 = (1 - 0.015) * parent + 0.015 * v61
    v61_w025 = (1 - 0.025) * parent + 0.025 * v61
    v61_w020 = base  # = (1 - 0.020) * parent + 0.020 * v61
    stack = np.stack([v61_w015, v61_w020, v61_w025])
    outputs.append(save("STACK_v61_mean", np.mean(stack, axis=0), target_col))
    outputs.append(save("STACK_v61_median", np.median(stack, axis=0), target_col))

    # Mean of (v34_w30 + v61_w020 + v35_v33_FRR_w50)
    arr_top3 = [
        load("submission_FBSv35_NB_FRR_w20.csv")[0],
        load("submission_FBSv34_NB_FRR_w30.csv")[0],
        base,
    ]
    arr_top3 = [a for a in arr_top3 if a is not None]
    if len(arr_top3) >= 3:
        outputs.append(save("MIX_top3", np.mean(arr_top3, axis=0), target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v40_v61_push_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v40 candidates")


if __name__ == "__main__":
    main()
