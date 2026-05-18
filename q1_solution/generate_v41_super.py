#!/usr/bin/env python3
"""v41: SUPER push. Base = FBSv40_v61_w050 = 7.20640.

New tools:
  - v61_alone partner (winning direction)
  - LGBM noresid (corr 0.978)
  - LGBM dart (corr 0.990)
  - LGBM v20 actuals_spatial (corr 0.987)

Strategy:
  S1. Push v61 dose further: w55, w60, w70, w80, w100
  S2. v61 + noresid combo
  S3. v61 + dart combo
  S4. NEW LGBM mixes (noresid+dart+spatial)
  S5. Triple: v61 + noresid + dart
  S6. v61 doses on previous winners (cross-base)
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv40_v61_w050.csv"
BASE_SCORE = 7.206400653558756
PARENT_FILE = "submission_FBSv35_NB_FRR_w20.csv"
PARENT_SCORE = 7.211396180429282
POOL = 'teamblend_v3_pool'


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
    path = ROOT / f"submission_FBSv41_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def main():
    base, target_col = load(BASE_FILE)
    parent, _ = load(PARENT_FILE)
    v61, _ = load("submission_v61_alone.csv")
    lgbm_spatial, _ = load("submission_LGBM_v20_actuals_spatial.csv")
    lgbm_noresid, _ = load("submission_LGBM_noresid.csv")
    lgbm_dart, _ = load("submission_LGBM_dart.csv")
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    champ_orig, _ = load("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    outputs = []
    delta_frr = frr - champ_orig

    # ====== S1: PUSH v61 dose w55, w60, w70, w80, w100 ======
    for w in (0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.090, 0.100, 0.120, 0.150, 0.200):
        blend = (1 - w) * parent + w * v61
        outputs.append(save(f"v61_w{int(w*1000):03d}", blend, target_col))

    # ====== S2: BASE + LGBM_dart small dose ======
    for w in (0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080):
        blend = (1 - w) * base + w * lgbm_dart
        outputs.append(save(f"NB_dart_w{int(w*1000):03d}", blend, target_col))

    # ====== S3: BASE + LGBM_noresid small dose ======
    for w in (0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080):
        blend = (1 - w) * base + w * lgbm_noresid
        outputs.append(save(f"NB_noresid_w{int(w*1000):03d}", blend, target_col))

    # ====== S4: BASE + LGBM_spatial small dose ======
    for w in (0.005, 0.010, 0.015, 0.020, 0.030, 0.050):
        blend = (1 - w) * base + w * lgbm_spatial
        outputs.append(save(f"NB_spat_w{int(w*1000):03d}", blend, target_col))

    # ====== S5: Triple LGBM mean partner ======
    lgbm_3mean = (lgbm_spatial + lgbm_noresid + lgbm_dart) / 3.0
    for w in (0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080):
        blend = (1 - w) * base + w * lgbm_3mean
        outputs.append(save(f"NB_LGBM3_w{int(w*1000):03d}", blend, target_col))

    # ====== S6: 3-WAY base + v61 + LGBM ======
    for w_v61 in (0.015, 0.020, 0.025):
        for w_lgbm in (0.010, 0.020, 0.030):
            for partner, ptag in [(lgbm_dart, 'dart'), (lgbm_noresid, 'noresid'),
                                   (lgbm_spatial, 'spat')]:
                blend = (1 - w_v61 - w_lgbm) * parent + w_v61 * v61 + w_lgbm * partner
                outputs.append(save(
                    f"TRI_v61{int(w_v61*1000):03d}_{ptag}{int(w_lgbm*1000):03d}",
                    blend, target_col))

    # ====== S7: 4-WAY base + v61 + LGBM_avg ======
    for w_v61 in (0.020, 0.030, 0.040):
        for w_lgbm in (0.010, 0.020, 0.030):
            blend = (1 - w_v61 - w_lgbm) * parent + w_v61 * v61 + w_lgbm * lgbm_3mean
            outputs.append(save(
                f"FOUR_v61{int(w_v61*1000):03d}_lgbm3{int(w_lgbm*1000):03d}",
                blend, target_col))

    # ====== S8: v61 + FRR with more aggressive doses ======
    for w_v61 in (0.030, 0.040, 0.050, 0.060):
        for w_frr in (0.10, 0.20, 0.30, 0.50):
            blend = (1 - w_v61) * parent + w_v61 * v61 + w_frr * delta_frr
            outputs.append(save(
                f"v61_FRR_v{int(w_v61*1000):03d}_f{int(w_frr*100):02d}",
                blend, target_col))

    # ====== S9: BASE + extra v61 + LGBM ======
    # base = (1-0.05)*parent + 0.05*v61 already
    # Add more v61 + small LGBM
    for w_v61_extra in (0.010, 0.020, 0.030):
        for w_lgbm in (0.010, 0.020):
            blend = (1 - w_v61_extra - w_lgbm) * base + w_v61_extra * v61 + w_lgbm * lgbm_dart
            outputs.append(save(
                f"BASE_v61{int(w_v61_extra*1000):03d}_dart{int(w_lgbm*1000):03d}",
                blend, target_col))

    # ====== S10: STACK of winners ======
    winners = [
        BASE_FILE,
        "submission_FBSv40_v61_w040.csv",
        "submission_FBSv40_v61_w035.csv",
        "submission_FBSv40_v61_w030.csv",
        "submission_FBSv40_NB_v61_w020.csv",
        "submission_FBSv40_NB_v61_w015.csv",
        "submission_FBSv40_TRI_v61030_FRR20.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 4:
        stk = np.asarray(arrs)
        outputs.append(save("STACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save("STACK_TOP5", np.mean(stk[:5], axis=0), target_col))
        outputs.append(save(f"STACK_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"STACK_MED_{len(arrs)}", np.median(stk, axis=0), target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v41_super_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v41 candidates")


if __name__ == "__main__":
    main()
