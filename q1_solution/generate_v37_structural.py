#!/usr/bin/env python3
"""v37: STRUCTURAL shift. Base = FBSv35_NB_FRR_w20 = 7.21140.

FRR direction saturating (each iteration gives -0.0003 now). Need new sources.

Strategy:
  S1. TOP6_MEDIAN micro-doses (NEW partner — not tested at this stage)
  S2. v71/v77 micro-doses
  S3. Mix winners across v34, v35
  S4. Combine NB_FRR + small TOP6
  S5. Curated proxy on new base with VERY tight rms window
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv35_NB_FRR_w20.csv"
BASE_SCORE = 7.211396180429282
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
    "submission_FBSv31_BASE_plus_FRR_w50.csv": 7.218465181225035,
    "submission_FBSv31_BASE_PROX_FRR_t05_p35_n55_f20.csv": 7.213629337187302,
    "submission_FBSv33_NB_FRR_w20.csv": 7.212619135451977,
    "submission_FBSv34_NB_FRR_w30.csv": 7.211678029707256,
    "submission_FBSv34_NB_FRR_w25.csv": 7.211795180481663,
    "submission_FBSv34_NB_FRR_w20.csv": 7.211930368730238,
    "submission_FBSv34_V31_FRR_w28.csv": 7.212309097479216,
    "submission_FBSv35_NB_FRR_w25.csv": 7.211429180583302,
    "submission_FBSv35_NB_FRR_w30.csv": 7.211495107650985,
    "submission_FBSv35_NB_FRR_w15.csv": 7.212942941006496,
    "submission_FBSv35_v33_FRR_w50.csv": 7.211396180429282,
    "submission_FBSv35_v33_FRR_w35.csv": 7.211571436852025,
    "submission_FBSv35_v33_FRR_w33.csv": 7.211608818939807,
    "submission_FBSv35_v33_FRR_w32.csv": 7.211631658625246,
    "submission_FBSv35_STACK_TOP3.csv": 7.211795713190034,
    "submission_FBSv35_STACK_TOP5.csv": 7.211930688355446,
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
    path = ROOT / f"submission_FBSv37_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def main():
    base, target_col = load(BASE_FILE)
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    champ_orig, _ = load("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_v89_covshift.csv', 'submission_v87_mlp.csv',
                'submission_v85_catboost.csv', 'submission_TOP6_MEDIAN.csv',
                'submission_v77_capped.csv',
                'submission_v71_sea_defaultmix_w25_dose35.csv']:
        fp = pool_path / fn
        if fp.exists():
            v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
            if len(v) == len(base):
                key = fn.replace('submission_', '').replace('.csv', '').split('_')[0]
                externals[key] = v
    externals['DIV3'] = (externals['v89'] + externals['v87'] + externals['v85']) / 3.0
    externals['covCB'] = (externals['v89'] + externals['v85']) / 2.0

    outputs = []

    # ====== S1: TOP6_MEDIAN micro-doses (NEW partner — never on FRR-laden bases) ======
    if 'TOP6' in externals:
        top6 = externals['TOP6']
        for w in (0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.080):
            blend = (1 - w) * base + w * top6
            outputs.append(save(f"TOP6_w{int(w*1000):03d}", blend, target_col))

    # ====== S2: v71 micro-doses (forensic manual-tuned teammate sub) ======
    if 'v71' in externals:
        v71 = externals['v71']
        for w in (0.003, 0.005, 0.010, 0.015, 0.020, 0.025, 0.040, 0.060):
            blend = (1 - w) * base + w * v71
            outputs.append(save(f"v71_w{int(w*1000):03d}", blend, target_col))

    # ====== S3: v77 micro-doses ======
    if 'v77' in externals:
        v77 = externals['v77']
        for w in (0.005, 0.010, 0.015, 0.020, 0.030):
            blend = (1 - w) * base + w * v77
            outputs.append(save(f"v77_w{int(w*1000):03d}", blend, target_col))

    # ====== S4: v89/v87/v85 individual at small dose ======
    for tag in ('v89', 'v87', 'v85', 'DIV3', 'covCB'):
        if tag in externals:
            for w in (0.005, 0.010, 0.015, 0.020, 0.030):
                blend = (1 - w) * base + w * externals[tag]
                outputs.append(save(f"{tag}_w{int(w*1000):03d}", blend, target_col))

    # ====== S5: MULTI-EXT (combine 3-4 partners at small doses) ======
    if all(k in externals for k in ('TOP6', 'DIV3', 'v71')):
        for w_top, w_div, w_v71 in [
            (0.010, 0.005, 0.003),
            (0.015, 0.005, 0.005),
            (0.020, 0.005, 0.005),
            (0.020, 0.010, 0.005),
            (0.025, 0.010, 0.005),
            (0.030, 0.010, 0.005),
            (0.020, 0.000, 0.010),
            (0.015, 0.010, 0.010),
        ]:
            total = w_top + w_div + w_v71
            blend = (1 - total) * base + w_top * externals['TOP6'] + w_div * externals['DIV3'] + w_v71 * externals['v71']
            outputs.append(save(
                f"MEXT_t{int(w_top*1000):03d}_d{int(w_div*1000):03d}_v{int(w_v71*1000):03d}",
                blend, target_col))

    # ====== S6: NB_FRR + small TOP6 (combine winning direction with new partner) ======
    if 'TOP6' in externals:
        delta_frr = frr - champ_orig
        for w_frr in (0.05, 0.10, 0.15):
            for w_top6 in (0.010, 0.020, 0.030):
                blend = base + w_frr * delta_frr + w_top6 * (externals['TOP6'] - base)
                outputs.append(save(
                    f"NB_FRR_TOP6_f{int(w_frr*100):02d}_t{int(w_top6*1000):03d}",
                    blend, target_col))

    # ====== S7: Stack v34 + v35 winners ======
    winners = [
        BASE_FILE,
        "submission_FBSv35_v33_FRR_w50.csv",
        "submission_FBSv35_NB_FRR_w25.csv",
        "submission_FBSv35_NB_FRR_w30.csv",
        "submission_FBSv34_NB_FRR_w30.csv",
        "submission_FBSv34_NB_FRR_w25.csv",
        "submission_FBSv34_NB_FRR_w20.csv",
        "submission_FBSv35_v33_FRR_w35.csv",
        "submission_FBSv34_V31_FRR_w28.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 5:
        stk = np.asarray(arrs)
        outputs.append(save("STACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save("STACK_TOP5", np.mean(stk[:5], axis=0), target_col))
        outputs.append(save(f"STACK_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"STACK_MED_{len(arrs)}", np.median(stk, axis=0), target_col))
        # Inv-LB weighted
        sc = np.array([7.21140, 7.21140, 7.21143, 7.21150, 7.21168, 7.21180, 7.21193, 7.21157, 7.21231])[:len(arrs)]
        w_lb = 1.0 / (sc - 7.211 + 0.0005)
        w_lb = w_lb / w_lb.sum()
        outputs.append(save("STACK_WLB", (stk.T @ w_lb).reshape(-1), target_col))

    # ====== S8: Apply tiny TOP6 dose on multiple v34/v35 winners ======
    if 'TOP6' in externals:
        for fn in [
            "submission_FBSv34_NB_FRR_w30.csv",
            "submission_FBSv35_NB_FRR_w25.csv",
            "submission_FBSv35_v33_FRR_w50.csv",
        ]:
            v, _ = load(fn)
            if v is not None:
                for w in (0.010, 0.020, 0.030):
                    blend = (1 - w) * v + w * externals['TOP6']
                    short = fn.replace('submission_FBSv', 'v').replace('.csv', '')[:15]
                    outputs.append(save(f"TOP6on_{short}_w{int(w*1000):03d}",
                                        blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v37_structural_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v37 candidates")


if __name__ == "__main__":
    main()
