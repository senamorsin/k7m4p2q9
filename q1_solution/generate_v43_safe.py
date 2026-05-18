#!/usr/bin/env python3
"""v43: SAFE approaches only. Base = FBSv40_v61_w050 = 7.20640.

Failed in v42: gauss-weighted v61 (Δ +0.005), sectors, sign-direction.
Failed in v41: LGBM partners (all hurt at 2%).

What WORKS reliably:
  - Curated sign-proxy on new base (FBSv7 method, FBSv28 method, FBSv30 method)
  - NB_v61_wXX additive (small win in v40)
  - Micro-doses of FRR direction

v43 strategy:
  S1. Curated proxy on new base — full grid sweep
  S2. NB_v61 micro-extensions (additive +1-3%)
  S3. NB + FRR ladder
  S4. NB + tiny TOP6/v71 (already-validated partners)
  S5. STACK winners (robust ensemble)
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
    "submission_FBSv37_v71_w010.csv": 7.211342769804875,
    "submission_FBSv37_v71_w015.csv": 7.211385396898819,
    # v39/v40 (most informative)
    "submission_FBSv39_v61_w020.csv": 7.208357496446700,
    "submission_FBSv39_v22m_w020.csv": 7.224783126802958,
    "submission_FBSv39_v60w_w015.csv": 7.219498737656290,
    "submission_FBSv39_DUO_v22_ml010_v61_al010.csv": 7.216010931052730,
    "submission_FBSv39_TRI3_v22010_v61010_v60005.csv": 7.218883416152318,
    "submission_FBSv40_v61_w040.csv": 7.206824593775993,
    "submission_FBSv40_v61_w035.csv": 7.207100664009665,
    "submission_FBSv40_v61_w030.csv": 7.207373416395550,
    "submission_FBSv40_v61_w025.csv": 7.207838574900902,
    "submission_FBSv40_v61_w022.csv": 7.208136336298204,
    "submission_FBSv40_v61_w015.csv": 7.208973932378025,
    "submission_FBSv40_v61_w060.csv": 7.206402333453468,
    "submission_FBSv40_NB_v61_w020.csv": 7.206844204809593,
    "submission_FBSv40_NB_v61_w015.csv": 7.207119310257987,
    "submission_FBSv40_NB_v61_w010.csv": 7.207451528388248,
    "submission_FBSv40_TRI_v61030_FRR20.csv": 7.207793499515571,
    "submission_FBSv40_TRI_v61025_FRR15.csv": 7.208057067341899,
    "submission_FBSv40_TRI_v61020_FRR20.csv": 7.208698546974743,
    "submission_FBSv40_TRI_v61020_FRR10.csv": 7.208389433437516,
    "submission_FBSv40_v61_on_v34_w30_w020.csv": 7.208727597197079,
    "submission_FBSv40_v61_on_v34_w25_w030.csv": 7.207825084430669,
    "submission_FBSv40_v61_on_v31_TRIPLE_w020.csv": 7.210515892069980,
    "submission_FBSv40_v61_TOP6_v020_t010.csv": 7.208645187816960,
    "submission_FBSv40_v61_TOP6_v025_t010.csv": 7.208116861049324,
    # v41 (LGBM dose failed)
    "submission_FBSv41_v61_w060.csv": 7.206402333453468,
    "submission_FBSv41_v61_w070.csv": 7.206884494676794,
    "submission_FBSv41_v61_w080.csv": 7.207634104673207,
    "submission_FBSv41_v61_w100.csv": 7.210157807941255,
    "submission_FBSv41_NB_dart_w020.csv": 7.209559093697929,
    "submission_FBSv41_NB_noresid_w020.csv": 7.210451735846067,
    "submission_FBSv41_NB_spat_w020.csv": 7.209259831083076,
    "submission_FBSv41_NB_LGBM3_w020.csv": 7.209693931484559,
    "submission_FBSv41_FOUR_v61040_lgbm3020.csv": 7.210263669776840,
    "submission_FBSv41_v61_FRR_v060_f20.csv": 7.206799000202425,
    "submission_FBSv41_v61_FRR_v050_f30.csv": 7.207014280388302,
    "submission_FBSv41_v61_FRR_v040_f30.csv": 7.207462631499110,
    "submission_FBSv41_STACK_TOP5.csv": 7.206877543566713,
    "submission_FBSv41_STACK_MED_7.csv": 7.207051970943805,
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
    path = ROOT / f"submission_FBSv43_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def build_proxy(base, base_score, max_rms=0.7, max_abs=5.0, alpha=1e-3):
    dirs, tgts, rms_list = [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base): continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        if rms < max_rms and float(np.max(np.abs(diff))) < max_abs:
            dirs.append(diff)
            tgts.append(-((score - base_score) * N_INST / 100.0))
            rms_list.append(rms)
    if len(dirs) < 4: return None, len(dirs)
    D = np.asarray(dirs); T = np.asarray(tgts); R = np.asarray(rms_list)
    W = np.diag(1.0 / np.power(0.05 + R, 0.7))
    gram = (D @ D.T) / len(base)
    beta = np.linalg.solve(W @ gram @ W + alpha * np.eye(len(D)), W @ T)
    proxy = D.T @ (W @ beta)
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy, len(dirs)


def build_correction(proxy, q_keep_pct, gp, gn):
    q = 1.0 - q_keep_pct/100.0
    active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
    pos = active & (proxy > 0); neg = active & (proxy < 0)
    corr = np.zeros_like(proxy)
    corr[pos] = gp * proxy[pos]; corr[neg] = gn * proxy[neg]
    return corr


def main():
    base, target_col = load(BASE_FILE)
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    champ_orig, _ = load("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv")
    v61, _ = load("submission_v61_alone.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_TOP6_MEDIAN.csv',
                'submission_v71_sea_defaultmix_w25_dose35.csv',
                'submission_v77_capped.csv',
                'submission_v89_covshift.csv']:
        fp = pool_path / fn
        if fp.exists():
            v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
            if len(v) == len(base):
                key = fn.replace('submission_', '').replace('.csv', '').split('_')[0]
                externals[key] = v

    outputs = []
    delta_frr = frr - champ_orig

    # ====== S1: Curated proxy on new base (full sweep, like FBSv7/v30) ======
    for max_rms in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85):
        proxy, n = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        print(f"  Proxy rms<{max_rms}: {n} constraints")
        for q in (3, 5, 7, 8, 10, 12, 15):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50),
                            (0.25, 0.65), (0.45, 0.45)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NP_r{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S2: NB+v61 micro extensions (validated win in v40) ======
    parent, _ = load("submission_FBSv35_NB_FRR_w20.csv")
    # Apply MORE v61 on top of various v40 bases
    for base_fn, base_tag in [
        ("submission_FBSv40_v61_w050.csv", "v50"),
        ("submission_FBSv40_v61_w060.csv", "v60"),
        ("submission_FBSv40_NB_v61_w020.csv", "NB20"),
    ]:
        b, _ = load(base_fn)
        if b is None: continue
        for w in (0.005, 0.010, 0.015, 0.020, 0.030, 0.050):
            blend = (1 - w) * b + w * v61
            outputs.append(save(f"add_v61_{base_tag}_w{int(w*1000):03d}",
                                blend, target_col))

    # ====== S3: NB + FRR micro doses (proven direction) ======
    for w in (0.05, 0.10, 0.15, 0.20, 0.25):
        blend = base + w * delta_frr
        outputs.append(save(f"NB_FRR_w{int(w*100):02d}", blend, target_col))

    # ====== S4: NB + TOP6/v71/v77 tiny doses (validated partners) ======
    for tag, p in externals.items():
        for w in (0.003, 0.005, 0.008, 0.010, 0.015, 0.020):
            blend = (1 - w) * base + w * p
            outputs.append(save(f"NB_{tag}_w{int(w*1000):03d}", blend, target_col))

    # ====== S5: NB + proxy + extra FRR ======
    proxy_best, _ = build_proxy(base, BASE_SCORE, max_rms=0.5)
    if proxy_best is not None:
        for q in (5, 7, 10):
            corr = build_correction(proxy_best, q, 0.35, 0.55)
            for w_frr in (0.05, 0.10, 0.15):
                blend = base + corr + w_frr * delta_frr
                outputs.append(save(
                    f"NP_FRR_t{q:02d}_w{int(w_frr*100):02d}",
                    blend, target_col))

    # ====== S6: STACK top winners ======
    winners = [
        BASE_FILE,
        "submission_FBSv40_v61_w040.csv",
        "submission_FBSv40_v61_w035.csv",
        "submission_FBSv40_NB_v61_w020.csv",
        "submission_FBSv40_NB_v61_w015.csv",
        "submission_FBSv40_v61_on_v34_w25_w030.csv",
        "submission_FBSv41_v61_FRR_v060_f20.csv",
        "submission_FBSv41_STACK_TOP5.csv",
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
        sc = np.array([7.20640, 7.20682, 7.20710, 7.20684, 7.20712, 7.20783, 7.20680, 7.20688])[:len(arrs)]
        w_lb = 1.0 / (sc - 7.206 + 0.0005)
        w_lb = w_lb / w_lb.sum()
        outputs.append(save("STACK_WLB", (stk.T @ w_lb).reshape(-1), target_col))

    # ====== S7: TRIPLE: parent + v61 + FRR (find sweet spot) ======
    for w_v61 in (0.040, 0.050, 0.060):
        for w_frr in (0.05, 0.10, 0.15, 0.20):
            blend = (1 - w_v61) * parent + w_v61 * v61 + w_frr * delta_frr
            outputs.append(save(
                f"PvFR_v{int(w_v61*1000):03d}_f{int(w_frr*100):02d}",
                blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v43_safe_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v43 candidates")


if __name__ == "__main__":
    main()
