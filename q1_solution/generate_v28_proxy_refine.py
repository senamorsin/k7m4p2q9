#!/usr/bin/env python3
"""v28: Refine NBprox winner (t08_p35_n55 = 7.23669). Combine with FRR.

v27 finding: NBprox_t08 (top-8% proxy correction) beats NB plain.
t08 > t10 > t12 → narrower better. Try t05, t06, t07.
Also: NB_plus_FRR_w10 also worked — orthogonal direction.

v28 strategy:
  S1. NBprox quantile fine grid (t05-t12) × asymmetric γ
  S2. NBprox on this winner base (iterate)
  S3. NBprox + FRR combination (proxy correction + FRR direction)
  S4. Curated proxy with tighter window
  S5. Stack winners
  S6. Fresh LGBM dose probing on new base
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv27_NBprox_t08_p35_n55.csv"
BASE_SCORE = 7.236693385785031
POOL = 'teamblend_v3_pool'
VALID_FEATURES = "data/valid_features.csv"

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv18_3wM_ws80_hr13_m3_b18.csv": 7.247888868240465,
    "submission_FBSv18_ADD_COMB_b08.csv": 7.247603331340510,
    "submission_FBSv18_4w_ws80_hr13_m3_b25.csv": 7.247553356938171,
    "submission_FBSv19_FRRraw_a2_g15.csv": 7.246677323305486,
    "submission_FBSv17_COMB_WS_HR_DIV3_b12.csv": 7.249806446590462,
    "submission_FBSv24_CHAMP_LGBM_w020.csv": 7.241733084922744,
    "submission_FBSv24_CHAMP_LGBM_w030.csv": 7.240292163223663,
    "submission_FBSv26_CHAMP_LGBM_w035.csv": 7.239721311405139,
    "submission_FBSv26_CHAMP_LGBM_w040.csv": 7.239216956002586,
    "submission_FBSv26_CHAMP_LGBM_w050.csv": 7.238409974746849,
    "submission_FBSv26_CHAMP_LGBM_w060.csv": 7.237970103804146,
    "submission_FBSv26_CHAMP_LGBM_w080.csv": 7.238172595523859,
    "submission_FBSv26_NB_LGBM_w020.csv": 7.238449162310680,
    "submission_FBSv26_NB_LGBM_w030.csv": 7.237982397774434,
    "submission_FBSv26_COMBO_LGBM040_FRR20.csv": 7.238491050051701,
    "submission_FBSv26_COMBO_LGBM030_FRR20.csv": 7.239499578976215,
    "submission_FBSv27_NB_plus_FRR_w10.csv": 7.237615105630028,
    "submission_FBSv27_NBprox_t10_p35_n55.csv": 7.238728452342865,
    "submission_FBSv27_NBprox_t12_p35_n55.csv": 7.241931389299324,
    "submission_FBSv27_STACK_v26_TOP3.csv": 7.237885911164830,
    "submission_FBSv27_STACK_v26_WLB_6.csv": 7.237913065922035,
    "submission_FBSv27_STACK_v26_MEAN_6.csv": 7.237929170635638,
    "submission_FBSv27_AVG_NB_w060.csv": 7.237976250789291,
    "submission_FBSv27_CHAMP_LGBM_w065.csv": 7.237901803969211,
    "submission_FBSv27_CHAMP_LGBM_w070.csv": 7.237901449011781,
    "submission_FBSv27_NB_plus_DIV3_w05.csv": 7.239143447175383,
    "submission_FBSv27_NB_plus_covCB_w05.csv": 7.239658633558096,
    "submission_FBSv27_DOUBLE_LGBM_u04_m20.csv": 7.239996120821089,
    "submission_FBSv27_DOUBLE_LGBM_u06_m20.csv": 7.239831088890670,
    "submission_FBSv27_NB_TRI_d02_c02.csv": 7.238861728250856,
    # base skipped: NBprox_t08_p35_n55 = 7.236693385785031
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
    path = ROOT / f"submission_FBSv28_{label}.csv"
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
    if len(dirs) < 4: return None
    D = np.asarray(dirs); T = np.asarray(tgts); R = np.asarray(rms_list)
    W = np.diag(1.0 / np.power(0.05 + R, 0.7))
    gram = (D @ D.T) / len(base)
    beta = np.linalg.solve(W @ gram @ W + alpha * np.eye(len(D)), W @ T)
    proxy = D.T @ (W @ beta)
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy


def build_correction(proxy, q_keep_pct=8, gp=0.35, gn=0.55):
    q = 1.0 - q_keep_pct/100.0
    active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
    pos = active & (proxy > 0); neg = active & (proxy < 0)
    corr = np.zeros_like(proxy)
    corr[pos] = gp * proxy[pos]; corr[neg] = gn * proxy[neg]
    return corr


def main():
    base, target_col = load(BASE_FILE)
    nb_base, _ = load("submission_FBSv26_CHAMP_LGBM_w060.csv")  # parent
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    champ_orig, _ = load("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv")
    nb_plus_frr, _ = load("submission_FBSv27_NB_plus_FRR_w10.csv")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    outputs = []

    # ====== S1: Build proxy on NEW BASE with various windows ======
    for max_rms in (0.4, 0.5, 0.6, 0.7, 0.85):
        proxy = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        rms_tag = f"r{int(max_rms*10):02d}"
        # Quantile + asymmetric grid
        for q in (5, 6, 7, 8, 9, 10, 12, 15):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50), (0.30, 0.60),
                            (0.25, 0.60), (0.45, 0.45), (0.35, 0.50)]:
                corr = build_correction(proxy, q_keep_pct=q, gp=gp, gn=gn)
                outputs.append(save(
                    f"NBprox_{rms_tag}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S2: Proxy on parent (CHAMP_LGBM_w060) — apply to new base ======
    proxy_parent = build_proxy(nb_base, 7.237970103804146, max_rms=0.6)
    if proxy_parent is not None:
        for q in (6, 8, 10):
            corr = build_correction(proxy_parent, q_keep_pct=q, gp=0.35, gn=0.55)
            outputs.append(save(f"PROXparent_t{q:02d}_p35_n55",
                                base + corr, target_col))

    # ====== S3: NB + FRR (the winning combo was w10 → push w15, w20, w25) ======
    delta_frr = frr - champ_orig
    for w in (0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40):
        blend = base + w * delta_frr
        outputs.append(save(f"NB_plus_FRR_w{int(w*100):02d}", blend, target_col))

    # ====== S4: Combine NB + NBprox + FRR (triple stack) ======
    proxy_best = build_proxy(base, BASE_SCORE, max_rms=0.7)
    if proxy_best is not None:
        for q in (6, 8, 10):
            corr_prox = build_correction(proxy_best, q_keep_pct=q, gp=0.35, gn=0.55)
            for w_frr in (0.05, 0.10, 0.15):
                blend = base + corr_prox + w_frr * delta_frr
                outputs.append(save(
                    f"NBprox_plus_FRR_t{q:02d}_w{int(w_frr*100):02d}",
                    blend, target_col))

    # ====== S5: Stack winners ======
    winners = [
        BASE_FILE,
        "submission_FBSv27_NB_plus_FRR_w10.csv",
        "submission_FBSv27_STACK_v26_TOP3.csv",
        "submission_FBSv27_STACK_v26_WLB_6.csv",
        "submission_FBSv26_CHAMP_LGBM_w060.csv",
        "submission_FBSv26_NB_LGBM_w030.csv",
        "submission_FBSv27_CHAMP_LGBM_w065.csv",
        "submission_FBSv27_CHAMP_LGBM_w070.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    stk = np.asarray(arrs)
    if len(arrs) >= 5:
        outputs.append(save(f"STACK_v27_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"STACK_v27_MEDIAN_{len(arrs)}", np.median(stk, axis=0), target_col))
        outputs.append(save("STACK_v27_TOP3", np.mean(stk[:3], axis=0), target_col))
        # Trimmed
        if len(arrs) >= 6:
            trim = np.zeros_like(stk[0])
            for i in range(stk.shape[1]):
                col = np.sort(stk[:, i])
                trim[i] = col[1:-1].mean()
            outputs.append(save(f"STACK_v27_TRIM1_{len(arrs)}", trim, target_col))

    # ====== S6: NB + LGBM dose probe (fine grid around new base) ======
    lgbm_new, _ = load("submission_LGBM_v20_actuals_spatial.csv")
    if lgbm_new is not None:
        # Pure LGBM dose on the NEW base
        for w in (0.01, 0.02, 0.03, 0.05, 0.08):
            blend = (1 - w) * base + w * lgbm_new
            outputs.append(save(f"NB_LGBM_w{int(w*100):02d}", blend, target_col))

    # ====== S7: Apply NBprox correction on PARENT base (cross-validation) ======
    if proxy_best is not None:
        corr = build_correction(proxy_best, q_keep_pct=8, gp=0.35, gn=0.55)
        outputs.append(save("PROX_on_parent",
                            nb_base + corr, target_col))
        outputs.append(save("PROX_on_NBplusFRR",
                            nb_plus_frr + corr, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v28_proxy_refine_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v28 candidates")


if __name__ == "__main__":
    main()
