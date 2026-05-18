#!/usr/bin/env python3
"""v23: Smart weighted ensembles of all winners + cross-base proxy on champ.

Top winners (LB sorted):
  7.24570: FBSv18_3wM_ws80_hr13_m3_b25  ← champion
  7.24667: FBSv19_FRRraw_a2_g15
  7.24755: FBSv18_4w_ws80_hr13_m3_b25
  7.24760: FBSv18_ADD_COMB_b08
  7.24770: FBSv18_3wM_ws90_hr13_m3_b18
  7.24806: FBSv18_ADD_COMB_b06
  7.24810: FBSv18_COMB_covCB_b18
  7.24846: FBSv18_COMB_v89_b18
  7.24886: FBSv18_COMB_v87_b18

Strategies:
  S1. LB-inverse-distance weighted stack (top 5, 7, 10)
  S2. Exponential decay weighted: w_i = exp(-(score_i - best) / τ) for various τ
  S3. Top-2 close-tie average (7.24570 + 7.24667 = should be ~7.244 if uncorrelated)
  S4. Champion + tiny FRRraw_a2 correction (combined two best mechanisms)
  S5. Diff-pattern: re-derive proxy on champion with new dense neighbours
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv"
BASE_SCORE = 7.245703367161376

WINNERS = [
    ("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv", 7.245703),
    ("submission_FBSv19_FRRraw_a2_g15.csv", 7.246677),
    ("submission_FBSv18_4w_ws80_hr13_m3_b25.csv", 7.247553),
    ("submission_FBSv18_ADD_COMB_b08.csv", 7.247603),
    ("submission_FBSv18_3wM_ws80_hr15_m3_b25.csv", 7.247699),
    ("submission_FBSv18_3wM_ws90_hr13_m3_b18.csv", 7.247710),
    ("submission_FBSv18_3wM_ws80_hr13_m3_b18.csv", 7.247889),
    ("submission_FBSv18_ADD_COMB_b06.csv", 7.248054),
    ("submission_FBSv18_COMB_covCB_b18.csv", 7.248100),
    ("submission_FBSv18_COMB_v89_b18.csv", 7.248462),
    ("submission_FBSv18_ADD_COMB_b04.csv", 7.248585),
    ("submission_FBSv18_COMB_v87_b18.csv", 7.248854),
    ("submission_FBSv18_4w_ws80_hr15_m3_b25.csv", 7.249634),
    ("submission_FBSv17_COMB_WS_HR_DIV3_b12.csv", 7.249806),
    ("submission_FBSv18_3wM_ws80_hr15_m3_b18.csv", 7.249882),
]


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    df = pd.read_csv(ROOT / name)
    return df.iloc[:, 0].to_numpy(dtype=float), df.columns[0]


def save(label, values, target_col):
    arr = np.clip(values, 0.0, N_INST)
    path = ROOT / f"submission_FBSv23_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def main():
    base, target_col = load(BASE_FILE)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    # Load all winners
    arrs = []
    scores = []
    for fn, sc in WINNERS:
        try:
            v, _ = load(fn)
            arrs.append(v)
            scores.append(sc)
        except FileNotFoundError:
            continue
    stk = np.asarray(arrs)
    sc_arr = np.asarray(scores)
    print(f"Loaded {len(arrs)} winners")

    outputs = []

    # ====== S1: LB-inverse-distance weighted (various N) ======
    for top_n in (3, 5, 7, 10, len(arrs)):
        if top_n > len(arrs): continue
        s_top = stk[:top_n]
        sc_top = sc_arr[:top_n]
        # Closer to best = larger weight
        w = 1.0 / (sc_top - 7.245 + 0.001)
        w = w / w.sum()
        weighted = (s_top.T @ w).reshape(-1)
        outputs.append(save(f"WLB_top{top_n}", weighted, target_col))

    # ====== S2: Exponential decay weighted ======
    for tau in (0.001, 0.0025, 0.005, 0.01, 0.02):
        w = np.exp(-(sc_arr - sc_arr.min()) / tau)
        w = w / w.sum()
        weighted = (stk.T @ w).reshape(-1)
        outputs.append(save(f"EXP_tau{int(tau*1000):03d}", weighted, target_col))

    # ====== S3: Plain top-N averages ======
    for top_n in (2, 3, 4, 5, 7):
        if top_n > len(arrs): continue
        outputs.append(save(f"MEAN_top{top_n}", np.mean(stk[:top_n], axis=0), target_col))
        outputs.append(save(f"MED_top{top_n}", np.median(stk[:top_n], axis=0), target_col))

    # ====== S4: Champion + tiny dose of FRRraw ======
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    delta_frr = frr - base
    for w in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        blend = base + w * delta_frr
        outputs.append(save(f"CHAMP_plus_FRR_w{int(w*100):02d}", blend, target_col))

    # ====== S5: Champion + dose of each top winner (cross-blend) ======
    for fn, sc in WINNERS[1:6]:
        v, _ = load(fn)
        delta = v - base
        for w in (0.20, 0.35, 0.50):
            blend = base + w * delta
            short = fn.replace('submission_FBSv', 'v').replace('.csv', '').replace('_', '')[:20]
            outputs.append(save(f"CHAMP_plus_{short}_w{int(w*100):02d}", blend, target_col))

    # ====== S6: Trimmed mean with various trim sizes ======
    for trim_n in (1, 2, 3):
        if len(arrs) > 2*trim_n + 2:
            trimmed = np.zeros_like(stk[0])
            for i in range(stk.shape[1]):
                col = np.sort(stk[:, i])
                trimmed[i] = col[trim_n:-trim_n].mean()
            outputs.append(save(f"TRIM{trim_n}_all", trimmed, target_col))

    # ====== S7: Best of two — top-2 average (could be close-to-true minimum) ======
    # If two top subs have uncorrelated errors, their average should beat both
    top2 = stk[:2]
    outputs.append(save("TOP2_MEAN", np.mean(top2, axis=0), target_col))
    # Top 2 weighted heavily toward champ
    for w in (0.55, 0.60, 0.65, 0.70):
        blend = w * stk[0] + (1-w) * stk[1]
        outputs.append(save(f"TOP2_W{int(w*100):02d}", blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v23_smart_ensemble_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v23 candidates")


if __name__ == "__main__":
    main()
