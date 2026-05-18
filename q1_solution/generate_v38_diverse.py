#!/usr/bin/env python3
"""v38: STRUCTURAL JUMP via new diverse partners.

Found teammate's:
  - v22_mlp (corr 0.949 — REAL MLP, highest diversity)
  - v32_deep_nn (corr 0.977 — different NN)
  - v33_transformer (corr 0.987)
  - v86_alone (corr 0.979)

These haven't been used as partners. Inject at varied doses.

Base = FBSv35_NB_FRR_w20 = 7.21140
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
    path = ROOT / f"submission_FBSv38_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def main():
    base, target_col = load(BASE_FILE)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    # Load NEW diverse partners
    new_partners = {}
    for fn in [
        'submission_v22_mlp.csv',          # corr 0.949 — most diverse!
        'submission_v32_deep_nn.csv',      # corr 0.977
        'submission_v33_transformer.csv',  # corr 0.987 (positive bias)
        'submission_v34_optuna_direct.csv',
        'submission_v35_nn_ensemble.csv',
        'submission_v25_q1_only.csv',      # Q1-only training
        'submission_v31_xgboost.csv',      # XGBoost variant
        'submission_v86_alone.csv',
        'submission_v36_rotor_avg.csv',
        'submission_v37_stability_curve.csv',
        'submission_v85_alone.csv',
        'submission_v87_alone.csv',
        'submission_v89_alone.csv',
        'submission_v23_no_lags.csv',
        'submission_v24_lgbm_no_lags.csv',
    ]:
        v, _ = load(fn)
        if v is not None:
            key = fn.replace('submission_', '').replace('.csv', '')
            new_partners[key] = v
    print(f"Loaded {len(new_partners)} new diverse partners")

    outputs = []

    # ====== S1: Individual partner micro-doses ======
    for tag, partner in new_partners.items():
        for w in (0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080):
            blend = (1 - w) * base + w * partner
            outputs.append(save(f"{tag}_w{int(w*1000):03d}", blend, target_col))

    # ====== S2: v22_mlp specifically (MOST diverse) at wider dose range ======
    if 'v22_mlp' in new_partners:
        p = new_partners['v22_mlp']
        for w in (0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.070, 0.100):
            blend = (1 - w) * base + w * p
            outputs.append(save(f"v22mlp_w{int(w*1000):03d}", blend, target_col))

    # ====== S3: Pairwise combos ======
    pairs = [
        ('v22_mlp', 'v32_deep_nn'),     # two NNs combined
        ('v22_mlp', 'v33_transformer'),
        ('v22_mlp', 'v86_alone'),
        ('v32_deep_nn', 'v33_transformer'),
        ('v22_mlp', 'v31_xgboost'),
        ('v22_mlp', 'v25_q1_only'),
        ('v32_deep_nn', 'v86_alone'),
    ]
    for tag1, tag2 in pairs:
        if tag1 in new_partners and tag2 in new_partners:
            p1 = new_partners[tag1]
            p2 = new_partners[tag2]
            for w1, w2 in [(0.010, 0.010), (0.015, 0.010), (0.020, 0.010),
                            (0.010, 0.020), (0.020, 0.020), (0.015, 0.015),
                            (0.030, 0.010), (0.010, 0.030)]:
                blend = (1 - w1 - w2) * base + w1 * p1 + w2 * p2
                outputs.append(save(
                    f"DUO_{tag1[:8]}_{int(w1*1000):03d}_{tag2[:8]}_{int(w2*1000):03d}",
                    blend, target_col))

    # ====== S4: NN-ensemble (combine all NN-like) ======
    nn_partners = ['v22_mlp', 'v32_deep_nn', 'v33_transformer', 'v35_nn_ensemble']
    if all(t in new_partners for t in nn_partners):
        nn_avg = np.mean([new_partners[t] for t in nn_partners], axis=0)
        for w in (0.010, 0.015, 0.020, 0.030, 0.050, 0.080):
            blend = (1 - w) * base + w * nn_avg
            outputs.append(save(f"NN4avg_w{int(w*1000):03d}", blend, target_col))
        # NN median (robust)
        nn_med = np.median([new_partners[t] for t in nn_partners], axis=0)
        for w in (0.020, 0.030, 0.050):
            blend = (1 - w) * base + w * nn_med
            outputs.append(save(f"NN4med_w{int(w*1000):03d}", blend, target_col))

    # ====== S5: BIG STACK — diverse partners + champion winners ======
    # Champion v34 winners + best NEW partners as ensemble
    champ_files = [
        BASE_FILE,
        "submission_FBSv34_NB_FRR_w30.csv",
        "submission_FBSv35_NB_FRR_w25.csv",
        "submission_FBSv35_v33_FRR_w50.csv",
    ]
    champ_arrs = [load(f)[0] for f in champ_files if load(f)[0] is not None]

    # Try various ratios
    if 'v22_mlp' in new_partners:
        v22 = new_partners['v22_mlp']
        # Take all-champ mean and blend with v22
        champ_mean = np.mean(champ_arrs, axis=0)
        for w_v22 in (0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040):
            blend = (1 - w_v22) * champ_mean + w_v22 * v22
            outputs.append(save(f"CHAMPmean_v22_w{int(w_v22*1000):03d}",
                                blend, target_col))

    # ====== S6: Triple inject (3 NEW partners at small doses) ======
    trips = [
        ('v22_mlp', 'v32_deep_nn', 'v86_alone'),
        ('v22_mlp', 'v33_transformer', 'v31_xgboost'),
        ('v32_deep_nn', 'v33_transformer', 'v22_mlp'),
        ('v22_mlp', 'v86_alone', 'v37_stability_curve'),
    ]
    for tag1, tag2, tag3 in trips:
        if all(t in new_partners for t in (tag1, tag2, tag3)):
            p1, p2, p3 = new_partners[tag1], new_partners[tag2], new_partners[tag3]
            for w in (0.010, 0.015, 0.020):
                total = 3 * w
                blend = (1 - total) * base + w * p1 + w * p2 + w * p3
                outputs.append(save(
                    f"TRI_{tag1[:6]}_{tag2[:6]}_{tag3[:6]}_w{int(w*1000):03d}",
                    blend, target_col))

    # ====== S7: Apply BIGGER doses on more divergent partners ======
    # v22_mlp at higher doses (corr=0.95 means it's a real different model)
    if 'v22_mlp' in new_partners:
        p = new_partners['v22_mlp']
        for w in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
            blend = (1 - w) * base + w * p
            outputs.append(save(f"v22BIG_w{int(w*100):02d}", blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v38_diverse_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v38 candidates")


if __name__ == "__main__":
    main()
