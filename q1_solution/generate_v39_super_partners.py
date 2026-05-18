#!/usr/bin/env python3
"""v39: Super-diverse partners injection.

Found:
  - v61_alone: corr=0.9749 (high diversity)
  - v60_q1_weighted_lgbm_direct: corr=0.9824
  - v22_mlp: corr=0.9490 (HIGHEST diversity)
  - v39_hub_114m: corr=0.9890

Strategy:
  S1. v61_alone micro-doses (NEW HIGH-DIV partner)
  S2. v60_weighted micro-doses
  S3. v22_mlp larger doses (0.949 corr)
  S4. v39_hub micro-doses
  S5. Multi-mix: champion + v61 + v22_mlp
  S6. NEW partners as constraints in NEW proxy

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
    path = ROOT / f"submission_FBSv39_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def main():
    base, target_col = load(BASE_FILE)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    # Top diverse partners by corr (low corr = high diversity)
    partners = {}
    for fn in [
        'submission_v22_mlp.csv',                  # corr 0.949 ⭐
        'submission_v61_alone.csv',                # corr 0.975 ⭐
        'submission_v32_deep_nn.csv',              # corr 0.977
        'submission_v60_q1_weighted_lgbm_direct.csv',  # corr 0.982
        'submission_v25_q1_only.csv',              # corr 0.985
        'submission_v34_optuna_direct.csv',        # corr 0.987
        'submission_v33_transformer.csv',          # corr 0.987
        'submission_v86_alone.csv',                # corr 0.979
        'submission_v35_nn_ensemble.csv',          # corr 0.986
        'submission_v36_rotor_avg.csv',
        'submission_v37_stability_curve.csv',
        'submission_v39_hub_114m.csv',
        'submission_v41_20_winner.csv',
        'submission_v45_v43_rotor_05_v41best_demean.csv',
    ]:
        v, _ = load(fn)
        if v is not None:
            key = fn.replace('submission_', '').replace('.csv', '')
            partners[key] = v
    print(f"Loaded {len(partners)} diverse partners")

    outputs = []

    # ====== S1: v61_alone (NEW high-div) ======
    if 'v61_alone' in partners:
        p = partners['v61_alone']
        for w in (0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.120):
            blend = (1 - w) * base + w * p
            outputs.append(save(f"v61_w{int(w*1000):03d}", blend, target_col))

    # ====== S2: v60_weighted (NEW q1-only LGBM direct) ======
    if 'v60_q1_weighted_lgbm_direct' in partners:
        p = partners['v60_q1_weighted_lgbm_direct']
        for w in (0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080):
            blend = (1 - w) * base + w * p
            outputs.append(save(f"v60w_w{int(w*1000):03d}", blend, target_col))

    # ====== S3: v22_mlp larger doses (corr 0.949) ======
    if 'v22_mlp' in partners:
        p = partners['v22_mlp']
        for w in (0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080, 0.100, 0.150):
            blend = (1 - w) * base + w * p
            outputs.append(save(f"v22m_w{int(w*1000):03d}", blend, target_col))

    # ====== S4: v39_hub_114m (rotor avg variant) ======
    if 'v39_hub_114m' in partners:
        p = partners['v39_hub_114m']
        for w in (0.010, 0.020, 0.030, 0.050):
            blend = (1 - w) * base + w * p
            outputs.append(save(f"v39h_w{int(w*1000):03d}", blend, target_col))

    # ====== S5: v41/v45 winner variants ======
    for tag in ('v41_20_winner', 'v45_v43_rotor_05_v41best_demean'):
        if tag in partners:
            p = partners[tag]
            short = tag[:10]
            for w in (0.010, 0.020, 0.030, 0.050):
                blend = (1 - w) * base + w * p
                outputs.append(save(f"{short}_w{int(w*1000):03d}",
                                    blend, target_col))

    # ====== S6: TRIPLE mix (v22 + v61 + v60w) — three most diverse ======
    if all(t in partners for t in ('v22_mlp', 'v61_alone', 'v60_q1_weighted_lgbm_direct')):
        v22 = partners['v22_mlp']
        v61 = partners['v61_alone']
        v60w = partners['v60_q1_weighted_lgbm_direct']
        for w22, w61, w60 in [
            (0.010, 0.010, 0.005),
            (0.015, 0.015, 0.005),
            (0.020, 0.010, 0.005),
            (0.010, 0.020, 0.005),
            (0.015, 0.010, 0.010),
            (0.025, 0.015, 0.005),
            (0.020, 0.015, 0.010),
        ]:
            total = w22 + w61 + w60
            blend = (1 - total) * base + w22 * v22 + w61 * v61 + w60 * v60w
            outputs.append(save(
                f"TRI3_v22{int(w22*1000):03d}_v61{int(w61*1000):03d}_v60{int(w60*1000):03d}",
                blend, target_col))

    # ====== S7: Pairwise mixes (most diverse pair) ======
    for tag1, tag2 in [
        ('v22_mlp', 'v61_alone'),
        ('v22_mlp', 'v60_q1_weighted_lgbm_direct'),
        ('v61_alone', 'v60_q1_weighted_lgbm_direct'),
        ('v22_mlp', 'v33_transformer'),
        ('v61_alone', 'v33_transformer'),
        ('v32_deep_nn', 'v22_mlp'),
        ('v32_deep_nn', 'v61_alone'),
    ]:
        if tag1 in partners and tag2 in partners:
            p1 = partners[tag1]; p2 = partners[tag2]
            for w1, w2 in [(0.010, 0.010), (0.015, 0.010), (0.020, 0.010), (0.010, 0.020),
                            (0.020, 0.020), (0.025, 0.025), (0.030, 0.015)]:
                blend = (1 - w1 - w2) * base + w1 * p1 + w2 * p2
                outputs.append(save(
                    f"DUO_{tag1[:6]}{int(w1*1000):03d}_{tag2[:6]}{int(w2*1000):03d}",
                    blend, target_col))

    # ====== S8: Mean of top-3 diverse + champion blend ======
    if all(t in partners for t in ('v22_mlp', 'v61_alone', 'v32_deep_nn')):
        div_mean = (partners['v22_mlp'] + partners['v61_alone'] + partners['v32_deep_nn']) / 3.0
        for w in (0.010, 0.015, 0.020, 0.030, 0.050, 0.080):
            blend = (1 - w) * base + w * div_mean
            outputs.append(save(f"DIV3new_w{int(w*1000):03d}", blend, target_col))

    # ====== S9: Apply same recipes on FBSv34_NB_FRR_w30 (parent base) ======
    parent, _ = load("submission_FBSv34_NB_FRR_w30.csv")
    if parent is not None and 'v22_mlp' in partners:
        for w in (0.010, 0.020, 0.030, 0.050):
            blend = (1 - w) * parent + w * partners['v22_mlp']
            outputs.append(save(f"v34_v22m_w{int(w*1000):03d}", blend, target_col))
        if 'v61_alone' in partners:
            for w in (0.010, 0.020, 0.030):
                blend = (1 - w) * parent + w * partners['v61_alone']
                outputs.append(save(f"v34_v61_w{int(w*1000):03d}", blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v39_super_partners_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v39 candidates")


if __name__ == "__main__":
    main()
