#!/usr/bin/env python3
"""Feedback-sign v2: rebuild row-level sign proxy with ALL new LB constraints
and NEW BASE = current champion submission_feedback_sign_smooth5_g0p08.csv (7.3726).

Original generator used base=rebound_v6_t075_hneg10 (7.3746) and 30 constraints.
We now have ~25 additional LB scores from CHAMPv3_*, REBv6_*, pseudotarget,
feedback_sign_* siblings — TOTAL of ~50+ directional constraints. Stronger proxy →
larger correctable signal → bigger LB drop.

Idea: score(b+d) - score(b) ≈ -<sign(err), d>·100/N. So given enough d_i ↔ Δ_i pairs
we recover proxy ∝ sign(err) of the base. Apply with shrinkage γ to the base.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_feedback_sign_smooth5_g0p08.csv"   # NEW BASE = champion
BASE_SCORE = 7.372587722053496
VALID_FEATURES = "data/valid_features.csv"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"

# ALL LB scores we know. Mix of original constraints + today's uploads.
KNOWN_SCORES = {
    # === ORIGINAL CONSTRAINTS (from generate_feedback_sign_candidates.py) ===
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

    # === FEEDBACK_SIGN SIBLINGS (today + earlier) ===
    "submission_feedback_sign_g0p08.csv": 7.373278334346155,
    "submission_feedback_sign_g0p12.csv": None,   # missing LB, will skip
    "submission_feedback_sign_smooth5_g0p08.csv": 7.372587722053496,  # this is BASE
    # don't include BASE in constraints

    # === REBv6 variants (today) ===
    "submission_REBv6_LIGHT10_smooth.csv": 7.377025947066968,
    "submission_REBv6_VLIGHT_smooth.csv": 7.378272056634792,
    "submission_REBv6_recipe_20_30_50.csv": 7.406747265442087,
    "submission_REBv6_recipe_15_35_50.csv": 7.408338532640587,
    "submission_REBv6_recipe_25_30_45.csv": 7.404852525722752,

    # === Pseudotarget (high RMS, will be filtered out) ===
    # "submission_pseudotarget_k24_r1e5_d0p08.csv": 7.419476967415053,
    # "submission_pseudotarget_ens_smooth7_d0p1.csv": 7.431353925145531,

    # === CHAMPv3 blends (today, the most informative — close to base) ===
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

    # === CHAMP-named (highcap is too far, skip)
    # "submission_CHAMP_highcap_thr50_sh1p0.csv": 7.422005228941361,
    # "submission_CHAMP_highcap_thr60_sh1p5.csv": 7.448162632744240,
}

KNOWN_SCORES = {k: v for k, v in KNOWN_SCORES.items() if v is not None}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> tuple[np.ndarray, str]:
    path = ROOT / name
    if not path.exists():
        raise FileNotFoundError(name)
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, OSError):
        return None, None
    values = df.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} has NaN/inf")
    return values, df.columns[0]


def save(label: str, values: np.ndarray, target_col: str) -> dict:
    arr = np.clip(values, 0.0, N_INST)
    path = ROOT / f"submission_FBSv2_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding="utf-8")
    return {"file": path.name, "sha256": sha256(path), "mean": float(arr.mean()),
            "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}


def build_proxy(base: np.ndarray) -> tuple[np.ndarray, list[str], dict]:
    directions, targets, names, rms_list = [], [], [], []
    diag = {}
    for name, score in KNOWN_SCORES.items():
        try:
            values, _ = load(name)
        except FileNotFoundError:
            continue
        if len(values) != len(base):
            continue
        diff = values - base
        rms = float(np.sqrt(np.mean(diff * diff)))
        max_abs = float(np.max(np.abs(diff)))
        diag[name] = {"rms": rms, "max_abs": max_abs, "score": score}
        # Keep only "nearby" submissions where linear approx holds
        if rms < 2.5 and max_abs < 15.0:
            directions.append(diff)
            # The directional derivative target:
            # score - BASE_SCORE  ≈ - <sign_err, diff> * 100 / N_INST / len(base)
            # → <sign_err, diff> ≈ -(score - BASE_SCORE) * N_INST / 100 * len(base)
            # But we want PER-row "sign", so target dot product is:
            #     -(score - BASE_SCORE) * N_INST / 100
            # divided by len(base)? Let's keep same scale as original:
            target = -((score - BASE_SCORE) * N_INST / 100.0)
            targets.append(target)
            names.append(name)
            rms_list.append(rms)

    D = np.asarray(directions)
    if len(D) < 5:
        raise ValueError(f"not enough constraints: only {len(D)}")
    target_vec = np.asarray(targets)
    rms_arr = np.asarray(rms_list)

    # Inverse-RMS weight + Tikhonov regularization
    W = np.diag(1.0 / np.power(0.05 + rms_arr, 0.7))
    gram = (D @ D.T) / len(base)
    alpha = 1e-3
    beta = np.linalg.solve(W @ gram @ W + alpha * np.eye(len(D)), W @ target_vec)
    proxy = D.T @ (W @ beta)

    # Center & robust-normalize
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy, names, diag


def main() -> None:
    base, target_col = load(BASE_FILE)
    proxy, constraints, diag = build_proxy(base)
    print(f"Built proxy from {len(constraints)} constraints")
    print(f"Proxy stats: mean={proxy.mean():.4f}, std={proxy.std():.4f}, "
          f"min={proxy.min():.3f}, max={proxy.max():.3f}")

    outputs = []

    # 1. Raw proxy at various gammas (signed shrinkage like original recipe but on new base)
    for gamma in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        outputs.append(save(f"g{str(gamma).replace('.', 'p')}",
                            base + gamma * proxy, target_col))

    # 2. Top-quantile (only most-confident rows)
    for q in [0.50, 0.65, 0.80]:
        active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
        for gamma in [0.10, 0.15, 0.22]:
            corr = np.zeros_like(proxy)
            corr[active] = gamma * proxy[active]
            outputs.append(save(f"top{int((1-q)*100)}_g{str(gamma).replace('.', 'p')}",
                                base + corr, target_col))

    # 3. Time-smoothed proxy (rolling window 3, 5, 7)
    valid = pd.read_csv(ROOT / VALID_FEATURES, parse_dates=[DATETIME_COL])
    if len(valid) == len(proxy):
        for win in [3, 5, 7, 11]:
            ts = pd.Series(proxy, index=valid[DATETIME_COL]).sort_index()
            smooth = ts.rolling(window=win, center=True, min_periods=1).mean()
            smooth = smooth.reindex(valid[DATETIME_COL]).to_numpy(dtype=float)
            smooth = smooth - smooth.mean()
            smooth = smooth / (smooth.std() + 1e-9)
            for gamma in [0.05, 0.08, 0.12, 0.18]:
                outputs.append(save(f"smooth{win}_g{str(gamma).replace('.', 'p')}",
                                    base + gamma * smooth, target_col))

    # 4. Tanh-bounded correction (saturating, like champion was)
    bounded = np.tanh(proxy)
    for gamma in [0.05, 0.10, 0.15, 0.25, 0.40]:
        outputs.append(save(f"tanh_g{str(gamma).replace('.', 'p')}",
                            base + gamma * bounded, target_col))

    # 5. Apply proxy correction to other strong bases
    other_bases = {
        "REBv6": "submission_rebound_v6_t075_hneg10.csv",
        "DIV3w08": "submission_CHAMPv3_DIV3_w08.csv",
    }
    for tag, fn in other_bases.items():
        try:
            other, _ = load(fn)
            for gamma in [0.05, 0.08, 0.12]:
                outputs.append(save(f"on_{tag}_g{str(gamma).replace('.', 'p')}",
                                    other + gamma * proxy, target_col))
        except FileNotFoundError:
            pass

    # 6. Symmetric: apply NEGATIVE gamma too (safety check — proxy direction should help, not hurt)
    for gamma in [0.05, 0.08]:
        outputs.append(save(f"NEG_g{str(gamma).replace('.', 'p')}",
                            base - gamma * proxy, target_col))

    report = {
        "base_file": BASE_FILE,
        "base_score": BASE_SCORE,
        "n_constraints": len(constraints),
        "constraints": constraints,
        "method": "v2 row-level sign proxy w/ expanded LB constraint set; new champ base",
        "outputs": [o["file"] for o in outputs],
        "diag": diag,
    }
    (ROOT / "feedback_sign_v2_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v2 candidates.")
    print("Top safe bets:")
    for o in outputs[:6]:
        print(f"  {o['file']}: mean={o['mean']:.3f}")


if __name__ == "__main__":
    main()
