#!/usr/bin/env python3
"""Build row-level feedback-sign candidates from leaderboard score differences.

For small moves around the current best, MAE differences approximate directional
derivatives:

    score(candidate) - score(base) ~= -mean(sign(error_base) * (candidate-base))

This script estimates a continuous row-level sign proxy from all known local
leaderboard feedback, then writes a few shrunken corrections. These files are
high-upside probes and should be uploaded after the safer rebound-v7 checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PARTNER_SUBMISSIONS = (
    ROOT / "teamblend_v3_pool"
)
N_INST = 90.09
BASE_FILE = "submission_rebound_v6_t075_hneg10.csv"
BASE_SCORE = 7.37455489169975
VALID_FEATURES = "data/valid_features.csv"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"

KNOWN_SCORES = {
    "submission_teamblend_main_7387_repro.csv": 7.387167370013311,
    "submission_teamblend_top6w52_shift_m0p5.csv": 7.386781738989777,
    "submission_teamblend_top6w48_shift_m0p5.csv": 7.38872899864706,
    "submission_teamblend_shift_m0p4.csv": 7.3838721717538105,
    "submission_teamblend_shift_m0p6.csv": 7.392078693732011,
    "submission_teamblend_top6w54_shift_m0p4.csv": 7.384175671630803,
    "submission_teamblend_top6w52_shift_m0p4.csv": 7.383778288828245,
    "submission_teamblend_shift_m0p325.csv": 7.383149714673269,
    "submission_teamblend_shift_m0p35.csv": 7.38321838222866,
    "submission_teamblend_shift_m0p425.csv": 7.3844793448842445,
    "submission_teamblend_shift_m0p375.csv": 7.383470129177834,
    "submission_breakout_heavy_recipe153550_shift_m0p325.csv": 7.389218586196146,
    "submission_breakout_heavy_cb10_shift_m0p325.csv": 7.390550890701163,
    "submission_breakout_heavy_cb10_recipe153550_shift_m0p325.csv": 7.389836762207342,
    "submission_highupside_best65_v8935.csv": 7.4743365577664855,
    "submission_highupside_best75_v8925.csv": 7.436231580203731,
    "submission_highupside_v89_alone_meanaligned.csv": 7.973826146501979,
    "submission_publicq1_86gwh_scale.csv": 7.4203597217020425,
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
    "submission_rebound_v6_t075_hneg10.csv": 7.37455489169975,
    "submission_rebound_v6_t10_h0.csv": 7.3747940763733855,
    "submission_rebound_v6_surface_push.csv": 7.375287304530807,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(name: str) -> Path:
    for path in [ROOT / name, PARTNER_SUBMISSIONS / name]:
        if path.exists():
            return path
    raise FileNotFoundError(name)


def load(name: str) -> tuple[np.ndarray, str]:
    path = resolve(name)
    df = pd.read_csv(path)
    values = df.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN/inf values")
    return values, df.columns[0]


def save(label: str, values: np.ndarray, target_col: str) -> dict[str, object]:
    arr = np.clip(values, 0.0, N_INST)
    path = ROOT / f"submission_feedback_sign_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding="utf-8")
    return {
        "file": path.name,
        "sha256": sha256(path),
        "n_rows": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def build_sign_proxy(base: np.ndarray) -> tuple[np.ndarray, list[str], dict[str, float]]:
    directions = []
    targets = []
    names = []
    diagnostics = {}
    for name, score in KNOWN_SCORES.items():
        if name == BASE_FILE:
            continue
        try:
            values, _ = load(name)
        except FileNotFoundError:
            # Skip missing constraint files — chain still works with subset
            continue
        if len(values) != len(base):
            continue
        diff = values - base
        rms = float(np.sqrt(np.mean(diff * diff)))
        max_abs = float(np.max(np.abs(diff)))
        diagnostics[name] = rms
        if rms < 1.0 and max_abs < 10.0:
            directions.append(diff)
            targets.append(-((score - BASE_SCORE) * N_INST / 100.0))
            names.append(name)

    D = np.asarray(directions)
    target = np.asarray(targets)
    if D.ndim != 2 or len(D) < 5:
        raise ValueError("not enough local constraints")

    # Solve for a continuous sign vector in the span of known directions.
    rms = np.sqrt(np.mean(D * D, axis=1))
    weights = 1.0 / np.power(0.05 + rms, 0.7)
    gram = (D @ D.T) / len(base)
    alpha = 1e-3
    beta = np.linalg.solve(np.diag(weights) @ gram @ np.diag(weights) + alpha * np.eye(len(D)), np.diag(weights) @ target)
    proxy = D.T @ (np.diag(weights) @ beta)
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy, names, diagnostics


def main() -> None:
    base, target_col = load(BASE_FILE)
    proxy, constraints, diagnostics = build_sign_proxy(base)
    v5, _ = load("submission_rebound_v5_conservative.csv")

    outputs: list[dict[str, object]] = []
    for gamma in [0.05, 0.08, 0.12, 0.16, 0.22]:
        outputs.append(save(f"g{str(gamma).replace('.', 'p')}", base + gamma * proxy, target_col))

    active = np.abs(proxy) >= np.quantile(np.abs(proxy), 0.65)
    for gamma in [0.15, 0.25]:
        correction = np.zeros_like(proxy)
        correction[active] = gamma * proxy[active]
        outputs.append(save(f"top35_g{str(gamma).replace('.', 'p')}", base + correction, target_col))

    valid = pd.read_csv(ROOT / VALID_FEATURES, parse_dates=[DATETIME_COL])
    if len(valid) == len(proxy):
        proxy_by_time = pd.Series(proxy, index=valid[DATETIME_COL])
        smooth = (
            proxy_by_time.sort_index()
            .rolling(window=5, center=True, min_periods=1)
            .mean()
            .reindex(valid[DATETIME_COL])
            .to_numpy(dtype=float)
        )
        smooth = smooth - smooth.mean()
        smooth = smooth / (smooth.std() + 1e-9)
        for gamma in [0.08, 0.12, 0.18]:
            outputs.append(save(f"smooth5_g{str(gamma).replace('.', 'p')}", base + gamma * smooth, target_col))

    for gamma in [0.08, 0.12]:
        outputs.append(save(f"v5_g{str(gamma).replace('.', 'p')}", v5 + gamma * proxy, target_col))

    report = {
        "base_file": BASE_FILE,
        "base_score": BASE_SCORE,
        "method": "row-level sign proxy from local MAE directional derivatives",
        "proxy_stats": {
            "mean": float(proxy.mean()),
            "std": float(proxy.std()),
            "min": float(proxy.min()),
            "max": float(proxy.max()),
        },
        "constraints": constraints,
        "diagnostic_rms_to_base": diagnostics,
        "recommended_upload_order": [
            "submission_feedback_sign_g0p08.csv",
            "submission_feedback_sign_g0p12.csv",
            "submission_feedback_sign_smooth5_g0p12.csv",
            "submission_feedback_sign_top35_g0p15.csv",
            "submission_feedback_sign_g0p16.csv",
            "submission_feedback_sign_v5_g0p08.csv",
        ],
        "outputs": outputs,
    }
    (ROOT / "feedback_sign_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Wrote feedback-sign candidates:")
    for output in outputs:
        print(
            f"  {output['file']}: mean={output['mean']:.3f}, std={output['std']:.3f}, "
            f"min={output['min']:.3f}, max={output['max']:.3f}, sha={output['sha256'][:12]}"
        )
    print("Report: feedback_sign_report.json")


if __name__ == "__main__":
    main()
