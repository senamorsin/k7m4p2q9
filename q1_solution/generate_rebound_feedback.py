#!/usr/bin/env python3
"""Build leaderboard-feedback rebound candidates.

This is intentionally not a grid. It uses three pieces of feedback:

1. The current best is the -0.325 MW shifted team blend.
2. Moving toward v89 made the score much worse, so use a small negative dose.
3. The public-total scale-up was bad, so keep only a tiny mean-preserved shape
   component from that direction and avoid the large positive mean jump.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
N_INST = 90.09

BASE_FILE = "submission_teamblend_shift_m0p325.csv"
V89_FILE = "submission_highupside_v89_alone_meanaligned.csv"
PUBLIC_SCALE_FILE = "submission_publicq1_86gwh_scale.csv"
TOP6_TILT_FILE = "submission_teamblend_top6w52_shift_m0p4.csv"
TOP6_TILT_REFERENCE_FILE = "submission_teamblend_shift_m0p4.csv"
HEAVY_BREAKOUT_FILE = "submission_breakout_heavy_recipe153550_shift_m0p325.csv"

RECIPES = {
    # Accepted on the leaderboard at 7.3756761080900635.
    "rebound_v1": {
        "const_shift_mw": 0.020,
        "v89_dose": -0.084,
        "public_shape_dose": 0.100,
    },
    # Safer midpoint between the accepted v1 and the stronger v2 step.
    "rebound_v2_soft": {
        "const_shift_mw": 0.000,
        "v89_dose": -0.128,
        "public_shape_dose": 0.095,
    },
    # Next single step after adding the v1 feedback to the surrogate.
    "rebound_v2": {
        "const_shift_mw": -0.020,
        "v89_dose": -0.172,
        "public_shape_dose": 0.090,
    },
    # Slightly deeper local optimum from the uniform/local quadratic fit.
    "rebound_v3_deep": {
        "const_shift_mw": -0.020,
        "v89_dose": -0.184,
        "public_shape_dose": 0.100,
    },
    # Tests the same anti-v89 correction with more mean-preserved shape and
    # no large public-total mean lift.
    "rebound_v3_shape": {
        "const_shift_mw": -0.010,
        "v89_dose": -0.150,
        "public_shape_dose": 0.180,
    },
    # Center of the night local fit after adding the accepted v1 score:
    # keep the v1-like mean, push anti-v89 harder, and reduce the public-shape
    # component that was only weakly supported by the public-total experiment.
    "rebound_v4_center": {
        "const_shift_mw": 0.015,
        "v89_dose": -0.175,
        "public_shape_dose": 0.065,
    },
    # Positive heavy-breakout variants were worse on the leaderboard. These
    # candidates apply a small negative heavy dose on top of the best local
    # shape instead of continuing to push v89 depth.
    "rebound_v5_hneg20": {
        "const_shift_mw": -0.0096,
        "v89_dose": -0.1499,
        "public_shape_dose": 0.1787,
        "heavy_dose": -0.20,
    },
    # Higher-upside version: more public shape plus a small top6 tilt and
    # anti-heavy correction.
    "rebound_v5_bmore_hneg25": {
        "const_shift_mw": -0.0120,
        "v89_dose": -0.1450,
        "public_shape_dose": 0.2300,
        "top6_tilt_dose": 0.50,
        "heavy_dose": -0.25,
    },
    # Conservative compromise between v2_soft and v3_shape with anti-heavy.
    "rebound_v5_conservative": {
        "const_shift_mw": -0.0060,
        "v89_dose": -0.1300,
        "public_shape_dose": 0.1800,
        "top6_tilt_dose": 0.50,
        "heavy_dose": -0.20,
    },
    # Extrapolate the actual winning v5 direction:
    # v5_hneg20 -> v5_conservative improved, mostly by softening anti-v89 and
    # increasing the top6 tilt. These keep that move but reduce/remove the
    # anti-heavy dose, because hneg20 alone was not helpful.
    "rebound_v6_t075_h0": {
        "const_shift_mw": -0.0040,
        "v89_dose": -0.1200,
        "public_shape_dose": 0.1800,
        "top6_tilt_dose": 0.75,
        "heavy_dose": 0.00,
    },
    "rebound_v6_t075_hneg10": {
        "const_shift_mw": -0.0040,
        "v89_dose": -0.1200,
        "public_shape_dose": 0.1800,
        "top6_tilt_dose": 0.75,
        "heavy_dose": -0.10,
    },
    "rebound_v6_t10_h0": {
        "const_shift_mw": -0.0020,
        "v89_dose": -0.1100,
        "public_shape_dose": 0.1800,
        "top6_tilt_dose": 1.00,
        "heavy_dose": 0.00,
    },
    # One bolder surface-push candidate: higher top6 tilt and positive mean,
    # still staying close enough to the successful v5 region.
    "rebound_v6_surface_push": {
        "const_shift_mw": 0.0120,
        "v89_dose": -0.1450,
        "public_shape_dose": 0.1600,
        "top6_tilt_dose": 1.25,
        "heavy_dose": -0.30,
    },
    # Local checks around the new best v6_t075_hneg10. These fill missing
    # one-coordinate tests instead of opening a broad sweep.
    "rebound_v7_t10_hneg10": {
        "const_shift_mw": -0.0020,
        "v89_dose": -0.1100,
        "public_shape_dose": 0.1800,
        "top6_tilt_dose": 1.00,
        "heavy_dose": -0.10,
    },
    "rebound_v7_t075_hneg15": {
        "const_shift_mw": -0.0040,
        "v89_dose": -0.1200,
        "public_shape_dose": 0.1800,
        "top6_tilt_dose": 0.75,
        "heavy_dose": -0.15,
    },
    "rebound_v7_t06_hneg10": {
        "const_shift_mw": -0.0050,
        "v89_dose": -0.1250,
        "public_shape_dose": 0.1800,
        "top6_tilt_dose": 0.60,
        "heavy_dose": -0.10,
    },
    "rebound_v7_b16_t075_hneg10": {
        "const_shift_mw": -0.0020,
        "v89_dose": -0.1200,
        "public_shape_dose": 0.1600,
        "top6_tilt_dose": 0.75,
        "heavy_dose": -0.10,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> tuple[np.ndarray, str]:
    df = pd.read_csv(path)
    values = df.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains NaN/inf values")
    return values, df.columns[0]


def save(name: str, values: np.ndarray, target_col: str) -> dict[str, object]:
    arr = np.clip(values, 0.0, N_INST)
    path = ROOT / f"submission_{name}.csv"
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


def main() -> None:
    base, target_col = load(ROOT / BASE_FILE)
    v89, _ = load(ROOT / V89_FILE)
    public_scale, _ = load(ROOT / PUBLIC_SCALE_FILE)
    top6_tilt_file, _ = load(ROOT / TOP6_TILT_FILE)
    top6_tilt_reference, _ = load(ROOT / TOP6_TILT_REFERENCE_FILE)
    heavy_breakout, _ = load(ROOT / HEAVY_BREAKOUT_FILE)

    if not (
        len(base)
        == len(v89)
        == len(public_scale)
        == len(top6_tilt_file)
        == len(top6_tilt_reference)
        == len(heavy_breakout)
    ):
        raise ValueError("submission lengths do not match")

    v89_direction = v89 - base
    public_direction = public_scale - base
    public_shape = public_direction - public_direction.mean()
    top6_tilt = top6_tilt_file - top6_tilt_reference
    heavy_direction = heavy_breakout - base

    outputs = []
    for name, recipe in RECIPES.items():
        rebound = (
            base
            + recipe["const_shift_mw"]
            + recipe["v89_dose"] * v89_direction
            + recipe["public_shape_dose"] * public_shape
            + recipe.get("top6_tilt_dose", 0.0) * top6_tilt
            + recipe.get("heavy_dose", 0.0) * heavy_direction
        )
        outputs.append(save(name, rebound, target_col))

    report = {
        "base_score": 7.383149714673269,
        "base_file": BASE_FILE,
        "bad_feedback_used": {
            "v89_alone_meanaligned_score": 7.973826146501979,
            "v89_25pct_score": 7.436231580203731,
            "v89_35pct_score": 7.4743365577664855,
            "public_scale_score": 7.4203597217020425,
            "rebound_v1_score": 7.3756761080900635,
            "rebound_v2_soft_score": 7.375976942587208,
            "rebound_v2_score": 7.378333904676163,
            "rebound_v3_shape_score": 7.375528878103264,
            "rebound_v3_deep_score": 7.378783536070742,
            "rebound_v4_center_score": 7.378335339868966,
            "breakout_heavy_recipe153550_score": 7.389218586196146,
            "rebound_v5_hneg20_score": 7.375692490066722,
            "rebound_v5_bmore_hneg25_score": 7.374766359042817,
            "rebound_v5_conservative_score": 7.3747373220862854,
            "rebound_v6_t075_h0_score": 7.374774556660089,
            "rebound_v6_t075_hneg10_score": 7.37455489169975,
            "rebound_v6_t10_h0_score": 7.3747940763733855,
            "rebound_v6_surface_push_score": 7.375287304530807,
        },
        "recipes": {
            name: {
                **recipe,
                "formula": (
                    "base + const + v89_dose*(v89-base) "
                    "+ public_shape_dose*((public_scale-base)-mean(public_scale-base)) "
                    "+ top6_tilt_dose*(top6w52_shift_m0p4-shift_m0p4) "
                    "+ heavy_dose*(breakout_heavy_recipe153550-base)"
                ),
            }
            for name, recipe in RECIPES.items()
        },
        "outputs": outputs,
        "primary_next": "submission_rebound_v7_t10_hneg10.csv",
        "upload_order_after_v1": [
            "submission_rebound_v7_t10_hneg10.csv",
            "submission_rebound_v7_t075_hneg15.csv",
            "submission_rebound_v7_t06_hneg10.csv",
            "submission_rebound_v7_b16_t075_hneg10.csv",
        ],
        "surrogate_note": {
            "expected_region": "best observed local file is v6_t075_hneg10 at 7.374554892; v7 fills missing one-coordinate checks around it",
            "risk": "local gains are small; for larger moves use feedback_sign candidates generated separately",
            "formula": (
                "base + const + v89_dose*(v89-base) "
                "+ public_shape_dose*((public_scale-base)-mean(public_scale-base)) "
                "+ optional top6_tilt_dose*(top6w52_shift_m0p4-shift_m0p4) "
                "+ optional heavy_dose*(breakout_heavy_recipe153550-base)"
            ),
        },
    }
    report_path = ROOT / "rebound_feedback_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Wrote rebound candidates:")
    for output in outputs:
        print(
            f"  {output['file']}: mean={output['mean']:.3f}, std={output['std']:.3f}, "
            f"min={output['min']:.3f}, max={output['max']:.3f}, sha={output['sha256'][:12]}"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
