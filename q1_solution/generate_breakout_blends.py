#!/usr/bin/env python3
"""Build non-micro blend candidates from the recovered TOP6 components.

`submission_TOP6_MEDIAN.csv` is exactly the median of six source submissions.
The median was a good safe blend, but it can also suppress one component that
has useful independent signal. These candidates replace the median with a full
component or a component family, while keeping the proven v77/v71 anchor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
N_INST = 90.09
TARGET_COL = "Выработка. Результирующий расчет"
SHIFT = -0.325

TOP6_COMPONENTS = {
    "10stta_tempd15": "submission_10sTTA_TEMPd15.csv",
    "new10s_w53": "submission_NEW10sCHAMP_w53.csv",
    "new10s_cb5": "submission_NEW10sCHAMP_CB5.csv",
    "new10s_recipe203050": "submission_NEW10sCHAMP_recipe_20_30_50.csv",
    "new10s_recipe153550": "submission_NEW10sCHAMP_recipe_15_35_50.csv",
    "new10s_cb10": "submission_NEW10sCHAMP_CB10.csv",
}


def load(name: str) -> np.ndarray:
    # Look in ROOT first, then teamblend_v3_pool/
    for cand in (ROOT / name, ROOT / "teamblend_v3_pool" / name):
        if cand.exists():
            path = cand
            break
    else:
        raise FileNotFoundError(name)
    values = pd.read_csv(path).iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN/inf values")
    return values


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name: str, values: np.ndarray) -> dict[str, object]:
    arr = np.clip(values, 0.0, N_INST)
    path = ROOT / f"submission_{name}.csv"
    pd.DataFrame({TARGET_COL: arr}).to_csv(path, index=False, encoding="utf-8")
    return {
        "file": path.name,
        "sha256": sha256(path),
        "n_rows": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def anchored(source: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return 0.50 * source + 0.50 * anchor + SHIFT


def source_heavy(source: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return 0.65 * source + 0.35 * anchor + SHIFT


def main() -> None:
    valid_rows = len(pd.read_csv(ROOT / "data" / "valid_features.csv"))
    anchor = load("submission_v77w30_v71.csv")
    top6_median = load("submission_TOP6_MEDIAN.csv")
    top6_avg = load("submission_TOP6_AVG.csv")
    top6_inv = load("submission_TOP6_INVSCORE.csv")

    components = {key: load(filename) for key, filename in TOP6_COMPONENTS.items()}
    for name, values in {
        "anchor": anchor,
        "TOP6_MEDIAN": top6_median,
        "TOP6_AVG": top6_avg,
        "TOP6_INVSCORE": top6_inv,
        **components,
    }.items():
        if len(values) != valid_rows:
            raise ValueError(f"{name}: expected {valid_rows} rows, got {len(values)}")

    exact_avg = np.mean(np.vstack(list(components.values())), axis=0)
    exact_median = np.median(np.vstack(list(components.values())), axis=0)
    checks = {
        "top6_avg_max_abs_diff": float(np.max(np.abs(exact_avg - top6_avg))),
        "top6_median_max_abs_diff": float(np.max(np.abs(exact_median - top6_median))),
    }

    outputs: list[dict[str, object]] = []
    outputs.append(save("breakout_top6avg_shift_m0p325", anchored(top6_avg, anchor)))
    outputs.append(save("breakout_top6inv_shift_m0p325", anchored(top6_inv, anchor)))

    for key, values in components.items():
        outputs.append(save(f"breakout_{key}_shift_m0p325", anchored(values, anchor)))

    cb_pair = 0.50 * components["new10s_cb5"] + 0.50 * components["new10s_cb10"]
    recipe_pair = (
        0.50 * components["new10s_recipe203050"]
        + 0.50 * components["new10s_recipe153550"]
    )
    diverse_pair = (
        0.50 * components["new10s_cb10"]
        + 0.50 * components["new10s_recipe153550"]
    )
    outputs.append(save("breakout_cbpair_shift_m0p325", anchored(cb_pair, anchor)))
    outputs.append(save("breakout_recipepair_shift_m0p325", anchored(recipe_pair, anchor)))
    outputs.append(save("breakout_cb10_recipe153550_shift_m0p325", anchored(diverse_pair, anchor)))

    # Coarse, higher-upside probes: if one recovered source is genuinely stronger
    # than the median, it needs more than 50% dose to move the score by ~0.05.
    outputs.append(save("breakout_heavy_cb10_shift_m0p325", source_heavy(components["new10s_cb10"], anchor)))
    outputs.append(
        save(
            "breakout_heavy_recipe153550_shift_m0p325",
            source_heavy(components["new10s_recipe153550"], anchor),
        )
    )
    outputs.append(save("breakout_heavy_cb10_recipe153550_shift_m0p325", source_heavy(diverse_pair, anchor)))
    outputs.append(save("breakout_heavy_top6avg_shift_m0p325", 0.60 * top6_avg + 0.40 * anchor + SHIFT))

    report = {
        "shift_mw": SHIFT,
        "anchor": "submission_v77w30_v71.csv",
        "checks": checks,
        "top6_components": TOP6_COMPONENTS,
        "recommended_upload_order": [
            "submission_breakout_heavy_cb10_recipe153550_shift_m0p325.csv",
            "submission_breakout_heavy_cb10_shift_m0p325.csv",
            "submission_breakout_heavy_recipe153550_shift_m0p325.csv",
            "submission_breakout_cb10_shift_m0p325.csv",
            "submission_breakout_new10s_recipe153550_shift_m0p325.csv",
            "submission_breakout_cb10_recipe153550_shift_m0p325.csv",
            "submission_breakout_top6avg_shift_m0p325.csv",
        ],
        "outputs": outputs,
    }
    report_path = ROOT / "breakout_blend_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Recovered TOP6 checks:")
    for key, value in checks.items():
        print(f"  {key}: {value:.3e}")
    print("Wrote breakout submissions:")
    for item in outputs:
        print(
            f"  {item['file']}: mean={item['mean']:.3f}, std={item['std']:.3f}, "
            f"sha={str(item['sha256'])[:12]}"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
