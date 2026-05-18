#!/usr/bin/env python3
"""High-upside candidates from late archive models.

These are intentionally not fine weight sweeps. They test whether a late,
diverse source (v89 covariate-shift, plus a small optional v85/v87 mix) has
real independent signal beyond the current best team blend.
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
TARGET_COL = "Выработка. Результирующий расчет"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> np.ndarray:
    values = pd.read_csv(path).iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains NaN/inf values")
    return values


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


def mean_aligned(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return source + (reference.mean() - source.mean())


def main() -> None:
    valid_rows = len(pd.read_csv(ROOT / "data" / "valid_features.csv"))
    best = load(ROOT / "submission_teamblend_shift_m0p325.csv")
    sources = {
        "v89": load(PARTNER_SUBMISSIONS / "submission_v89_alone.csv"),
        "v85": load(PARTNER_SUBMISSIONS / "submission_v85_alone.csv"),
        "v87": load(PARTNER_SUBMISSIONS / "submission_v87_alone.csv"),
    }
    for name, values in {"best": best, **sources}.items():
        if len(values) != valid_rows:
            raise ValueError(f"{name}: expected {valid_rows}, got {len(values)}")

    v89 = mean_aligned(sources["v89"], best)
    v85 = mean_aligned(sources["v85"], best)
    v87 = mean_aligned(sources["v87"], best)
    diverse_mix = np.median(np.vstack([v89, v85, v87]), axis=0)

    outputs: list[dict[str, object]] = []

    # Full-source probe: risky, but this is the only kind of candidate that can
    # reveal a genuinely better late model instead of another 0.001 tweak.
    outputs.append(save("highupside_v89_alone_meanaligned", v89))

    # Coarse doses, not a sweep. If v89 is useful, 20-35% should move the score.
    outputs.append(save("highupside_best75_v8925", 0.75 * best + 0.25 * v89))
    outputs.append(save("highupside_best65_v8935", 0.65 * best + 0.35 * v89))

    # Multi-source late-model median, also coarse.
    outputs.append(save("highupside_best75_latemix25", 0.75 * best + 0.25 * diverse_mix))
    outputs.append(save("highupside_best65_latemix35", 0.65 * best + 0.35 * diverse_mix))

    report = {
        "base": "submission_teamblend_shift_m0p325.csv",
        "sources": {
            "v89": str(PARTNER_SUBMISSIONS / "submission_v89_alone.csv"),
            "v85": str(PARTNER_SUBMISSIONS / "submission_v85_alone.csv"),
            "v87": str(PARTNER_SUBMISSIONS / "submission_v87_alone.csv"),
        },
        "recommended_upload_order": [
            "submission_highupside_v89_alone_meanaligned.csv",
            "submission_highupside_best75_v8925.csv",
            "submission_highupside_best65_v8935.csv",
            "submission_highupside_best75_latemix25.csv",
        ],
        "outputs": outputs,
    }
    report_path = ROOT / "highupside_blend_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Wrote high-upside submissions:")
    for item in outputs:
        print(
            f"  {item['file']}: mean={item['mean']:.3f}, std={item['std']:.3f}, "
            f"sha={str(item['sha256'])[:12]}"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
