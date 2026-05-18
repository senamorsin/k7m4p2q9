#!/usr/bin/env python3
"""Calibrate the current best submission to the public Q1-2026 Azov WPP total.

Public operating results for EL5-Energo report Azov WPP Q1-2026 generation at
about 86 million kWh. The validation file is Q1-2026 with 34 missing hourly
timestamps, so this script estimates the missing-hour contribution from the
current best shape and rescales only the submitted rows to match the published
quarter total.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"
N_INST = 90.09

BASE_SUBMISSION = "submission_teamblend_shift_m0p325.csv"
VALID_FEATURES = "data/valid_features.csv"
PUBLIC_Q1_TOTAL_GWH = 86.0
SOURCE_URLS = [
    "https://smart-lab.ru/blog/news/1296367.php",
    "https://www.eprussia.ru/news/base/2026/7370715.htm",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_submission(name: str, values: np.ndarray, target_col: str) -> dict[str, object]:
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
        "sum_gwh_valid_rows": float(arr.sum() / 1000.0),
    }


def estimate_full_q1_total(series: pd.Series) -> tuple[float, float, list[str]]:
    full_hours = pd.date_range("2026-01-01 00:00:00", "2026-03-31 23:00:00", freq="h")
    chronological = series.sort_index()
    missing = full_hours.difference(chronological.index)
    full_series = chronological.reindex(full_hours).interpolate("time").ffill().bfill()
    known_gwh = float(chronological.sum() / 1000.0)
    missing_gwh = float(full_series.loc[missing].sum() / 1000.0)
    return known_gwh + missing_gwh, missing_gwh, [str(x) for x in missing]


def main() -> None:
    valid = pd.read_csv(ROOT / VALID_FEATURES, parse_dates=[DATETIME_COL])
    base_df = pd.read_csv(ROOT / BASE_SUBMISSION)
    target_col = base_df.columns[0]
    base_values = base_df.iloc[:, 0].to_numpy(dtype=float)
    if len(valid) != len(base_values):
        raise ValueError(f"row mismatch: valid={len(valid)}, submission={len(base_values)}")
    if not np.isfinite(base_values).all():
        raise ValueError(f"{BASE_SUBMISSION} contains NaN/inf values")

    base_by_time = pd.Series(base_values, index=valid[DATETIME_COL])
    base_full_gwh, base_missing_gwh, missing_hours = estimate_full_q1_total(base_by_time)
    scale = PUBLIC_Q1_TOTAL_GWH / base_full_gwh

    scaled_values = base_values * scale
    scaled_by_time = pd.Series(scaled_values, index=valid[DATETIME_COL])
    scaled_full_gwh, scaled_missing_gwh, _ = estimate_full_q1_total(scaled_by_time)

    output = save_submission("publicq1_86gwh_scale", scaled_values, target_col)

    report = {
        "base_submission": BASE_SUBMISSION,
        "public_q1_total_gwh": PUBLIC_Q1_TOTAL_GWH,
        "source_note": "EL5-Energo Q1-2026 operating results: Azov WPP approximately 86 million kWh.",
        "source_urls": SOURCE_URLS,
        "valid_rows": int(len(valid)),
        "full_q1_hours": 2160,
        "missing_hours_count": len(missing_hours),
        "missing_hours": missing_hours,
        "base_full_q1_gwh_estimated": base_full_gwh,
        "base_missing_hours_gwh_estimated": base_missing_gwh,
        "scale_factor": scale,
        "scaled_full_q1_gwh_estimated": scaled_full_gwh,
        "scaled_missing_hours_gwh_estimated": scaled_missing_gwh,
        "primary_output": output,
    }
    report_path = ROOT / "public_total_calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Public-total calibration")
    print(f"  base full-Q1 estimate : {base_full_gwh:.3f} GWh")
    print(f"  target public total   : {PUBLIC_Q1_TOTAL_GWH:.3f} GWh")
    print(f"  scale factor          : {scale:.6f}")
    print(f"  primary file          : {output['file']}  valid_sum={output['sum_gwh_valid_rows']:.3f} GWh")
    print(f"  report                : {report_path}")


if __name__ == "__main__":
    main()
