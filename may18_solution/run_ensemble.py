#!/usr/bin/env python3
"""Neutral two-model inference entrypoint for the 18.05 forecast.

The announced final scoring window is 00:00-08:00 MSK because planned repair
works started at 09:00.  The default strategy therefore uses the best validated
single model on scored hours, while keeping the rest of the day valid and
physically clipped for submission format completeness.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COL = "Выработка"
CAPACITY = 90.09


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_pred(path: Path, n: int) -> np.ndarray:
    df = pd.read_csv(path)
    if TARGET_COL in df.columns:
        values = df[TARGET_COL]
    elif len(df.columns) == 1:
        values = df.iloc[:, 0]
    else:
        raise ValueError(f"{path} must contain one prediction column")
    pred = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if pred.shape != (n,):
        raise ValueError(f"{path}: expected {n} rows, got {len(pred)}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{path}: non-finite predictions")
    return np.clip(pred, 0.0, CAPACITY)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run model1/model2 May 18 forecast.")
    parser.add_argument("--features", type=Path, default=Path("data/raw/may18_features_actualized_bestmatch.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/submission_may18_current.csv"))
    parser.add_argument("--no-header-output", type=Path, default=Path("submissions/submission_may18_current_no_header.csv"))
    parser.add_argument("--model-root", type=Path, default=Path("model_weights"))
    parser.add_argument("--model-source-dir", type=Path, default=Path("model_sources"))
    parser.add_argument(
        "--method",
        choices=("scored-window-model1", "mean"),
        default="scored-window-model1",
        help="default uses model1 on scored hours 00-08 and mean on 09-23",
    )
    args = parser.parse_args()

    features = args.features
    feature_df = pd.read_csv(features)
    n = len(feature_df)
    parts_dir = args.output.parent / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    preds = []
    for model_name in ("model1", "model2"):
        module = load_module(args.model_source_dir / f"{model_name}.py", model_name)
        part_path = parts_dir / f"{model_name}.csv"
        module.predict_day(
            str(features),
            str(args.model_root / model_name),
            str(part_path),
            None,
        )
        preds.append(read_pred(part_path, n))

    mean_pred = np.mean(np.vstack(preds), axis=0)
    if args.method == "scored-window-model1":
        if "hour_of_day" in feature_df.columns:
            hours = pd.to_numeric(feature_df["hour_of_day"], errors="raise").to_numpy(dtype=int)
        else:
            hours = pd.to_datetime(feature_df["METEOFORECASTHOUR_OPENM_Datetime"]).dt.hour.to_numpy(dtype=int)
        scored_mask = (hours >= 0) & (hours <= 8)
        final = mean_pred.copy()
        final[scored_mask] = preds[0][scored_mask]
    else:
        final = mean_pred
    final = np.clip(final, 0.0, CAPACITY)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({TARGET_COL: final}).to_csv(args.output, index=False, float_format="%.6f")
    pd.DataFrame({TARGET_COL: final}).to_csv(args.no_header_output, index=False, header=False, float_format="%.6f")
    print(f"Wrote {args.output} and {args.no_header_output}; rows={len(final)} mean={final.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
