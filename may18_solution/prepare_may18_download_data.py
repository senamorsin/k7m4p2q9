#!/usr/bin/env python3
"""Prepare official May 18 download data for the final pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATE_COL = "METEOFORECASTHOUR_OPENM_Datetime"
TARGET_COL = "Выработка. Результирующий расчет"
TARGET_DATE = "2026-05-18"

DEFAULT_SOURCE = ROOT / "data" / "raw" / "may18_download_raw.csv"
DEFAULT_BASE_TRAIN = ROOT / "data" / "raw" / "train_dataset.csv"
DEFAULT_FEATURES = ROOT / "data" / "raw" / "may18_features.csv"
DEFAULT_RECENT_TRAIN = ROOT / "data" / "raw" / "may18_recent_train.csv"
DEFAULT_COMBINED_TRAIN = ROOT / "data" / "raw" / "train_dataset_plus_may18_recent.csv"
DEFAULT_METADATA = ROOT / "data" / "raw" / "may18_download_preparation.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def normalize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL]).dt.round("h")
    out[DATE_COL] = out[DATE_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare official May 18 download data.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--recent-train-out", type=Path, default=DEFAULT_RECENT_TRAIN)
    parser.add_argument("--combined-train-out", type=Path, default=DEFAULT_COMBINED_TRAIN)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--target-date", default=TARGET_DATE)
    args = parser.parse_args()

    source = args.source.resolve()
    base_train_path = args.base_train.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if not base_train_path.exists():
        raise FileNotFoundError(base_train_path)

    raw = pd.read_csv(source)
    if DATE_COL not in raw.columns or TARGET_COL not in raw.columns:
        raise ValueError(f"{source} must contain {DATE_COL!r} and {TARGET_COL!r}")
    raw = normalize_datetime(raw)

    missing_target = raw[TARGET_COL].isna()
    features = raw.loc[missing_target].copy()
    if len(features) != 24:
        raise ValueError(f"Expected exactly 24 blank target rows, got {len(features)}")
    if set(features["month"].astype(int)) != {5}:
        raise ValueError("Blank target rows are not all month 5")
    if set(features["hour_of_day"].astype(int)) != set(range(24)):
        raise ValueError("Blank target rows must contain each hour_of_day 0..23 exactly once")

    features[DATE_COL] = pd.to_datetime(args.target_date) + pd.to_timedelta(features["hour_of_day"].astype(int), unit="h")
    features[DATE_COL] = features[DATE_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    features = features.sort_values("hour_of_day").drop(columns=[TARGET_COL]).reset_index(drop=True)

    recent_train = raw.loc[~missing_target].copy()
    recent_train[TARGET_COL] = pd.to_numeric(recent_train[TARGET_COL], errors="raise")
    recent_train = recent_train.sort_values(DATE_COL).reset_index(drop=True)

    base_train = pd.read_csv(base_train_path)
    if list(base_train.columns) != list(recent_train.columns):
        raise ValueError("Base train and recent train columns differ")
    combined_train = pd.concat([base_train, recent_train], ignore_index=True)
    combined_train[DATE_COL] = pd.to_datetime(combined_train[DATE_COL]).dt.strftime("%Y-%m-%d %H:%M:%S")
    combined_train = combined_train.sort_values(DATE_COL).reset_index(drop=True)

    features_out = args.features_out.resolve()
    recent_train_out = args.recent_train_out.resolve()
    combined_train_out = args.combined_train_out.resolve()
    metadata_out = args.metadata_out.resolve()
    write_csv(features_out, features)
    write_csv(recent_train_out, recent_train)
    write_csv(combined_train_out, combined_train)

    feature_dates = pd.to_datetime(features[DATE_COL]).dt.date.astype(str)
    metadata = {
        "script": Path(__file__).name,
        "source": rel(source),
        "source_sha256": sha256_file(source),
        "base_train": rel(base_train_path),
        "base_train_sha256": sha256_file(base_train_path),
        "target_date": args.target_date,
        "outputs": {
            "features": {
                "path": rel(features_out),
                "sha256": sha256_file(features_out),
                "rows": int(len(features)),
                "date_counts": {str(k): int(v) for k, v in feature_dates.value_counts().sort_index().items()},
            },
            "recent_train": {
                "path": rel(recent_train_out),
                "sha256": sha256_file(recent_train_out),
                "rows": int(len(recent_train)),
                "min_datetime": str(pd.to_datetime(recent_train[DATE_COL]).min()),
                "max_datetime": str(pd.to_datetime(recent_train[DATE_COL]).max()),
            },
            "combined_train": {
                "path": rel(combined_train_out),
                "sha256": sha256_file(combined_train_out),
                "rows": int(len(combined_train)),
                "min_datetime": str(pd.to_datetime(combined_train[DATE_COL]).min()),
                "max_datetime": str(pd.to_datetime(combined_train[DATE_COL]).max()),
            },
        },
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
