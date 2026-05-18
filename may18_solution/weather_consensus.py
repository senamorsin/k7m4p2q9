#!/usr/bin/env python3
"""Canonical Open-Meteo weather consensus for judge-day feature files.

The final models should not each interpret the weather directory differently.
This module applies the weather once, upstream, and writes a regular
valid-like feature CSV that every model can consume identically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATE_COL = "METEOFORECASTHOUR_OPENM_Datetime"
DEFAULT_TARGET_DATE = "2026-05-18"
DEFAULT_OUTPUT = Path("tmp") / "may18_features_openmeteo_consensus.csv"

WEATHER_COLUMNS = (
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_direction_120m",
    "wind_direction_180m",
    "wind_gusts_10m",
    "temperature_80m",
    "temperature_120m",
    "pressure_msl",
    "rain",
    "showers",
    "snowfall",
    "cloud_cover_low",
)

DIRECTION_COLUMNS = tuple(c for c in WEATHER_COLUMNS if c.startswith("wind_direction"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path, root: Path | None = None) -> str:
    if root is None:
        root = Path.cwd()
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def direction_to_degrees(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) and np.nanmax(np.abs(finite)) < 1.5:
        arr = arr * 1000.0
    return arr % 360.0


def circular_mean_degrees(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    out = np.full(matrix.shape[0], np.nan, dtype=float)
    for i, row in enumerate(matrix):
        row = row[np.isfinite(row)]
        if len(row) == 0:
            continue
        rad = np.deg2rad(row)
        s = float(np.mean(np.sin(rad)))
        c = float(np.mean(np.cos(rad)))
        if abs(s) < 1e-12 and abs(c) < 1e-12:
            out[i] = float(np.nanmedian(row) % 360.0)
        else:
            out[i] = float(np.rad2deg(np.arctan2(s, c)) % 360.0)
    return out


def load_weather_source(path: Path, target_date: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    if DATE_COL not in df.columns:
        raise ValueError(f"{path} lacks {DATE_COL}")
    df = df[df[DATE_COL].dt.date.astype(str) == target_date].copy()
    if df.empty:
        raise ValueError(f"{path} has no rows for {target_date}")
    df = df.sort_values(DATE_COL)
    if df[DATE_COL].duplicated().any():
        df = df.drop_duplicates(DATE_COL, keep="last")
    df = df.set_index(DATE_COL)
    return df


def weather_source_name(path: Path, target_date: str) -> str:
    prefix = f"openmeteo_forecast_{target_date}_"
    stem = path.stem
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def build_weather_consensus(
    weather_dir: Path,
    target_date: str = DEFAULT_TARGET_DATE,
    min_sources: int = 5,
    source_filter: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_paths = sorted(p for p in weather_dir.glob("*.csv") if p.is_file())
    if source_filter is not None:
        csv_paths = [p for p in csv_paths if weather_source_name(p, target_date) in source_filter]
    loaded_sources: list[tuple[Path, pd.DataFrame]] = []
    failures: list[dict[str, str]] = []

    for path in csv_paths:
        try:
            loaded_sources.append((path, load_weather_source(path, target_date)))
        except Exception as exc:
            failures.append({"path": str(path), "error": repr(exc)})

    if len(loaded_sources) < min_sources:
        raise RuntimeError(
            f"Need at least {min_sources} usable weather sources; got {len(loaded_sources)}. "
            f"Failures: {failures}"
        )

    all_index = sorted(set().union(*(set(df.index) for _, df in loaded_sources)))
    consensus = pd.DataFrame(index=pd.DatetimeIndex(all_index, name=DATE_COL))
    source_names = [weather_source_name(path, target_date) for path, _ in loaded_sources]

    for col in WEATHER_COLUMNS:
        arrays = []
        for _, df in loaded_sources:
            if col not in df.columns:
                continue
            if col in DIRECTION_COLUMNS:
                values = pd.Series(direction_to_degrees(df[col]), index=df.index).reindex(consensus.index)
            else:
                values = pd.to_numeric(df[col], errors="coerce").reindex(consensus.index)
            arrays.append(values.to_numpy(dtype=float))
        if not arrays:
            continue
        stack = np.vstack(arrays).T
        if col in DIRECTION_COLUMNS:
            consensus[col] = circular_mean_degrees(stack) / 1000.0
        else:
            consensus[col] = np.nanmedian(stack, axis=1)

    consensus = consensus.reset_index()
    numeric = consensus.drop(columns=[DATE_COL]).apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad_cols = sorted(numeric.columns[numeric.isna().any()].tolist())
        raise ValueError(f"consensus weather contains NaNs in columns {bad_cols}")

    metadata = {
        "target_date": target_date,
        "method": {
            "numeric": "median_across_sources",
            "direction": "circular_mean_across_sources_then_degrees_div_1000",
        },
        "source_count": len(loaded_sources),
        "sources": [
            {
                "path": rel(path),
                "sha256": sha256_file(path),
                "rows": int(len(df)),
            }
            for path, df in loaded_sources
        ],
        "source_names": source_names,
        "failures": failures,
        "rows": int(len(consensus)),
        "columns": list(consensus.columns),
    }
    return consensus, metadata


def apply_weather_consensus(
    features_path: Path,
    weather_dir: Path,
    output_path: Path,
    target_date: str = DEFAULT_TARGET_DATE,
    min_sources: int = 5,
    sources: set[str] | None = None,
) -> dict[str, Any]:
    features = pd.read_csv(features_path, parse_dates=[DATE_COL])
    if DATE_COL not in features.columns:
        raise ValueError(f"{features_path} lacks {DATE_COL}")

    consensus, weather_meta = build_weather_consensus(
        weather_dir,
        target_date,
        min_sources,
        source_filter=sources,
    )
    consensus_by_time = consensus.set_index(DATE_COL)

    out = features.copy()
    feature_times = pd.to_datetime(out[DATE_COL], errors="coerce")
    if feature_times.isna().any():
        raise ValueError(f"{features_path} contains invalid datetimes")

    overlay_columns = [c for c in WEATHER_COLUMNS if c in out.columns and c in consensus_by_time.columns]
    matched = feature_times.isin(consensus_by_time.index).to_numpy()
    for col in overlay_columns:
        mapped = feature_times.map(consensus_by_time[col])
        out.loc[matched, col] = mapped[matched].to_numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, float_format="%.6f")

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = {
        "script": Path(__file__).name,
        "features_input": {
            "path": rel(features_path),
            "sha256": sha256_file(features_path),
            "rows": int(len(features)),
        },
        "weather_dir": rel(weather_dir),
        "weather": weather_meta,
        "overlay": {
            "target_date": target_date,
            "columns": overlay_columns,
            "matched_rows": int(matched.sum()),
            "unmatched_rows": int((~matched).sum()),
        },
        "output": {
            "path": rel(output_path),
            "sha256": sha256_file(output_path),
            "rows": int(len(out)),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply canonical Open-Meteo consensus to a feature CSV.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--weather-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-date", default=DEFAULT_TARGET_DATE)
    parser.add_argument("--min-sources", type=int, default=5)
    parser.add_argument("--sources", nargs="*", default=None, help="optional exact Open-Meteo source names to include")
    args = parser.parse_args()

    metadata = apply_weather_consensus(
        features_path=args.features.resolve(),
        weather_dir=args.weather_dir.resolve(),
        output_path=args.output.resolve(),
        target_date=args.target_date,
        min_sources=args.min_sources,
        sources=set(args.sources) if args.sources else None,
    )
    print(json.dumps({
        "output": metadata["output"]["path"],
        "metadata": rel(args.output.resolve().with_suffix(".metadata.json")),
        "matched_rows": metadata["overlay"]["matched_rows"],
        "source_count": metadata["weather"]["source_count"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
