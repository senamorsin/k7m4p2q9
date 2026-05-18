#!/usr/bin/env python3
"""
Fetch live Open-Meteo forecast weather for the judge-day pipeline.

This script fetches future forecast-model outputs for May 18, 2026 or any
requested target date.

Outputs one CSV per successful model in a directory that can be passed to:

    weather_consensus.py --weather-dir <output-dir>

Each CSV uses the competition datetime column name and weather column names so
the downstream feature-preparation step can consume it deterministically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
WEATHER_DIR = ROOT / "data" / "weather"

DATE_COL = "METEOFORECASTHOUR_OPENM_Datetime"
LATITUDE = 46.8268455973
LONGITUDE = 38.7179393185
TIMEZONE = "Europe/Moscow"
DEFAULT_TARGET_DATE = "2026-05-18"
DEFAULT_MODELS = (
    "best_match",
    "gfs_seamless",
    "ecmwf_ifs025",
    "icon_seamless",
    "gem_seamless",
    "ukmo_seamless",
    "meteofrance_seamless",
)
HOURLY = (
    "temperature_2m",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_direction_120m",
    "wind_direction_180m",
    "wind_gusts_10m",
    "rain",
    "showers",
    "snowfall",
    "cloud_cover_low",
    "cloud_cover",
)
ARCHIVE_HOURLY = (
    "temperature_2m",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "cloud_cover_low",
    "cloud_cover",
)
OUTPUT_COLUMNS = [
    DATE_COL,
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
    "surface_pressure",
    "rain",
    "showers",
    "snowfall",
    "cloud_cover_low",
    "cloud_cover",
]
ALPHA_DEFAULT = 0.143


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def wind_at_height(v_ref: pd.Series, h_ref: float, h_out: float) -> pd.Series:
    return pd.to_numeric(v_ref, errors="coerce").astype(float) * (h_out / h_ref) ** ALPHA_DEFAULT


def normalize_weather_frame(raw: pd.DataFrame, target_date: str, source_name: str) -> pd.DataFrame:
    df = raw.copy()
    if "time" not in df.columns:
        raise ValueError(f"Open-Meteo response for {source_name} has no time column")
    df[DATE_COL] = pd.to_datetime(df["time"])
    df["temperature_80m"] = pd.to_numeric(df.get("temperature_2m"), errors="coerce")
    df["temperature_120m"] = df["temperature_80m"] - 0.3
    if "wind_speed_100m" in df.columns and "wind_speed_80m" not in df.columns:
        df["wind_speed_80m"] = wind_at_height(df["wind_speed_100m"], 100.0, 80.0)
    if "wind_speed_100m" in df.columns and "wind_speed_120m" not in df.columns:
        df["wind_speed_120m"] = wind_at_height(df["wind_speed_100m"], 100.0, 120.0)
    if "wind_speed_100m" in df.columns and "wind_speed_180m" not in df.columns:
        df["wind_speed_180m"] = wind_at_height(df["wind_speed_100m"], 100.0, 180.0)
    if "wind_direction_100m" in df.columns and "wind_direction_80m" not in df.columns:
        df["wind_direction_80m"] = df["wind_direction_100m"]
    if "wind_direction_100m" in df.columns and "wind_direction_120m" not in df.columns:
        df["wind_direction_120m"] = df["wind_direction_100m"]
    if "wind_direction_100m" in df.columns and "wind_direction_180m" not in df.columns:
        df["wind_direction_180m"] = df["wind_direction_100m"]
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[OUTPUT_COLUMNS].copy()
    for col in [c for c in OUTPUT_COLUMNS if c != DATE_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["wind_speed_80m"].isna().all() and not df["wind_speed_120m"].isna().all():
        df["wind_speed_80m"] = df["wind_speed_120m"] / 1.05
    if df["wind_speed_80m"].isna().all() and not df["wind_speed_10m"].isna().all():
        df["wind_speed_80m"] = df["wind_speed_10m"] / 0.75
    df["wind_speed_10m"] = df["wind_speed_10m"].fillna(df["wind_speed_80m"] * 0.75)
    df["wind_speed_120m"] = df["wind_speed_120m"].fillna(df["wind_speed_80m"] * 1.05)
    df["wind_speed_180m"] = df["wind_speed_180m"].fillna(df["wind_speed_120m"] * 1.05)
    if df["wind_direction_80m"].isna().all():
        df["wind_direction_80m"] = df["wind_direction_120m"].fillna(df["wind_direction_10m"])
    df["wind_direction_10m"] = df["wind_direction_10m"].fillna(df["wind_direction_80m"])
    df["wind_direction_120m"] = df["wind_direction_120m"].fillna(df["wind_direction_80m"])
    df["wind_direction_180m"] = df["wind_direction_180m"].fillna(df["wind_direction_120m"])
    df["surface_pressure"] = df["surface_pressure"].fillna(df["pressure_msl"])
    df["temperature_80m"] = df["temperature_80m"].fillna(df["temperature_120m"] + 0.3)
    df["temperature_120m"] = df["temperature_120m"].fillna(df["temperature_80m"] - 0.3)
    zero_fill = ["rain", "showers", "snowfall", "cloud_cover_low", "cloud_cover"]
    for col in zero_fill:
        df[col] = df[col].fillna(0.0)
    for col in ["cloud_cover_low", "cloud_cover"]:
        if df[col].max(skipna=True) > 1.5:
            df[col] = df[col] / 100.0
    numeric_cols = [c for c in df.columns if c != DATE_COL]
    df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both").ffill().bfill()
    if len(df) != 24:
        raise ValueError(f"{source_name} returned {len(df)} rows for {target_date}; expected 24")
    if df[DATE_COL].dt.strftime("%Y-%m-%d").nunique() != 1:
        raise ValueError(f"{source_name} returned multiple dates for {target_date}")
    if df[DATE_COL].dt.strftime("%Y-%m-%d").iloc[0] != target_date:
        raise ValueError(f"{source_name} returned {df[DATE_COL].iloc[0]} instead of {target_date}")
    if df[DATE_COL].duplicated().any():
        raise ValueError(f"{source_name} returned duplicate timestamps")
    if not df[numeric_cols].apply(pd.to_numeric, errors="coerce").notna().all().all():
        raise ValueError(f"{source_name} returned non-numeric weather values")
    return df


def get_openmeteo_json(
    url: str,
    params: dict[str, Any],
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> tuple[dict[str, Any], str]:
    last_exc: Exception | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code < 500 and response.status_code not in {408, 429}:
                response.raise_for_status()
                return response.json(), response.url
            response.raise_for_status()
            return response.json(), response.url
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and status < 500 and status not in {408, 429}:
                raise
            last_exc = exc
        except (requests.ConnectionError, requests.Timeout, ValueError) as exc:
            last_exc = exc
        if attempt < max(0, retries):
            time.sleep(retry_sleep * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def fetch_model(model: str, target_date: str, timeout: int, retries: int, retry_sleep: float) -> tuple[pd.DataFrame, str]:
    params: dict[str, Any] = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": target_date,
        "end_date": target_date,
        "timezone": TIMEZONE,
        "wind_speed_unit": "ms",
        "hourly": ",".join(HOURLY),
        "models": model,
    }
    payload, request_url = get_openmeteo_json(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
    )
    if "hourly" not in payload:
        raise ValueError(f"Open-Meteo response for {model} has no hourly key: {payload}")
    df = pd.DataFrame(payload["hourly"])
    return normalize_weather_frame(df, target_date, model), request_url


def fetch_archive_actual(target_date: str, timeout: int, retries: int, retry_sleep: float) -> tuple[pd.DataFrame, str]:
    params: dict[str, Any] = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": target_date,
        "end_date": target_date,
        "timezone": TIMEZONE,
        "wind_speed_unit": "ms",
        "hourly": ",".join(ARCHIVE_HOURLY),
    }
    payload, request_url = get_openmeteo_json(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
    )
    if "hourly" not in payload:
        raise ValueError(f"Open-Meteo archive response has no hourly key: {payload}")
    df = pd.DataFrame(payload["hourly"])
    return normalize_weather_frame(df, target_date, "archive_actual"), request_url


def actual_cutoff(target_date: str, lag_hours: int, now: pd.Timestamp | None = None) -> pd.Timestamp:
    now = now or pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)
    target_start = pd.Timestamp(target_date)
    target_end = target_start + pd.Timedelta(hours=23)
    cutoff = now.floor("h") - pd.Timedelta(hours=lag_hours)
    return min(max(cutoff, target_start - pd.Timedelta(hours=1)), target_end)


def overlay_actual_rows(forecast: pd.DataFrame, actual: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    out = forecast.copy()
    cols = [c for c in OUTPUT_COLUMNS if c != DATE_COL]
    actual_indexed = actual.set_index(DATE_COL)
    mask = out[DATE_COL] <= cutoff
    if not mask.any():
        return out, 0
    aligned = actual_indexed.reindex(out.loc[mask, DATE_COL])
    replace_mask = aligned[cols].notna().all(axis=1).to_numpy()
    if replace_mask.any():
        row_idx = out.index[mask][replace_mask]
        out.loc[row_idx, cols] = aligned.loc[replace_mask, cols].to_numpy()
    return out, int(replace_mask.sum())


def should_require_actualization(target_date: str, after_hour: int, now: pd.Timestamp | None = None) -> bool:
    now = now or pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)
    threshold = pd.Timestamp(target_date) + pd.Timedelta(hours=after_hour)
    return now >= threshold


def write_source(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.6f")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch live Open-Meteo forecast models for judge day.")
    parser.add_argument("--target-date", default=DEFAULT_TARGET_DATE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=2, help="per-source retries for transient Open-Meteo failures")
    parser.add_argument("--retry-sleep", type=float, default=1.0, help="base seconds between retries")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--keep-going", action="store_true", help="skip failed models if at least one succeeds")
    parser.add_argument("--min-successful-sources", type=int, default=5, help="fail if fewer forecast sources succeed")
    parser.add_argument(
        "--no-actualize-elapsed",
        action="store_true",
        help="do not overlay elapsed target-day hours with Open-Meteo archive/analysis weather when available",
    )
    parser.add_argument(
        "--actual-lag-hours",
        type=int,
        default=1,
        help="elapsed-hour safety lag for archive overlay; default keeps the current incomplete hour on forecast",
    )
    parser.add_argument(
        "--require-actualization-after-hour",
        type=int,
        default=23,
        help="if local time is at/after this target-day hour, fail when no actual rows were overlaid",
    )
    args = parser.parse_args()

    target_date = str(pd.Timestamp(args.target_date).date())
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else WEATHER_DIR / f"openmeteo_forecast_{target_date}"
    )
    metadata_path = output_dir / "metadata.json"

    successes: dict[str, Any] = {}
    failures: dict[str, str] = {}
    actual_meta: dict[str, Any] = {"enabled": not args.no_actualize_elapsed, "used": False}

    actual_df: pd.DataFrame | None = None
    actual_url: str | None = None
    cutoff: pd.Timestamp | None = None
    if not args.no_actualize_elapsed:
        cutoff = actual_cutoff(target_date, max(0, args.actual_lag_hours))
        actual_meta.update({
            "lag_hours": int(max(0, args.actual_lag_hours)),
            "cutoff": str(cutoff),
        })
        try:
            actual_df, actual_url = fetch_archive_actual(
                target_date=target_date,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            actual_meta.update({
                "available": True,
                "source": "https://archive-api.open-meteo.com/v1/archive",
                "url": actual_url,
                "rows": int(len(actual_df)),
            })
        except Exception as exc:
            actual_meta.update({"available": False, "error": repr(exc)})

    for model in args.models:
        try:
            df, url = fetch_model(
                model,
                target_date=target_date,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            overlay_rows = 0
            if actual_df is not None and cutoff is not None:
                df, overlay_rows = overlay_actual_rows(df, actual_df, cutoff)
                actual_meta["used"] = bool(actual_meta["used"] or overlay_rows > 0)
            out_path = output_dir / f"openmeteo_forecast_{target_date}_{model}.csv"
            write_source(df, out_path)
            successes[model] = {
                "path": rel(out_path),
                "sha256": sha256_file(out_path),
                "rows": int(len(df)),
                "time_min": str(df[DATE_COL].min()),
                "time_max": str(df[DATE_COL].max()),
                "url": url,
                "actual_overlay_rows": int(overlay_rows),
            }
            print(f"fetched {model}: {out_path}")
        except Exception as exc:
            failures[model] = repr(exc)
            print(f"failed {model}: {exc}")
            if not args.keep_going:
                raise
        time.sleep(args.sleep)

    if not successes:
        raise RuntimeError(f"No forecast models fetched. Failures: {failures}")

    total_overlay_rows = int(sum(item["actual_overlay_rows"] for item in successes.values()))
    check_errors: list[str] = []
    if len(successes) < args.min_successful_sources:
        check_errors.append(
            f"Only {len(successes)} weather sources succeeded; required at least {args.min_successful_sources}."
        )
    if (
        not args.no_actualize_elapsed
        and should_require_actualization(target_date, args.require_actualization_after_hour)
        and total_overlay_rows == 0
    ):
        check_errors.append(
            "No Open-Meteo archive/analysis rows were overlaid even though the run is "
            f"at/after {target_date} {args.require_actualization_after_hour:02d}:00 {TIMEZONE}."
        )

    metadata = {
        "script": Path(__file__).name,
        "source": "https://api.open-meteo.com/v1/forecast",
        "target_date": target_date,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "hourly": list(HOURLY),
        "request_retries": int(max(0, args.retries)),
        "request_retry_sleep": float(args.retry_sleep),
        "output_dir": rel(output_dir),
        "successes": successes,
        "failures": failures,
        "actualization": actual_meta,
        "checks": {
            "min_successful_sources": int(args.min_successful_sources),
            "successful_sources": int(len(successes)),
            "total_actual_overlay_rows": total_overlay_rows,
            "require_actualization_after_hour": int(args.require_actualization_after_hour),
            "passed": not check_errors,
            "errors": check_errors,
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": rel(output_dir),
        "metadata": rel(metadata_path),
        "success_count": len(successes),
        "failures": failures,
        "check_errors": check_errors,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    if check_errors:
        raise RuntimeError("; ".join(check_errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
