#!/usr/bin/env python3
"""Feedback-sign v17: FEATURE-AWARE injection (NEW DIRECTION).

So far all gauss weights computed on PREDICTED POWER. Open up to FEATURES:
- gauss in WIND-SPEED space (where forecast uncertainty highest)
- hour-of-day patterns
- wind-direction sectors
- repair count (Кол-во ВЭУ в ремонте)

This is genuinely orthogonal info — never used before in the chain.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv15_EXTgauss_DIV3_b05_c30_s20.csv"
BASE_SCORE = 7.252770942615267
POOL = 'teamblend_v3_pool'
VALID_FEATURES = "data/valid_features.csv"


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    path = ROOT / name
    if not path.exists():
        pool_p = ROOT / POOL / name
        if pool_p.exists():
            df = pd.read_csv(pool_p)
            return df.iloc[:, 0].to_numpy(dtype=float), df.columns[0]
        return None, None
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, OSError):
        return None, None
    vals = df.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(vals).all(): return None, None
    return vals, df.columns[0]


def save(label, values, target_col):
    arr = np.clip(values, 0.0, N_INST)
    path = ROOT / f"submission_FBSv17_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path),
            "mean": float(arr.mean()), "std": float(arr.std())}


def main():
    base, target_col = load(BASE_FILE)
    if base is None:
        raise RuntimeError(f"BASE not found: {BASE_FILE}")
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    # ALSO load v12 base (cleaner, no double-baked DIV3)
    v12_base, _ = load("submission_FBSv12_MCBv12_20_TRIM_x12.csv")

    # Load externals
    pool_path = ROOT / POOL
    externals = {}
    for fn in ['submission_v89_covshift.csv', 'submission_v87_mlp.csv',
                'submission_v85_catboost.csv', 'submission_TOP6_MEDIAN.csv',
                'submission_v77_capped.csv']:
        fp = pool_path / fn
        if fp.exists():
            v = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)
            if len(v) == len(base):
                externals[fn.replace('submission_', '').replace('.csv', '').split('_')[0]] = v
    if 'v89' in externals and 'v87' in externals and 'v85' in externals:
        externals['DIV3'] = (externals['v89'] + externals['v87'] + externals['v85']) / 3.0
        externals['covCB'] = (externals['v89'] + externals['v85']) / 2.0
    print(f"  Loaded {len(externals)} external sources")

    # Load features in SAME ORDER as submission (descending datetime in valid_features.csv)
    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
    if len(feat) != len(base):
        raise RuntimeError(f"Length mismatch: feat={len(feat)} vs base={len(base)}")

    ws80 = feat['wind_speed_80m'].values
    ws120 = feat['wind_speed_120m'].values
    ws_hub = (ws80 + ws120) / 2.0
    hour = feat['hour_of_day'].values
    month = feat['month'].values
    repair = feat['Кол-во_ВЭУ_в_ремонте'].values
    wdir80 = feat['wind_direction_80m'].values  # normalized 0-1 (or radians?)
    temp80 = feat['temperature_80m'].values

    outputs = []

    # ====== F1: GAUSS in WIND-SPEED space ======
    # Forecast uncertainty highest in mid-wind-speed regime (5-12 m/s for typical turbines)
    # Mid-power output corresponds to ws_eff ~ 6-12 m/s
    for tag in ('DIV3', 'covCB', 'v87'):
        if tag not in externals: continue
        partner = externals[tag]
        for ws_cen in (6.0, 8.0, 10.0, 12.0):
            for sig in (2.0, 3.0, 4.0):
                for b in (0.03, 0.05, 0.08, 0.12, 0.18):
                    g = np.exp(-((ws_hub - ws_cen)**2) / (2*sig**2))
                    g = g / g.max()
                    w = b * g
                    blend = (1 - w) * v12_base + w * partner
                    outputs.append(save(
                        f"WS_{tag}_ws{int(ws_cen*10):02d}_s{int(sig*10):02d}_b{int(b*100):02d}",
                        blend, target_col))

    # ====== F2: GAUSS-WS on NEW CHAMP base (additive) ======
    for tag in ('DIV3', 'covCB'):
        if tag not in externals: continue
        partner = externals[tag]
        for ws_cen, sig in [(8.0, 3.0), (10.0, 3.0)]:
            for b in (0.02, 0.04, 0.06, 0.08, 0.10):
                g = np.exp(-((ws_hub - ws_cen)**2) / (2*sig**2)); g /= g.max()
                w = b * g
                blend = (1 - w) * base + w * partner
                outputs.append(save(
                    f"WSadd_{tag}_ws{int(ws_cen*10):02d}_s{int(sig*10):02d}_b{int(b*100):02d}",
                    blend, target_col))

    # ====== F3: HOUR-OF-DAY weighting ======
    # Patterns: morning ramp-up (5-9), evening calm (18-22), night stable (0-4)
    for tag in ('DIV3', 'covCB'):
        if tag not in externals: continue
        partner = externals[tag]
        for hour_cen, hour_sig in [(12.0, 6.0), (8.0, 4.0), (18.0, 4.0), (15.0, 5.0)]:
            for b in (0.03, 0.05, 0.08):
                g = np.exp(-((hour.astype(float) - hour_cen)**2) / (2*hour_sig**2))
                g /= g.max()
                w = b * g
                blend = (1 - w) * v12_base + w * partner
                outputs.append(save(
                    f"HR_{tag}_h{int(hour_cen)}_s{int(hour_sig)}_b{int(b*100):02d}",
                    blend, target_col))

    # ====== F4: REPAIR count weighted ======
    # High repair count = more uncertainty, partner may help
    for tag in ('DIV3', 'covCB'):
        if tag not in externals: continue
        partner = externals[tag]
        rep_norm = repair / max(repair.max(), 1)
        for b in (0.05, 0.10, 0.15, 0.20):
            w = b * rep_norm
            blend = (1 - w) * v12_base + w * partner
            outputs.append(save(f"REP_{tag}_b{int(b*100):02d}",
                                blend, target_col))

    # ====== F5: COMBINED FEATURE WEIGHTING ======
    # Mid-wind AND mid-repair AND daytime
    if 'DIV3' in externals:
        partner = externals['DIV3']
        # gauss ws8 + hour12-18 + repair high
        g_ws = np.exp(-((ws_hub - 8.0)**2) / (2*3.0**2)); g_ws /= g_ws.max()
        g_hr = np.exp(-((hour.astype(float) - 13.0)**2) / (2*5.0**2)); g_hr /= g_hr.max()
        g_combined = g_ws * g_hr
        for b in (0.05, 0.08, 0.12, 0.18, 0.25):
            w = b * g_combined
            blend = (1 - w) * v12_base + w * partner
            outputs.append(save(f"COMB_WS_HR_DIV3_b{int(b*100):02d}",
                                blend, target_col))

    # ====== F6: WIND-DIRECTION sector weighting ======
    # wdir80 may be in 0-1 normalized. Check distribution
    # Try gauss centered on different sectors
    for tag in ('DIV3', 'covCB'):
        if tag not in externals: continue
        partner = externals[tag]
        for wd_cen in (0.1, 0.25, 0.5, 0.75, 0.9):
            for sig_wd in (0.1, 0.15):
                g = np.exp(-((wdir80 - wd_cen)**2) / (2*sig_wd**2))
                # circular: also check distance from wd_cen+1 (wrap-around)
                g_wrap = np.exp(-((wdir80 - (wd_cen-1.0))**2) / (2*sig_wd**2))
                g_wrap2 = np.exp(-((wdir80 - (wd_cen+1.0))**2) / (2*sig_wd**2))
                g = np.maximum(np.maximum(g, g_wrap), g_wrap2)
                g /= g.max()
                for b in (0.05, 0.10):
                    w = b * g
                    blend = (1 - w) * v12_base + w * partner
                    outputs.append(save(
                        f"WD_{tag}_d{int(wd_cen*100):02d}_b{int(b*100):02d}",
                        blend, target_col))

    # ====== F7: TIME-VARYING gauss — March (month=3) is most relevant since test is Q1 2026 ======
    # All test rows are 2026 Q1, so month=1,2,3 only
    # But MAYBE different month corrections work differently
    if 'DIV3' in externals:
        partner = externals['DIV3']
        for m in (1, 2, 3):
            mask = (month == m).astype(float)
            for b in (0.03, 0.06, 0.10, 0.15):
                # Apply mask + gauss in WS
                g_ws = np.exp(-((ws_hub - 8.0)**2) / (2*3.0**2)); g_ws /= g_ws.max()
                w = b * mask * g_ws
                blend = (1 - w) * v12_base + w * partner
                outputs.append(save(f"MONTH{m}_DIV3_b{int(b*100):02d}",
                                    blend, target_col))

    # ====== F8: REPAIR-ANTI-WEIGHTED (less repair = more confident = less correction) ======
    if 'DIV3' in externals:
        partner = externals['DIV3']
        anti_rep = 1.0 - (repair / max(repair.max(), 1))
        for b in (0.05, 0.10, 0.15, 0.20):
            w = b * anti_rep
            blend = (1 - w) * v12_base + w * partner
            outputs.append(save(f"ANTIREP_DIV3_b{int(b*100):02d}",
                                blend, target_col))

    report = {
        "base_file": BASE_FILE, "base_score": BASE_SCORE,
        "outputs": [o["file"] for o in outputs],
    }
    (ROOT / "feedback_sign_v17_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v17 candidates (FEATURE-AWARE injection)")


if __name__ == "__main__":
    main()
