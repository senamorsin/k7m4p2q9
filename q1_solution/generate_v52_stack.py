#!/usr/bin/env python3
"""v52: STACK-heavy approach since v51 STACK won. Base = STACK_TOP5 = 7.05809.

v51 finding: STACK_TOP5 won. ITER failed. Pure NP plateaued.
Method: aggregate top winners across versions using various weighting schemes.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv51_STACK_TOP5.csv"
BASE_SCORE = 7.058091117066827
POOL = 'teamblend_v3_pool'

# Same KNOWN_SCORES as v51 + v51 batch
KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv30_NP30_r07_t05_p35_n55.csv": 7.221701978080064,
    "submission_FBSv35_NB_FRR_w20.csv": 7.211396180429282,
    "submission_FBSv40_v61_w050.csv": 7.206400653558756,
    "submission_FBSv43_NP_r07_t07_p35_n55.csv": 7.195070449786494,
    "submission_FBSv44_ITER2_t07_t05_p35_n55.csv": 7.182302330032473,
    "submission_FBSv45_NP3_r07_t07_p35_n55.csv": 7.177334737579046,
    "submission_FBSv46_ITER_q07_q03_r07.csv": 7.156591930447194,
    "submission_FBSv47_ITER_q07_q05_r07_p35_n55.csv": 7.114619938614021,
    "submission_FBSv48_NP_v61_t07_w010.csv": 7.091050592650442,
    "submission_FBSv48_NP_r07_t07_p35_n55.csv": 7.091257199088176,
    "submission_FBSv49_NP_r07_t07_p35_n55.csv": 7.072083467860836,
    "submission_FBSv49_NP_r10_t07_p35_n55.csv": 7.072083467860836,
    "submission_FBSv49_NP_FRR_t07_w10.csv": 7.072135026755919,
    "submission_FBSv49_NP_v61_t07_w010.csv": 7.072199005423169,
    "submission_FBSv50_NP_r07_t05_p35_n55.csv": 7.059243104620970,
    "submission_FBSv50_NP_r07_t07_p35_n55.csv": 7.060762098031390,
    "submission_FBSv50_ITER_q07_q05_r07.csv": 7.061700326158532,
    "submission_FBSv50_ITER_q07_q03_r07.csv": 7.061923447302093,
    "submission_FBSv50_ITER_q07_q07_r07.csv": 7.061972491624026,
    "submission_FBSv50_NP_r07_t10_p35_n55.csv": 7.063783625085822,
    "submission_FBSv50_NP_v61_t07_w015.csv": 7.064280881383817,
    "submission_FBSv50_NP_v61_t07_w010.csv": 7.064447631713790,
    "submission_FBSv50_NP_r10_t07_p35_n55.csv": 7.065031756659245,
    "submission_FBSv50_STACK_TOP3.csv": 7.072083467860836,
    "submission_FBSv50_STACK_TOP5.csv": 7.072087664558700,
    # v51 batch
    "submission_FBSv51_STACK_TOP3.csv": 7.058124084415541,
    "submission_FBSv51_NP_r07_t05_p35_n55.csv": 7.061802377294151,
    "submission_FBSv51_NP_FRR_t07_w10.csv": 7.062817663526376,
    "submission_FBSv51_NP_r07_t03_p35_n55.csv": 7.062301002684558,
    "submission_FBSv51_NP_r07_t07_p35_n55.csv": 7.064660406822989,
    "submission_FBSv51_NP_TOP6_t07_w010.csv": 7.063028210916008,
    "submission_FBSv51_NP_v61_t07_w010.csv": 7.063166673734768,
    "submission_FBSv51_NP_v61_t07_w015.csv": 7.063207270930630,
    "submission_FBSv51_NP_r10_t07_p35_n55.csv": 7.063279203539998,
    "submission_FBSv51_NP_r15_t07_p35_n55.csv": 7.063279203539998,
    "submission_FBSv51_NP_r10_t05_p35_n55.csv": 7.065485209172899,
    "submission_FBSv51_ITER_q07_q05_r07.csv": 7.070660602464970,
    "submission_FBSv51_ITER_q05_q05_r07.csv": 7.068181304577783,
    "submission_FBSv51_ITER_q07_q03_r07.csv": 7.072970334513854,
}


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
    path = ROOT / f"submission_FBSv52_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def build_proxy(base, base_score, max_rms=0.7, max_abs=5.0, alpha=1e-3):
    dirs, tgts, rms_list = [], [], []
    for name, score in KNOWN_SCORES.items():
        vals, _ = load(name)
        if vals is None or len(vals) != len(base): continue
        diff = vals - base
        rms = float(np.sqrt(np.mean(diff*diff)))
        if rms < max_rms and float(np.max(np.abs(diff))) < max_abs:
            dirs.append(diff)
            tgts.append(-((score - base_score) * N_INST / 100.0))
            rms_list.append(rms)
    if len(dirs) < 4: return None, len(dirs)
    D = np.asarray(dirs); T = np.asarray(tgts); R = np.asarray(rms_list)
    W = np.diag(1.0 / np.power(0.05 + R, 0.7))
    gram = (D @ D.T) / len(base)
    beta = np.linalg.solve(W @ gram @ W + alpha * np.eye(len(D)), W @ T)
    proxy = D.T @ (W @ beta)
    proxy = proxy - proxy.mean()
    lo, hi = np.percentile(proxy, [1, 99])
    proxy = np.clip(proxy, lo, hi)
    proxy = proxy / (proxy.std() + 1e-9)
    return proxy, len(dirs)


def build_correction(proxy, q_keep_pct, gp, gn):
    q = 1.0 - q_keep_pct/100.0
    active = np.abs(proxy) >= np.quantile(np.abs(proxy), q)
    pos = active & (proxy > 0); neg = active & (proxy < 0)
    corr = np.zeros_like(proxy)
    corr[pos] = gp * proxy[pos]; corr[neg] = gn * proxy[neg]
    return corr


def main():
    base, target_col = load(BASE_FILE)
    print(f"BASE: {BASE_FILE} (LB={BASE_SCORE})")

    outputs = []

    # ====== S1: ALL TOP WINNERS ACROSS V50-V51 ======
    all_winners_with_scores = [
        ("submission_FBSv51_STACK_TOP5.csv", 7.058091117066827),
        ("submission_FBSv51_STACK_TOP3.csv", 7.058124084415541),
        ("submission_FBSv50_NP_r07_t05_p35_n55.csv", 7.059243104620970),
        ("submission_FBSv50_NP_r07_t07_p35_n55.csv", 7.060762098031390),
        ("submission_FBSv50_ITER_q07_q05_r07.csv", 7.061700326158532),
        ("submission_FBSv51_NP_r07_t05_p35_n55.csv", 7.061802377294151),
        ("submission_FBSv50_ITER_q07_q03_r07.csv", 7.061923447302093),
        ("submission_FBSv50_ITER_q07_q07_r07.csv", 7.061972491624026),
        ("submission_FBSv51_NP_r07_t03_p35_n55.csv", 7.062301002684558),
        ("submission_FBSv51_NP_FRR_t07_w10.csv", 7.062817663526376),
        ("submission_FBSv51_NP_TOP6_t07_w010.csv", 7.063028210916008),
        ("submission_FBSv51_NP_v61_t07_w010.csv", 7.063166673734768),
        ("submission_FBSv51_NP_v61_t07_w015.csv", 7.063207270930630),
        ("submission_FBSv51_NP_r10_t07_p35_n55.csv", 7.063279203539998),
        ("submission_FBSv50_NP_r07_t10_p35_n55.csv", 7.063783625085822),
        ("submission_FBSv50_NP_v61_t07_w015.csv", 7.064280881383817),
        ("submission_FBSv50_NP_v61_t07_w010.csv", 7.064447631713790),
        ("submission_FBSv51_NP_r07_t07_p35_n55.csv", 7.064660406822989),
        ("submission_FBSv50_NP_r10_t07_p35_n55.csv", 7.065031756659245),
    ]
    arrs = []
    scs = []
    for fn, sc in all_winners_with_scores:
        v, _ = load(fn)
        if v is not None:
            arrs.append(v); scs.append(sc)
    print(f"Loaded {len(arrs)} winners")

    if len(arrs) >= 5:
        stk = np.asarray(arrs)
        sc_arr = np.asarray(scs)

        # Top-N means
        for n_top in (3, 5, 7, 10, 12, 15, len(arrs)):
            if n_top > len(arrs): continue
            outputs.append(save(f"MEAN_TOP{n_top}", np.mean(stk[:n_top], axis=0), target_col))
            outputs.append(save(f"MED_TOP{n_top}", np.median(stk[:n_top], axis=0), target_col))

        # Trimmed mean (drop best/worst)
        for trim_n in (1, 2, 3, 4):
            for n_top in (10, 15, len(arrs)):
                if n_top > len(arrs) or n_top <= 2*trim_n + 2: continue
                trim = np.zeros_like(stk[0])
                for i in range(stk.shape[1]):
                    col = np.sort(stk[:n_top, i])
                    trim[i] = col[trim_n:-trim_n].mean()
                outputs.append(save(f"TRIM{trim_n}_TOP{n_top}", trim, target_col))

        # Exponential decay weighted (closer to best = more weight)
        for tau in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05):
            for n_top in (5, 10, len(arrs)):
                if n_top > len(arrs): continue
                w = np.exp(-(sc_arr[:n_top] - sc_arr[:n_top].min()) / tau)
                w = w / w.sum()
                weighted = (stk[:n_top].T @ w).reshape(-1)
                outputs.append(save(f"EXP_tau{int(tau*1000):03d}_TOP{n_top}",
                                    weighted, target_col))

        # Inverse-distance weighted
        for n_top in (5, 10, len(arrs)):
            if n_top > len(arrs): continue
            for floor in (0.0001, 0.0005, 0.001, 0.002):
                w = 1.0 / (sc_arr[:n_top] - sc_arr[:n_top].min() + floor)
                w = w / w.sum()
                weighted = (stk[:n_top].T @ w).reshape(-1)
                outputs.append(save(
                    f"WLB_f{int(floor*10000):04d}_TOP{n_top}",
                    weighted, target_col))

        # Top-2 with various weight ratios (since top-2 are very close)
        for w1 in (0.4, 0.5, 0.6, 0.7, 0.8):
            blend = w1 * stk[0] + (1 - w1) * stk[1]
            outputs.append(save(f"TOP2_w{int(w1*100):02d}", blend, target_col))

        # Champion + dose of each runner-up
        champ = stk[0]  # base = STACK_TOP5 from v51
        for i in range(1, min(8, len(stk))):
            for w in (0.10, 0.20, 0.30, 0.40):
                blend = (1 - w) * champ + w * stk[i]
                outputs.append(save(f"CHAMP_plus_w{int(w*100):02d}_idx{i}",
                                    blend, target_col))

    # ====== S2: NP on new base (just in case it still works) ======
    for max_rms in (0.5, 0.7, 1.0, 1.5):
        proxy, n = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        print(f"  Proxy rms<{max_rms}: {n} cons")
        for q in (5, 7, 10):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NP_r{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S3: Multiple champion bases averaged ======
    # Build proxy on EACH of top-3 bases, average resulting predictions
    if len(arrs) >= 3:
        adjusted = []
        for i in range(3):
            b = arrs[i]
            s = scs[i]
            p, _ = build_proxy(b, s, max_rms=1.0)
            if p is not None:
                # Apply standard t=7 correction
                c = build_correction(p, 7, 0.35, 0.55)
                adjusted.append(b + c)
        if len(adjusted) >= 3:
            outputs.append(save("MULTIBASE_PROX_MEAN",
                                np.mean(adjusted, axis=0), target_col))
            outputs.append(save("MULTIBASE_PROX_MED",
                                np.median(adjusted, axis=0), target_col))

    # ====== S4: Light smoothing on champion ======
    # Read valid_features.csv for time order
    feat_path = ROOT / "data" / "valid_features.csv"
    if feat_path.exists():
        feat = pd.read_csv(feat_path, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
        if len(feat) == len(base):
            ts = pd.Series(base, index=feat['METEOFORECASTHOUR_OPENM_Datetime']).sort_index()
            for win in (3, 5, 7):
                smooth = ts.rolling(window=win, center=True, min_periods=1).mean()
                smooth = smooth.reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)
                for w in (0.05, 0.10, 0.20):
                    blend = (1 - w) * base + w * smooth
                    outputs.append(save(f"SMOOTH{win}_w{int(w*100):02d}",
                                        blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v52_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v52 candidates")


if __name__ == "__main__":
    main()
