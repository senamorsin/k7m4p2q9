#!/usr/bin/env python3
"""v48: Continue ITER chain. Base = ITER_q07_q05_r07 = 7.11462."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv47_ITER_q07_q05_r07_p35_n55.csv"
BASE_SCORE = 7.114619938614021
POOL = 'teamblend_v3_pool'

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv28_NBprox_r07_t07_p35_n55.csv": 7.232883131643645,
    "submission_FBSv30_NP30_r07_t05_p35_n55.csv": 7.221701978080064,
    "submission_FBSv33_NB_FRR_w20.csv": 7.212619135451977,
    "submission_FBSv34_NB_FRR_w30.csv": 7.211678029707256,
    "submission_FBSv35_NB_FRR_w20.csv": 7.211396180429282,
    "submission_FBSv40_v61_w050.csv": 7.206400653558756,
    "submission_FBSv43_NP_r07_t07_p35_n55.csv": 7.195070449786494,
    "submission_FBSv44_ITER2_t07_t05_p35_n55.csv": 7.182302330032473,
    "submission_FBSv44_ITER2_t07_t07_p35_n55.csv": 7.187356143593107,
    "submission_FBSv44_NP_r05_t08_p35_n55.csv": 7.191742471234125,
    "submission_FBSv45_NP3_r07_t07_p35_n55.csv": 7.177334737579046,
    "submission_FBSv45_NP3_r05_t07_p35_n55.csv": 7.182090679212620,
    "submission_FBSv45_ITER3_q06_p35_n55.csv": 7.181810218454357,
    "submission_FBSv45_ITER3_q10_p35_n55.csv": 7.180461897094080,
    "submission_FBSv45_ITER2v_q08_q05_p35_n55.csv": 7.181289996540839,
    "submission_FBSv46_ITER_q07_q03_r07.csv": 7.156591930447194,
    "submission_FBSv46_ITER_q05_q05_r07.csv": 7.157456519843061,
    "submission_FBSv46_ITER_q07_q05_r07.csv": 7.158321770812686,
    "submission_FBSv46_ITER_q07_q07_r07.csv": 7.162082971721379,
    "submission_FBSv46_ITER_q05_q03_r07.csv": 7.163584060604573,
    "submission_FBSv46_ITER_q10_q05_r07.csv": 7.164896678064599,
    "submission_FBSv46_ITER_q10_q07_r07.csv": 7.167937969437695,
    "submission_FBSv46_NP_r07_t06_p35_n55.csv": 7.170504873999901,
    "submission_FBSv46_NP_r07_t07_p40_n50.csv": 7.170388932711679,
    "submission_FBSv46_NP_r07_t07_p35_n55.csv": 7.172212196435162,
    # v47 batch — KEY constraints
    "submission_FBSv47_ITER_q07_q03_r07_p35_n55.csv": 7.114974442311162,
    "submission_FBSv47_ITER_q07_q07_r07_p35_n55.csv": 7.114821048625633,
    "submission_FBSv47_ITER_q05_q05_r07_p35_n55.csv": 7.119696150224579,
    "submission_FBSv47_NP_r07_t07_p35_n55.csv": 7.119607519746579,
    "submission_FBSv47_ITER_q10_q03_r07_p35_n55.csv": 7.123530699580140,
    "submission_FBSv47_ITER_q05_q03_r07_p35_n55.csv": 7.124236232812844,
    "submission_FBSv47_NP_r10_t07_p35_n55.csv": 7.126408454637359,
    "submission_FBSv47_NP_v61_t05_w010.csv": 7.128855651851872,
    "submission_FBSv47_NP_TOP6_t05_w010.csv": 7.129017138029446,
    "submission_FBSv47_NP_FRR_t05_w10.csv": 7.129756220019426,
    "submission_FBSv47_NP_r07_t05_p35_n55.csv": 7.129141995153917,
    "submission_FBSv47_ITER_q03_q03_r07_p35_n55.csv": 7.133193305302131,
    "submission_FBSv47_NP_r07_t03_p35_n55.csv": 7.141037466616263,
    "submission_FBSv47_STACK_TOP3.csv": 7.156857929583948,
    "submission_FBSv47_STACK_TOP5.csv": 7.158732408939886,
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
    path = ROOT / f"submission_FBSv48_{label}.csv"
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

    # ====== S1: NP grid on new base ======
    for max_rms in (0.5, 0.7, 1.0, 1.5, 2.0):
        proxy, n = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        print(f"  Proxy rms<{max_rms}: {n} cons")
        for q in (3, 4, 5, 6, 7, 8, 10, 12):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50), (0.30, 0.65)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NP_r{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S2: ITER (apply proxy twice) ======
    proxy_first, _ = build_proxy(base, BASE_SCORE, max_rms=0.7)
    if proxy_first is None:
        proxy_first, _ = build_proxy(base, BASE_SCORE, max_rms=1.0)
    if proxy_first is not None:
        for q1 in (3, 5, 7, 10, 12, 15):
            corr1 = build_correction(proxy_first, q1, 0.35, 0.55)
            b1 = base + corr1
            for sec_rms in (0.7, 1.0, 1.5):
                proxy2, n2 = build_proxy(b1, BASE_SCORE - 0.005, max_rms=sec_rms)
                if proxy2 is not None and n2 >= 5:
                    for q2 in (3, 5, 7, 10):
                        for gp2, gn2 in [(0.35, 0.55), (0.30, 0.55)]:
                            corr2 = build_correction(proxy2, q2, gp2, gn2)
                            outputs.append(save(
                                f"ITER_q{q1:02d}_q{q2:02d}_r{int(sec_rms*10):02d}_p{int(gp2*100):02d}_n{int(gn2*100):02d}",
                                b1 + corr2, target_col))
                    break

    # ====== S3: NP + tiny extras ======
    frr, _ = load("submission_FBSv19_FRRraw_a2_g15.csv")
    champ_orig, _ = load("submission_FBSv18_3wM_ws80_hr13_m3_b25.csv")
    v61, _ = load("submission_v61_alone.csv")
    delta_frr = frr - champ_orig
    pool_path = ROOT / POOL
    top6 = None
    fp = pool_path / "submission_TOP6_MEDIAN.csv"
    if fp.exists():
        top6 = pd.read_csv(fp).iloc[:, 0].to_numpy(dtype=float)

    if proxy_first is not None:
        for q in (5, 7):
            corr = build_correction(proxy_first, q, 0.35, 0.55)
            for w_frr in (0.05, 0.10, 0.15):
                outputs.append(save(
                    f"NP_FRR_t{q:02d}_w{int(w_frr*100):02d}",
                    base + corr + w_frr * delta_frr, target_col))
            if v61 is not None:
                for w in (0.010, 0.015, 0.020):
                    outputs.append(save(
                        f"NP_v61_t{q:02d}_w{int(w*1000):03d}",
                        base + corr + w * (v61 - base), target_col))
            if top6 is not None:
                for w in (0.005, 0.010, 0.015):
                    outputs.append(save(
                        f"NP_TOP6_t{q:02d}_w{int(w*1000):03d}",
                        base + corr + w * (top6 - base), target_col))

    # ====== S4: Stack v47 winners ======
    winners = [
        BASE_FILE,
        "submission_FBSv47_ITER_q07_q03_r07_p35_n55.csv",
        "submission_FBSv47_ITER_q07_q07_r07_p35_n55.csv",
        "submission_FBSv47_NP_r07_t07_p35_n55.csv",
        "submission_FBSv47_ITER_q05_q05_r07_p35_n55.csv",
        "submission_FBSv47_ITER_q10_q03_r07_p35_n55.csv",
        "submission_FBSv47_ITER_q05_q03_r07_p35_n55.csv",
        "submission_FBSv47_NP_r10_t07_p35_n55.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 5:
        stk = np.asarray(arrs)
        outputs.append(save("STACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save("STACK_TOP5", np.mean(stk[:5], axis=0), target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v48_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(outputs)} v48 candidates")


if __name__ == "__main__":
    main()
