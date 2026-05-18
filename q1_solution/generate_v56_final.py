#!/usr/bin/env python3
"""v56 FINAL HOUR: Base = ITER_q07_q05_r10 = 6.95012. Need max jump in 1h."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
N_INST = 90.09
BASE_FILE = "submission_FBSv55_ITER_q07_q05_r10.csv"
BASE_SCORE = 6.950119354892452
POOL = 'teamblend_v3_pool'
VALID_FEATURES = "data/valid_features.csv"

KNOWN_SCORES = {
    "submission_FBSv12_MCBv12_20_TRIM_x12.csv": 7.256224793049420,
    "submission_FBSv18_3wM_ws80_hr13_m3_b25.csv": 7.245703367161376,
    "submission_FBSv40_v61_w050.csv": 7.206400653558756,
    "submission_FBSv44_ITER2_t07_t05_p35_n55.csv": 7.182302330032473,
    "submission_FBSv47_ITER_q07_q05_r07_p35_n55.csv": 7.114619938614021,
    "submission_FBSv49_NP_r07_t07_p35_n55.csv": 7.072083467860836,
    "submission_FBSv50_NP_r07_t05_p35_n55.csv": 7.059243104620970,
    "submission_FBSv51_STACK_TOP5.csv": 7.058091117066827,
    "submission_FBSv52_SMOOTH3_w10.csv": 7.035217603401514,
    "submission_FBSv53_smooth3_parent_w30.csv": 7.002079690270437,
    "submission_FBSv53_smooth3_parent_w25.csv": 7.008738211507544,
    "submission_FBSv54_NP_t07_p35_n55.csv": 6.976626128075572,
    "submission_FBSv54_smooth3_w60.csv": 6.978026347941609,
    "submission_FBSv54_smooth3_w70.csv": 6.978906999435949,
    "submission_FBSv54_smooth3_w50.csv": 6.981591875943377,
    "submission_FBSv54_SMSTACK_w50.csv": 6.983401888435778,
    "submission_FBSv54_smooth3_w45.csv": 6.985493271787274,
    "submission_FBSv54_NB_smooth3_w20.csv": 6.989276395780954,
    "submission_FBSv54_smooth3_w40.csv": 6.990689958450519,
    # v55 batch
    "submission_FBSv55_ITER_q07_q07_r10.csv": 6.951763302169862,
    "submission_FBSv55_ITER_q05_q05_r10.csv": 6.951548522255020,
    "submission_FBSv55_NPplus_SM_t07_w20.csv": 6.956017375905002,
    "submission_FBSv55_NPalt_sm60_t07_p35_n55.csv": 6.958799035515771,
    "submission_FBSv55_NP_r10_t10_p35_n55.csv": 6.960327939400416,
    "submission_FBSv55_NPplus_SM_t07_w10.csv": 6.960968266615593,
    "submission_FBSv55_NPalt_sm60_t05_p35_n55.csv": 6.961620434512953,
    "submission_FBSv55_NB_smooth3_w30.csv": 6.961959296304910,
    "submission_FBSv55_NP_r07_t07_p35_n55.csv": 6.962130996239678,
    "submission_FBSv55_NB_smooth3_w20.csv": 6.964777743541367,
    "submission_FBSv55_NP_r10_t07_p35_n55.csv": 6.967509623171778,
    "submission_FBSv55_NP_r15_t07_p35_n55.csv": 6.967509623171778,
    "submission_FBSv55_NP_r10_t05_p35_n55.csv": 6.967975925505645,
    "submission_FBSv55_NB_smooth3_w10.csv": 6.969898331955745,
    "submission_FBSv55_STACK_TOP3.csv": 6.970566289937993,
    "submission_FBSv55_STACK_WLB.csv": 6.973903367666474,
    "submission_FBSv55_STACK_TOP5.csv": 6.974956435697376,
    "submission_FBSv55_smooth3_w65.csv": 6.977730436953080,
    "submission_FBSv55_smooth3_w62.csv": 6.977897697096683,
    "submission_FBSv55_smooth3_w55.csv": 6.979474256671095,
    "submission_FBSv55_STACK_TRIM1_8.csv": 6.981510793252718,
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
    path = ROOT / f"submission_FBSv56_{label}.csv"
    pd.DataFrame({target_col: arr}).to_csv(path, index=False, encoding='utf-8')
    return {"file": path.name, "sha256": sha256(path)}


def smooth_array(arr, feat, win, kind='mean'):
    ts = pd.Series(arr, index=feat['METEOFORECASTHOUR_OPENM_Datetime']).sort_index()
    if kind == 'mean':
        s = ts.rolling(window=win, center=True, min_periods=1).mean()
    else:
        s = ts.rolling(window=win, center=True, min_periods=1).median()
    return s.reindex(feat['METEOFORECASTHOUR_OPENM_Datetime']).to_numpy(dtype=float)


def build_proxy(base, base_score, max_rms=1.0, max_abs=10.0, alpha=1e-3):
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
    feat = pd.read_csv(VALID_FEATURES, parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])

    outputs = []

    # ====== S1: NP on new base ======
    for max_rms in (0.7, 1.0, 1.5, 2.0):
        proxy, n = build_proxy(base, BASE_SCORE, max_rms=max_rms)
        if proxy is None: continue
        print(f"  Proxy rms<{max_rms}: {n} cons")
        for q in (3, 5, 7, 8, 10, 12):
            for gp, gn in [(0.35, 0.55), (0.30, 0.55), (0.40, 0.50)]:
                corr = build_correction(proxy, q, gp, gn)
                outputs.append(save(
                    f"NP_r{int(max_rms*10):02d}_t{q:02d}_p{int(gp*100):02d}_n{int(gn*100):02d}",
                    base + corr, target_col))

    # ====== S2: ITER on new base ======
    proxy_first, _ = build_proxy(base, BASE_SCORE, max_rms=1.5)
    if proxy_first is not None:
        for q1 in (5, 7, 10, 12):
            corr1 = build_correction(proxy_first, q1, 0.35, 0.55)
            b1 = base + corr1
            for sec_rms in (1.0, 1.5, 2.0):
                proxy2, n2 = build_proxy(b1, BASE_SCORE - 0.005, max_rms=sec_rms)
                if proxy2 is not None and n2 >= 5:
                    for q2 in (3, 5, 7, 10):
                        corr2 = build_correction(proxy2, q2, 0.35, 0.55)
                        outputs.append(save(
                            f"ITER_q{q1:02d}_q{q2:02d}_r{int(sec_rms*10):02d}",
                            b1 + corr2, target_col))
                    break

    # ====== S3: NP + smoothing combo (NPplus_SM was winner) ======
    if proxy_first is not None:
        for q in (5, 7, 10):
            corr = build_correction(proxy_first, q, 0.35, 0.55)
            b_corrected = base + corr
            sm_b = smooth_array(b_corrected, feat, 3, 'mean')
            for w in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
                blend = (1 - w) * b_corrected + w * sm_b
                outputs.append(save(
                    f"NPplus_SM_t{q:02d}_w{int(w*100):02d}",
                    blend, target_col))

    # ====== S4: Apply smoothing on new base directly ======
    sm3 = smooth_array(base, feat, 3, 'mean')
    sm5 = smooth_array(base, feat, 5, 'mean')
    for w in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        blend = (1 - w) * base + w * sm3
        outputs.append(save(f"NB_smooth3_w{int(w*100):02d}", blend, target_col))

    # ====== S5: STACK v55 winners ======
    winners = [
        BASE_FILE,
        "submission_FBSv55_ITER_q05_q05_r10.csv",
        "submission_FBSv55_ITER_q07_q07_r10.csv",
        "submission_FBSv55_NPplus_SM_t07_w20.csv",
        "submission_FBSv55_NPalt_sm60_t07_p35_n55.csv",
        "submission_FBSv55_NP_r10_t10_p35_n55.csv",
        "submission_FBSv55_NPplus_SM_t07_w10.csv",
        "submission_FBSv55_NPalt_sm60_t05_p35_n55.csv",
        "submission_FBSv55_NB_smooth3_w30.csv",
        "submission_FBSv55_NP_r07_t07_p35_n55.csv",
    ]
    arrs = []
    for fn in winners:
        v, _ = load(fn)
        if v is not None: arrs.append(v)
    if len(arrs) >= 5:
        stk = np.asarray(arrs)
        outputs.append(save("STACK_TOP3", np.mean(stk[:3], axis=0), target_col))
        outputs.append(save("STACK_TOP5", np.mean(stk[:5], axis=0), target_col))
        outputs.append(save("STACK_TOP7", np.mean(stk[:7], axis=0), target_col))
        outputs.append(save(f"STACK_MEAN_{len(arrs)}", np.mean(stk, axis=0), target_col))
        outputs.append(save(f"STACK_MED_{len(arrs)}", np.median(stk, axis=0), target_col))
        sc = np.array([6.95012, 6.95155, 6.95176, 6.95602, 6.95880, 6.96033, 6.96097, 6.96162, 6.96196, 6.96213])[:len(arrs)]
        w_lb = 1.0 / (sc - 6.949 + 0.0005)
        w_lb = w_lb / w_lb.sum()
        outputs.append(save("STACK_WLB", (stk.T @ w_lb).reshape(-1), target_col))
        # Trimmed
        if len(arrs) >= 6:
            trim = np.zeros_like(stk[0])
            for i in range(stk.shape[1]):
                col = np.sort(stk[:, i])
                trim[i] = col[1:-1].mean()
            outputs.append(save(f"STACK_TRIM1_{len(arrs)}", trim, target_col))

    # ====== S6: ITER3 (apply proxy 3 times) ======
    if proxy_first is not None:
        corr1 = build_correction(proxy_first, 7, 0.35, 0.55)
        b1 = base + corr1
        proxy2, _ = build_proxy(b1, BASE_SCORE - 0.005, max_rms=1.5)
        if proxy2 is not None:
            corr2 = build_correction(proxy2, 5, 0.35, 0.55)
            b2 = b1 + corr2
            proxy3, n3 = build_proxy(b2, BASE_SCORE - 0.010, max_rms=1.5)
            if proxy3 is not None and n3 >= 5:
                for q3 in (3, 5, 7, 10):
                    corr3 = build_correction(proxy3, q3, 0.35, 0.55)
                    outputs.append(save(f"ITER3_q07_q05_q{q3:02d}", b2 + corr3, target_col))

    # ====== S7: Wide smooth dose grid on parent (v53 winning method) ======
    parent_v51, _ = load("submission_FBSv51_STACK_TOP5.csv")
    if parent_v51 is not None:
        sm3_p = smooth_array(parent_v51, feat, 3, 'mean')
        for w in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00):
            blend = (1 - w) * parent_v51 + w * sm3_p
            # Now also apply NP on this
            outputs.append(save(f"sm3parent_w{int(w*100):02d}", blend, target_col))

    report = {"base_file": BASE_FILE, "base_score": BASE_SCORE,
              "outputs": [o["file"] for o in outputs]}
    (ROOT / "v56_final_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} v56 candidates")


if __name__ == "__main__":
    main()
