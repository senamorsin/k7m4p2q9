#!/usr/bin/env python3
"""Blend current champion (feedback_sign_smooth5_g0p08 = 7.3726) with diverse teammate v3 submissions.

Strategy: champion has best signal but is correlated to local stack. Teammate v77/v85/v87/v89 are
trained on different feature set (their lineage, multi-source OM, covshift) — small dose adds
independent signal that local stack lacks.
"""
import pandas as pd
import numpy as np
import os

CHAMP = 'submission_feedback_sign_smooth5_g0p08.csv'
POOL = 'teamblend_v3_pool'
TARGET_COL = pd.read_csv(CHAMP).columns[0]

champ = pd.read_csv(CHAMP).iloc[:, 0].values

# Pool of diverse partners (corr < 0.995)
PARTNERS = {
    'v89cov':   pd.read_csv(f'{POOL}/submission_v89_covshift.csv').iloc[:, 0].values,
    'v87mlp':   pd.read_csv(f'{POOL}/submission_v87_mlp.csv').iloc[:, 0].values,
    'v85cb':    pd.read_csv(f'{POOL}/submission_v85_catboost.csv').iloc[:, 0].values,
    'v77cap':   pd.read_csv(f'{POOL}/submission_v77_capped.csv').iloc[:, 0].values,
    'wshift05': pd.read_csv(f'{POOL}/submission_winner_shift_-05.csv').iloc[:, 0].values,
    'top6':    pd.read_csv(f'{POOL}/submission_TOP6_MEDIAN.csv').iloc[:, 0].values,
    'v77v71':  pd.read_csv(f'{POOL}/submission_top6_w50_v77w30_v71.csv').iloc[:, 0].values,
    'v71':     pd.read_csv(f'{POOL}/submission_v71_sea_defaultmix_w25_dose35.csv').iloc[:, 0].values,
}

def clip(p):
    return np.clip(p, 0, 89.5)

out = {}

# 1. Champion + low dose of each diverse partner
for tag, p in PARTNERS.items():
    for w in (0.05, 0.08, 0.10, 0.12, 0.15):
        blend = (1-w)*champ + w*p
        out[f'submission_CHAMPv3_{tag}_w{int(w*100):02d}.csv'] = clip(blend)

# 2. Global shift micro-tune (teammate's -0.5 transfers slightly)
for sh in (-0.15, -0.20, -0.25, -0.30, -0.40, -0.50):
    out[f'submission_CHAMPv3_shift_m{abs(sh):.2f}.csv'.replace('.', 'p').replace('csvp','csv')] = clip(champ + sh)

# 3. Triple blend: champ + diverse + top6
for w_d, w_t in [(0.06, 0.08), (0.08, 0.08), (0.10, 0.10), (0.06, 0.12), (0.05, 0.15)]:
    for div_tag in ('v89cov', 'v87mlp', 'v85cb'):
        w_c = 1 - w_d - w_t
        blend = w_c*champ + w_d*PARTNERS[div_tag] + w_t*PARTNERS['top6']
        out[f'submission_CHAMPv3_TRI_{div_tag}_d{int(w_d*100):02d}_t{int(w_t*100):02d}.csv'] = clip(blend)

# 4. MLP/CatBoost/Cov ensemble at low dose (the "diverse triplet")
mlp_cb_cov = (PARTNERS['v87mlp'] + PARTNERS['v85cb'] + PARTNERS['v89cov']) / 3.0
for w in (0.08, 0.12, 0.15, 0.18):
    out[f'submission_CHAMPv3_DIV3_w{int(w*100):02d}.csv'] = clip((1-w)*champ + w*mlp_cb_cov)

# 5. Champ + diverse + shift combo
for div_tag, w in [('v89cov', 0.10), ('v87mlp', 0.10), ('v85cb', 0.10), ('top6', 0.10)]:
    for sh in (-0.15, -0.25, -0.35):
        b = (1-w)*champ + w*PARTNERS[div_tag] + sh
        sh_s = f'm{abs(sh):.2f}'.replace('.', 'p')
        out[f'submission_CHAMPv3_{div_tag}_w{int(w*100):02d}_{sh_s}.csv'] = clip(b)

# 6. Pure shift variants of champion (fine grid)
for sh in (-0.10, -0.20, -0.30, -0.40):
    fn = f'submission_CHAMP_shift_m0p{abs(sh)*100:.0f}.csv'.replace('m0p10', 'm0p10').replace('m0p20','m0p20').replace('m0p30','m0p30').replace('m0p40','m0p40')
    pass  # Already covered above

# 7. Adaptive: shift only HIGH-power rows down (teammate noted high-ws over-pred)
for thr, sh in [(50.0, -1.0), (50.0, -1.5), (60.0, -1.5), (60.0, -2.0), (70.0, -2.0), (40.0, -1.0)]:
    b = champ.copy()
    mask = b > thr
    b[mask] = b[mask] + sh
    b = clip(b)
    out[f'submission_CHAMP_highcap_thr{int(thr)}_sh{abs(sh):.1f}.csv'.replace('.','p')] = b

# 8. Adaptive: low-power floor cap
for thr, mult in [(2.0, 0.8), (3.0, 0.8), (3.0, 0.9), (5.0, 0.85)]:
    b = champ.copy()
    mask = b < thr
    b[mask] = b[mask] * mult
    b = clip(b)
    out[f'submission_CHAMP_lowmult_thr{int(thr)}_m{int(mult*100)}.csv'] = b

# Write all
print(f'Generating {len(out)} blends...')
for fn, arr in out.items():
    fn_clean = fn
    # Fix any double extensions caused by the shift names
    if not fn_clean.endswith('.csv'):
        fn_clean = fn_clean + '.csv'
    pd.DataFrame({TARGET_COL: arr}).to_csv(fn_clean, index=False, encoding='utf-8')

print(f'Done. Files in current dir starting with submission_CHAMPv3_ or submission_CHAMP_')
print(f'CHAMP baseline: mean={champ.mean():.3f}')
print(f'Sample blend means:')
for k in list(out.keys())[:8]:
    print(f'  {k}: mean={out[k].mean():.3f}')
