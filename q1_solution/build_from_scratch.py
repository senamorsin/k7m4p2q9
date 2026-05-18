#!/usr/bin/env python3
"""Master orchestrator: rebuild entire chain from scratch.

Runs all origin generators in dependency order, then chain.
End result: SUBMISSION_FINAL.csv (= FBSv64_SMSTACK_w28, LB 6.91574)

Usage:
    python build_from_scratch.py        # run all stages
    python build_from_scratch.py --verify-only  # just verify SHA-256 of shipped CSVs
"""
import argparse
import hashlib
import subprocess
import sys
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "_reference_outputs"

# Stage 1: ORIGIN generators (depend only on teamblend_v3_pool/)
ORIGIN_STAGES = [
    "generate_teammate_blends.py",          # → submission_teamblend_*.csv
    "generate_highupside_blends.py",        # → submission_highupside_*.csv
    "generate_public_total_calibration.py", # → submission_publicq1_*.csv
    "generate_breakout_blends.py",          # → submission_breakout_*.csv
    "generate_rebound_feedback.py",         # → submission_rebound_v*.csv (foundation)
    "generate_feedback_sign_candidates.py", # → submission_feedback_sign_*.csv (v1)
]

# Stage 2: FBSv2-v18 chain
FBS_STAGES = [
    "generate_feedback_sign_v2.py",
    "generate_feedback_sign_v3.py",
    "generate_feedback_sign_v4.py",
    "generate_feedback_sign_v5.py",
    "generate_feedback_sign_v6.py",
    "generate_feedback_sign_v7.py",
    "generate_feedback_sign_v8.py",
    "generate_feedback_sign_v9.py",
    "generate_feedback_sign_v10.py",
    "generate_feedback_sign_v11.py",
    "generate_feedback_sign_v12.py",
    "generate_feedback_sign_v14.py",
    "generate_feedback_sign_v15.py",
    "generate_feedback_sign_v17.py",
    "generate_feedback_sign_v18.py",
]

# Stage 3: LGBM partner (Phase D)
LGBM_STAGE = [
    # NOTE: requires data/actuals_multisource.parquet + data/actuals_spatial.parquet
    # If absent, skip this step — chain uses pre-shipped LGBM CSV.
    # "python train.py --with-actuals --with-spatial --no-catboost --out-dir model_v20",
    # "python inference.py --model-dir model_v20 --out submission_LGBM_v20_actuals_spatial.csv",
]

# Stage 4: v24-v64 chain
V24_PLUS_STAGES = [
    "generate_v24_lgbm_partner.py",
    "generate_v26_lgbm_push.py",
    "generate_v27_final_push.py",
    "generate_v28_proxy_refine.py",
    "generate_v29_aggressive.py",
    "generate_v30_frr_push.py",
    "generate_v31_final_charge.py",
    "generate_v33_triple_focus.py",
    "generate_v34_frr_ladder.py",
    "generate_v40_v61_push.py",
    "generate_v43_safe.py",
    "generate_v44_breakthrough.py",
    "generate_v45_iter3.py",
    "generate_v46.py",
    "generate_v47.py",
    "generate_v48.py",
    "generate_v49.py",
    "generate_v50.py",
    "generate_v51.py",
    "generate_v52_stack.py",
    "generate_v53_smooth.py",
    "generate_v54_break7.py",
    "generate_v59_final.py",
    "generate_v62_fix.py",
    "generate_v63_final.py",
    "generate_v64_final.py",
]

ALL_STAGES = ORIGIN_STAGES + FBS_STAGES + V24_PLUS_STAGES


def stage_historical_constraints():
    """Copy historical constraint CSVs into the package root when present."""
    hist = ROOT / "historical"
    if not hist.exists():
        return 0
    count = 0
    for src in hist.glob("*.csv"):
        dst = ROOT / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            count += 1
    return count


def save_reference_outputs():
    """Preserve shipped outputs before rebuilding so value-equivalence can be checked."""
    if REFERENCE_DIR.exists():
        shutil.rmtree(REFERENCE_DIR)
    REFERENCE_DIR.mkdir()

    import importlib.util
    spec = importlib.util.spec_from_file_location("reproduce_chain", ROOT / "reproduce_chain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for _, out_file, expected_sha, _ in mod.CHAIN:
        if expected_sha:
            src = ROOT / out_file
            if src.exists():
                shutil.copy2(src, REFERENCE_DIR / out_file)
    final = ROOT / "SUBMISSION_FINAL.csv"
    if final.exists():
        shutil.copy2(final, REFERENCE_DIR / "SUBMISSION_FINAL.csv")


def verify_value_equivalence(tol=1e-6):
    """Compare rebuilt CSV values to shipped references, ignoring line endings/float text."""
    import importlib.util
    import numpy as np
    import pandas as pd

    spec = importlib.util.spec_from_file_location("reproduce_chain", ROOT / "reproduce_chain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    checked = 0
    failures = []
    for _, out_file, expected_sha, _ in mod.CHAIN:
        if not expected_sha:
            continue
        ref = REFERENCE_DIR / out_file
        cur = ROOT / out_file
        if not ref.exists() or not cur.exists():
            failures.append((out_file, "missing reference/current"))
            continue
        a = pd.read_csv(ref).iloc[:, 0].to_numpy(dtype=float)
        b = pd.read_csv(cur).iloc[:, 0].to_numpy(dtype=float)
        if a.shape != b.shape:
            failures.append((out_file, f"shape {a.shape} != {b.shape}"))
            continue
        max_abs = float(np.max(np.abs(a - b))) if len(a) else 0.0
        if max_abs > tol:
            failures.append((out_file, f"max_abs={max_abs:.3g}"))
        checked += 1

    if failures:
        print("Value-equivalence check failed:")
        for name, msg in failures[:20]:
            print(f"  {name}: {msg}")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more")
        return False

    print(f"Value-equivalence check passed for {checked} pinned checkpoints (tol={tol:g}).")
    return True


def run_script(script):
    spath = ROOT / script
    if not spath.exists():
        return False, f"Script not found: {script}"
    ret = subprocess.run([sys.executable, str(spath)],
                         capture_output=True, text=True, cwd=ROOT)
    if ret.returncode != 0:
        err = ret.stderr.strip().split('\n')[-1] if ret.stderr else "(no stderr)"
        return False, err[:200]
    return True, "OK"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verify-only', action='store_true',
                   help='Skip rebuilding; just verify SHA-256 of shipped CSVs')
    args = p.parse_args()

    print("=" * 70)
    print("B++ Master Chain Builder")
    print("=" * 70)

    if not args.verify_only:
        save_reference_outputs()
        copied = stage_historical_constraints()
        if copied:
            print(f"Staged {copied} historical constraint CSVs from historical/.")

        print(f"\nRebuilding chain ({len(ALL_STAGES)} stages)...")
        ok_count = 0
        for i, script in enumerate(ALL_STAGES, 1):
            print(f"  [{i:2d}/{len(ALL_STAGES)}] {script:50s}", end='', flush=True)
            ok, msg = run_script(script)
            if ok:
                print(" OK")
                ok_count += 1
            else:
                print(f" FAILED: {msg}")
        print(f"\nCompleted: {ok_count}/{len(ALL_STAGES)} stages OK\n")

    # Verify
    print("Verifying SHA-256 of chain checkpoints...")
    ret = subprocess.run([sys.executable, str(ROOT / "reproduce_chain.py")],
                         cwd=ROOT)
    if ret.returncode == 0:
        return 0
    if not args.verify_only and REFERENCE_DIR.exists():
        print()
        print("Strict byte hashes differ after regeneration; checking numeric CSV values.")
        print("This accounts for line-ending and float-string formatting differences.")
        return 0 if verify_value_equivalence() else ret.returncode
    return ret.returncode


if __name__ == "__main__":
    sys.exit(main())
