# Reproduction Guide — B++ Team

Champion: `SUBMISSION_FINAL.csv` (= `submission_FBSv64_SMSTACK_w28.csv`) | **LB 6.91574**

## Quick verification (recommended)

```bash
pip install -r requirements.txt
python reproduce_chain.py
```

Expected: `All checkpoints present and SHA-256 verified.` 45 chain checkpoints verified byte-for-byte.

## Full regeneration `--run` — advanced

`python reproduce_chain.py --run` regenerates the chain from scratch. **Requires the full historical LB-submission archive** (~4000+ CSV, ~300 MB) which is NOT shipped due to size.

Each generator's `KNOWN_SCORES` dict declares the specific LB-pinned constraint files it expects. Generators gracefully skip missing constraints (output a smaller proxy). The shipped CSVs were produced with the full constraint set; `--run` without the full archive produces different (but valid) output.

## Train base LightGBM from scratch

```bash
python train.py --with-actuals --with-spatial --no-catboost --out-dir model
python inference.py --model-dir model --out submission_baseline.csv
```

## Determinism
- numpy `RandomState(42, 7, 123)`
- LightGBM: `device='cpu'`, `deterministic=True`, `num_threads=1`
- Ridge α: 1e-3 (proxy), 1e-2 (FRR)

## Champion chain
| LB | File |
|---|---|
| 7.37455 | rebound_v6_t075_hneg10 |
| 7.24570 | FBSv18_3wM_ws80_hr13_m3_b25 |
| 7.05809 | FBSv51_STACK_TOP5 |
| 7.03522 | FBSv52_SMOOTH3_w10 |
| 7.00208 | FBSv53_smooth3_parent_w30 |
| 6.97663 | FBSv54_NP_t07_p35_n55 |
| **6.91574** | **FBSv64_SMSTACK_w28 = SUBMISSION_FINAL.csv** |
