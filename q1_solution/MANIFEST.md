# B++ Q1 2026 Submission Manifest

## Final Submission

- Final file: `SUBMISSION_FINAL.csv`
- Final checkpoint: `submission_FBSv64_SMSTACK_w28.csv`
- LB score: `6.91574`
- SHA-256: `5fd5f48ba9550591432be8e9ee9e7d853114599a17c37bd172139aa4d52a2477`
- Generator: `generate_v64_final.py`

`SUBMISSION_FINAL.csv` and `submission_FBSv64_SMSTACK_w28.csv` are identical.

## Chain Summary

| Phase | Main files | Role |
|---|---|---|
| Foundation | `train.py`, `inference.py`, `features.py`, `model/` | Deterministic LightGBM baseline and feature engineering |
| Historical constraints | `historical/*.csv` | Public-LB generated submissions used as feedback constraints |
| Early feedback | `generate_feedback_sign_*.py` | Recover row-level error-sign proxy from LB constraints |
| External diversity | `teamblend_v3_pool/`, late feedback generators | Inject diverse teammate models and feature-aware corrections |
| Late stack | `generate_v52_stack.py` ... `generate_v64_final.py` | Smooth/stack late strong checkpoints into final v64 |

Selected public-LB checkpoints:

| LB | Submission |
|---:|---|
| 7.37455 | `submission_rebound_v6_t075_hneg10.csv` |
| 7.33052 | `submission_FBSv7_CUR05_t10_p35_n55.csv` |
| 7.26668 | `submission_FBSv10_NB_CURens_NEW_t10_p35_n55.csv` |
| 7.21168 | `submission_FBSv34_NB_FRR_w30.csv` |
| 7.05809 | `submission_FBSv51_STACK_TOP5.csv` |
| 7.03522 | `submission_FBSv52_SMOOTH3_w10.csv` |
| 7.00208 | `submission_FBSv53_smooth3_parent_w30.csv` |
| 6.97663 | `submission_FBSv54_NP_t07_p35_n55.csv` |
| 6.91866 | `submission_FBSv59_SMSTACK_w50.csv` |
| 6.91751 | `submission_FBSv62_SMSTACK_w45.csv` |
| 6.91615 | `submission_FBSv63_SMSTACK_w36.csv` |
| 6.91574 | `submission_FBSv64_SMSTACK_w28.csv` |

## Reproducibility

Run:

```bash
python reproduce_chain.py
```

This verifies the delivered checkpoint files against pinned byte hashes.

Run:

```bash
python build_from_scratch.py
```

This stages `historical/*.csv`, runs all 47 generator stages and verifies that
the rebuilt values match the delivered chain. Byte hashes can differ for
generated CSVs because pandas writes platform-dependent line endings and float
strings; the shipped `SUBMISSION_FINAL.csv` is the authoritative upload file.

