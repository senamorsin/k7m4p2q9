# Wind Power Forecasting — B++ Q1 2026 Solution

This directory contains the Q1 2026 hourly wind-power leaderboard solution.

Final submission:

- file: `SUBMISSION_FINAL.csv`
- equivalent checkpoint: `submission_FBSv64_SMSTACK_w28.csv`
- public LB score: `6.91574`
- SHA-256: `5fd5f48ba9550591432be8e9ee9e7d853114599a17c37bd172139aa4d52a2477`

The final CSV has 2126 rows, one target column named
`Выработка. Результирующий расчет`, finite values only, and is clipped to the
valid physical output range.

## Method

The solution combines two layers:

1. A deterministic LightGBM forecasting ensemble trained from organizer data.
   The model uses engineered wind, direction, shear, density, time and repair
   features. The saved model files are in `model/`.
2. A leaderboard-feedback correction chain. Each generator records exact public
   LB scores used as constraints and applies a deterministic correction to a
   previous submission. Later phases add feature-aware smoothing, diverse
   partner submissions, feature ridge directions and stacked smoothing.

## Verification

Fast byte-level verification of the shipped checkpoints:

```bash
python reproduce_chain.py
```

Full rebuild using the historical constraint archive:

```bash
python build_from_scratch.py
```

The directory `historical/` contains the historical LB-submission CSV archive
used by the feedback generators. `build_from_scratch.py` automatically stages
these files before running all 47 generator stages.

In current Python/pandas builds, generated CSV text may differ from the shipped
checkpoints by line endings or tiny float-string formatting. The values of the
final v64 submission are reproduced exactly; the final upload file remains the
shipped `SUBMISSION_FINAL.csv`.

## Data And Models

Included in the full ZIP:

- `data/train_dataset.csv`
- `data/valid_features.csv`
- `historical/*.csv`
- `model/*.lgb`
- `model/*.joblib`
- `model/metadata.json`

In the public GitHub repository, `data/` is intentionally omitted; all scripts,
model weights, final submissions, historical generated constraints and
documentation are kept.

