# Churn XAI System

## Overview
This repo is a small, opinionated pipeline for the IBM Telco Customer Churn dataset. It pulls the CSV (non‑Kaggle), cleans it up, trains a few baseline models, evaluates them, and produces SHAP plots for explainability.

I kept the stack minimal (pandas + scikit‑learn + SHAP). You should be able to run everything end‑to‑end with a single command.

## Dataset
- Dataset: IBM Telco Customer Churn
- Expected local path: `data/telco_churn.csv`
- Required column: `Churn`
- Typical original filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

Options for getting the data:
1) Set `RAW_URL` in `src/config.py` to a direct CSV URL, or
2) Manually place the CSV at `data/telco_churn.csv`.

If you download it manually, make sure the file still has the `Churn` column.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run
```bash
python -m src.main
```

You can also override defaults:
```bash
python -m src.main --data data/telco_churn.csv --raw-url "" --skip-download
```

## Expected outputs
- `artifacts/` (trained models)
- `reports/roc_curve.png`
- `reports/pr_curve.png`
- `reports/shap_summary.png`
- `reports/shap_local.png`
- `reports/permutation_importance.png` (only if SHAP fails)

## Notes
- If the HTTP download fails, the pipeline will tell you exactly where to put the CSV and which column is required.
- The `reports/` folder is intentionally simple; no custom colors or seaborn to keep it dependency‑light.
