# Churn XAI System

## Overview
End-to-end Python pipeline for the IBM Telco Customer Churn dataset. It downloads the data (non-Kaggle), preprocesses it, trains multiple models, evaluates them, and generates SHAP explainability outputs.

## Dataset
- Dataset: IBM Telco Customer Churn
- Expected local path: `data/telco_churn.csv`
- Expected column name: `Churn`
- Typical original filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

You can either:
1) Set `RAW_URL` in `src/config.py` to a direct CSV URL, or
2) Manually place the CSV at `data/telco_churn.csv`.

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

## Expected outputs
Artifacts and reports are written to:
- `artifacts/` (trained models)
- `reports/roc_curve.png`
- `reports/pr_curve.png`
- `reports/shap_summary.png`
- `reports/shap_local.png`
- `reports/permutation_importance.png` (only if SHAP fails)

If the dataset is missing and `RAW_URL` is not set or fails, the pipeline will instruct you to download the CSV manually.
