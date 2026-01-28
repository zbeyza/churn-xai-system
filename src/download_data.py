from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .config import DATA_DIR, DATA_FILE, RAW_URL, TARGET_COL


def _validate_csv(path: Path) -> None:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Downloaded CSV is missing expected column '{TARGET_COL}'. "
            "Please verify you have the IBM Telco Customer Churn dataset."
        )


def download_data(raw_url: Optional[str] = None) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = raw_url if raw_url is not None else RAW_URL

    if not url:
        raise RuntimeError(
            "RAW_URL is empty. Please set RAW_URL in src/config.py or manually "
            f"place the file at {DATA_FILE}. Expected column: '{TARGET_COL}'."
        )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        tmp_path = DATA_FILE.with_suffix(".tmp")
        tmp_path.write_bytes(resp.content)
        _validate_csv(tmp_path)
        tmp_path.replace(DATA_FILE)
        return DATA_FILE
    except Exception as exc:
        raise RuntimeError(
            "HTTP download failed. Please manually download the IBM Telco Customer "
            "Churn CSV and place it at data/telco_churn.csv. Expected column: 'Churn'. "
            "Typical filename: WA_Fn-UseC_-Telco-Customer-Churn.csv. "
            "Suggested search phrase: \"IBM Telco Customer Churn WA_Fn-UseC_-Telco-Customer-Churn.csv\". "
            "The pipeline will run after placing the file."
        ) from exc
