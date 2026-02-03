"""Project paths and dataset configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

# Direct CSV URL; can be overridden via CLI or left empty for manual download.
RAW_URL = "https://raw.githubusercontent.com/SaeidRostami/Customer_Churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_FILE = DATA_DIR / "telco_churn.csv"
TARGET_COL = "Churn"
