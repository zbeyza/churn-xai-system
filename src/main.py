from __future__ import annotations

from pathlib import Path

from .config import DATA_FILE, REPORTS_DIR, ARTIFACTS_DIR
from .download_data import download_data
from .preprocess import preprocess
from .train import train_models
from .evaluate import evaluate_models
from .explain import explain_best_model


def main() -> None:
    if not DATA_FILE.exists():
        print(f"Dataset not found at {DATA_FILE}. Attempting download...")
        download_data()
    else:
        print(f"Using existing dataset at {DATA_FILE}.")

    X_train, X_test, y_train, y_test, feature_names, _ = preprocess()
    models = train_models(X_train, y_train)
    results_sorted = evaluate_models(models, X_test, y_test)
    explain_best_model(models, results_sorted, X_train, X_test, y_test, feature_names)

    print("\nOutputs written to:")
    print(f"- {ARTIFACTS_DIR}")
    print(f"- {REPORTS_DIR}")


if __name__ == "__main__":
    main()
