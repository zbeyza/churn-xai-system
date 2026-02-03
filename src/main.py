from __future__ import annotations

"""Command-line entrypoint for the churn pipeline."""

from argparse import ArgumentParser
from pathlib import Path

from .config import DATA_FILE, REPORTS_DIR, ARTIFACTS_DIR, RAW_URL
from .download_data import download_data
from .preprocess import preprocess
from .train import train_models
from .evaluate import evaluate_models
from .explain import explain_best_model


def _build_parser() -> ArgumentParser:
    """CLI flags for data location and download behavior."""
    parser = ArgumentParser(description="IBM Telco churn training + XAI pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_FILE),
        help="Path to telco churn CSV (default: data/telco_churn.csv).",
    )
    parser.add_argument(
        "--raw-url",
        type=str,
        default=RAW_URL,
        help="Direct CSV URL for download. Leave empty to force manual placement.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip HTTP download and assume the CSV already exists.",
    )
    return parser


def main() -> None:
    """Run the full pipeline from raw CSV to reports."""
    parser = _build_parser()
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        if args.skip_download:
            raise RuntimeError(f"Dataset not found at {data_path}. Download was skipped.")
        print(f"Dataset not found at {data_path}. Attempting download...")
        download_data(args.raw_url, data_path)
    else:
        print(f"Using existing dataset at {data_path}.")

    # Orchestrate the full flow: preprocess -> train -> evaluate -> explain.
    X_train, X_test, y_train, y_test, feature_names, _ = preprocess(str(data_path))
    models = train_models(X_train, y_train)
    ranked_results = evaluate_models(models, X_test, y_test)
    explain_best_model(models, ranked_results, X_train, X_test, y_test, feature_names)

    print("\nDone. Outputs written to:")
    print(f"- {ARTIFACTS_DIR}")
    print(f"- {REPORTS_DIR}")


if __name__ == "__main__":
    main()
