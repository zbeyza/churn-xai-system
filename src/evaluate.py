from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .config import REPORTS_DIR


def evaluate_models(models: Dict[str, object], X_test, y_test) -> List[dict]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    plt.figure(figsize=(7, 5))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

        y_pred = (y_proba >= 0.5).astype(int)
        print(f"\nModel: {name}")
        print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        print(classification_report(y_test, y_pred, digits=3))

        results.append(
            {
                "name": name,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "y_proba": y_proba,
            }
        )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    for result in results:
        precision, recall, _ = precision_recall_curve(y_test, result["y_proba"])
        plt.plot(recall, precision, label=f"{result['name']} (AP={result['pr_auc']:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pr_curve.png")
    plt.close()

    results_sorted = sorted(results, key=lambda x: x["pr_auc"], reverse=True)
    print("\nSummary (sorted by PR-AUC):")
    for r in results_sorted:
        print(f"- {r['name']}: ROC-AUC={r['roc_auc']:.4f}, PR-AUC={r['pr_auc']:.4f}")

    return results_sorted
