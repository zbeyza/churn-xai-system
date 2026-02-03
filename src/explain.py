from __future__ import annotations

"""Explainability helpers for the best-performing model."""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import permutation_importance

from .config import REPORTS_DIR


def _is_tree_model(model: object) -> bool:
    return model.__class__.__name__ in {"RandomForestClassifier", "GradientBoostingClassifier"}


def explain_best_model(
    models: Dict[str, object],
    ranked_results: List[dict],
    X_train,
    X_test,
    y_test,
    feature_names: list[str],
) -> None:
    """Generate SHAP plots (or permutation fallback) for the best model."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    best = ranked_results[0]
    best_name = best["name"]
    model = models[best_name]

    print(f"\nBest model by PR-AUC: {best_name}")

    y_proba = model.predict_proba(X_test)[:, 1]
    top_idx = int(np.argmax(y_proba))

    try:
        # Prefer native SHAP explainers when available.
        if _is_tree_model(model):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_train)

        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

        shap.summary_plot(shap_values_to_plot, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_summary.png")
        plt.close()

        base_value = getattr(explainer, "expected_value", 0.0)
        if isinstance(base_value, list):
            base_value = base_value[1]

        if hasattr(shap, "Explanation"):
            expl = shap.Explanation(
                values=shap_values_to_plot[top_idx],
                base_values=base_value,
                data=X_test.iloc[top_idx].values,
                feature_names=feature_names,
            )
            shap.plots.waterfall(expl, show=False)
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "shap_local.png")
            plt.close()
        else:
            # Fallback for older SHAP versions without Explanation.
            shap.summary_plot(
                shap_values_to_plot[top_idx: top_idx + 1],
                X_test.iloc[top_idx: top_idx + 1],
                feature_names=feature_names,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "shap_local.png")
            plt.close()

        mean_abs = np.abs(shap_values_to_plot).mean(axis=0)
        top_idx_sorted = np.argsort(mean_abs)[::-1][:10]
        print("Top 10 features (mean |SHAP|):")
        for i in top_idx_sorted:
            print(f"- {feature_names[i]}: {mean_abs[i]:.4f}")
    except Exception as exc:
        print(f"SHAP failed ({exc}). Falling back to permutation importance.")
        # Model-agnostic fallback when SHAP can't run.
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
        )
        importances = result.importances_mean
        idxs = np.argsort(importances)[::-1][:10]

        plt.figure(figsize=(7, 5))
        plt.barh([feature_names[i] for i in idxs][::-1], importances[idxs][::-1])
        plt.xlabel("Permutation Importance")
        plt.title("Top 10 Features")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "permutation_importance.png")
        plt.close()

        print("Top 10 features (permutation importance):")
        for i in idxs:
            print(f"- {feature_names[i]}: {importances[i]:.4f}")
